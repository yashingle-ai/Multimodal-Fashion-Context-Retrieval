"""Query decomposition — the ML idea that differentiates Approach 1.

Vanilla CLIP encodes a whole sentence into ONE vector and behaves like a
"bag of concepts": it cannot reliably bind an attribute to the right object,
so "red tie and white shirt" collapses onto "white tie and red shirt".

We instead parse the natural-language query into structured facets and,
critically, into (color -> garment) BINDINGS. Each binding becomes its own
short phrase ("a red tie", "a white shirt") that is encoded separately, so the
attribute stays glued to its garment. The retriever scores each facet against
the image independently and fuses the scores (see retriever.py).

Two backends:
    - rule-based (default): a curated fashion vocabulary + dependency-free
      heuristics. Deterministic, no API key, fully reproducible.
    - LLM (optional): set OPENAI_API_KEY and pass backend="llm" for a more
      general parser. Falls back to rule-based on any error.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List

# ---------------------------------------------------------------------------
# Curated fashion vocabulary
# ---------------------------------------------------------------------------
COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "black",
    "white", "grey", "gray", "brown", "beige", "navy", "maroon", "teal",
    "olive", "cream", "tan", "gold", "silver", "khaki",
]
MODIFIERS = ["bright", "dark", "light", "pastel", "neon", "deep", "pale", "vivid"]

GARMENTS = [
    "raincoat", "coat", "jacket", "blazer", "hoodie", "sweater", "cardigan",
    "t-shirt", "tshirt", "shirt", "button-down", "button down", "blouse",
    "top", "dress", "skirt", "jeans", "trousers", "pants", "shorts", "suit",
    "tie", "scarf", "hat", "cap", "boots", "shoes", "sneakers", "gown",
    "outerwear", "vest", "jumpsuit", "leggings",
]
ACCESSORIES = ["tie", "scarf", "hat", "cap", "bag", "handbag", "sunglasses",
               "belt", "watch", "gloves", "necklace"]

ENVIRONMENTS = {
    "office": ["office", "workplace", "desk", "meeting room", "corporate"],
    "street": ["street", "urban", "city", "sidewalk", "downtown"],
    "park": ["park", "bench", "garden", "outdoor greenery"],
    "home": ["home", "indoor", "living room", "bedroom", "couch"],
}

STYLES = {
    "formal": ["formal", "business", "professional", "office attire", "elegant"],
    "casual": ["casual", "weekend", "relaxed", "everyday", "streetwear"],
    "sporty": ["sporty", "athletic", "gym", "activewear"],
}

# facet -> weight used during score fusion (tunable)
DEFAULT_WEIGHTS = {
    "bindings": 1.0,     # color+garment pairs — the compositionality signal
    "garments": 0.5,     # garments mentioned without a bound color
    "environment": 0.6,  # scene / place
    "style": 0.5,        # formal / casual vibe
    "full": 0.4,         # the whole original sentence (global context safety net)
}


@dataclass
class ParsedQuery:
    raw: str
    bindings: List[str] = field(default_factory=list)      # ["a red tie", ...]
    garments: List[str] = field(default_factory=list)      # unbound garments
    environment: List[str] = field(default_factory=list)   # ["modern office", ...]
    style: List[str] = field(default_factory=list)         # ["formal setting", ...]

    def facet_prompts(self) -> Dict[str, List[str]]:
        """Group parsed items into the facets the retriever scores."""
        prompts = {
            "bindings": self.bindings,
            "garments": [f"a photo of a {g}" for g in self.garments],
            "environment": [f"a person in a {e}" for e in self.environment],
            "style": [f"{s} outfit" for s in self.style],
            "full": [self.raw],
        }
        return {k: v for k, v in prompts.items() if v}

    def as_dict(self) -> dict:
        return {
            "raw": self.raw,
            "bindings": self.bindings,
            "garments": self.garments,
            "environment": self.environment,
            "style": self.style,
        }


# ---------------------------------------------------------------------------
# Rule-based parser
# ---------------------------------------------------------------------------
def _find_bindings(text: str) -> tuple[list[str], set[str]]:
    """Extract '(modifier) color + garment' phrases near each other.

    Returns the binding phrases and the set of garments already bound (so the
    unbound-garment pass doesn't double-count them).
    """
    bindings, bound = [], set()
    color_alt = "|".join(sorted(COLORS, key=len, reverse=True))
    mod_alt = "|".join(MODIFIERS)
    garm_alt = "|".join(sorted(GARMENTS, key=len, reverse=True))
    # (bright)? (yellow) ... (raincoat)  — allow a few filler words between.
    pat = re.compile(
        rf"\b(?:(?P<mod>{mod_alt})\s+)?(?P<color>{color_alt})\s+(?:\w+\s+){{0,2}}?(?P<garm>{garm_alt})\b",
        re.IGNORECASE,
    )
    for m in pat.finditer(text):
        mod = (m.group("mod") + " ") if m.group("mod") else ""
        color, garm = m.group("color").lower(), m.group("garm").lower()
        bindings.append(f"a {mod}{color} {garm}".strip())
        bound.add(garm)
    return bindings, bound


def parse_rule_based(query: str) -> ParsedQuery:
    text = query.lower()
    bindings, bound = _find_bindings(text)

    garments = []
    for g in GARMENTS:
        if re.search(rf"\b{re.escape(g)}\b", text) and g not in bound:
            garments.append(g)
    # de-dup near-synonyms (tshirt/t-shirt)
    garments = sorted(set(g.replace("tshirt", "t-shirt") for g in garments))

    environment = []
    for canon, kws in ENVIRONMENTS.items():
        if any(re.search(rf"\b{re.escape(k)}\b", text) for k in kws):
            # keep a descriptive phrase if present, else the canonical word
            environment.append(f"modern {canon}" if canon == "office" else canon)

    style = []
    for canon, kws in STYLES.items():
        if any(re.search(rf"\b{re.escape(k)}\b", text) for k in kws):
            style.append(f"{canon} setting" if canon == "formal" else canon)

    return ParsedQuery(
        raw=query,
        bindings=bindings,
        garments=garments,
        environment=environment,
        style=style,
    )


# ---------------------------------------------------------------------------
# Optional LLM parser
# ---------------------------------------------------------------------------
_LLM_SYS = (
    "You decompose a fashion image-search query into JSON with keys: "
    "bindings (list of 'color garment' noun phrases, each preserving which "
    "color belongs to which garment), garments (garments mentioned with no "
    "color), environment (place/scene phrases), style (formal/casual/etc). "
    "Return ONLY JSON."
)


def parse_llm(query: str) -> ParsedQuery:
    from openai import OpenAI  # lazy import

    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _LLM_SYS},
            {"role": "user", "content": query},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    data = json.loads(resp.choices[0].message.content)
    return ParsedQuery(
        raw=query,
        bindings=[f"a {b}" if not b.startswith("a ") else b
                  for b in data.get("bindings", [])],
        garments=data.get("garments", []),
        environment=data.get("environment", []),
        style=data.get("style", []),
    )


def parse_query(query: str, backend: str = "rule") -> ParsedQuery:
    """Parse a query. backend='rule' (default) or 'llm'.

    LLM backend transparently falls back to rule-based on any error/missing key.
    """
    if backend == "llm" and os.getenv("OPENAI_API_KEY"):
        try:
            return parse_llm(query)
        except Exception as e:  # pragma: no cover
            print(f"[query_parser] LLM backend failed ({e}); using rule-based.")
    return parse_rule_based(query)
