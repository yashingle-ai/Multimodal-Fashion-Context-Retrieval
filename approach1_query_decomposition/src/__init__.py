# --- Environment guard: stub _lzma when absent (some pyenv builds lack it) so
# transformers' optional torchvision import doesn't cascade into failures.
import sys as _sys, types as _types
try:
    import _lzma  # noqa
except ModuleNotFoundError:
    _m = _types.ModuleType("_lzma")
    for _n in ["LZMADecompressor","LZMACompressor","LZMAError","FORMAT_XZ","FORMAT_ALONE","FORMAT_RAW","FORMAT_AUTO","CHECK_CRC64","CHECK_NONE","CHECK_CRC32","CHECK_SHA256","CHECK_ID_MAX","CHECK_UNKNOWN","FILTER_LZMA1","FILTER_LZMA2","FILTER_DELTA","FILTER_X86","MF_HC3","MODE_FAST","PRESET_DEFAULT","PRESET_EXTREME"]:
        setattr(_m,_n,type("_S",(),{}))
    _m.is_check_supported=lambda *a,**k: True
    _m._encode_filter_properties=lambda *a,**k: b""
    _m._decode_filter_properties=lambda *a,**k: {}
    _sys.modules["_lzma"]=_m

"""Approach 1 — Query Decomposition + Facet Scoring for fashion retrieval."""
