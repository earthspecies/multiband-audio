"""Band selectors: entropy, spectral flux, and GMM-based band scoring."""

from multiband_audio.selectors.base import BaseBandSelector
from multiband_audio.selectors.entropy import EntropyBandSelector
from multiband_audio.selectors.flux import FluxBandSelector

__all__ = [
    "BaseBandSelector",
    "EntropyBandSelector",
    "FluxBandSelector",
]

# GMMBandSelector requires scikit-learn (optional dependency)
try:
    from multiband_audio.selectors.gmm import GMMBandSelector  # noqa: F401

    __all__.append("GMMBandSelector")
except ImportError:
    pass
