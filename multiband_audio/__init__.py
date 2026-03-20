"""Multiband audio transforms and fusion for PyTorch.

Split audio into frequency bands via heterodyning, score bands, and
fuse representations.

Examples
--------
>>> import multiband_audio as mba
>>> import torch
>>> t = mba.MultibandTransform(sample_rate=48000)
>>> x = torch.randn(2, 48000)
>>> bands = t(x)
>>> bands.ndim
3
"""

from multiband_audio._version import __version__
from multiband_audio.data import collate_fn
from multiband_audio.fusion import (
    FUSION_REGISTRY,
    BaseFusion,
    BilinearPoolingFusion,
    ConcatLinearFusion,
    CrossAttentionFusion,
    GatedPoolFusion,
    HybridFusion,
    MaxPoolFusion,
    MeanPoolFusion,
    MoEFusion,
    SelfAttentionFusion,
    TopKMoEFusion,
    build_fusion,
)
from multiband_audio.nn import LinearHead, MultibandWrapper
from multiband_audio.selectors import BaseBandSelector, EntropyBandSelector, FluxBandSelector
from multiband_audio.transforms import (
    BandGridConfig,
    BandScores,
    HeterodyneCfg,
    HeterodyneToBaseband,
    MultibandSelectiveTransform,
    MultibandTransform,
    MultibandTransformDynamic,
    Spectrogram,
    SpectrogramCfg,
    make_band_grid,
)

__all__ = [
    # Version
    "__version__",
    # Data utilities
    "collate_fn",
    # Transforms
    "BandGridConfig",
    "BandScores",
    "HeterodyneCfg",
    "HeterodyneToBaseband",
    "MultibandSelectiveTransform",
    "MultibandTransform",
    "MultibandTransformDynamic",
    "SpectrogramCfg",
    "Spectrogram",
    "make_band_grid",
    # Selectors
    "BaseBandSelector",
    "EntropyBandSelector",
    "FluxBandSelector",
    # Fusion
    "BaseFusion",
    "BilinearPoolingFusion",
    "ConcatLinearFusion",
    "CrossAttentionFusion",
    "FUSION_REGISTRY",
    "GatedPoolFusion",
    "HybridFusion",
    "MaxPoolFusion",
    "MeanPoolFusion",
    "MoEFusion",
    "SelfAttentionFusion",
    "TopKMoEFusion",
    "build_fusion",
    # NN
    "LinearHead",
    "MultibandWrapper",
]
