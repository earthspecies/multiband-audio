"""Fusion modules for combining multi-band embeddings."""

from multiband_audio.fusion.advanced import BilinearPoolingFusion, ConcatLinearFusion, TopKMoEFusion
from multiband_audio.fusion.attention import CrossAttentionFusion, SelfAttentionFusion
from multiband_audio.fusion.base import BaseFusion
from multiband_audio.fusion.gated import GatedPoolFusion, HybridFusion
from multiband_audio.fusion.logit import MoEFusion
from multiband_audio.fusion.pooling import MaxPoolFusion, MeanPoolFusion
from multiband_audio.fusion.registry import FUSION_REGISTRY, build_fusion

__all__ = [
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
]
