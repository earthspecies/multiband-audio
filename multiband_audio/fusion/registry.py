"""Fusion module registry and factory function."""

from __future__ import annotations

import inspect
from typing import Any

from multiband_audio.fusion.advanced import BilinearPoolingFusion, ConcatLinearFusion, TopKMoEFusion
from multiband_audio.fusion.attention import CrossAttentionFusion, SelfAttentionFusion
from multiband_audio.fusion.base import BaseFusion
from multiband_audio.fusion.gated import GatedPoolFusion, HybridFusion
from multiband_audio.fusion.logit import MoEFusion
from multiband_audio.fusion.pooling import MaxPoolFusion, MeanPoolFusion

FUSION_REGISTRY: dict[str, type[BaseFusion]] = {
    # Paper names (primary keys)
    "mp": MeanPoolFusion,  # Mean-Pool
    "gp": GatedPoolFusion,  # Gated-Pool
    "moe": MoEFusion,  # Mixture-of-Experts
    "hyb": HybridFusion,  # Hybrid
    "sa": SelfAttentionFusion,  # Self-Attention
    # Additional (not in paper)
    "max_pool": MaxPoolFusion,
    "cross_attention": CrossAttentionFusion,
    "moe_topk": TopKMoEFusion,
    "bilinear": BilinearPoolingFusion,
    "concat_linear": ConcatLinearFusion,
}


def build_fusion(name: str, **kwargs: Any) -> BaseFusion:
    """Build a fusion module by name.

    Parameters
    ----------
    name : str
        Name of the fusion method (see :data:`FUSION_REGISTRY`).
    **kwargs : Any
        Arguments forwarded to the fusion class constructor.

    Returns
    -------
    BaseFusion
        Instantiated fusion module.

    Raises
    ------
    ValueError
        If ``name`` is not in the registry.

    Examples
    --------
    >>> f = build_fusion("gp", embed_dim=256)
    >>> f.__class__.__name__
    'GatedPoolFusion'
    """
    if name not in FUSION_REGISTRY:
        raise ValueError(f"Unknown fusion type: {name}. Available: {list(FUSION_REGISTRY.keys())}")
    cls = FUSION_REGISTRY[name]
    # Filter kwargs to only those accepted by the constructor
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_keyword:
        filtered = kwargs
    else:
        filtered = {k: v for k, v in kwargs.items() if k in valid_params}
    return cls(**filtered)
