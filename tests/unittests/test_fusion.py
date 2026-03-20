"""Unit tests for fusion modules."""

from __future__ import annotations

import pytest
import torch

from multiband_audio.fusion.advanced import BilinearPoolingFusion, ConcatLinearFusion, TopKMoEFusion
from multiband_audio.fusion.attention import CrossAttentionFusion, SelfAttentionFusion
from multiband_audio.fusion.gated import GatedPoolFusion, HybridFusion
from multiband_audio.fusion.logit import MoEFusion
from multiband_audio.fusion.pooling import MaxPoolFusion, MeanPoolFusion
from multiband_audio.fusion.registry import FUSION_REGISTRY, build_fusion

N, B, D = 2, 3, 256


@pytest.fixture
def embeddings() -> torch.Tensor:
    """Return random band embeddings of shape (N, B, D).

    Returns
    -------
    torch.Tensor
        Random ``(N, B, D)`` tensor.
    """
    return torch.randn(N, B, D)


@pytest.fixture
def scores() -> torch.Tensor:
    """Return random handcrafted scores of shape (N, B, 2).

    Returns
    -------
    torch.Tensor
        Random ``(N, B, 2)`` tensor.
    """
    return torch.randn(N, B, 2)


class TestMeanPoolFusion:
    def test_shape(self, embeddings: torch.Tensor) -> None:
        f = MeanPoolFusion()
        out = f(embeddings)
        assert out.shape == (N, D)

    def test_weights(self, embeddings: torch.Tensor) -> None:
        f = MeanPoolFusion()
        f(embeddings)
        w = f.get_band_weights()
        assert w is not None
        assert w.shape == (B,)


class TestMaxPoolFusion:
    def test_shape(self, embeddings: torch.Tensor) -> None:
        f = MaxPoolFusion()
        out = f(embeddings)
        assert out.shape == (N, D)


class TestGatedPoolFusion:
    def test_shape(self, embeddings: torch.Tensor) -> None:
        f = GatedPoolFusion(embed_dim=D)
        out = f(embeddings)
        assert out.shape == (N, D)

    def test_weights_sum_to_one(self, embeddings: torch.Tensor) -> None:
        f = GatedPoolFusion(embed_dim=D)
        f(embeddings)
        w = f.get_band_weights()
        assert abs(w.sum().item() - 1.0) < 1e-5


class TestHybridFusion:
    def test_with_scores(self, embeddings: torch.Tensor, scores: torch.Tensor) -> None:
        f = HybridFusion(embed_dim=D, num_handcrafted=2)
        out = f(embeddings, scores)
        assert out.shape == (N, D)

    def test_without_scores(self, embeddings: torch.Tensor) -> None:
        f = HybridFusion(embed_dim=D, num_handcrafted=2)
        out = f(embeddings, None)
        assert out.shape == (N, D)


class TestSelfAttentionFusion:
    def test_shape(self, embeddings: torch.Tensor) -> None:
        f = SelfAttentionFusion(embed_dim=D, num_heads=4, num_layers=1)
        out = f(embeddings)
        assert out.shape == (N, D)


class TestCrossAttentionFusion:
    def test_shape(self, embeddings: torch.Tensor) -> None:
        f = CrossAttentionFusion(embed_dim=D, num_heads=4)
        out = f(embeddings)
        assert out.shape == (N, D)

    def test_multi_query(self, embeddings: torch.Tensor) -> None:
        f = CrossAttentionFusion(embed_dim=D, num_heads=4, num_queries=4)
        out = f(embeddings)
        assert out.shape == (N, D)


class TestMoEFusion:
    def test_shape(self, embeddings: torch.Tensor) -> None:
        f = MoEFusion(embed_dim=D, num_classes=10, num_bands=B)
        out = f(embeddings)
        assert out.shape == (N, 10)

    def test_returns_logits(self) -> None:
        f = MoEFusion(embed_dim=D, num_classes=10, num_bands=B)
        assert f.returns_logits is True


class TestTopKMoEFusion:
    def test_shape(self, embeddings: torch.Tensor) -> None:
        f = TopKMoEFusion(embed_dim=D, num_experts=4, top_k=2, hidden_dim=128)
        out = f(embeddings)
        assert out.shape == (N, D)


class TestBilinearPoolingFusion:
    def test_shape(self, embeddings: torch.Tensor) -> None:
        f = BilinearPoolingFusion(embed_dim=D, output_dim=D, rank=32)
        out = f(embeddings)
        assert out.shape == (N, D)


class TestConcatLinearFusion:
    def test_shape(self, embeddings: torch.Tensor) -> None:
        f = ConcatLinearFusion(embed_dim=D, hidden_dim=512)
        out = f(embeddings)
        assert out.shape == (N, D)


class TestRegistry:
    def test_paper_keys_registered(self) -> None:
        for key in ("mp", "gp", "moe", "hyb", "sa"):
            assert key in FUSION_REGISTRY, f"Paper key '{key}' missing from registry"

    def test_build_fusion(self) -> None:
        f = build_fusion("gp", embed_dim=D)
        assert isinstance(f, GatedPoolFusion)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown fusion type"):
            build_fusion("does_not_exist")
