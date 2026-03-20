"""Unit tests for MultibandWrapper and LinearHead."""

from __future__ import annotations

import torch
from torch import nn

from multiband_audio.nn.heads import LinearHead
from multiband_audio.nn.wrapper import MultibandWrapper


class ToyBackbone(nn.Module):
    """Maps (N, T) -> (N, embed_dim)."""

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.pool(x).squeeze(1)


class TestLinearHead:
    def test_shape(self) -> None:
        h = LinearHead(256, 10)
        x = torch.randn(2, 256)
        out = h(x)
        assert out.shape == (2, 10)

    def test_num_classes(self) -> None:
        h = LinearHead(128, 5)
        assert h.num_classes == 5


class TestMultibandWrapper:
    def test_with_string_fusion_and_head(self) -> None:
        backbone = ToyBackbone(embed_dim=256)
        head = LinearHead(256, 10)
        w = MultibandWrapper(backbone=backbone, fusion="mp", head=head, embed_dim=256)
        x = torch.randn(2, 3, 16000)
        out = w(x)
        assert out.shape == (2, 10)

    def test_with_string_fusion_no_head(self) -> None:
        backbone = ToyBackbone(embed_dim=256)
        w = MultibandWrapper(backbone=backbone, fusion="gp", embed_dim=256)
        x = torch.randn(2, 3, 16000)
        out = w(x)
        assert out.shape == (2, 256)

    def test_with_moe_fusion(self) -> None:
        backbone = ToyBackbone(embed_dim=256)
        w = MultibandWrapper(
            backbone=backbone,
            fusion="moe",
            embed_dim=256,
            num_classes=10,
            num_bands=3,
        )
        x = torch.randn(2, 3, 16000)
        out = w(x)
        assert out.shape == (2, 10)

    def test_single_band_input(self) -> None:
        backbone = ToyBackbone(embed_dim=256)
        w = MultibandWrapper(backbone=backbone, fusion="mp", embed_dim=256)
        x = torch.randn(2, 16000)  # single band, no B dim
        out = w(x)
        assert out.shape == (2, 256)

    def test_tuple_input(self) -> None:
        backbone = ToyBackbone(embed_dim=256)
        w = MultibandWrapper(backbone=backbone, fusion="mp", embed_dim=256)
        bands = torch.randn(2, 3, 16000)
        scores = torch.randn(2, 3, 2)
        out = w((bands, scores))
        assert out.shape == (2, 256)

    def test_get_band_weights(self) -> None:
        backbone = ToyBackbone(embed_dim=256)
        w = MultibandWrapper(backbone=backbone, fusion="gp", embed_dim=256)
        x = torch.randn(2, 3, 16000)
        w(x)
        weights = w.get_band_weights()
        assert weights is not None
        assert weights.shape == (3,)

    def test_get_fusion_name(self) -> None:
        backbone = ToyBackbone(embed_dim=256)
        w = MultibandWrapper(backbone=backbone, fusion="gp", embed_dim=256)
        assert w.get_fusion_name() == "GatedPoolFusion"
