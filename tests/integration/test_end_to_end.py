"""Integration tests: full pipeline from waveform to output."""

from __future__ import annotations

import torch
from torch import nn

from multiband_audio import (
    LinearHead,
    MultibandSelectiveTransform,
    MultibandTransform,
    MultibandWrapper,
    build_fusion,
)
from multiband_audio.selectors import EntropyBandSelector


class ToyBackbone(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.pool(x).squeeze(1)


class TestEndToEnd:
    def test_transform_to_wrapper(self):
        """Full pipeline: transform -> wrapper -> logits."""
        transform = MultibandTransform(sample_rate=48000, target_sr=16000)
        backbone = ToyBackbone(embed_dim=256)
        head = LinearHead(256, 10)
        wrapper = MultibandWrapper(
            backbone=backbone,
            fusion="gp",
            head=head,
            embed_dim=256,
        )

        waveform = torch.randn(2, 48000)
        bands = transform(waveform)
        logits = wrapper(bands)

        assert logits.shape == (2, 10)

    def test_transform_with_scores_to_hybrid_wrapper(self):
        """Pipeline with handcrafted scores for hybrid fusion."""
        transform = MultibandTransform(
            sample_rate=48000,
            target_sr=16000,
            return_scores=True,
        )
        backbone = ToyBackbone(embed_dim=256)
        wrapper = MultibandWrapper(
            backbone=backbone,
            fusion="hyb",
            embed_dim=256,
            num_handcrafted=2,
        )

        waveform = torch.randn(2, 48000)
        bands, scores = transform(waveform)
        # Hybrid fusion expects scores, wrapper handles tuple
        out = wrapper((bands, scores))
        assert out.shape == (2, 256)

    def test_selective_transform_pipeline(self):
        """Selective transform -> wrapper pipeline."""
        selector = EntropyBandSelector(sample_rate=48000, max_freq_hz=24000)
        transform = MultibandSelectiveTransform(
            band_selector=selector,
            sample_rate=48000,
            target_sr=16000,
            max_bands=2,
        )
        backbone = ToyBackbone(embed_dim=256)
        head = LinearHead(256, 10)
        wrapper = MultibandWrapper(
            backbone=backbone,
            fusion="mp",
            head=head,
            embed_dim=256,
        )

        waveform = torch.randn(2, 48000)
        bands = transform(waveform)
        logits = wrapper(bands)

        assert logits.shape == (2, 10)

    def test_build_fusion_factory(self):
        """Verify all registered fusions can be instantiated."""
        from multiband_audio.fusion.registry import FUSION_REGISTRY

        emb = torch.randn(2, 3, 256)
        for name in FUSION_REGISTRY:
            kwargs = {"embed_dim": 256}
            if name == "moe":
                kwargs["num_classes"] = 10
                kwargs["num_bands"] = 3
            elif name == "hyb":
                kwargs["num_handcrafted"] = 2
            elif name == "moe_topk":
                kwargs["hidden_dim"] = 128
            elif name == "bilinear":
                kwargs["output_dim"] = 256
                kwargs["rank"] = 32
            elif name == "concat_linear":
                kwargs["hidden_dim"] = 512

            f = build_fusion(name, **kwargs)
            out = f(emb)
            assert out.ndim == 2, f"Failed for {name}: expected 2D, got {out.ndim}D"
