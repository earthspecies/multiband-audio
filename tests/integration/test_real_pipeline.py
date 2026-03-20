"""Real end-to-end pipeline tests.

Uses an actual audio file from disk, a realistic 1D CNN backbone with
learnable parameters, and verifies the full extract → fuse → classify
pipeline including backward passes and all paper fusion strategies.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import pytest
import torch
import torch.nn as nn

from multiband_audio import LinearHead, MultibandTransform, MultibandWrapper
from multiband_audio.fusion.registry import FUSION_REGISTRY

DATA_DIR = Path(__file__).parent.parent / "data"
CHIRP_WAV = DATA_DIR / "test_chirp.wav"

NUM_CLASSES = 5
EMBED_DIM = 64


# ---------------------------------------------------------------------------
# Realistic 1D CNN backbone (mimics EfficientNet-style: waveform → embedding)
# ---------------------------------------------------------------------------


class SmallConvBackbone(nn.Module):
    """Lightweight 1D CNN: (N, T) -> (N, embed_dim). Trainable parameters."""

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, embed_dim, kernel_size=8, stride=2, padding=3),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)  # (N, 1, T)
        return self.net(x).squeeze(-1)  # (N, embed_dim)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_real_audio(n_samples: int = 2) -> tuple[torch.Tensor, int]:
    """Load real audio from disk, stack into batch.

    Returns
    -------
    tuple[torch.Tensor, int]
        Waveform batch and sample rate.
    """
    audio, sr = librosa.load(str(CHIRP_WAV), sr=None, mono=True)
    waveform = torch.from_numpy(audio).float()
    batch = waveform.unsqueeze(0).expand(n_samples, -1)
    return batch, sr


def make_wrapper(fusion_name: str, **fusion_kwargs: object) -> MultibandWrapper:
    backbone = SmallConvBackbone(embed_dim=EMBED_DIM)
    head = None if fusion_name == "moe" else LinearHead(EMBED_DIM, NUM_CLASSES)
    return MultibandWrapper(
        backbone=backbone,
        fusion=fusion_name,
        head=head,
        embed_dim=EMBED_DIM,
        freeze_backbone=False,
        **fusion_kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRealAudioLoading:
    def test_load_and_transform(self) -> None:
        """Load real audio, split into bands, verify shape and finite values."""
        waveform, sr = load_real_audio(n_samples=2)
        transform = MultibandTransform(sample_rate=sr)

        bands = transform(waveform)

        assert bands.ndim == 3, f"Expected (N, B, T), got {bands.shape}"
        assert bands.shape[0] == 2
        assert bands.shape[1] >= 1  # at least one band
        assert torch.isfinite(bands).all(), "NaN/Inf in band output"

    def test_expected_num_bands(self) -> None:
        """48 kHz → 3 bands (0–8, 8–16, 16–24 kHz)."""
        waveform, sr = load_real_audio()
        assert sr == 48000
        transform = MultibandTransform(sample_rate=sr)
        bands = transform(waveform)
        assert bands.shape[1] == 3, f"Expected 3 bands for 48 kHz, got {bands.shape[1]}"

    def test_band_values_differ(self) -> None:
        """Bands should not all be identical — each captures different content."""
        waveform, sr = load_real_audio(n_samples=1)
        transform = MultibandTransform(sample_rate=sr)
        bands = transform(waveform)  # (1, B, T)
        # Adjacent bands should differ
        diff = (bands[0, 0] - bands[0, 1]).abs().mean()
        assert diff > 1e-6, "Band 0 and Band 1 are suspiciously identical"


class TestRealForwardPass:
    def test_full_pipeline_gp(self) -> None:
        """Real audio → transform → backbone → gated-pool fusion → logits."""
        waveform, sr = load_real_audio(n_samples=2)
        transform = MultibandTransform(sample_rate=sr)
        wrapper = make_wrapper("gp")

        bands = transform(waveform)
        logits = wrapper(bands)

        assert logits.shape == (2, NUM_CLASSES)
        assert torch.isfinite(logits).all()

    def test_full_pipeline_moe(self) -> None:
        """MoE fusion returns logits directly (no head needed)."""
        waveform, sr = load_real_audio(n_samples=2)
        transform = MultibandTransform(sample_rate=sr)
        wrapper = make_wrapper("moe", num_classes=NUM_CLASSES, num_bands=3)

        bands = transform(waveform)
        logits = wrapper(bands)

        assert logits.shape == (2, NUM_CLASSES)
        assert torch.isfinite(logits).all()

    def test_full_pipeline_sa(self) -> None:
        """Self-attention fusion end-to-end."""
        waveform, sr = load_real_audio(n_samples=2)
        transform = MultibandTransform(sample_rate=sr)
        wrapper = make_wrapper("sa", num_heads=4)

        bands = transform(waveform)
        logits = wrapper(bands)

        assert logits.shape == (2, NUM_CLASSES)
        assert torch.isfinite(logits).all()

    def test_full_pipeline_mp(self) -> None:
        """Mean-pool fusion end-to-end."""
        waveform, sr = load_real_audio(n_samples=2)
        transform = MultibandTransform(sample_rate=sr)
        wrapper = make_wrapper("mp")

        bands = transform(waveform)
        logits = wrapper(bands)

        assert logits.shape == (2, NUM_CLASSES)

    def test_full_pipeline_hybrid_with_real_scores(self) -> None:
        """Hybrid fusion receives real entropy+flux scores from the transform."""
        waveform, sr = load_real_audio(n_samples=2)
        transform = MultibandTransform(sample_rate=sr, return_scores=True)
        wrapper = make_wrapper("hyb", num_handcrafted=2)

        result = transform(waveform)
        bands, scores = result

        assert scores.shape == (2, bands.shape[1], 2), f"Expected scores (N, B, 2), got {scores.shape}"
        assert torch.isfinite(scores).all(), "NaN/Inf in handcrafted scores"

        # Hybrid should use real scores — pass as tuple
        out = wrapper((bands, scores))
        assert out.shape == (2, NUM_CLASSES)
        assert torch.isfinite(out).all()

    def test_hybrid_scores_actually_used(self) -> None:
        """Verify hybrid output differs when scores are vs. are not passed."""
        waveform, sr = load_real_audio(n_samples=2)
        transform = MultibandTransform(sample_rate=sr, return_scores=True)
        wrapper = make_wrapper("hyb", num_handcrafted=2)
        wrapper.eval()

        bands, scores = transform(waveform)

        with torch.no_grad():
            out_with_scores = wrapper((bands, scores))
            out_no_scores = wrapper(bands)  # falls back to embeddings.mean

        diff = (out_with_scores - out_no_scores).abs().mean().item()
        assert diff > 1e-6, "Hybrid output identical with/without scores — scores not being used"


class TestTrainingLoop:
    def test_backward_pass(self) -> None:
        """Full forward + backward + optimizer step with real audio."""
        waveform, sr = load_real_audio(n_samples=2)
        transform = MultibandTransform(sample_rate=sr)
        wrapper = make_wrapper("gp")
        optimizer = torch.optim.Adam(wrapper.parameters(), lr=1e-3)

        bands = transform(waveform)
        labels = torch.zeros(2, dtype=torch.long)

        optimizer.zero_grad()
        logits = wrapper(bands)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_gradient_flows_to_backbone(self) -> None:
        """Gradients reach the backbone when freeze_backbone=False."""
        waveform, sr = load_real_audio(n_samples=2)
        transform = MultibandTransform(sample_rate=sr)

        backbone = SmallConvBackbone(embed_dim=EMBED_DIM)
        head = LinearHead(EMBED_DIM, NUM_CLASSES)
        wrapper = MultibandWrapper(
            backbone=backbone, fusion="mp", head=head, embed_dim=EMBED_DIM, freeze_backbone=False
        )

        bands = transform(waveform)
        labels = torch.zeros(2, dtype=torch.long)
        logits = wrapper(bands)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

        first_conv = list(backbone.net.parameters())[0]
        assert first_conv.grad is not None, "No gradient reached backbone"
        assert first_conv.grad.abs().sum() > 0

    def test_frozen_backbone_no_grad(self) -> None:
        """When freeze_backbone=True, backbone parameters have no gradient."""
        waveform, sr = load_real_audio(n_samples=2)
        transform = MultibandTransform(sample_rate=sr)

        backbone = SmallConvBackbone(embed_dim=EMBED_DIM)
        head = LinearHead(EMBED_DIM, NUM_CLASSES)
        wrapper = MultibandWrapper(backbone=backbone, fusion="mp", head=head, embed_dim=EMBED_DIM, freeze_backbone=True)

        bands = transform(waveform)
        labels = torch.zeros(2, dtype=torch.long)
        logits = wrapper(bands)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

        for p in backbone.parameters():
            assert p.grad is None, "Frozen backbone should have no gradients"

    def test_moe_backward_pass(self) -> None:
        """MoE fusion (logit-level) backward pass."""
        waveform, sr = load_real_audio(n_samples=2)
        transform = MultibandTransform(sample_rate=sr)
        wrapper = make_wrapper("moe", num_classes=NUM_CLASSES, num_bands=3)
        optimizer = torch.optim.Adam(wrapper.parameters(), lr=1e-3)

        bands = transform(waveform)
        labels = torch.zeros(2, dtype=torch.long)
        optimizer.zero_grad()
        logits = wrapper(bands)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)

    def test_hybrid_backward_pass(self) -> None:
        """Hybrid fusion backward pass with real scores."""
        waveform, sr = load_real_audio(n_samples=2)
        transform = MultibandTransform(sample_rate=sr, return_scores=True)

        backbone = SmallConvBackbone(embed_dim=EMBED_DIM)
        head = LinearHead(EMBED_DIM, NUM_CLASSES)
        wrapper = MultibandWrapper(backbone=backbone, fusion="hyb", head=head, embed_dim=EMBED_DIM, num_handcrafted=2)
        optimizer = torch.optim.Adam(wrapper.parameters(), lr=1e-3)

        bands, scores = transform(waveform)
        labels = torch.zeros(2, dtype=torch.long)
        optimizer.zero_grad()
        logits = wrapper((bands, scores))
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)


class TestPaddingMask:
    def test_variable_length_batch(self) -> None:
        """Simulate variable-length audio: second sample is shorter (zero-padded)."""
        waveform, sr = load_real_audio(n_samples=1)
        T = waveform.shape[-1]

        # Simulate a shorter second recording by zero-padding
        short = waveform.clone()
        short[:, T // 2 :] = 0.0
        batch = torch.cat([waveform, short], dim=0)  # (2, T)

        transform = MultibandTransform(sample_rate=sr)
        backbone = SmallConvBackbone(embed_dim=EMBED_DIM)
        head = LinearHead(EMBED_DIM, NUM_CLASSES)
        wrapper = MultibandWrapper(backbone=backbone, fusion="mp", head=head, embed_dim=EMBED_DIM)

        bands = transform(batch)  # (2, B, T_out)
        T_out = bands.shape[-1]

        # Mark second sample's latter half as padding
        padding_mask = torch.zeros(2, T_out, dtype=torch.bool)
        padding_mask[1, T_out // 2 :] = True

        logits = wrapper(bands, padding_mask=padding_mask)
        assert logits.shape == (2, NUM_CLASSES)
        assert torch.isfinite(logits).all()

    def test_mask_shape_tiling(self) -> None:
        """Mask (N, T) is correctly tiled to (N*B, T) inside wrapper."""
        waveform, sr = load_real_audio(n_samples=3)
        transform = MultibandTransform(sample_rate=sr)
        wrapper = make_wrapper("gp")

        bands = transform(waveform)
        N, B, T = bands.shape

        padding_mask = torch.zeros(N, T, dtype=torch.bool)
        padding_mask[0, T // 2 :] = True  # first sample has padding

        # Should not raise, shape should be correct
        logits = wrapper(bands, padding_mask=padding_mask)
        assert logits.shape == (3, NUM_CLASSES)


class TestAllFusionsEndToEnd:
    """Smoke test: all registered fusions run forward+backward on real audio."""

    @pytest.mark.parametrize("fusion_name", list(FUSION_REGISTRY.keys()))
    def test_fusion_forward_backward(self, fusion_name: str) -> None:
        waveform, sr = load_real_audio(n_samples=2)
        transform = MultibandTransform(sample_rate=sr, return_scores=True)
        bands, scores = transform(waveform)
        B = bands.shape[1]

        extra = {}
        if fusion_name == "moe":
            extra = {"num_classes": NUM_CLASSES, "num_bands": B}
        elif fusion_name == "hyb":
            extra = {"num_handcrafted": 2}
        elif fusion_name == "moe_topk":
            extra = {"hidden_dim": 32}
        elif fusion_name == "bilinear":
            extra = {"output_dim": EMBED_DIM, "rank": 16}
        elif fusion_name == "concat_linear":
            extra = {"hidden_dim": 128}

        backbone = SmallConvBackbone(embed_dim=EMBED_DIM)
        head = None if fusion_name == "moe" else LinearHead(EMBED_DIM, NUM_CLASSES)
        wrapper = MultibandWrapper(backbone=backbone, fusion=fusion_name, head=head, embed_dim=EMBED_DIM, **extra)
        optimizer = torch.optim.Adam(wrapper.parameters(), lr=1e-3)

        inp = (bands, scores) if fusion_name == "hyb" else bands
        labels = torch.zeros(2, dtype=torch.long)

        optimizer.zero_grad()
        logits = wrapper(inp)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss), f"{fusion_name}: loss is not finite"
        assert logits.shape == (2, NUM_CLASSES), f"{fusion_name}: expected (2, {NUM_CLASSES}), got {logits.shape}"
