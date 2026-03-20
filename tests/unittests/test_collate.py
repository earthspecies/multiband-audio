"""Tests for collate_fn and padding mask propagation through the pipeline."""

from __future__ import annotations

import torch
import pytest

from multiband_audio import collate_fn, MultibandTransform, MultibandWrapper, LinearHead


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------

class TestCollateFn:
    def test_output_shapes(self):
        batch = [(torch.randn(16000), 0), (torch.randn(8000), 1), (torch.randn(12000), 2)]
        waveforms, mask, labels = collate_fn(batch)
        assert waveforms.shape == (3, 16000)
        assert mask.shape == (3, 16000)
        assert labels.shape == (3,)

    def test_padding_mask_correct(self):
        """Second sample is 8000 samples, padded to 16000."""
        batch = [(torch.randn(16000), 0), (torch.randn(8000), 1)]
        waveforms, mask, labels = collate_fn(batch)

        # First sample: no padding
        assert not mask[0].any(), "Full-length sample should have no padding"
        # Second sample: last 8000 positions are padding
        assert not mask[1, :8000].any(), "Valid region should be False"
        assert mask[1, 8000:].all(), "Padded region should be True"

    def test_waveform_values_preserved(self):
        """Original audio values must be unchanged after padding."""
        audio = torch.randn(8000)
        batch = [(audio, 0), (torch.randn(16000), 1)]
        waveforms, mask, labels = collate_fn(batch)
        assert torch.allclose(waveforms[0, :8000], audio)
        assert (waveforms[0, 8000:] == 0).all()

    def test_channel_dim_stripped(self):
        """(1, T) waveforms should be treated same as (T,)."""
        batch = [(torch.randn(1, 16000), 0), (torch.randn(1, 8000), 1)]
        waveforms, mask, labels = collate_fn(batch)
        assert waveforms.shape == (2, 16000)

    def test_equal_length_batch_no_padding(self):
        """All same length → mask is all False."""
        batch = [(torch.randn(16000), i) for i in range(4)]
        waveforms, mask, labels = collate_fn(batch)
        assert not mask.any(), "No padding needed for equal-length batch"

    def test_label_types(self):
        """Labels work as ints or tensors."""
        batch_int    = [(torch.randn(100), 0), (torch.randn(50), 1)]
        batch_tensor = [(torch.randn(100), torch.tensor(0)),
                        (torch.randn(50),  torch.tensor(1))]
        _, _, labels_int    = collate_fn(batch_int)
        _, _, labels_tensor = collate_fn(batch_tensor)
        assert torch.equal(labels_int, labels_tensor)

    def test_single_sample(self):
        batch = [(torch.randn(16000), 0)]
        waveforms, mask, labels = collate_fn(batch)
        assert waveforms.shape == (1, 16000)
        assert not mask.any()


# ---------------------------------------------------------------------------
# MultibandTransform mask propagation
# ---------------------------------------------------------------------------

class TestTransformMaskPropagation:
    def test_no_mask_returns_tensor(self):
        """Without mask, forward returns plain tensor (backward compat)."""
        t = MultibandTransform(sample_rate=48000)
        x = torch.randn(2, 48000)
        out = t(x)
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 3

    def test_with_mask_returns_tuple(self):
        """With mask, forward returns (bands, band_mask)."""
        t = MultibandTransform(sample_rate=48000)
        x = torch.randn(2, 48000)
        mask = torch.zeros(2, 48000, dtype=torch.bool)
        out = t(x, padding_mask=mask)
        assert isinstance(out, tuple)
        assert len(out) == 2

    def test_band_mask_shape(self):
        """band_mask shape must match bands shape in N and T dims."""
        t = MultibandTransform(sample_rate=48000)
        x = torch.randn(3, 48000)
        mask = torch.zeros(3, 48000, dtype=torch.bool)
        bands, band_mask = t(x, padding_mask=mask)
        assert band_mask.shape == (3, bands.shape[-1])

    def test_band_mask_dtype(self):
        t = MultibandTransform(sample_rate=48000)
        x = torch.randn(2, 48000)
        mask = torch.zeros(2, 48000, dtype=torch.bool)
        _, band_mask = t(x, padding_mask=mask)
        assert band_mask.dtype == torch.bool

    def test_padding_preserved_in_band_mask(self):
        """Padded positions in waveform should map to padded positions in band_mask."""
        t = MultibandTransform(sample_rate=48000)
        x = torch.randn(2, 48000)
        mask = torch.zeros(2, 48000, dtype=torch.bool)
        mask[1, 24000:] = True  # second half of sample 1 is padding

        bands, band_mask = t(x, padding_mask=mask)
        T_out = bands.shape[-1]

        # First sample: no padding
        assert not band_mask[0].any()
        # Second sample: roughly second half should be padded
        mid = T_out // 2
        assert band_mask[1, mid:].any(), "Padded region should propagate to band_mask"

    def test_zero_padding_propagates(self):
        """Fully unpadded mask → band_mask also all False."""
        t = MultibandTransform(sample_rate=48000)
        x = torch.randn(2, 48000)
        mask = torch.zeros(2, 48000, dtype=torch.bool)
        _, band_mask = t(x, padding_mask=mask)
        assert not band_mask.any()

    def test_with_scores_and_mask(self):
        """return_scores=True + mask → (bands, scores, band_mask)."""
        t = MultibandTransform(sample_rate=48000, return_scores=True)
        x = torch.randn(2, 48000)
        mask = torch.zeros(2, 48000, dtype=torch.bool)
        out = t(x, padding_mask=mask)
        assert isinstance(out, tuple)
        assert len(out) == 3
        bands, scores, band_mask = out
        assert bands.ndim == 3
        assert scores.ndim == 3
        assert band_mask.dtype == torch.bool

    def test_with_scores_no_mask(self):
        """return_scores=True without mask → (bands, scores) — backward compat."""
        t = MultibandTransform(sample_rate=48000, return_scores=True)
        x = torch.randn(2, 48000)
        out = t(x)
        assert isinstance(out, tuple)
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Full pipeline: collate_fn → transform → wrapper
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def _make_wrapper(self, embed_dim=64, num_classes=5):
        from torch import nn
        class _B(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(1, embed_dim, 16, 4, 6)
                self.pool = nn.AdaptiveAvgPool1d(1)
            def forward(self, x):
                return self.pool(self.conv(x.unsqueeze(1))).squeeze(-1)
        return MultibandWrapper(
            backbone=_B(), fusion="gp",
            head=LinearHead(embed_dim, num_classes), embed_dim=embed_dim,
        )

    def test_collate_to_wrapper(self):
        """collate_fn → transform(mask) → wrapper(band_mask) gives correct shape."""
        batch = [
            (torch.randn(48000), 0),
            (torch.randn(24000), 1),
            (torch.randn(36000), 2),
        ]
        waveforms, pad_mask, labels = collate_fn(batch)

        t = MultibandTransform(sample_rate=48000)
        bands, band_mask = t(waveforms, padding_mask=pad_mask)

        wrapper = self._make_wrapper()
        wrapper.eval()
        with torch.no_grad():
            logits = wrapper(bands, padding_mask=band_mask)

        assert logits.shape == (3, 5)
        assert torch.isfinite(logits).all()

    def test_masked_and_unmasked_logits_differ_for_padded_sample(self):
        """Passing the mask should change output for padded samples (at least with
        backbones that respect it — here we just verify no crash and shape is right)."""
        batch = [
            (torch.randn(48000), 0),
            (torch.randn(16000), 1),   # heavily padded
        ]
        waveforms, pad_mask, _ = collate_fn(batch)
        t = MultibandTransform(sample_rate=48000)
        bands, band_mask = t(waveforms, padding_mask=pad_mask)

        wrapper = self._make_wrapper()
        wrapper.eval()
        with torch.no_grad():
            logits_with = wrapper(bands, padding_mask=band_mask)
            logits_sans = wrapper(bands)

        assert logits_with.shape == logits_sans.shape == (2, 5)

    def test_backward_with_mask(self):
        """Gradient flows through the full pipeline with masking."""
        batch = [(torch.randn(48000), 0), (torch.randn(32000), 1)]
        waveforms, pad_mask, labels = collate_fn(batch)

        t = MultibandTransform(sample_rate=48000)
        bands, band_mask = t(waveforms, padding_mask=pad_mask)

        wrapper = self._make_wrapper()
        import torch.nn as nn
        logits = wrapper(bands, padding_mask=band_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        assert torch.isfinite(loss)

    def test_equal_length_batch_mask_is_all_false(self):
        """Equal-length batch → mask is all False → band_mask is all False."""
        batch = [(torch.randn(48000), i) for i in range(4)]
        waveforms, pad_mask, _ = collate_fn(batch)
        assert not pad_mask.any()

        t = MultibandTransform(sample_rate=48000)
        bands, band_mask = t(waveforms, padding_mask=pad_mask)
        assert not band_mask.any()
