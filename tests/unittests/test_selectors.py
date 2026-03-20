"""Unit tests for band selectors."""

from __future__ import annotations

import torch

from multiband_audio.selectors.entropy import EntropyBandSelector
from multiband_audio.selectors.flux import FluxBandSelector


class TestEntropyBandSelector:
    def test_select_top1(self) -> None:
        s = EntropyBandSelector(sample_rate=48000, max_freq_hz=24000)
        spec = torch.randn(2, 128, 64)
        bands = s.select(spec, top_k=1)
        assert len(bands) == 1
        f_low, f_high = bands[0]
        assert f_low >= 0
        assert f_high <= 24000

    def test_select_top3(self) -> None:
        s = EntropyBandSelector(sample_rate=48000, max_freq_hz=24000)
        spec = torch.randn(2, 128, 64)
        bands = s.select(spec, top_k=3)
        assert len(bands) == 3

    def test_forward_returns_single(self) -> None:
        s = EntropyBandSelector(sample_rate=48000, max_freq_hz=24000)
        spec = torch.randn(2, 128, 64)
        band = s(spec)
        assert isinstance(band, tuple)
        assert len(band) == 2

    def test_4d_input(self) -> None:
        s = EntropyBandSelector(sample_rate=48000, max_freq_hz=24000)
        spec = torch.randn(2, 1, 128, 64)  # (B, C, F, T)
        bands = s.select(spec, top_k=1)
        assert len(bands) == 1


class TestFluxBandSelector:
    def test_select_top1(self) -> None:
        s = FluxBandSelector(sample_rate=48000, max_freq_hz=24000)
        spec = torch.randn(2, 128, 64)
        bands = s.select(spec, top_k=1)
        assert len(bands) == 1

    def test_select_top2(self) -> None:
        s = FluxBandSelector(sample_rate=48000, max_freq_hz=24000)
        spec = torch.randn(2, 128, 64)
        bands = s.select(spec, top_k=2)
        assert len(bands) == 2
