"""Unit tests for transforms."""

from __future__ import annotations

import torch

from multiband_audio._configs import BandGridConfig, HeterodyneCfg, SpectrogramCfg, make_band_grid
from multiband_audio.transforms._heterodyne import HeterodyneToBaseband
from multiband_audio.transforms._spectrogram import Spectrogram
from multiband_audio.transforms.multiband import MultibandTransform, MultibandTransformDynamic


class TestBandGridConfig:
    def test_make_band_grid_basic(self):
        cfg = BandGridConfig(sample_rate=48000, max_freq_hz=24000)
        bands = make_band_grid(cfg)
        assert len(bands) == 3
        assert bands[0] == (0.0, 8000.0)
        assert bands[-1] == (16000.0, 24000.0)

    def test_make_band_grid_with_step(self):
        cfg = BandGridConfig(sample_rate=48000, max_freq_hz=24000, band_width_hz=8000, step_hz=4000)
        bands = make_band_grid(cfg)
        # Overlapping bands: 0-8k, 4k-12k, 8k-16k, 12k-20k, 16k-24k, 20k-24k
        assert len(bands) == 6
        assert bands[0] == (0.0, 8000.0)
        assert bands[1] == (4000.0, 12000.0)

    def test_make_band_grid_empty(self):
        cfg = BandGridConfig(sample_rate=48000, max_freq_hz=0)
        bands = make_band_grid(cfg)
        assert bands == []


class TestSpectrogram:
    def test_shape_2d_input(self):
        spec = Spectrogram(SpectrogramCfg(sample_rate=16000, n_mels=64))
        x = torch.randn(2, 16000)
        out = spec(x)
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[1] == 64  # n_mels

    def test_shape_3d_input(self):
        spec = Spectrogram(SpectrogramCfg(sample_rate=16000, n_mels=64))
        x = torch.randn(2, 1, 16000)
        out = spec(x)
        assert out.ndim == 3
        assert out.shape[0] == 2

    def test_short_signal_padded(self):
        spec = Spectrogram(SpectrogramCfg(sample_rate=16000, n_fft=1024))
        x = torch.randn(1, 100)  # shorter than n_fft
        out = spec(x)
        assert out.ndim == 3


class TestHeterodyne:
    def test_basic_downsample(self):
        het = HeterodyneToBaseband(HeterodyneCfg(baseband_sr=16000))
        x = torch.randn(2, 1, 48000)
        out, sr = het(x, sr_in=48000, f_low=8000.0, f_high=16000.0)
        assert sr == 16000
        assert out.shape[0] == 2

    def test_same_sr_no_resample(self):
        het = HeterodyneToBaseband(HeterodyneCfg(baseband_sr=16000))
        x = torch.randn(1, 1, 16000)
        out, sr = het(x, sr_in=16000, f_low=0.0, f_high=8000.0)
        assert sr == 16000
        assert out.shape[-1] == 16000

    def test_invalid_bandwidth(self):
        het = HeterodyneToBaseband(HeterodyneCfg(baseband_sr=16000))
        x = torch.randn(1, 1, 16000)
        try:
            het(x, sr_in=16000, f_low=8000.0, f_high=8000.0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestMultibandTransform:
    def test_basic_output_shape(self):
        t = MultibandTransform(sample_rate=48000, target_sr=16000)
        x = torch.randn(2, 48000)
        out = t(x)
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[1] == 3  # 3 bands: 0-8k, 8k-16k, 16k-24k

    def test_num_bands(self):
        t = MultibandTransform(sample_rate=48000, target_sr=16000)
        assert t.get_num_bands() == 3

    def test_band_info(self):
        t = MultibandTransform(sample_rate=48000, target_sr=16000)
        info = t.get_band_info()
        assert len(info) == 3
        assert info[0] == (0.0, 8000.0)

    def test_with_scores(self):
        t = MultibandTransform(sample_rate=48000, target_sr=16000, return_scores=True)
        x = torch.randn(2, 48000)
        bands, scores = t(x)
        assert bands.ndim == 3
        assert scores.ndim == 3
        assert scores.shape[0] == 2
        assert scores.shape[1] == 3  # num_bands

    def test_custom_max_freq(self):
        t = MultibandTransform(sample_rate=48000, target_sr=16000, max_freq=16000)
        assert t.get_num_bands() == 2  # 0-8k, 8k-16k


class TestMultibandTransformDynamic:
    def test_basic(self):
        t = MultibandTransformDynamic(target_sr=16000)
        x = torch.randn(2, 48000)
        bands, scores, info = t(x, sample_rate=48000)
        assert bands.ndim == 3
        assert bands.shape[0] == 2
        assert scores is None
        assert len(info) == 3

    def test_low_sr_fallback(self):
        t = MultibandTransformDynamic(target_sr=16000, band_width=8000)
        x = torch.randn(1, 600)
        bands, scores, info = t(x, sample_rate=600)
        assert bands.ndim == 3
        assert len(info) == 1  # single fallback band
        assert info[0] == (0.0, 300.0)  # nyquist = 300
