"""Configuration dataclasses for multiband transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SpectrogramCfg:
    """Configuration for mel spectrogram computation.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz.
    n_fft : int
        FFT window size.
    hop_length : int
        Hop length between STFT frames.
    n_mels : int
        Number of mel filter banks.
    normalize : bool
        Whether to z-score normalize the spectrogram.

    Examples
    --------
    >>> cfg = SpectrogramCfg(sample_rate=48000)
    >>> cfg.n_mels
    128
    """

    sample_rate: int = 48000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    normalize: bool = True


@dataclass
class HeterodyneCfg:
    """Configuration for heterodyne down-conversion.

    Parameters
    ----------
    baseband_sr : int
        Target baseband sample rate in Hz after down-conversion.
    lowpass_factor : float
        Anti-alias low-pass cutoff as fraction of baseband_sr.

    Examples
    --------
    >>> cfg = HeterodyneCfg(baseband_sr=16000)
    >>> cfg.lowpass_factor
    0.45
    """

    baseband_sr: int = 16000
    lowpass_factor: float = 0.45


@dataclass
class BandGridConfig:
    """Configuration for the frequency band grid.

    Parameters
    ----------
    sample_rate : int
        Original audio sample rate in Hz.
    max_freq_hz : int
        Upper frequency limit in Hz. Must be set explicitly
        (typically ``sample_rate // 2``).
    band_width_hz : int
        Width of each band in Hz.
    step_hz : int
        Step between band start frequencies in Hz.
    min_freq_hz : int
        Lower frequency limit in Hz.

    Examples
    --------
    >>> cfg = BandGridConfig(sample_rate=48000, max_freq_hz=24000)
    >>> cfg.band_width_hz
    8000
    """

    sample_rate: int
    max_freq_hz: int
    band_width_hz: int = 8000
    step_hz: int = 8000
    min_freq_hz: int = 0


def make_band_grid(cfg: BandGridConfig) -> List[Tuple[float, float]]:
    """Build a list of ``(f_low, f_high)`` band edges from a grid config.

    Parameters
    ----------
    cfg : BandGridConfig
        Band grid configuration.

    Returns
    -------
    List[Tuple[float, float]]
        List of ``(f_low, f_high)`` tuples in Hz.

    Examples
    --------
    >>> cfg = BandGridConfig(sample_rate=48000, max_freq_hz=24000)
    >>> make_band_grid(cfg)
    [(0.0, 8000.0), (8000.0, 16000.0), (16000.0, 24000.0)]
    """
    bands: List[Tuple[float, float]] = []
    f = cfg.min_freq_hz
    while f < cfg.max_freq_hz:
        f_low = f
        f_high = min(f + cfg.band_width_hz, cfg.max_freq_hz)
        bands.append((float(f_low), float(f_high)))
        f += cfg.step_hz
    return bands
