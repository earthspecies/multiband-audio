"""Heterodyne down-conversion to baseband."""

from __future__ import annotations

from typing import Literal, Tuple

import librosa
import torch
import torchaudio
from torch import nn

from multiband_audio._configs import HeterodyneCfg


def resample_audio(
    audio: torch.Tensor,
    orig_sr: int,
    target_sr: int,
    method: Literal["librosa", "torchaudio"] = "librosa",
) -> torch.Tensor:
    """Resample audio tensor.

    Parameters
    ----------
    audio : torch.Tensor
        Audio tensor of shape ``(C, T)`` or ``(T,)``.
    orig_sr : int
        Original sample rate.
    target_sr : int
        Target sample rate.
    method : str
        ``"librosa"`` (default, kaiser_best sinc interpolation) or
        ``"torchaudio"`` (faster).

    Returns
    -------
    torch.Tensor
        Resampled audio tensor.
    """
    if orig_sr == target_sr:
        return audio

    if method == "librosa":
        audio_np = audio.numpy()
        resampled = librosa.resample(
            y=audio_np,
            orig_sr=orig_sr,
            target_sr=target_sr,
            res_type="kaiser_best",
        )
        return torch.from_numpy(resampled).float()
    else:
        return torchaudio.transforms.Resample(orig_sr, target_sr)(audio)


class HeterodyneToBaseband(nn.Module):
    """Down-convert a frequency band to baseband via heterodyning.

    Applies bandpass filtering, mixes with a cosine at the band center
    frequency, low-pass filters, and resamples to the target baseband
    sample rate.

    For the baseband band (f_low=0), skips bandpass/heterodyne/lowpass
    and directly resamples — matching the standard baseline resample path.

    Parameters
    ----------
    cfg : HeterodyneCfg
        Heterodyne configuration.

    Examples
    --------
    >>> het = HeterodyneToBaseband(HeterodyneCfg(baseband_sr=16000))
    >>> x = torch.randn(2, 1, 48000)
    >>> out, sr = het(x, sr_in=48000, f_low=8000.0, f_high=16000.0)
    >>> sr
    16000
    """

    def __init__(self, cfg: HeterodyneCfg) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        x: torch.Tensor,
        sr_in: int,
        f_low: float,
        f_high: float,
    ) -> Tuple[torch.Tensor, int]:
        """Heterodyne a frequency band to baseband.

        Parameters
        ----------
        x : torch.Tensor
            Waveform of shape ``(B, 1, T)`` or ``(B, T)``.
        sr_in : int
            Input sample rate in Hz.
        f_low : float
            Lower band edge in Hz.
        f_high : float
            Upper band edge in Hz.

        Returns
        -------
        Tuple[torch.Tensor, int]
            Baseband waveform ``(B, 1, T')`` and output sample rate.
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        B, C, T = x.shape
        device = x.device

        bandwidth = f_high - f_low
        if bandwidth <= 0:
            raise ValueError("Bandwidth must be positive")

        # Baseband: already at [0, f_high] — just resample directly.
        # No bandpass or heterodyne needed. Librosa kaiser_best handles
        # anti-aliasing, matching the baseline resample path exactly.
        if f_low == 0:
            if sr_in != self.cfg.baseband_sr:
                x_squeezed = x.squeeze(1)
                x_resampled = resample_audio(x_squeezed.cpu(), sr_in, self.cfg.baseband_sr)
                x_out = x_resampled.unsqueeze(1).to(device)
            else:
                x_out = x
            return x_out, self.cfg.baseband_sr

        # Non-baseband: bandpass → heterodyne → lowpass → resample
        center = 0.5 * (f_low + f_high)
        Q = max(center / bandwidth, 0.5)

        # 1) band-pass
        x_bp = torchaudio.functional.bandpass_biquad(
            x,
            sample_rate=sr_in,
            central_freq=center,
            Q=Q,
        )

        # 2) heterodyne: mix down by center freq
        t = torch.arange(T, device=device, dtype=x.dtype) / sr_in
        cos = torch.cos(2.0 * torch.pi * center * t).view(1, 1, T)
        x_mix = x_bp * cos

        # 3) low-pass in baseband
        cutoff = self.cfg.lowpass_factor * self.cfg.baseband_sr
        x_lp = torchaudio.functional.lowpass_biquad(
            x_mix,
            sample_rate=sr_in,
            cutoff_freq=cutoff,
        )

        # 4) resample to baseband_sr (librosa kaiser_best for quality)
        if sr_in != self.cfg.baseband_sr:
            x_squeezed = x_lp.squeeze(1)  # (B, T)
            x_resampled = resample_audio(x_squeezed.cpu(), sr_in, self.cfg.baseband_sr)
            x_out = x_resampled.unsqueeze(1).to(device)  # (B, 1, T_out)
        else:
            x_out = x_lp

        return x_out, self.cfg.baseband_sr
