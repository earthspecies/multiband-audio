"""Mel spectrogram extraction module."""

from __future__ import annotations

import torch
import torchaudio
from torch import nn

from multiband_audio._configs import SpectrogramCfg


class Spectrogram(nn.Module):
    """Compute a log-mel spectrogram with optional z-score normalization.

    Parameters
    ----------
    cfg : SpectrogramCfg
        Spectrogram configuration.

    Examples
    --------
    >>> spec = Spectrogram(SpectrogramCfg(sample_rate=16000))
    >>> x = torch.randn(2, 1, 16000)
    >>> out = spec(x)
    >>> out.shape[0]
    2
    """

    def __init__(self, cfg: SpectrogramCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            center=True,
            power=2.0,
        )
        self.eps = 1e-10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log-mel spectrogram.

        Parameters
        ----------
        x : torch.Tensor
            Waveform of shape ``(B, T)`` or ``(B, 1, T)``.

        Returns
        -------
        torch.Tensor
            Log-mel spectrogram of shape ``(B, n_mels, T')``.
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        # Pad short signals to at least one FFT window
        if x.shape[-1] < self.cfg.n_fft:
            pad_len = self.cfg.n_fft - x.shape[-1]
            x = torch.nn.functional.pad(x, (0, pad_len))
        spec = self.mel(x)  # (B, [C,] F, T)
        # Squeeze channel dim if MelSpectrogram preserved it
        if spec.ndim == 4:
            spec = spec.squeeze(1)
        spec = torch.log(spec + self.eps)
        if self.cfg.normalize:
            mean = spec.mean(dim=(1, 2), keepdim=True)
            std = spec.std(dim=(1, 2), keepdim=True) + 1e-5
            spec = (spec - mean) / std
        return spec
