"""Spectral entropy band selector."""

from __future__ import annotations

from typing import List, Tuple

import torch

from multiband_audio.selectors.base import BaseBandSelector


class EntropyBandSelector(BaseBandSelector):
    """Select bands with highest spectral entropy (averaged over time and batch).

    Higher entropy indicates a more uniform spectral distribution within a
    band, suggesting richer information content.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz.
    max_freq_hz : int
        Upper frequency limit in Hz. Must be set explicitly.
    band_width_hz : int
        Width of each candidate band in Hz.
    step_hz : int
        Step between band start frequencies in Hz.
    min_freq_hz : int
        Lower frequency limit in Hz.
    name : str or None
        Optional display name.

    Examples
    --------
    >>> selector = EntropyBandSelector(sample_rate=48000, max_freq_hz=24000)
    >>> spec = torch.randn(2, 128, 64)
    >>> bands = selector.select(spec, top_k=1)
    >>> len(bands)
    1
    """

    def __init__(
        self,
        sample_rate: int,
        max_freq_hz: int,
        band_width_hz: int = 8000,
        step_hz: int = 8000,
        min_freq_hz: int = 0,
        name: str | None = None,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            band_width_hz=band_width_hz,
            step_hz=step_hz,
            min_freq_hz=min_freq_hz,
            max_freq_hz=max_freq_hz,
            name=name or "entropy",
        )

    def select(self, spec: torch.Tensor, top_k: int = 1) -> List[Tuple[float, float]]:
        """Select bands by spectral entropy.

        Parameters
        ----------
        spec : torch.Tensor
            Log-power spectrogram of shape ``(B, F, T)`` or ``(B, C, F, T)``.
        top_k : int
            Number of bands to return.

        Returns
        -------
        List[Tuple[float, float]]
            Top-k ``(f_low, f_high)`` tuples sorted by entropy (descending).
        """
        spec = self._as_bft(spec)
        B, F, T = spec.shape
        device = spec.device

        power = spec.exp()
        power_sum_t = power.sum(dim=-1) + 1e-12  # (B, F)

        freqs = torch.linspace(0, self.grid_cfg.sample_rate / 2, F, device=device)

        scores = []
        for f_low, f_high in self.bands:
            mask = (freqs >= f_low) & (freqs < f_high)
            if mask.sum() < 2:
                scores.append(torch.tensor(-1e9, device=device))
                continue

            band_pow = power_sum_t[:, mask]  # (B, F_band)
            band_pow = band_pow / (band_pow.sum(dim=-1, keepdim=True) + 1e-12)
            entropy = -(band_pow * (band_pow + 1e-12).log()).sum(dim=-1)  # (B,)
            scores.append(entropy.mean())

        scores_tensor = torch.stack(scores)
        k = min(top_k, len(self.bands))
        topk_idx = torch.topk(scores_tensor, k).indices.tolist()
        return [self.bands[int(i)] for i in topk_idx]
