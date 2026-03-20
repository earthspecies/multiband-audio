"""Abstract base class for band selectors."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from multiband_audio._configs import BandGridConfig, make_band_grid


class BaseBandSelector(nn.Module):
    """Abstract interface for frequency band selection.

    Takes a log-power spectrogram and returns the best ``(f_low, f_high)``
    band(s) according to a scoring criterion.

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
        Optional display name for this selector.

    Examples
    --------
    >>> # BaseBandSelector is abstract; see EntropyBandSelector for usage.
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
        super().__init__()
        self.grid_cfg = BandGridConfig(
            sample_rate=sample_rate,
            band_width_hz=band_width_hz,
            step_hz=step_hz,
            min_freq_hz=min_freq_hz,
            max_freq_hz=max_freq_hz,
        )
        self.bands = make_band_grid(self.grid_cfg)
        self.name = name or type(self).__name__

    def select(self, spec: torch.Tensor, top_k: int = 1) -> List[Tuple[float, float]]:
        """Select the best band(s) from a log-power spectrogram.

        Parameters
        ----------
        spec : torch.Tensor
            Log-power spectrogram of shape ``(B, F, T)``.
        top_k : int
            Number of bands to return, sorted best to worst.

        Returns
        -------
        List[Tuple[float, float]]
            List of ``(f_low, f_high)`` tuples in Hz.
        """
        raise NotImplementedError

    def forward(self, spec: torch.Tensor) -> Tuple[float, float]:
        """Return the single best band (legacy interface).

        Parameters
        ----------
        spec : torch.Tensor
            Log-power spectrogram of shape ``(B, F, T)``.

        Returns
        -------
        Tuple[float, float]
            ``(f_low, f_high)`` of the best band in Hz.
        """
        return self.select(spec, top_k=1)[0]

    @staticmethod
    def _as_bft(spec: torch.Tensor) -> torch.Tensor:
        """Normalize spectrogram shape to ``(B, F, T)``.

        Handles 4-D ``(B, C, F, T)`` input by averaging over channels,
        and 2-D ``(F, T)`` input by adding a batch dimension.

        Parameters
        ----------
        spec : torch.Tensor
            Spectrogram of shape ``(F, T)``, ``(B, F, T)``, or ``(B, C, F, T)``.

        Returns
        -------
        torch.Tensor
            Spectrogram of shape ``(B, F, T)``.
        """
        if spec.ndim == 4:
            spec = spec.mean(dim=1)
        if spec.ndim == 2:
            spec = spec.unsqueeze(0)
        return spec
