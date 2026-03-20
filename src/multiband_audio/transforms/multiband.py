"""Main multiband transforms: split waveforms into frequency bands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torchaudio
from torch import nn

from multiband_audio._configs import BandGridConfig, HeterodyneCfg, SpectrogramCfg, make_band_grid
from multiband_audio.transforms._heterodyne import HeterodyneToBaseband
from multiband_audio.transforms._spectrogram import Spectrogram


@dataclass
class BandScores:
    """Container for handcrafted band scores.

    Parameters
    ----------
    entropy : torch.Tensor or None
        Per-band spectral entropy of shape ``(B, num_bands)``.
    flux : torch.Tensor or None
        Per-band spectral flux of shape ``(B, num_bands)``.

    Examples
    --------
    >>> scores = BandScores(entropy=torch.randn(2, 3), flux=torch.randn(2, 3))
    >>> scores.to_tensor().shape
    torch.Size([2, 3, 2])
    """

    entropy: Optional[torch.Tensor] = None
    flux: Optional[torch.Tensor] = None

    def to_tensor(self) -> torch.Tensor:
        """Stack available scores into ``(B, num_bands, num_scores)`` tensor.

        Returns
        -------
        torch.Tensor
            Stacked scores tensor.

        Raises
        ------
        ValueError
            If no scores are available.
        """
        parts = []
        if self.entropy is not None:
            parts.append(self.entropy)
        if self.flux is not None:
            parts.append(self.flux)
        if not parts:
            raise ValueError("No scores available")
        return torch.stack(parts, dim=-1)


class MultibandTransform(nn.Module):
    """Split a waveform into all frequency bands via heterodyning.

    Returns all bands below Nyquist. This is the primary transform for
    post-fusion approaches where all bands are processed through a shared
    backbone and then fused with learned weights.

    Parameters
    ----------
    sample_rate : int
        Input audio sample rate in Hz.
    target_sr : int
        Target baseband sample rate in Hz after down-conversion.
    band_width : int
        Width of each frequency band in Hz.
    step : int
        Step between band start frequencies in Hz.
    max_freq : int or None
        Upper frequency limit in Hz. Defaults to Nyquist (``sample_rate // 2``).
    n_fft : int
        FFT window size for spectrogram (used for score computation).
    hop_length : int
        Hop length for spectrogram.
    n_mels : int
        Number of mel filter banks.
    lowpass_factor : float
        Anti-alias low-pass cutoff as fraction of ``target_sr``.
    return_scores : bool
        Whether to return handcrafted scores alongside bands.
    score_types : list of str or None
        Score types to compute when ``return_scores=True``.

    Examples
    --------
    >>> t = MultibandTransform(sample_rate=48000, target_sr=16000)
    >>> x = torch.randn(2, 48000)
    >>> out = t(x)
    >>> out.ndim
    3
    """

    def __init__(
        self,
        sample_rate: int,
        target_sr: int = 16000,
        band_width: int = 8000,
        step: int = 8000,
        max_freq: Optional[int] = None,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        lowpass_factor: float = 0.45,
        return_scores: bool = False,
        score_types: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        max_freq = max_freq if max_freq is not None else sample_rate // 2

        self.spec = Spectrogram(
            SpectrogramCfg(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )
        )
        self.heterodyne = HeterodyneToBaseband(
            HeterodyneCfg(baseband_sr=target_sr, lowpass_factor=lowpass_factor)
        )

        self.grid_cfg = BandGridConfig(
            sample_rate=sample_rate,
            max_freq_hz=max_freq,
            band_width_hz=band_width,
            step_hz=step,
        )
        self.bands = make_band_grid(self.grid_cfg)
        self.return_scores = return_scores
        self.score_types = score_types or ["entropy", "flux"]

        nyquist = sample_rate / 2.0
        self.valid_bands = [(f_low, f_high) for f_low, f_high in self.bands if f_high <= nyquist]
        self.num_bands = len(self.valid_bands)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        """Process waveform and return all valid bands.

        Parameters
        ----------
        x : torch.Tensor
            Waveform of shape ``(N, T)`` or ``(N, 1, T)``.
        padding_mask : torch.Tensor or None
            Boolean mask of shape ``(N, T)`` where ``True`` marks padded
            (invalid) positions, as returned by
            :func:`~multiband_audio.collate_fn`.  When provided, the mask
            is scaled to match the band output length and returned as an
            extra element so it can be forwarded directly to
            :class:`~multiband_audio.MultibandWrapper`.

        Returns
        -------
        torch.Tensor or tuple
            * ``padding_mask=None, return_scores=False``: ``(N, B, T_out)``
            * ``padding_mask=None, return_scores=True``: ``(bands, scores)``
            * ``padding_mask given, return_scores=False``: ``(bands, band_mask)``
            * ``padding_mask given, return_scores=True``: ``(bands, scores, band_mask)``
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)

        sr_in = self.spec.cfg.sample_rate

        outputs: List[torch.Tensor] = []
        for f_low, f_high in self.valid_bands:
            x_band, _ = self.heterodyne(x, sr_in=sr_in, f_low=f_low, f_high=f_high)
            outputs.append(x_band)

        all_bands = torch.cat(outputs, dim=1)

        if not torch.isfinite(all_bands).all():
            nan_count = torch.isnan(all_bands).sum().item()
            inf_count = torch.isinf(all_bands).sum().item()
            raise ValueError(f"NaN/Inf in multiband output (nan={nan_count}, inf={inf_count})")

        # Scale padding mask from waveform length to band output length
        band_mask: Optional[torch.Tensor] = None
        if padding_mask is not None:
            T_out = all_bands.shape[-1]
            import torch.nn.functional as F
            band_mask = F.interpolate(
                padding_mask.float().unsqueeze(1),
                size=T_out,
                mode="nearest",
            ).squeeze(1).bool()

        if not self.return_scores:
            return (all_bands, band_mask) if band_mask is not None else all_bands

        spec = self.spec(x)
        scores = self._compute_scores(spec)
        if band_mask is not None:
            return all_bands, scores.to_tensor(), band_mask
        return all_bands, scores.to_tensor()

    def _compute_scores(self, spec: torch.Tensor) -> BandScores:
        """Compute entropy and flux scores for each band."""
        scores = BandScores()
        if spec.ndim == 4:
            spec = spec.mean(dim=1)

        power = spec.exp()

        if "entropy" in self.score_types:
            scores.entropy = self._compute_entropy(power)
        if "flux" in self.score_types:
            scores.flux = self._compute_flux(power)

        return scores

    def _compute_entropy(self, power: torch.Tensor) -> torch.Tensor:
        """Compute spectral entropy for each band."""
        if power.ndim == 4:
            power = power.mean(dim=1)
        N, F, T = power.shape
        nyquist = self.grid_cfg.sample_rate / 2.0
        freq_per_bin = nyquist / F

        entropies = []
        for f_low, f_high in self.valid_bands:
            bin_low = int(f_low / freq_per_bin)
            bin_high = min(int(f_high / freq_per_bin), F)

            if bin_high <= bin_low:
                entropies.append(torch.zeros(N, device=power.device))
                continue

            band_power = power[:, bin_low:bin_high, :]
            band_sum = band_power.sum(dim=-1)
            band_prob = band_sum / (band_sum.sum(dim=-1, keepdim=True) + 1e-10)
            entropy = -(band_prob * (band_prob + 1e-10).log()).sum(dim=-1)
            entropies.append(entropy)

        return torch.stack(entropies, dim=1)

    def _compute_flux(self, power: torch.Tensor) -> torch.Tensor:
        """Compute spectral flux for each band."""
        if power.ndim == 4:
            power = power.mean(dim=1)
        N, F, T = power.shape
        nyquist = self.grid_cfg.sample_rate / 2.0
        freq_per_bin = nyquist / F

        fluxes = []
        for f_low, f_high in self.valid_bands:
            bin_low = int(f_low / freq_per_bin)
            bin_high = min(int(f_high / freq_per_bin), F)

            if bin_high <= bin_low or T < 2:
                fluxes.append(torch.zeros(N, device=power.device))
                continue

            band_power = power[:, bin_low:bin_high, :]
            diff = band_power[:, :, 1:] - band_power[:, :, :-1]
            pos_diff = torch.clamp(diff, min=0.0)
            flux = pos_diff.sum(dim=(1, 2))
            fluxes.append(flux)

        return torch.stack(fluxes, dim=1)

    def get_band_info(self) -> List[Tuple[float, float]]:
        """Return list of ``(f_low, f_high)`` for valid bands.

        Returns
        -------
        List[Tuple[float, float]]
            Band edges in Hz.
        """
        return self.valid_bands.copy()

    def get_num_bands(self) -> int:
        """Return number of valid bands.

        Returns
        -------
        int
            Number of bands below Nyquist.
        """
        return self.num_bands


class MultibandTransformDynamic(nn.Module):
    """Dynamic multiband transform that computes bands at runtime.

    Use this when the dataset has variable sample rates.  Bands are
    determined per-call based on the provided ``sample_rate``.

    Parameters
    ----------
    target_sr : int
        Target baseband sample rate in Hz.
    band_width : int
        Width of each frequency band in Hz.
    step : int
        Step between band start frequencies in Hz.
    lowpass_factor : float
        Anti-alias low-pass cutoff as fraction of ``target_sr``.
    return_scores : bool
        Whether to return scores alongside bands.
    score_types : list of str or None
        Score types to compute when ``return_scores=True``.

    Examples
    --------
    >>> t = MultibandTransformDynamic(target_sr=16000)
    >>> x = torch.randn(2, 48000)
    >>> bands, scores, info = t(x, sample_rate=48000)
    >>> bands.ndim
    3
    """

    def __init__(
        self,
        target_sr: int = 16000,
        band_width: int = 8000,
        step: int = 8000,
        lowpass_factor: float = 0.45,
        return_scores: bool = False,
        score_types: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.heterodyne = HeterodyneToBaseband(
            HeterodyneCfg(baseband_sr=target_sr, lowpass_factor=lowpass_factor)
        )
        self.band_width = band_width
        self.step = step
        self.return_scores = return_scores
        self.score_types = score_types or ["entropy", "flux"]
        self._spec_cache: dict[int, Spectrogram] = {}

    def _get_spec(self, sample_rate: int) -> Spectrogram:
        """Get or create spectrogram extractor for a given sample rate."""
        if sample_rate not in self._spec_cache:
            self._spec_cache[sample_rate] = Spectrogram(
                SpectrogramCfg(sample_rate=sample_rate)
            )
        return self._spec_cache[sample_rate]

    def _compute_bands(self, sample_rate: int) -> List[Tuple[float, float]]:
        """Compute valid bands for given sample rate.

        Always returns at least one band. If Nyquist < band_width,
        returns a single band ``[0, nyquist]``.

        Parameters
        ----------
        sample_rate : int
            Audio sample rate in Hz.

        Returns
        -------
        List[Tuple[float, float]]
            Band edges in Hz.
        """
        nyquist = sample_rate / 2.0
        bands: List[Tuple[float, float]] = []
        f = 0.0
        while f < nyquist:
            f_high = min(f + self.band_width, nyquist)
            bands.append((f, f_high))
            f += self.step
        if not bands and nyquist > 0:
            bands.append((0.0, float(nyquist)))
        return bands

    def forward(
        self,
        x: torch.Tensor,
        sample_rate: int,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Process waveform and return all valid bands for a given sample rate.

        Parameters
        ----------
        x : torch.Tensor
            Waveform of shape ``(N, T)`` or ``(N, 1, T)``.
        sample_rate : int
            Input sample rate in Hz.
        padding_mask : torch.Tensor or None
            Boolean mask of shape ``(N, T)`` where ``True`` marks padded
            (invalid) positions, as returned by
            :func:`~multiband_audio.collate_fn`.  When provided, the mask
            is scaled to match the band output length and returned as the
            second element of the output tuple.

        Returns
        -------
        tuple
            * ``padding_mask=None``: ``(bands, scores, band_info)``
            * ``padding_mask given``: ``(bands, band_mask, scores, band_info)``

            ``bands`` is ``(N, num_bands, T_baseband)``, ``band_mask`` is
            ``(N, T_baseband)`` bool, ``scores`` is
            ``(N, num_bands, num_scores)`` or ``None``, and ``band_info``
            is a list of ``(f_low, f_high)`` tuples.
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)

        valid_bands = self._compute_bands(sample_rate)

        outputs = []
        for f_low, f_high in valid_bands:
            x_band, _ = self.heterodyne(x, sr_in=sample_rate, f_low=f_low, f_high=f_high)
            outputs.append(x_band)

        all_bands = torch.cat(outputs, dim=1)

        scores = None
        if self.return_scores:
            N = x.shape[0]
            num_bands = len(valid_bands)
            scores = torch.ones(N, num_bands, len(self.score_types), device=x.device)

        if padding_mask is not None:
            T_out = all_bands.shape[-1]
            import torch.nn.functional as F
            band_mask = F.interpolate(
                padding_mask.float().unsqueeze(1),
                size=T_out,
                mode="nearest",
            ).squeeze(1).bool()
            return all_bands, band_mask, scores, valid_bands

        return all_bands, scores, valid_bands
