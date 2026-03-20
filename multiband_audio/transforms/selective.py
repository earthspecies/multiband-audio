"""Selective multiband transform: spectrogram -> band selector -> heterodyne."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torchaudio
from torch import nn

from multiband_audio._configs import HeterodyneCfg, SpectrogramCfg
from multiband_audio.selectors.base import BaseBandSelector
from multiband_audio.transforms._heterodyne import HeterodyneToBaseband
from multiband_audio.transforms._spectrogram import Spectrogram


class MultibandSelectiveTransform(nn.Module):
    """Select top-k bands from a waveform using a band selector.

    Pipeline: waveform -> spectrogram -> band selector -> heterodyne.
    Unlike :class:`MultibandTransform` which returns all bands, this
    transform uses a scoring function to pick the most informative bands.

    Parameters
    ----------
    band_selector : BaseBandSelector
        Band selector instance (entropy, flux, or GMM).
    sample_rate : int
        Input audio sample rate in Hz.
    target_sr : int
        Target baseband sample rate in Hz.
    max_bands : int
        Maximum number of bands to select.
    n_fft : int
        FFT window size for spectrogram.
    hop_length : int
        Hop length for spectrogram.
    n_mels : int
        Number of mel filter banks.
    lowpass_factor : float
        Anti-alias low-pass cutoff as fraction of ``target_sr``.
    return_band_info : bool
        Whether to also return the selected band edges.

    Examples
    --------
    >>> from multiband_audio.selectors import EntropyBandSelector
    >>> selector = EntropyBandSelector(sample_rate=48000, max_freq_hz=24000)
    >>> t = MultibandSelectiveTransform(
    ...     band_selector=selector,
    ...     sample_rate=48000,
    ...     target_sr=16000,
    ...     max_bands=2,
    ... )
    >>> x = torch.randn(2, 48000)
    >>> out = t(x)
    >>> out.ndim
    3
    """

    def __init__(
        self,
        band_selector: BaseBandSelector,
        sample_rate: int,
        target_sr: int = 16000,
        max_bands: int = 1,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        lowpass_factor: float = 0.45,
        return_band_info: bool = False,
    ) -> None:
        super().__init__()
        self.spec = Spectrogram(
            SpectrogramCfg(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )
        )
        self.heterodyne = HeterodyneToBaseband(HeterodyneCfg(baseband_sr=target_sr, lowpass_factor=lowpass_factor))
        self.band_selector = band_selector
        self.max_bands = max(1, int(max_bands))
        self.return_band_info = return_band_info

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        """Select top-k bands and down-convert to baseband.

        Parameters
        ----------
        x : torch.Tensor
            Waveform of shape ``(N, 1, T)`` or ``(N, T)``.
        padding_mask : torch.Tensor or None
            Boolean mask of shape ``(N, T)`` where ``True`` marks padded
            (invalid) positions, as returned by
            :func:`~multiband_audio.collate_fn`.  When provided, the mask
            is scaled to match the band output length and returned as the
            second element of the output tuple.

        Returns
        -------
        torch.Tensor or tuple
            * ``padding_mask=None, return_band_info=False``: ``(N, max_bands, T_baseband)``
            * ``padding_mask=None, return_band_info=True``: ``(bands, band_info)``
            * ``padding_mask given, return_band_info=False``: ``(bands, band_mask)``
            * ``padding_mask given, return_band_info=True``: ``(bands, band_mask, band_info)``

        Raises
        ------
        ValueError
            If the band output contains NaN or Inf values.
        """
        spec = self.spec(x)
        sr_in = self.spec.cfg.sample_rate
        sr_out = self.heterodyne.cfg.baseband_sr
        bands = self.band_selector.select(spec, top_k=self.max_bands)
        x_for_het = x if x.ndim == 3 else x.unsqueeze(1)

        def _zero_band() -> torch.Tensor:
            zeros = torch.zeros_like(x_for_het)
            if sr_in != sr_out:
                zeros = torchaudio.functional.resample(zeros, orig_freq=sr_in, new_freq=sr_out)
            return zeros

        outputs: List[torch.Tensor] = []
        for f_low, f_high in bands:
            if f_high >= 0.5 * sr_in:
                outputs.append(_zero_band())
                continue
            x_band, _ = self.heterodyne(x_for_het, sr_in=sr_in, f_low=f_low, f_high=f_high)
            outputs.append(x_band)

        x_out = torch.cat(outputs, dim=1)
        if not torch.isfinite(x_out).all():
            nan_count = torch.isnan(x_out).sum().item()
            inf_count = torch.isinf(x_out).sum().item()
            x_min = torch.nanmin(x_out).item()
            x_max = torch.nanmax(x_out).item()
            raise ValueError(
                f"NaN/Inf in multiband output (nan={nan_count}, inf={inf_count}, min={x_min}, max={x_max})"
            )

        band_mask: Optional[torch.Tensor] = None
        if padding_mask is not None:
            T_out = x_out.shape[-1]
            import torch.nn.functional as F

            band_mask = (
                F.interpolate(
                    padding_mask.float().unsqueeze(1),
                    size=T_out,
                    mode="nearest",
                )
                .squeeze(1)
                .bool()
            )

        if band_mask is not None:
            return (x_out, band_mask, bands) if self.return_band_info else (x_out, band_mask)
        if self.return_band_info:
            return x_out, bands
        return x_out
