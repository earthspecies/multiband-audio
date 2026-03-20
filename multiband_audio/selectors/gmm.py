"""GMM-based band selector (requires scikit-learn)."""

from __future__ import annotations

from typing import List, Tuple

import torch
from sklearn.mixture import GaussianMixture

from multiband_audio.selectors.base import BaseBandSelector


class GMMBandSelector(BaseBandSelector):
    """Select bands by fitting a GMM on the long-term spectrum.

    Fits a small Gaussian Mixture Model to the frequency-weighted mean
    spectrum and assigns mixture weights to the nearest pre-defined bands.

    Requires ``scikit-learn``. Install with ``pip install "multiband-audio[gmm]"``.

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
    n_components : int
        Number of GMM components.
    name : str or None
        Optional display name.

    Examples
    --------
    >>> selector = GMMBandSelector(sample_rate=48000, max_freq_hz=24000, n_components=3)
    >>> spec = torch.randn(2, 128, 64).abs()  # positive values
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
        n_components: int = 3,
        name: str | None = None,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            band_width_hz=band_width_hz,
            step_hz=step_hz,
            min_freq_hz=min_freq_hz,
            max_freq_hz=max_freq_hz,
            name=name or "gmm",
        )
        self.n_components = n_components

    def select(self, spec: torch.Tensor, top_k: int = 1) -> List[Tuple[float, float]]:
        """Select bands using GMM on the mean spectrum.

        Parameters
        ----------
        spec : torch.Tensor
            Log-power spectrogram of shape ``(B, F, T)`` or ``(B, C, F, T)``.
        top_k : int
            Number of bands to return.

        Returns
        -------
        List[Tuple[float, float]]
            Top-k ``(f_low, f_high)`` tuples sorted by GMM weight (descending).
        """
        spec = self._as_bft(spec)
        _, F, T = spec.shape

        power = spec[0].exp()  # (F, T) - use first example
        mean_spec = power.mean(dim=-1)  # (F,)

        freqs = torch.linspace(0, self.grid_cfg.sample_rate / 2, F, device=spec.device).cpu().numpy().reshape(-1, 1)
        weights = (mean_spec / (mean_spec.sum() + 1e-12)).cpu().numpy()

        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            random_state=0,
        )
        try:
            gmm.fit(freqs, sample_weight=weights)
        except TypeError:
            # Older sklearn may not support sample_weight
            gmm.fit(freqs)

        # Accumulate mixture weights for each pre-defined band
        band_scores = torch.zeros(len(self.bands))
        means = gmm.means_.flatten()
        for k in range(self.n_components):
            center = means[k]
            w_k = gmm.weights_[k]
            dists = [abs((b[0] + b[1]) / 2 - center) for b in self.bands]
            idx = int(torch.tensor(dists).argmin().item())
            band_scores[idx] += float(w_k)

        k = min(top_k, len(self.bands))
        topk_idx = band_scores.topk(k).indices.tolist()
        return [self.bands[int(i)] for i in topk_idx]
