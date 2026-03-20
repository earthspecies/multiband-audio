"""Simple pooling-based fusion methods."""

from __future__ import annotations

from typing import Optional

import torch

from multiband_audio.fusion.base import BaseFusion


class MeanPoolFusion(BaseFusion):
    """Mean pooling over bands (non-parametric baseline).

    Examples
    --------
    >>> f = MeanPoolFusion()
    >>> emb = torch.randn(2, 3, 256)
    >>> f(emb).shape
    torch.Size([2, 256])
    """

    def forward(
        self,
        embeddings: torch.Tensor,
        handcrafted_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse by averaging over bands.

        Parameters
        ----------
        embeddings : torch.Tensor
            ``(N, B, D)`` band embeddings.
        handcrafted_scores : torch.Tensor or None
            Unused.

        Returns
        -------
        torch.Tensor
            ``(N, D)`` fused embedding.
        """
        N, B, D = embeddings.shape
        self._last_weights = torch.ones(B, device=embeddings.device) / B
        return embeddings.mean(dim=1)


class MaxPoolFusion(BaseFusion):
    """Max pooling over bands (non-parametric baseline).

    Examples
    --------
    >>> f = MaxPoolFusion()
    >>> emb = torch.randn(2, 3, 256)
    >>> f(emb).shape
    torch.Size([2, 256])
    """

    def forward(
        self,
        embeddings: torch.Tensor,
        handcrafted_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse by taking the max over bands.

        Parameters
        ----------
        embeddings : torch.Tensor
            ``(N, B, D)`` band embeddings.
        handcrafted_scores : torch.Tensor or None
            Unused.

        Returns
        -------
        torch.Tensor
            ``(N, D)`` fused embedding.
        """
        fused, indices = embeddings.max(dim=1)

        N, B, D = embeddings.shape
        band_counts = torch.zeros(B, device=embeddings.device)
        for b in range(B):
            band_counts[b] = (indices == b).float().mean()
        self._last_weights = band_counts / (band_counts.sum() + 1e-8)

        return fused
