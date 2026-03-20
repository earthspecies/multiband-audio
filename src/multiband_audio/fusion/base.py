"""Abstract base class for all fusion methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn


class BaseFusion(nn.Module, ABC):
    """Abstract base class for all fusion methods.

    All fusion modules take band embeddings ``(N, B, D)`` and produce a
    fused embedding ``(N, D)`` or logits ``(N, C)`` for classification.

    Subclasses must implement :meth:`forward`.  They may optionally
    store per-band weights in ``_last_weights`` for interpretability.

    Examples
    --------
    >>> # BaseFusion is abstract; see MeanPoolFusion for usage.
    """

    def __init__(self) -> None:
        super().__init__()
        self._last_weights: Optional[torch.Tensor] = None

    @abstractmethod
    def forward(
        self,
        embeddings: torch.Tensor,
        handcrafted_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse band embeddings into a single representation.

        Parameters
        ----------
        embeddings : torch.Tensor
            Band embeddings of shape ``(N, B, D)``.
        handcrafted_scores : torch.Tensor or None
            Optional handcrafted features ``(N, B, S)``.

        Returns
        -------
        torch.Tensor
            Fused embedding ``(N, D)`` or logits ``(N, C)``.
        """
        raise NotImplementedError

    def get_band_weights(self) -> Optional[torch.Tensor]:
        """Get the last computed band attention weights for logging.

        Returns
        -------
        torch.Tensor or None
            ``(B,)`` tensor of mean band weights, or ``None``.
        """
        return self._last_weights
