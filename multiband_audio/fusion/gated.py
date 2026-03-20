"""Gated fusion methods: Gated-Pool (GP) and Hybrid (HYB)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from multiband_audio.fusion.base import BaseFusion


class GatedPoolFusion(BaseFusion):
    """Softmax gating over band embeddings.

    Learns a linear gate ``D -> 1`` per band, applies softmax, and
    computes a weighted sum of band embeddings.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    temperature : float
        Softmax temperature (lower = sharper).

    Examples
    --------
    >>> f = GatedPoolFusion(embed_dim=256)
    >>> emb = torch.randn(2, 3, 256)
    >>> f(emb).shape
    torch.Size([2, 256])
    """

    def __init__(self, embed_dim: int = 1280, temperature: float = 1.0) -> None:
        super().__init__()
        self.gate = nn.Linear(embed_dim, 1)
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        handcrafted_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Gated fusion over band embeddings.

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
        scores = self.gate(embeddings).squeeze(-1)  # (N, B)
        weights = F.softmax(scores / self.temperature, dim=-1)
        self._last_weights = weights.detach().mean(dim=0)
        fused = (weights.unsqueeze(-1) * embeddings).sum(dim=1)
        return fused


class HybridFusion(BaseFusion):
    """Gating that uses both learned embeddings and handcrafted scores.

    The gate network takes concatenated ``[embedding, entropy, flux]``
    features to compute band importance weights.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    num_handcrafted : int
        Number of handcrafted score channels.
    hidden_dim : int
        Hidden layer size in the gate network.
    temperature : float
        Softmax temperature.

    Examples
    --------
    >>> f = HybridFusion(embed_dim=256, num_handcrafted=2)
    >>> emb = torch.randn(2, 3, 256)
    >>> scores = torch.randn(2, 3, 2)
    >>> f(emb, scores).shape
    torch.Size([2, 256])
    """

    def __init__(
        self,
        embed_dim: int = 1280,
        num_handcrafted: int = 2,
        hidden_dim: int = 256,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.num_handcrafted = num_handcrafted

        self.gate = nn.Sequential(
            nn.Linear(embed_dim + num_handcrafted, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.score_scale = nn.Parameter(torch.ones(num_handcrafted))
        self.score_bias = nn.Parameter(torch.zeros(num_handcrafted))

    def forward(
        self,
        embeddings: torch.Tensor,
        handcrafted_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Hybrid gated fusion using embeddings and handcrafted scores.

        Parameters
        ----------
        embeddings : torch.Tensor
            ``(N, B, D)`` band embeddings.
        handcrafted_scores : torch.Tensor or None
            ``(N, B, S)`` handcrafted features. Falls back to embed-only
            gating if ``None``.

        Returns
        -------
        torch.Tensor
            ``(N, D)`` fused embedding.
        """
        if handcrafted_scores is None:
            scores = embeddings.mean(dim=-1)
        else:
            hc = handcrafted_scores * self.score_scale + self.score_bias
            combined = torch.cat([embeddings, hc], dim=-1)
            scores = self.gate(combined).squeeze(-1)

        weights = F.softmax(scores / self.temperature, dim=-1)
        self._last_weights = weights.detach().mean(dim=0)
        fused = (weights.unsqueeze(-1) * embeddings).sum(dim=1)
        return fused
