"""Late fusion at the logit level with per-band classifiers."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from multiband_audio.fusion.base import BaseFusion


class MoEFusion(BaseFusion):
    """Late fusion with separate per-band classifiers.

    Each band gets its own classifier head, then outputs are combined
    with learned weights.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    num_classes : int
        Number of output classes.
    num_bands : int
        Maximum expected number of bands.
    temperature : float
        Softmax temperature for band weighting.

    Examples
    --------
    >>> f = MoEFusion(embed_dim=256, num_classes=10, num_bands=3)
    >>> emb = torch.randn(2, 3, 256)
    >>> f(emb).shape
    torch.Size([2, 10])
    """

    def __init__(
        self,
        embed_dim: int = 1280,
        num_classes: int = 11,
        num_bands: int = 12,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.num_classes = num_classes
        self.temperature = temperature

        self.classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, num_classes),
                )
                for _ in range(num_bands)
            ]
        )
        self.weight_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        handcrafted_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Late fusion at the logit level.

        Parameters
        ----------
        embeddings : torch.Tensor
            ``(N, B, D)`` band embeddings.
        handcrafted_scores : torch.Tensor or None
            Unused.

        Returns
        -------
        torch.Tensor
            ``(N, C)`` fused logits.
        """
        N, B, D = embeddings.shape

        all_logits = []
        all_weights = []

        for b in range(B):
            clf_idx = min(b, self.num_bands - 1)
            logits_b = self.classifiers[clf_idx](embeddings[:, b])
            weight_b = self.weight_net(embeddings[:, b])
            all_logits.append(logits_b)
            all_weights.append(weight_b)

        logits = torch.stack(all_logits, dim=1)  # (N, B, C)
        weights = torch.cat(all_weights, dim=1)  # (N, B)
        weights = F.softmax(weights / self.temperature, dim=-1)

        self._last_weights = weights.detach().mean(dim=0)

        fused_logits = (weights.unsqueeze(-1) * logits).sum(dim=1)
        return fused_logits

    @property
    def returns_logits(self) -> bool:
        """Indicate that this fusion returns logits, not embeddings."""
        return True
