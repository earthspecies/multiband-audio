"""Advanced fusion methods: MoE, bilinear pooling, concat-linear."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from multiband_audio.fusion.base import BaseFusion


class TopKMoEFusion(BaseFusion):
    """Mixture of Experts with top-k routing.

    A router network selects top-k experts to process the pooled band
    embedding.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    num_experts : int
        Total number of expert networks.
    top_k : int
        Number of experts to route to per sample.
    hidden_dim : int
        Hidden layer size in expert and router networks.

    Examples
    --------
    >>> f = TopKMoEFusion(embed_dim=256, num_experts=4, top_k=2)
    >>> emb = torch.randn(2, 3, 256)
    >>> f(emb).shape
    torch.Size([2, 256])
    """

    def __init__(
        self,
        embed_dim: int = 1280,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, embed_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.band_attn = nn.Linear(embed_dim, 1)

    def forward(
        self,
        embeddings: torch.Tensor,
        handcrafted_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MoE fusion over band embeddings.

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

        band_scores = self.band_attn(embeddings).squeeze(-1)
        band_weights = F.softmax(band_scores, dim=-1)
        self._last_weights = band_weights.detach().mean(dim=0)

        pooled = (band_weights.unsqueeze(-1) * embeddings).sum(dim=1)

        router_logits = self.router(pooled)
        top_k_logits, top_k_idx = router_logits.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        output = torch.zeros(N, D, device=embeddings.device, dtype=embeddings.dtype)
        for k in range(self.top_k):
            for e_idx in range(self.num_experts):
                mask = top_k_idx[:, k] == e_idx
                if mask.any():
                    expert_input = pooled[mask]
                    expert_out = self.experts[e_idx](expert_input)
                    output[mask] += top_k_weights[mask, k : k + 1] * expert_out

        return output


class BilinearPoolingFusion(BaseFusion):
    """Compact bilinear pooling for pairwise band interactions.

    Captures second-order interactions between adjacent bands using
    low-rank bilinear factors.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    output_dim : int
        Output dimension.
    rank : int
        Rank of bilinear decomposition.

    Examples
    --------
    >>> f = BilinearPoolingFusion(embed_dim=256, output_dim=256, rank=32)
    >>> emb = torch.randn(2, 3, 256)
    >>> f(emb).shape
    torch.Size([2, 256])
    """

    def __init__(
        self,
        embed_dim: int = 1280,
        output_dim: int = 1280,
        rank: int = 64,
    ) -> None:
        super().__init__()
        self.U = nn.Linear(embed_dim, rank, bias=False)
        self.V = nn.Linear(embed_dim, rank, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(rank, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.band_attn = nn.Linear(embed_dim, 1)

    def forward(
        self,
        embeddings: torch.Tensor,
        handcrafted_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Bilinear pooling fusion over band embeddings.

        Parameters
        ----------
        embeddings : torch.Tensor
            ``(N, B, D)`` band embeddings.
        handcrafted_scores : torch.Tensor or None
            Unused.

        Returns
        -------
        torch.Tensor
            ``(N, output_dim)`` fused embedding.
        """
        N, B, D = embeddings.shape

        band_scores = self.band_attn(embeddings).squeeze(-1)
        band_weights = F.softmax(band_scores, dim=-1)
        self._last_weights = band_weights.detach().mean(dim=0)

        first_order = (band_weights.unsqueeze(-1) * embeddings).sum(dim=1)

        if embeddings.shape[1] > 1:
            u = self.U(embeddings[:, :-1])
            v = self.V(embeddings[:, 1:])
            bilinear = (u * v).mean(dim=1)
        else:
            # Single band: no adjacent pairs, fall back to zero second-order term
            bilinear = torch.zeros(embeddings.shape[0], self.U.out_features, device=embeddings.device, dtype=embeddings.dtype)

        second_order = self.proj(bilinear)
        return first_order + second_order


class ConcatLinearFusion(BaseFusion):
    """Concatenate all band embeddings and project down.

    Uses per-band projection with attention-weighted pooling to handle
    variable number of bands.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    hidden_dim : int
        Hidden dimension in projection network.
    dropout : float
        Dropout rate.

    Examples
    --------
    >>> f = ConcatLinearFusion(embed_dim=256, hidden_dim=512)
    >>> emb = torch.randn(2, 3, 256)
    >>> f(emb).shape
    torch.Size([2, 256])
    """

    def __init__(
        self,
        embed_dim: int = 1280,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.band_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 4),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.band_attn = nn.Linear(hidden_dim // 4, 1)

    def forward(
        self,
        embeddings: torch.Tensor,
        handcrafted_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Concat-linear fusion over band embeddings.

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

        band_features = self.band_proj(embeddings)
        attn_scores = self.band_attn(band_features).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=-1)
        self._last_weights = attn_weights.detach().mean(dim=0)

        pooled = (attn_weights.unsqueeze(-1) * band_features).sum(dim=1)
        return self.proj(pooled)
