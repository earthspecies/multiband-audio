"""Attention-based fusion methods: self-attention and cross-attention."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from multiband_audio.fusion.base import BaseFusion


class SelfAttentionFusion(BaseFusion):
    """Transformer self-attention on band embeddings.

    Prepends a learnable ``[CLS]`` token and uses its output as the fused
    representation.  Supports variable number of bands via interpolated
    positional embeddings.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of transformer encoder layers.
    dropout : float
        Dropout rate.
    max_bands : int
        Maximum expected bands (for positional embedding allocation).

    Examples
    --------
    >>> f = SelfAttentionFusion(embed_dim=256, num_heads=4, num_layers=1)
    >>> emb = torch.randn(2, 3, 256)
    >>> f(emb).shape
    torch.Size([2, 256])
    """

    def __init__(
        self,
        embed_dim: int = 1280,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_bands: int = 12,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + max_bands, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        embeddings: torch.Tensor,
        handcrafted_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Self-attention fusion over band embeddings.

        Parameters
        ----------
        embeddings : torch.Tensor
            ``(N, B, D)`` band embeddings.
        handcrafted_scores : torch.Tensor or None
            Unused.

        Returns
        -------
        torch.Tensor
            ``(N, D)`` fused embedding from the CLS token.
        """
        N, B, D = embeddings.shape

        # Normalize input embeddings
        x = self.input_norm(embeddings)

        # Prepend CLS token
        cls = self.cls_token.expand(N, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add positional embeddings
        if x.shape[1] <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, : x.shape[1]]
        else:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=x.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            x = x + pos

        # Transformer encoding (force fp32 to avoid bf16 overflow in attention)
        with torch.autocast("cuda", enabled=False):
            out = self.transformer(x.float())
            out = self.norm(out)

        # Extract CLS token output
        cls_out = out[:, 0]

        # Band weights: scaled dot-product between CLS and band outputs
        with torch.no_grad():
            band_out = out[:, 1:]  # (N, B, D)
            sim = torch.bmm(cls_out.unsqueeze(1), band_out.transpose(1, 2)).squeeze(1) / (D**0.5)  # (N, B)
            weights = F.softmax(sim, dim=-1)
            self._last_weights = weights.mean(dim=0)  # (B,)

        return cls_out


class CrossAttentionFusion(BaseFusion):
    """Cross-attention with learnable query tokens attending to band embeddings.

    A single learnable query attends to all bands to produce the fused
    output.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    num_queries : int
        Number of learnable query tokens.
    dropout : float
        Dropout rate.

    Examples
    --------
    >>> f = CrossAttentionFusion(embed_dim=256, num_heads=4)
    >>> emb = torch.randn(2, 3, 256)
    >>> f(emb).shape
    torch.Size([2, 256])
    """

    def __init__(
        self,
        embed_dim: int = 1280,
        num_heads: int = 8,
        num_queries: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        embeddings: torch.Tensor,
        handcrafted_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-attention fusion over band embeddings.

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
        N = embeddings.shape[0]
        query = self.query.expand(N, -1, -1)

        attn_out, attn_weights = self.cross_attn(
            query, embeddings, embeddings, need_weights=True, average_attn_weights=True
        )

        self._last_weights = attn_weights.detach().mean(dim=(0, 1))

        x = self.norm(query + attn_out)
        x = x + self.mlp(x)
        x = self.norm2(x)

        if self.num_queries > 1:
            x = x.mean(dim=1)
        else:
            x = x.squeeze(1)

        return x
