"""Multiband wrapper: backbone + fusion + optional head."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import nn

from multiband_audio.fusion.base import BaseFusion
from multiband_audio.fusion.logit import MoEFusion
from multiband_audio.fusion.registry import build_fusion


class MultibandWrapper(nn.Module):
    """Composite model for multi-band audio classification with learned fusion.

    Architecture::

        Input: (N, B, T) - N samples, B bands, T timesteps
            |
        Reshape to (N*B, T) - treat each band as independent sample
            |
        Backbone (shared) - e.g., EfficientNet feature extractor
            |
        Reshape to (N, B, D) - D = embedding dim
            |
        Fusion module - combines band embeddings
            |
        Head (if not LogitFusion) - classification head
            |
        Output: (N, C) logits  or  (N, D) embeddings if no head

    Accepts the fusion as either a :class:`BaseFusion` instance or a
    string name that gets looked up in the fusion registry.

    Parameters
    ----------
    backbone : nn.Module
        Feature extractor that maps ``(N, T) -> (N, D)``.
    fusion : BaseFusion or str
        Fusion module instance, or string name for :func:`build_fusion`.
    head : nn.Module or None
        Classification head.  If ``None`` and the fusion does not return
        logits, the wrapper returns fused embeddings ``(N, D)``.
    embed_dim : int
        Embedding dimension (forwarded to :func:`build_fusion` when
        ``fusion`` is a string).
    gradient_checkpointing : bool
        Enable gradient checkpointing on the backbone.
    chunk_size : int or None
        If set, process bands in chunks to save memory.
    freeze_backbone : bool
        If ``True``, run backbone under ``torch.no_grad()`` and keep it in
        eval mode (linear probing). If ``False``, backbone is trained
        end-to-end.
    **fusion_kwargs
        Extra keyword arguments forwarded to :func:`build_fusion`.

    Examples
    --------
    >>> import torch
    >>> backbone = torch.nn.Linear(16000, 256)  # toy backbone
    >>> w = MultibandWrapper(backbone=backbone, fusion="mp", embed_dim=256)
    >>> x = torch.randn(2, 3, 16000)
    >>> # No head -> returns fused embeddings
    >>> w(x).shape
    torch.Size([2, 256])
    """

    def __init__(
        self,
        backbone: nn.Module,
        fusion: Union[BaseFusion, str],
        head: Optional[nn.Module] = None,
        embed_dim: int = 1280,
        gradient_checkpointing: bool = False,
        chunk_size: Optional[int] = None,
        freeze_backbone: bool = False,
        **fusion_kwargs: object,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.gradient_checkpointing = gradient_checkpointing
        self.chunk_size = chunk_size
        self.freeze_backbone = freeze_backbone

        if isinstance(fusion, str):
            self.fusion = build_fusion(fusion, embed_dim=embed_dim, **fusion_kwargs)
        else:
            self.fusion = fusion

        if gradient_checkpointing and hasattr(backbone, "enable_gradient_checkpointing"):
            backbone.enable_gradient_checkpointing()

        if freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True) -> "MultibandWrapper":
        """Override to keep frozen backbone in eval mode."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]],
        handcrafted_scores: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the multiband model.

        Parameters
        ----------
        x : torch.Tensor or Tuple[torch.Tensor, Optional[torch.Tensor]]
            Either ``(N, B, T)`` multi-band waveforms, ``(N, T)``
            single-band for baseline compatibility, or a tuple
            ``(bands, scores)`` from a transform.
        handcrafted_scores : torch.Tensor or None
            ``(N, B, S)`` optional scores for hybrid fusion.
        padding_mask : torch.Tensor or None
            ``(N, T)`` bool mask where ``True`` = padded (invalid) position.
            Critical for transformer-based backbones (BEATs/EAT): without
            this, attention over zero-padded frames corrupts representations.
            CNNs (EffNet) are unaffected but the mask is passed through
            harmlessly if the backbone supports it.

        Returns
        -------
        torch.Tensor
            ``(N, C)`` logits if a head is present, otherwise ``(N, D)``
            fused embeddings.
        """
        # Unpack tuple input from transforms
        if isinstance(x, tuple):
            bands, scores = x
            if handcrafted_scores is None:
                handcrafted_scores = scores
            x = bands

        if x.ndim == 2:
            x = x.unsqueeze(1)

        N, B, T = x.shape
        # Use reshape instead of view — tensor may not be contiguous after padding
        x_flat = x.reshape(N * B, T)

        # Tile padding mask from (N, T) -> (N*B, T): same mask for all bands
        # of a given sample since all bands share the same valid audio length.
        mask_flat = None
        if padding_mask is not None:
            mask_flat = padding_mask.unsqueeze(1).expand(N, B, T).reshape(N * B, T)

        # Process through backbone
        if self.chunk_size is not None and N * B > self.chunk_size:
            embeddings_list = []
            for i in range(0, N * B, self.chunk_size):
                chunk = x_flat[i : i + self.chunk_size]
                chunk_mask = mask_flat[i : i + self.chunk_size] if mask_flat is not None else None
                emb = self._backbone_forward(chunk, padding_mask=chunk_mask)
                embeddings_list.append(emb)
            embeddings_flat = torch.cat(embeddings_list, dim=0)
        else:
            embeddings_flat = self._backbone_forward(x_flat, padding_mask=mask_flat)

        D = embeddings_flat.shape[-1]
        embeddings = embeddings_flat.reshape(N, B, D)

        # Fusion
        if handcrafted_scores is not None:
            fused = self.fusion(embeddings, handcrafted_scores)
        else:
            fused = self.fusion(embeddings)

        # Head
        if isinstance(self.fusion, MoEFusion) or getattr(self.fusion, "returns_logits", False):
            return fused
        if self.head is not None:
            return self.head(fused)
        return fused

    def _backbone_forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward through backbone with optional gradient checkpointing and masking."""
        def _call(x: torch.Tensor) -> torch.Tensor:
            if padding_mask is not None and hasattr(self.backbone, "__call__"):
                try:
                    return self.backbone(x, padding_mask=padding_mask)
                except TypeError:
                    return self.backbone(x)
            return self.backbone(x)

        if self.freeze_backbone:
            with torch.no_grad():
                return _call(x)
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self.backbone, x, use_reentrant=False)
        return _call(x)

    def get_band_weights(self) -> Optional[torch.Tensor]:
        """Get learned band weights from fusion for logging.

        Returns
        -------
        torch.Tensor or None
            ``(B,)`` tensor of mean band weights, or ``None``.
        """
        return self.fusion.get_band_weights()

    def get_fusion_name(self) -> str:
        """Get name of the fusion module.

        Returns
        -------
        str
            Class name of the fusion module.
        """
        return self.fusion.__class__.__name__
