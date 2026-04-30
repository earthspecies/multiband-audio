"""Adapter that wraps a 2D image CNN to consume audio waveforms.

The :class:`MultibandWrapper` feeds raw waveforms ``(N, T)`` to the
backbone, but image CNNs from torchvision (EfficientNet, ResNet, ViT,
ConvNeXt, ...) expect 4D image input ``(N, C, H, W)``. This adapter
bridges the gap: it computes a mel-spectrogram, log-normalizes, expands
to N channels, and forwards through the inner model so that any image
CNN can be used as a frozen (or trainable) feature extractor on
multi-band audio.

Native-waveform models (BEATs, EAT, wav2vec2, HuBERT) do *not* need this
adapter — pass them directly to ``MultibandWrapper(backbone=...)``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchaudio.transforms as T


class SpectrogramBackbone(nn.Module):
    """Wrap a 2D image CNN to accept ``(N, T)`` audio waveforms.

    Pipeline::

        (N, T) waveform
            -> MelSpectrogram (n_mels, T')
            -> log(1 + .)
            -> per-sample standardization (zero-mean, unit-std)
            -> expand to (N, n_channels, n_mels, T')
            -> backbone
            -> (N, D)

    Designed for the common case of using a torchvision image CNN as a
    frozen feature extractor on bioacoustic audio (the pattern used by
    ``EfficientNetFeatureExtractor`` in the multi-sr-encoder paper code).

    Parameters
    ----------
    backbone : nn.Module
        Any image CNN mapping ``(N, C, H, W) -> (N, D)``. The output
        dimension ``D`` is whatever the model produces — for torchvision
        models with ``num_classes=K``, ``D == K``.
    sample_rate : int
        Sample rate of the input waveforms (must match the rate of the
        bands produced by :class:`MultibandTransform`, i.e. the baseband
        target SR).
    n_fft : int
        STFT window size.
    hop_length : int
        STFT hop size.
    n_mels : int
        Number of mel filterbank bins.
    n_channels : int
        Channel count to expand the (mono) spectrogram to. ``3`` for
        ImageNet-pretrained CNNs, ``1`` for single-channel models.
    normalize : bool
        If ``True``, apply ``log1p`` + per-sample standardization to the
        spectrogram before the backbone. Mirrors the preprocessing used
        by ``EfficientNetFeatureExtractor`` in multi-sr-encoder.

    Examples
    --------
    Wrap a torchvision EfficientNet-B0 for use with
    :class:`MultibandWrapper`::

        >>> import torchvision.models as tv
        >>> import multiband_audio as mba
        >>> img_cnn = tv.efficientnet_b0(num_classes=1280)
        >>> backbone = mba.SpectrogramBackbone(img_cnn, sample_rate=16000)
        >>> wrapper = mba.MultibandWrapper(
        ...     backbone=backbone,
        ...     fusion="moe",
        ...     head=mba.LinearHead(1280, num_classes=10),
        ...     embed_dim=1280,
        ...     freeze_backbone=True,
        ... )
    """

    def __init__(
        self,
        backbone: nn.Module,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        n_channels: int = 3,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.n_channels = n_channels
        self.normalize = normalize

        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run waveforms through mel-spec + image CNN.

        Parameters
        ----------
        x : torch.Tensor
            Audio waveforms ``(N, T)`` or ``(N, 1, T)``.
        padding_mask : torch.Tensor or None
            Currently ignored — image CNNs don't use it. Accepted in the
            signature for compatibility with :class:`MultibandWrapper`'s
            optional padding-mask forwarding.

        Returns
        -------
        torch.Tensor
            Backbone output ``(N, D)``.
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        # (N, T) -> (N, n_mels, T')
        spec = self.mel(x)
        if self.normalize:
            spec = torch.log1p(spec)
            mean = spec.mean(dim=(-2, -1), keepdim=True)
            std = spec.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
            spec = (spec - mean) / std
        # (N, n_mels, T') -> (N, n_channels, n_mels, T')
        spec = spec.unsqueeze(1).expand(-1, self.n_channels, -1, -1)
        return self.backbone(spec)
