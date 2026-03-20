"""Data utilities: collate function for variable-length audio batches."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch


def collate_fn(
    batch: List[Tuple[torch.Tensor, Union[int, torch.Tensor]]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length waveforms into a padded batch with a padding mask.

    Designed for use as the ``collate_fn`` argument to a PyTorch
    ``DataLoader`` when recordings have different durations.  Shorter
    recordings are zero-padded to the length of the longest one in the
    batch.  The returned mask can be passed directly to
    :class:`~multiband_audio.MultibandTransform` and then to
    :class:`~multiband_audio.MultibandWrapper` without any further
    manipulation.

    Parameters
    ----------
    batch : list of (waveform, label)
        Each ``waveform`` is a 1-D ``(T,)`` or 2-D ``(1, T)`` float
        tensor.  ``label`` is an int or a scalar tensor.

    Returns
    -------
    waveforms : torch.Tensor
        Zero-padded waveforms of shape ``(N, T_max)``.
    padding_mask : torch.Tensor
        Boolean mask of shape ``(N, T_max)`` where ``True`` marks
        padded (invalid) positions.
    labels : torch.Tensor
        Labels of shape ``(N,)``.

    Examples
    --------
    >>> import torch
    >>> from torch.utils.data import DataLoader, TensorDataset
    >>> from multiband_audio.data import collate_fn
    >>> samples = [(torch.randn(16000), 0), (torch.randn(8000), 1)]
    >>> waveforms, mask, labels = collate_fn(samples)
    >>> waveforms.shape
    torch.Size([2, 16000])
    >>> mask.shape
    torch.Size([2, 16000])
    >>> mask[1, 8000:].all()   # second sample is padded after 8000
    tensor(True)
    """
    waveforms, labels = zip(*batch)

    # Normalise to (T,) — strip leading channel dim if present
    waveforms = [w.squeeze(0) if w.ndim == 2 else w for w in waveforms]

    T_max = max(w.shape[-1] for w in waveforms)
    N = len(waveforms)

    padded = torch.zeros(N, T_max, dtype=waveforms[0].dtype)
    mask = torch.zeros(N, T_max, dtype=torch.bool)  # False = valid

    for i, w in enumerate(waveforms):
        L = w.shape[-1]
        padded[i, :L] = w
        mask[i, L:] = True  # True = padded

    if isinstance(labels[0], torch.Tensor):
        labels_t = torch.stack(labels)
    else:
        labels_t = torch.tensor(labels)

    return padded, mask, labels_t
