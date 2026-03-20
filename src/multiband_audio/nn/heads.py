"""Classification heads."""

from __future__ import annotations

import torch.nn as nn


class LinearHead(nn.Module):
    """Simple linear classification head with layer normalization.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    num_classes : int
        Number of output classes.

    Examples
    --------
    >>> import torch
    >>> head = LinearHead(256, 10)
    >>> x = torch.randn(2, 256)
    >>> head(x).shape
    torch.Size([2, 10])
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.flatten = nn.Flatten()
        self.ln = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: nn.Module) -> nn.Module:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape ``(N, input_dim)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(N, num_classes)``.
        """
        x = self.flatten(x)
        x = self.ln(x)
        return self.fc(x)
