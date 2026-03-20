"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn


@pytest.fixture
def test_data_dir() -> Path:
    """Path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def test_chirp_path(test_data_dir: Path) -> Path:
    """Path to the test chirp wav file."""
    return test_data_dir / "test_chirp.wav"


@pytest.fixture
def dummy_backbone() -> nn.Module:
    """A simple linear backbone for testing: (N, T) -> (N, 256)."""
    return nn.Sequential(
        nn.AdaptiveAvgPool1d(1),
    )


class ToyBackbone(nn.Module):
    """Backbone that maps (N, T) -> (N, embed_dim) for testing."""

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.pool(x).squeeze(1)


@pytest.fixture
def toy_backbone() -> ToyBackbone:
    """Toy backbone that maps arbitrary waveform to (N, 256)."""
    return ToyBackbone(embed_dim=256)
