"""Neural network wrappers and classification heads."""

from multiband_audio.nn.heads import LinearHead
from multiband_audio.nn.wrapper import MultibandWrapper

__all__ = [
    "LinearHead",
    "MultibandWrapper",
]
