"""Generate a small synthetic chirp for tests."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import torch


def generate_chirp(
    path: Path,
    sample_rate: int = 48000,
    duration: float = 1.0,
    f_start: float = 100.0,
    f_end: float = 20000.0,
) -> None:
    """Write a linear chirp signal as a mono 16-bit WAV file."""
    n_samples = int(sample_rate * duration)
    t = torch.linspace(0, duration, n_samples)
    freq = f_start + (f_end - f_start) * t / duration
    phase = 2 * torch.pi * torch.cumsum(freq, dim=0) / sample_rate
    waveform = 0.5 * torch.sin(phase)

    # Convert to 16-bit PCM
    samples_int16 = (waveform * 32767).clamp(-32768, 32767).to(torch.int16).numpy()

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(samples_int16)}h", *samples_int16))


if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "test_chirp.wav"
    generate_chirp(out)
    print(f"Generated {out} ({out.stat().st_size} bytes)")
