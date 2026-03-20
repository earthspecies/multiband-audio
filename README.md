# multiband-audio

[![arXiv](https://img.shields.io/badge/arXiv--b31b1b.svg)](https://arxiv.org/abs/)

Animals hear and vocalize across frequency ranges that differ substantially from humans, often extending into the ultrasonic domain. Yet most computational bioacoustics systems currently rely on audio models pre-trained at 16 kHz, corresponding to the human audible range, and resample any given input to the 0-8 kHz baseband and discarding higher-frequency information present in many bioacoustic recordings.

<img src="img/bands.jpg" alt="header" width="1000"/>

Typical approaches either discard this high-frequency content entirely (*baseband*) or slow down the recording to lower the high-frequency information (*time-expansion*), which expands the signal and reduces spectral resolution. 

This toolkit provides a third option: **adaptive multi-band encoding**, allowing pre-trained audio models to access the full-spectrum of bioacoustic recordings through heterodyning and learned **fusion** strategy.

## Adaptive Multi-Band Encoding

<img src="img/pipeline.jpg" alt="header" width="1000"/>

Given a recording at any sample rate, the input signal is split into *B* non-overlapping frequency bands (e.g. of 8 kHz each). Each non-baseband band is then heterodyned (mixed) down to the 0–8 kHz baseband, making it compatible with any standard pre-trained audio model. 

Applying this to each band produces *B* baseband waveforms, each representing a distinct portion of the original spectrum. We resample them to match the SR expected by the pre-trained model, and then pass them individually through the frozen encoder to obtain one embedding per band. Finally, a learned fusion module combines them into a single representation for classification.

## Installation

This package requires python >= 3.10.

Install with pip:
```bash
pip install multiband-audio
```

Install with [uv](https://github.com/astral-sh/uv):

```bash
uv add multiband-audio
```

## Usage

### 1. Split a recording into frequency bands

```python
import multiband_audio as mba
import librosa
import torch

# Load any recording in its native sample rate (e.g. 250 kHz)
audio, sample_rate = librosa.load("bat_call.wav", sr=None)
waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, T)

transform = mba.MultibandTransform(sample_rate=sample_rate)
bands = transform(waveform) # (1, 16, 16000)
```

The number of bands is determined automatically from the sample rate:

| Vocalization | Recording SR    | # Bands |
|--------------|-----------------|-------|
| Bat call     | 250 kHz         | 16    |
| Dog bark     | 44.1 kHz        | 3     |
| Bird song    | 44.1 kHz        | 3     |

### 2. Extract embeddings from a pre-trained model

Run your frozen pre-trained encoder on each band independently. `MultibandWrapper` handles this automatically, reshaping `(N, B, T)` → `(N * B, T)` for efficient batched inference:

```python
# Take pre-trained model
model = torchvision.models.efficientnet_b0()

# Wrap with mba
wrapper = mba.MultibandWrapper(
    backbone=model,
    fusion="moe",
    head=mba.LinearHead(1280, num_classes=10),
    embed_dim=1280,
    freeze_backbone=True,
)

# Get logits
logits = wrapper(bands)
```

### 3. Use a fusion module directly

If you already have extracted band-level embeddings:

```python
# Embeddings: (batch, num_bands, embed_dim)
embeddings = torch.randn(8, 16, 1280)

# GP fusion and linear probing
fusion = mba.GatedPoolFusion(embed_dim=1280)
head = mba.LinearHead(input_dim=1280, num_classes=10)
logits = head(fusion(embeddings)) # (8, 10)
```

### 4. Variable-length batches and padding masks

This toolkit also contains a [`collate_fn`](multiband_audio/data.py) which can be used for padding masks.

```python
from torch.utils.data import DataLoader
import multiband_audio as mba

loader = DataLoader(dataset, batch_size=16, collate_fn=mba.collate_fn)

transform = mba.MultibandTransform(sample_rate=sr)

for waveforms, padding_mask, labels in loader:
    bands, band_mask = transform(waveforms, padding_mask=padding_mask)
    logits = wrapper(bands, padding_mask=band_mask)
```

When input recordings have different lengths, `collate_fn` zero-pads them to the longest sample in a batch, and creates a mask marking the invalid positions. Giving it to `MultibandTransform` with `padding_mask` returns a scaled `band_mask` alongside the bands that can be forwarded to the wrapper.

## Fusion Strategies

Five fusion methods are evaluated in the paper and implemented in this toolkit:

| Name | Key | Class | Description |
|------|-----|-------|-------------|
| **Mean-Pool** | `mp` | `MeanPoolFusion` | Unweighted average. No learnable parameters. |
| **Gated-Pool** | `gp` | `GatedPoolFusion` | Softmax-weighted sum, one learned gate per band. |
| **Mixture-of-Experts** | `moe` | `MoEFusion` | Per-band classifiers, learned logit weighting. |
| **Hybrid** | `hyb` | `HybridFusion` | Gate uses both embeddings and spectral features (entropy, flux). |
| **Self-Attention** | `sa` | `SelfAttentionFusion` | Transformer over band embeddings with [CLS] token. |

Example:

```python
# Build any fusion by name
fusion = mba.build_fusion("gp", embed_dim=1280)
```

## Variable Sample Rates

If your dataset contains recordings at different sample rates, use `MultibandTransformDynamic` which computes the number of bands at runtime:

```python
# Target_sr=16_000 by default
transform = mba.MultibandTransformDynamic()

# Each file can have a different sample rate
audio_bird, sr_bird = librosa.load("bird.wav", sr=None)
audio_bat,  sr_bat  = librosa.load("bat.wav",  sr=None)

waveform_bird = torch.from_numpy(audio_bird).unsqueeze(0)
waveform_bat  = torch.from_numpy(audio_bat).unsqueeze(0)

bands_bird, _, band_info = transform(waveform_bird, sample_rate=sr_bird)  # 3 bands
bands_bat,  _, band_info = transform(waveform_bat,  sample_rate=sr_bat)   # 16 bands
```

## Cite

This repository contains the source code used for the paper *Beyond the Baseband: Adaptive Multi-Band Encoding for Full-Spectrum Bioacoustics Classification* by Sarkar et al. (2026). If you use this toolkit, please cite:

<!-- ```bib
@INPROCEEDINGS{Sarkar_Baseband_2026,
         author = {Sarkar, Eklavya},
          title = {Beyond the Baseband: Adaptive Multi-Band Encoding for Full-Spectrum Bioacoustics Classification},
      booktitle = {},
           year = {2026},
} 
```-->

## Contact

For any questions or issues, kindly contact the [author](mailto:eklavya@earthspecies.org) or open a GitHub issue.
