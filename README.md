# multiband-audio

[![arXiv](https://img.shields.io/badge/arXiv-2604.27936-b31b1b.svg)](https://arxiv.org/abs/2604.27936)
[![PyPI](https://img.shields.io/pypi/v/multiband-audio.svg)](https://pypi.org/project/multiband-audio/)
![CI status](https://github.com/earthspecies/multiband-audio/actions/workflows/publish.yml/badge.svg?branch=main)
![Pre-commit status](https://github.com/earthspecies/multiband-audio/actions/workflows/pre-commit.yml/badge.svg?branch=main)

A PyTorch-based toolkit for splitting bioacoustic audio samples into frequency bands via heterodyning and fusing per-band embeddings into a single representation.

## Description

Animals hear and vocalize across frequency ranges that differ substantially from humans, often extending into the ultrasonic domain. Yet most computational bioacoustics systems currently rely on standard audio models pre-trained at 16 kHz, corresponding to the human audible range. Typical approaches either resample a given input to the 0-8 kHz baseband and discard this high-frequency content entirely (*baseband*), or slow down the recording to lower the high-frequency information (*time-expansion*), which expands the signal and reduces spectral resolution.

This toolkit provides a _third_ option: **adaptive multi-band encoding**, allowing pre-trained audio models to access the full-spectrum of bioacoustic recordings through heterodyning and learned fusion strategy.

<img src="img/bands.jpg" alt="header" width="1000"/>

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
audio, sample_rate = librosa.load("assets/bat_call.wav", sr=None)
waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, T)

transform = mba.MultibandTransform(sample_rate=sample_rate)
bands = transform(waveform)
print(bands.shape)
# torch.Size([1, B, T']), B = num_bands, and T' = resampled length

```

The number of bands is determined automatically from the sample rate:

| Vocalization | Recording SR    | # Bands |
|--------------|-----------------|-------|
| Bat call     | 250 kHz         | 16    |
| Dog bark     | 44.1 kHz        | 3     |
| Bird song    | 44.1 kHz        | 3     |

### 2. Extract embeddings from a pre-trained model

Run your frozen pre-trained encoder on each band independently.

#### Backbone requirements

A `backbone` encoder model must simply contain the following properties:

- A `forward(x)` method which accepts a 2-D tensor `x` of shape `(N, T)`. These waveforms are at the **baseband sample rate** (default 16 kHz), *not* the original recording's sample rate. The baseband SR is set by `HeterodyneCfg.baseband_sr` and should match whatever sample rate your backbone expects.
- Returns a 2-D tensor of shape `(N, D)` where `D` is a fixed embedding dimension matching `embed_dim`.
- *(Optional)* `forward(x, padding_mask=mask)` accepts a `(N, T)` boolean mask if you want padding-mask support.

Most audio backbones don't work out of the box, as they either return a sequence of frames or expect images. A thin wrapper is usually needed. Two common cases:

#### A) Waveform-based models

`torchaudio.pipelines.WAV2VEC2_BASE` provides a pretrained `Wav2Vec2Model` (16 kHz, 768-d). Its `forward(x)` returns `(features, lengths)` with `features.shape == (N, frames, 768)`. We mean-pool over frames to get `(N, 768)`:

```python
import torch.nn as nn
import torchaudio

# Backbone
class Wav2Vec2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
    def forward(self, x):
        feats, _ = self.model(x)        # (N, frames, 768)
        return feats.mean(dim=1)        # (N, 768)

# Wrapper
wrapper = mba.MultibandWrapper(
    backbone=Wav2Vec2Backbone(),
    fusion="gp",
    head=mba.LinearHead(768, num_classes=10),
    embed_dim=768,
    freeze_backbone=True,
)

logits = wrapper(bands)
print(logits.shape)
# torch.Size([1, 10])
```

#### B) Image-based CNN models (e.g. EfficientNet, ResNet, ViT)

Image CNNs from `torchvision` expect a 4-D image input `(N, C, H, W)`. We can use `mba.SpectrogramBackbone` to bridge: it computes a log mel-spectrogram, and expands to N channels before forwarding:

```python
import torchvision

# Backbone
img_cnn = torchvision.models.efficientnet_b0(num_classes=1280)
backbone = mba.SpectrogramBackbone(img_cnn, sample_rate=16000)

# Wrapper
wrapper = mba.MultibandWrapper(
    backbone=backbone,
    fusion="gp",
    head=mba.LinearHead(1280, num_classes=10),
    embed_dim=1280,
    freeze_backbone=True,
)
logits = wrapper(bands)
# torch.Size([1, 10])
```

### 3. Use a fusion module directly

If you already have extracted band-level embeddings, you can directly learn a fusion module:

```python
# Embeddings: (batch, num_bands, embed_dim)
embeddings = torch.randn(8, 16, 1280)

# GP fusion and linear probing
fusion = mba.GatedPoolFusion(embed_dim=1280)
head = mba.LinearHead(input_dim=1280, num_classes=10)
logits = head(fusion(embeddings))
# torch.Size([8, 10])
```

### 4. Variable-length batches and padding masks

This toolkit also contains a [`collate_fn`](multiband_audio/data.py) which can be used for padding masks.

```python
from torch.utils.data import DataLoader
import multiband_audio as mba

# `dataset` is any torch Dataset returning (waveform, label) tuples,
# where waveform is a 1-D tensor (T,) at sample rate `sample_rate`.
loader = DataLoader(dataset, batch_size=16, collate_fn=mba.collate_fn)
transform = mba.MultibandTransform(sample_rate=sample_rate)

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
| **Self-Attention** | `sa` | `SelfAttentionFusion` | Transformer over band embeddings; `[CLS]` token output is the fused representation. |

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
audio_bird, sr_bird = librosa.load("assets/bird_call.wav", sr=None)
audio_bat,  sr_bat  = librosa.load("assets/bat_call.wav",  sr=None)

waveform_bird = torch.from_numpy(audio_bird).unsqueeze(0)
waveform_bat  = torch.from_numpy(audio_bat).unsqueeze(0)

bands_bird, _, band_info = transform(waveform_bird, sample_rate=sr_bird)  # 3 bands
bands_bat,  _, band_info = transform(waveform_bat,  sample_rate=sr_bat)   # 16 bands
```

## Notebook

A more detailed walkthrough example is provided in a Jupyter [notebook](notebooks/demo.ipynb) on sample bat, bird, and dog vocalizations.

## Cite

This repository contains the source code used for the paper *Beyond the Baseband: Adaptive Multi-Band Encoding for Full-Spectrum Bioacoustics Classification* by Sarkar et al. (2026). If you use this toolkit, please cite:

```
@misc{sarkar2026basebandadaptivemultibandencoding,
      title={Beyond the Baseband: Adaptive Multi-Band Encoding for Full-Spectrum Bioacoustics Classification},
      author={Eklavya Sarkar and Marius Miron and David Robinson and Gagan Narula and Milad Alizadeh and Ellen Gilsenan-McMahon and Emmanuel Chemla and Olivier Pietquin and Matthieu Geist},
      year={2026},
      eprint={2604.27936},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.27936},
}
```

## Contact

For any questions or issues, kindly contact the [author](mailto:eklavya@earthspecies.org) or open a GitHub issue.
