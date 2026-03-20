# Changelog

## [0.1.0] — 2026-03-20

Initial release.

### Added

- `MultibandTransform` — splits a waveform at any sample rate into non-overlapping 8 kHz frequency bands via heterodyne down-conversion and librosa kaiser_best resampling
- `MultibandTransformDynamic` — same as above but determines the number of bands at runtime, for datasets with variable sample rates
- Five fusion strategies matching the Interspeech 2026 paper:
  - `MeanPoolFusion` (`"mp"`) — unweighted average of band embeddings
  - `GatedPoolFusion` (`"gp"`) — learned softmax-weighted sum
  - `MoEFusion` (`"moe"`) — per-band classifiers with learned logit weighting
  - `HybridFusion` (`"hyb"`) — gating conditioned on both embeddings and handcrafted spectral features (entropy, flux)
  - `SelfAttentionFusion` (`"sa"`) — transformer encoder with [CLS] token over band embeddings
- `MultibandWrapper` — end-to-end module combining backbone, fusion, and classification head; handles `(N, B, T)` → `(N*B, T)` reshaping for efficient shared backbone inference
- `collate_fn` — DataLoader collate function for variable-length recordings; zero-pads to longest in batch and returns a boolean padding mask (`True` = padded position)
- `padding_mask` support in `MultibandTransform`, `MultibandTransformDynamic`, and `MultibandSelectiveTransform` — accepts `(N, T)` mask, scales it to band output length via nearest interpolation, returns it alongside bands for direct forwarding to `MultibandWrapper`
- `padding_mask` support in `MultibandWrapper` — correctly tiles `(N, T)` mask to `(N*B, T)` for transformer backbones (BEATs, EAT)
- `freeze_backbone` mode in `MultibandWrapper` — keeps backbone in eval mode and disables gradients for linear probing
- Baseband bypass in the heterodyne transform — band 0 (0–8 kHz) is resampled directly without bandpass/heterodyne, matching the standard baseline path
- `build_fusion(name, **kwargs)` factory with registry of all fusion strategies
- `LinearHead` classification head
- Band selector utilities: `EntropyBandSelector`, `FluxBandSelector`
- `MultibandSelectiveTransform` — selects top-k informative bands before encoding
- `py.typed` marker — package ships inline type annotations (PEP 561)
