# Connecting the reference implementation to a VLM

The released package exposes readout and selection operations. It does not
replace a model's tokenizer, image processor, prompt template, or decoder.

## CLIP-based LLaVA

Use the exact native vision weights, preprocessing, and native projected patch
features from the model under evaluation. Create `Readout` before the native
visual forward so its hook captures the final block input. The paired local
CLIP checkpoint must contain `model.safetensors`, `config.json`, and tokenizer
files; native visual tensors are compared strictly at their native dtype.

```python
from readprune import select_tokens
from readprune.readout import Readout

reader = Readout(native_clip_vision_model, paired_clip_directory, device)
try:
    # Run your model's unchanged native visual path once.
    X = native_visual_forward(image)   # [N, decoder_feature_dim]
    H = reader.pop_H()
    R = reader.image(H)["R"]
    t = reader.text(raw_question)["t"]
    selected = select_tokens(X, R @ t, k=64)
    X_selected = selected["tokens"]
    # Feed X_selected through the model's native multimodal sequence assembly.
finally:
    reader.close()
```

The names `native_visual_forward` and `native_clip_vision_model` above denote
your model's integration points, not functions implemented by this package.
The selected vectors must use the same image view and patch ordering as `R`.
Retain all nonpatch tokens and follow the model's position/mask conventions.
For LLaVA-NeXT, native crops, spatial unpadding, and separator placement require
the model-specific mapping; do not apply a flat single-view mapping to them.

## Qwen

Qwen has no CLIP CLS token. The paper's adapter uses an **additional frozen
CLIP** branch and transfers signed response vectors onto the native merger
grid. `readprune.spatial.map_letterbox_features` implements the original
single-image letterbox-to-grid bilinear transfer, and `retain_budget` gives
`floor(ratio * N)` with bounds. Apply transfer before absolute value/deadzone.
Use native Qwen vectors for geometry and output; do not replace them with CLIP
features. Position IDs, grid metadata, and attention-mask assembly remain the
responsibility of the model adapter.

The geometry helper is not a complete Qwen generation adapter. Its validity
requires the documented mean-color square letterbox and matching target-grid
convention; other preprocessing or video pooling needs a different mapping.
The auxiliary branch costs time and memory and must be included in timings.

## Scope of this first release

This is a reference core release, with synthetic tests and a standalone CLIP
example. Full benchmark launchers, optimized NeXT execution graphs, and video
inference adapters are not part of this release. Consequently, running this
CPU reference selector does not by itself reproduce every paper table or its
optimized end-to-end latency numbers.
