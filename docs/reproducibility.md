# Source provenance and validation

## Frozen algorithm

This anonymous snapshot contains the reference implementation used for the ReadPrune image experiments. Checksums for the included files are recorded in `RELEASE_MANIFEST.json`.

Public `_math.py` preserves the retained function bodies and removes the unused
exchange-repair branch. `selection.py` makes the FRA-Q2 call explicit:
`select(B, q**2, K)`. This gives the quality term `4 sum(log(q))`, not `2 sum(log(q))`.

Public `readout.py` uses the subsequently validated safetensors serialization
adapter, retains strict pairing with the native visual weights, and expands
the legacy attention `_shape` helper into the equivalent view/transpose
operations for Transformers 4.45.2. The response equations and content-length
weighted text pooling are unchanged.

The Qwen integration update additionally hashes native tensors through a byte
view so BF16 values need no unsupported NumPy conversion, and adapts the inner
CLIP text `return_dict` argument for Transformers 4.57.6 without changing its
pooling. The original 4.45.2 readout environment remains tested separately.

The spatial component preserves the original integer letterbox, bilinear
patch-center mapping, and floor-retention rules. It transfers signed vectors
before absolute-value sensitivity is calculated.

## Validation scope

The tests do not download models or data and do not require a GPU. They cover
direct greedy log-determinant selection, the q² convention, exact budget and
original vector preservation, attention-edge masking, saturated-attention
fallback, nonlinear readout equivalence, paired-checkpoint validation, long
text segmentation, and spatial mapping.

Run the complete suite with the optional readout dependencies installed:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m unittest discover -s tests -v
```

Tests with tiny random CLIP fixtures establish implementation properties;
they do not replace full-model numerical validation or benchmark evaluation.
Use the model's native preprocessing, prompt, feature layer, position IDs,
decoding settings, and official scoring procedure for reported experiments.

The [Qwen adapter](qwen.md) adds actual tiny Qwen2.5-VL and Qwen3-VL model tests
in its separate `qwen` environment. These include native full-budget logits
and greedy-ID parity, FP32/BF16 execution, compact KV caches, preserved mRoPE,
DeepStack support alignment, and local checkpoint/processor loading. Current
counts and source checksums are recorded in `RELEASE_MANIFEST.json`.

## Timing

The public selector is a CPU FP64 reference with numerical checks. It does not
include the optimized NeXT execution-graph runtime. Do not attribute that
runtime's reported latency numbers to this package without rerunning matched
complete-request measurements. Qwen additionally incurs the auxiliary CLIP
cost; shared-image caching and single-image requests should be reported
separately.
