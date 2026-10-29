# Source provenance and validation

## Frozen algorithm

The original reference sources were frozen for the ReadPrune/FRA-Q2 image
experiments. SHA-256 identifiers of those original files are:

| Component | Original SHA-256 |
|---|---|
| FP64 set mathematics | `00ad89f24eb62a12b7ac94341f7d1a887373484f6dd3b702ebdc5a066dfa3054` |
| CLIP readout | `05c0113bf66ebef47d573e256c8fbd1ec0eaea1566a6e4a3dd3fa261af8de8dd` |

Public `_math.py` preserves the retained function bodies and removes the unused
exchange-repair branch. `selection.py` makes the FRA-Q2 call explicit:
`select(B, q**2, K)`. This gives the quality term `4 sum(log(q))`, not `2 sum(log(q))`.

Public `readout.py` uses the subsequently validated safetensors serialization
adapter, retains strict pairing with the native visual weights, and expands
the legacy attention `_shape` helper into the equivalent view/transpose
operations for Transformers 4.45.2. The response equations and content-length
weighted text pooling are unchanged.

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

## Timing

The public selector is a CPU FP64 reference with numerical checks. It does not
include the optimized NeXT execution-graph runtime. Do not attribute that
runtime's reported latency numbers to this package without rerunning matched
complete-request measurements. Qwen additionally incurs the auxiliary CLIP
cost; shared-image caching and single-image requests should be reported
separately.
