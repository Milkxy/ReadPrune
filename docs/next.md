# LLaVA-NeXT with uniform crop budgets

This adapter runs single-image greedy generation with a local Hugging Face
`LlavaNextForConditionalGeneration` checkpoint, its native CLIP vision tower,
projector and language model. It supports CLIP feature layer `-2` and the
`default` patch-only strategy on one device. Model weights remain frozen.

## Installation and inputs

Use a separate Python 3.10 environment:

```bash
python -m pip install -e '.[next]'
python examples/next_generate.py \
  --model-dir checkpoints/llava-next \
  --clip-dir checkpoints/clip-vit-large-patch14-336 \
  --image example.jpg --question 'What is the person holding?' \
  --fixed-grid-size 672 --k 160 --device cuda:0 \
  --output outputs/next_answer.json
```

The model directory must be in Hugging Face format and include its processor,
tokenizer and chat template. All weights are loaded locally. The paired full
CLIP directory must contain `model.safetensors`, configuration and tokenizer
files. The adapter checks every native vision tensor against this checkpoint
at the native dtype; an unrelated CLIP checkpoint is rejected. Use `--prompt`
for a separate decoder prompt/options; selection always uses the raw question.

With a 336-pixel vision tower, `--fixed-grid-size 672` gives one base view plus
four tiles. Omit this option to use the checkpoint's native any-resolution
candidate grid. Changing the grid changes the experiment protocol: record it
and keep it identical for the baseline and pruned runs. `--full` replaces
`--k` and runs native unpruned generation without loading the readout branch.

## Allocation and selection

For total patch budget `K` and `C` crops, compute `base, remainder = divmod(K, C)`.
Crop `i` receives `base + (i < remainder)` patches, in processor crop order.
The base view participates in exactly the same allocation as every tile.
Require `C <= K <= C * patches_per_crop`; each crop retains at least one patch.

| Total budget | Five-crop allocation |
| --- | --- |
| 160 | 32, 32, 32, 32, 32 |
| 320 | 64, 64, 64, 64, 64 |
| 640 | 128, 128, 128, 128, 128 |
| 163 | 33, 33, 33, 32, 32 |

Each crop uses its own captured final-block input for the paired CLIP readout.
Signed readout changes are projected onto the same raw-question embedding.
The shared CPU FP64 selector combines these responses with that crop's native
projected features. There is only one allocation policy: uniform.

## Sequence packing

The compressed path gathers unchanged base-view vectors first, then selected
tile vectors in global spatial raster order. The tile support mask undergoes
the same spatial reshape and permutation as the features. This path uses
spatial packing **without unpadding or row separators**, retaining exactly
`K` visual patch vectors. It replaces the processor-expanded image span with
these vectors and runs the native language model with a compact attention mask
and KV cache. Text embeddings are preserved.

The unpruned `--full` path keeps native `spatial_unpad` packing and row
separators. Thus `K = C * patches_per_crop` is not a claim of sequence parity
with `--full`: their separator/unpadding conventions differ. Report patch
counts separately from native packed sequence lengths. The JSON audit includes
both lengths, per-crop budgets, selected indices and grid shape.

## Validation and scope

`tests/test_next.py` covers allocation including remainders, spatial ordering
with noncontiguous supports, unchanged vectors, pairing with a real tiny CLIP,
real tiny NeXT prefill/KV-cache generation in FP32 and BF16, native baseline
parity and rejection of inconsistent image spans/crops. Run the full suite:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest discover -s tests -v
```

This is a portable reference integration of the uniform selection and packing
rule. It does not include the optimized execution-graph implementation used
for latency experiments, dataset scorers, model weights or benchmark data.
Tiny-model validation does not establish pretrained-model accuracy or GPU
speedups. Measure the full request, including readout and CPU selection, when
using this implementation for timing.
