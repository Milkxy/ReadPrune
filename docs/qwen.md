# Qwen2.5-VL and Qwen3-VL

The `qwen` extra adds a complete **single-image, batch-one, greedy generation**
adapter for dense Qwen2.5-VL and Qwen3-VL checkpoints. It includes auxiliary
CLIP scoring, native merger-token selection, physical prompt compression,
three-axis mRoPE preservation, and KV-cached decoding. Qwen3 DeepStack branches
are sliced at exactly the same support as the main merger output.

## Install in a separate environment

```bash
python3.10 -m venv .venv-qwen
source .venv-qwen/bin/activate
python -m pip install -e '.[qwen]'
```

This pins PyTorch 2.6.0, Torchvision 0.21.0, and Transformers 4.57.6. Use a
PyTorch build appropriate to your CUDA platform. The original `readout` extra
pins an older Transformers release for the original CLIP reference; install
**one extra per environment**, not `.[readout,qwen]`. Qwen includes the CLIP
readout dependencies itself.

Provide local Qwen weights, processor/tokenizer assets, and a paired CLIP
ViT-L/14-336 checkpoint containing `model.safetensors`. Model loading is local
only and uses safetensors. Weights and benchmark data are obtained separately.

## Generate an answer

Qwen2.5-VL, retaining `max(1, floor(0.2 * N))` native merger tokens:

```bash
python examples/qwen_generate.py \
  --model-dir checkpoints/Qwen2.5-VL-7B-Instruct \
  --clip-dir checkpoints/clip-vit-large-patch14-336 \
  --image example.jpg --question 'What is the person holding?' \
  --retain-ratio 0.2 --rounding floor --device cuda:0 \
  --check-parity --output outputs/qwen25.json
```

Qwen3-VL uses the same entry point, including its native DeepStack features:

```bash
python examples/qwen_generate.py \
  --model-dir checkpoints/Qwen3-VL-8B-Instruct \
  --clip-dir checkpoints/clip-vit-large-patch14-336 \
  --image example.jpg --question 'What is the person holding?' \
  --retain-ratio 0.2 --rounding ceil --use-fast-processor --device cuda:0 \
  --check-parity --output outputs/qwen3.json
```

Use `--k 128` instead of `--retain-ratio` for an exact token count; K must not
exceed the native count. The output records the original/retained counts,
indices, sequence lengths, budget rule, mRoPE delta, and DeepStack lengths.
`--full` runs an unpruned baseline without loading CLIP. `--check-parity`
compares the adapter's full-budget logits and greedy IDs with native HF
execution before running the requested budget. It adds validation work and
should not be included in timed requests.

`--question` supplies the raw selection question. `--prompt` can separately
supply the decoder's formatted prompt, including multiple-choice options.
The selector does not receive that formatted prompt or answer labels.

## Python API

```python
from PIL import Image
from readprune.qwen import QwenReadPrune

with QwenReadPrune.from_pretrained(qwen_directory, clip_directory) as runtime:
    with Image.open(image_path) as image:
        result = runtime.generate(
            image, 'What is the person holding?',
            retain_ratio=0.2, rounding='floor', max_new_tokens=64,
            check_parity=True,
        )
    print(result['answer'])
```

For existing model integrations, `generate_from_inputs` accepts native
processor tensors, the identical observed RGB image, and a `QwenSelector`.
`visual_features`, `pack_native`, and `greedy_decode` expose the individual
stages. Inputs must be unpadded and contain one image. The adapter uses one
device with matching native text/vision dtypes. CUDA defaults to Qwen BF16 and
auxiliary CLIP FP16; the readout tail is FP32 and selection is CPU FP64.

## Protocol choices and boundaries

- The original Qwen2.5 7B four-benchmark run used **floor** budgets; subsequent
  Qwen transfer/control runs used **ceil**. Choose explicitly for the specific
  table. For example, at N=17 and ratio=0.1, these retain 1 and 2 tokens.
- Both model branches observe the same RGB image. The auxiliary branch uses
  integer mean-color letterboxing and transfers signed response vectors before
  the absolute-value/deadzone quality transform. Native Qwen merger features
  supply both diversity and the unchanged decoder vectors.
- The CLI uses direct RGB input. For protocols that re-encoded images at JPEG
  quality 85, pass `--jpeg-quality 85`; both Qwen and CLIP then see the same
  decoded image. Resolution limits and the model's chat template still affect
  results. Image processing defaults to `use_fast=False`; select
  `--use-fast-processor` for protocols using the fast processor (including the
  later Qwen transfer runs). Fixed-resolution experiments require their corresponding prepared
  images and resolution configuration.
- Full-budget requests bypass auxiliary scoring and selection. Pruned
  requests recompute both vision branches; this implementation does not cache
  responses across requests or include the later optimized GPU selector.
- The decoder implements greedy argmax, EOS stopping, and an optional minimum
  token count. Sampling, beams, custom logits processors, batched/multiple
  images, video, Qwen3 MoE, device sharding, and training are outside this adapter.
- This code supplies model inference, not dataset downloaders or official
  benchmark scorers. It does not alone reproduce every manuscript table.
  Account for auxiliary CLIP, mapping, CPU transfers, selection, and generation
  in complete-request timing; token reduction alone does not establish speedup.

## Validation and provenance

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m unittest discover -s tests -v
```

`tests/test_qwen.py` uses real tiny random Hugging Face models, not downloaded
weights. It tests both families in FP32/BF16, native K=N logits and greedy-ID
parity, non-prefix supports, mRoPE and cache positions, DeepStack alignment,
consecutive requests, invalid layouts, and real auxiliary CLIP-to-Qwen
generation. These CPU tests do not establish pretrained-model accuracy or GPU
latency. Run `--check-parity` with the actual checkpoint before evaluation.

The adapter is extracted from the existing Qwen2.5 experiment implementation
and the subsequent Qwen2.5/Qwen3 transfer runtime. The latter source SHA-256
and the original selector SHA-256 are recorded in `RELEASE_MANIFEST.json`.
Public extraction removes experiment schedulers, labels/scorers, machine
paths, alternative baseline methods, and NDR aggregation. It reuses the
published ReadPrune quality/DPP and signed spatial mapping functions.

The pinned native interfaces are documented by
[Hugging Face Qwen2.5-VL](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_vl)
and [Qwen3-VL source](https://github.com/huggingface/transformers/tree/v4.57.6/src/transformers/models/qwen3_vl).
