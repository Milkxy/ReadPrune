# Evaluation

[← Back to ReadPrune](README.md)

This page distinguishes the validation commands available in **v0.1.0** from the model-specific integration needed for benchmark evaluation.

## 1. Validate the reference implementation

Install the package with the readout dependencies as described in the [README](README.md#-installation), then run:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m unittest discover -s tests -v
```

The release passed **25 tests with no skips** in the recorded reference environment. The tests use synthetic tensors and tiny random CLIP fixtures; they require neither pretrained weights nor benchmark data.

| Test file | Coverage |
|:--|:--|
| [`tests/test_core.py`](tests/test_core.py) | Greedy log-determinant selection, the `q²` convention, exact budget, and original token preservation. |
| [`tests/test_readout.py`](tests/test_readout.py) | Attention connection ablation, nonlinear readout equivalence, checkpoint pairing, long-question encoding, and tensor preservation. |
| [`tests/test_spatial.py`](tests/test_spatial.py) | Letterbox-to-grid transfer and retention-budget rules. |

An installed-package smoke check is also available:

```bash
python examples/select_from_features.py \
  --demo --k 64 --output outputs/demo.npz
```

This is a software check, not a benchmark score. Environment versions and source provenance are documented in [docs/reproducibility.md](docs/reproducibility.md) and [requirements-reference.txt](requirements-reference.txt).

## 2. Connect a benchmark model

Full benchmark launchers are not included in this release. Follow [docs/integration.md](docs/integration.md) to connect the reference operations to a model's native visual and generation paths.

For a matched comparison:

1. Keep model weights, image preprocessing, prompts, decoding settings, and evaluator versions fixed between the unpruned model and ReadPrune.
2. Align the readout response with the native projected patch features. Use the native features for both diversity and the selected output vectors.
3. Preserve all nonpatch tokens, spatial order, position conventions, masks, and model-specific separator tokens. Count patch and separator budgets separately.
4. Evaluate with the benchmark's prescribed split and metric. For server-scored benchmarks, predictions alone are not a final accuracy result.
5. Record the checkpoint, dataset split, token budget, configuration, and scoring procedure alongside each result.

For Qwen, the auxiliary frozen CLIP branch and grid transfer are required parts of the described adaptation. The spatial helper alone does not implement model generation. For NeXT, multi-view cropping and unpadding require a model-specific adapter.

## 3. Measure inference cost

The public selector is a **CPU FP64 reference implementation**. The optimized NeXT execution-graph runtime used in latency experiments is not included in v0.1.0, so its reported speedup must not be attributed directly to this package.

For complete-request measurements, include preprocessing, the native visual forward, readout, selection, sequence assembly, generation, and output decoding. Include auxiliary CLIP image/text computation and spatial transfer when applicable. Warm up the model and synchronize GPU work at the timing boundaries; keep the same execution optimizations, generation settings, and hardware for both methods.

Report single-image requests and shared-image caching separately. A reduction in retained patches establishes a token-budget reduction, not an end-to-end speedup by itself.

## 4. Release traceability

The [v0.1.0 release](https://github.com/Milkxy/ReadPrune/releases/tag/v0.1.0) provides the tested source ZIP and its SHA-256 checksum. `RELEASE_MANIFEST.json` records source-file checksums and validation metadata. Documentation updates on `main` do not replace the versioned archive or rerun the underlying experiments.
