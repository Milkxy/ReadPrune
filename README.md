<div align="center">

# Read Before You Prune
### Training-Free Visual Token Compression via Question-Conditioned Readout Sensitivity

**ReadPrune · Reference Implementation**

[Quick Start](#-quick-start) · [Core Implementation](#-core-implementation) · [Evaluation](Evaluation.md) · [Source Release](https://github.com/Milkxy/ReadPrune/releases/tag/v0.1.0)

</div>

ReadPrune selects visual tokens by measuring **how a frozen visual readout responds to the removal of a patch connection, conditioned on the question**. It combines this response with native-feature diversity to retain an exact budget of informative, complementary patches. Selected tokens keep their original representations and spatial order.

## 🎯 News

- **2026-10-29:** Released [v0.1.0](https://github.com/Milkxy/ReadPrune/releases/tag/v0.1.0), including the reference selector, frozen CLIP readout, spatial mapping, examples, and offline tests.

## 💡 Highlights

- **Question-conditioned readout.** Suppress one CLS-to-patch attention connection at a time in the final frozen CLIP block, recompute the nonlinear CLS readout, and measure its change along the question embedding.
- **Joint quality and diversity.** Use response magnitude to weight a DPP selector built from native VLM features, balancing informative patches with complementary visual evidence.
- **Exact token budget.** Retain exactly `K` original patch vectors in their original order, without merging or updating their representations.
- **Training-free selection.** Use frozen encoders and forward computations, without gradient estimation or model training.

The released method is **ReadPrune**, previously named **FRA-Q2**. Its readout measures a local CLIP intervention; it is not an attribution of the language model's final answer. See the [method specification](docs/method.md) for the exact equations and `q²` convention.

## 👨‍💻 Release Checklist

- [x] Frozen CLIP readout and question encoding
- [x] Exact-budget, CPU FP64 quality–diversity selector
- [x] Auxiliary CLIP-to-Qwen spatial mapping component
- [x] Feature-selection and standalone CLIP examples
- [x] Offline validation suite — 25 tests passed in the reference environment
- [ ] Complete model-specific benchmark runners
- [ ] Optimized LLaVA-NeXT inference runtime
- [ ] Video inference adapter

This release contains the reference core. Model weights, benchmark datasets, and machine-specific launch configurations are obtained or configured separately. NDR/aggregation is not part of ReadPrune in this repository.

## 🔧 Installation

The reference environment uses **Python 3.10**. Start with the selector and its NumPy/SciPy dependencies:

```bash
git clone https://github.com/Milkxy/ReadPrune.git
cd ReadPrune
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For CLIP readout and the complete test suite, install the optional dependencies:

```bash
python -m pip install -e '.[readout]'
```

**Compatibility.** Readout is validated with PyTorch **2.1.2** and Transformers **4.45.2**. Use a PyTorch build compatible with your platform and CUDA runtime. The tested package versions are recorded in [requirements-reference.txt](requirements-reference.txt); compatibility with later Transformers releases has not been established.

## ⭐️ Quick Start

### 1. Select tokens without downloading a model

```bash
python examples/select_from_features.py \
  --demo --k 64 --output outputs/demo.npz
```

This seeded synthetic example selects 64 of 576 patches and verifies that the returned vectors are unchanged. It checks the software path, not benchmark accuracy.

For your own features, provide an NPZ file with `features` of shape `[N, D]` and signed `response_deltas` of shape `[N]`:

```bash
python examples/select_from_features.py \
  --input features.npz --k 64 --output outputs/selected.npz
```

### 2. Use the Python API

```python
from readprune import select_tokens

# X: original native projected patch features, [N, D]
# delta: (full_readout - ablated_readout) @ question_embedding, [N]
result = select_tokens(X, delta, k=64)

indices = result["indices"]  # sorted original patch positions
selected = result["tokens"] # X[indices]; original values, dtype, and device
```

`select_tokens` computes the quality scores and applies their square exactly once. Its output `quality` contains the raw scores; see [the method contract](docs/method.md) before substituting another DPP implementation.

### 3. Run a standalone CLIP example

Provide an image and a local, paired CLIP checkpoint with `model.safetensors`, `config.json`, processor configuration, and tokenizer files:

```bash
python examples/clip_readout.py \
  --clip-dir checkpoints/clip-vit-large-patch14-336 \
  --image example.jpg \
  --question 'What is the person holding?' \
  --k 64 --device cuda \
  --output outputs/readout_demo.npz
```

The example loads local safetensors weights and does not download a checkpoint. It demonstrates readout and selection using **CLIP patch geometry**. The paper's VLM integration instead uses **native projected VLM features** with matching views and patch positions.

## 🧩 Core Implementation

| Component | Code entry point | Purpose |
|:--|:--|:--|
| Connection ablation | [`ablate_cls_connections`](readprune/readout.py#L69) | Remove a patch connection and renormalize attention while retaining the CLS self-connection. |
| Visual readout | [`Readout.image`](readprune/readout.py#L311) | Recompute the nonlinear CLS tail and return per-patch readout changes. |
| Question encoding | [`Readout.text`](readprune/readout.py#L345) | Encode the question, with explicit segmentation for long text. |
| Token selection | [`select_tokens`](readprune/selection.py#L10) | Combine response-derived quality with native-feature diversity and gather exactly `K` original vectors. |
| Qwen grid transfer | [`map_letterbox_features`](readprune/spatial.py#L30) | Transfer signed auxiliary responses to the native grid before computing sensitivity. |

```text
ReadPrune/
├── readprune/                 # Readout, selector, and spatial mapping
├── examples/                  # Runnable feature and CLIP demonstrations
├── tests/                     # Offline numerical and interface checks
├── docs/                      # Method, integration, and provenance
├── Evaluation.md              # Validation and benchmark integration guide
├── pyproject.toml             # Package and optional dependencies
└── requirements-reference.txt # Tested environment
```

## 🔌 Model Integration

ReadPrune operates between native visual feature extraction and multimodal sequence assembly. The [integration guide](docs/integration.md) explains the required hooks, feature alignment, and sequence conventions.

| Model family | Integration requirement |
|:--|:--|
| CLIP-based LLaVA | Pair the readout with the native vision weights and select native projected patches from the same image view. |
| LLaVA-NeXT | Preserve crop ordering, spatial unpadding, and separator placement through a model-specific adapter. |
| Qwen | Use an additional frozen CLIP branch for the response, transfer it to the native grid, and retain Qwen's original features and position conventions. |

The Qwen mapping helper is a component, not a complete generation adapter. The current examples do not provide a one-command reproduction of all manuscript tables. Benchmark and timing requirements are collected in [Evaluation.md](Evaluation.md).

## 📊 Evaluation

Run the offline tests after installing the readout dependencies:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m unittest discover -s tests -v
```

The release passed **25 tests with no skips**, covering selection, readout, checkpoint pairing, long questions, and spatial mapping. Without the optional dependencies, readout-related tests are skipped. These checks establish implementation properties and do not substitute for full-model evaluation.

For benchmark setup and complete-request timing, see the [evaluation guide](Evaluation.md). The public CPU reference selector does not include the optimized NeXT runtime; Qwen timing must also account for its auxiliary CLIP branch.

## 📚 Documentation

- [Method specification](docs/method.md)
- [Model integration](docs/integration.md)
- [Evaluation guide](Evaluation.md)
- [Source provenance and validation](docs/reproducibility.md)
- [Third-party dependencies](THIRD_PARTY_NOTICES.md)
- [Versioned source archive](https://github.com/Milkxy/ReadPrune/releases/tag/v0.1.0)

## ❤️ Acknowledgments

We thank the developers of [CLIP](https://github.com/openai/CLIP), [Transformers](https://github.com/huggingface/transformers), [LLaVA](https://github.com/haotian-liu/LLaVA), [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT), and [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) for their open research software. Repository documentation is organized with reference to [PyramidDrop](https://github.com/Cooperx521/PyramidDrop).

<details>
<summary>中文说明</summary>

本仓库发布 ReadPrune（原 FRA-Q2）的核心参考实现。方法通过冻结 CLIP 的连接消融响应衡量视觉 patch 与问题的关联，并结合原生特征的多样性，选择固定数量的原始 Token，保持其表示和空间顺序。

当前版本包含选择器、读出模块、Qwen 空间映射组件、可运行示例和离线测试。完整模型评测脚本、NeXT 优化运行时与视频接口尚未发布；Qwen 适配需要额外冻结 CLIP，其成本应纳入效率测量。安装与使用请按上方说明进行。

</details>
