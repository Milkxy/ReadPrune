# ReadPrune

**Read Before You Prune: Training-Free Visual Token Compression via
Question-Conditioned Readout Sensitivity**

ReadPrune scores visual tokens by how a frozen CLIP readout changes when one
CLS-to-patch attention connection is suppressed. It combines the magnitude of
the question-conditioned response with native-feature diversity, selects
exactly K patches, and preserves their original vectors and spatial order.

This is the **reference core implementation** of ReadPrune, previously named
FRA-Q2. The first source release includes:

- Frozen CLIP connection ablation and nonlinear image/text readout.
- The paper's `q²` quality-weighted, CPU FP64 DPP selector.
- The auxiliary CLIP-to-Qwen spatial mapping component.
- Runnable feature-selection and standalone CLIP examples.
- Offline tests using synthetic tensors and tiny random model fixtures.

The release excludes NDR/aggregation, pretrained weights, benchmark data, and
server launch scripts. Model-specific benchmark runners and the optimized
NeXT execution-graph path are not included; this is not yet a one-command
reproduction of all manuscript tables. See [integration scope](docs/integration.md).

## Installation

Use Python 3.10. For selection from existing features:

```bash
git clone https://github.com/Milkxy/ReadPrune.git
cd ReadPrune
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For CLIP readout and the complete test suite:

```bash
python -m pip install -e '.[readout]'
```

Install a PyTorch wheel compatible with your platform/CUDA runtime. The
reference environment is recorded in `requirements-reference.txt`; readout is
validated with Transformers 4.45.2, not arbitrary later versions.

## Quick start: no model download

```bash
python examples/select_from_features.py --demo --k 64 --output outputs/demo.npz
python -m unittest discover -s tests -v
```

The demo uses seeded random features, so it is a software check, not an
accuracy experiment. Tests requiring the optional readout dependencies are
skipped if those dependencies are absent.

For real features, provide an NPZ containing `features` of shape `[N,D]` and
signed `response_deltas` of shape `[N]`:

```bash
python examples/select_from_features.py --input features.npz --k 64 --output outputs/selected.npz
```

Python API:

```python
from readprune import select_tokens

# X: original native projector features, [N, D]
# delta: (e_full - e_ablated) @ question_embedding, [N]
result = select_tokens(X, delta, k=64)
indices = result['indices']   # sorted original patch positions
tokens = result['tokens']     # X[indices], unchanged dtype/device/values
```

`select_tokens` applies the squared quality exactly once. Passing its raw
`quality` into an unrelated DPP implementation without matching this convention
does not reproduce FRA-Q2.

## Standalone CLIP readout

Provide a local, paired CLIP checkpoint containing `model.safetensors`,
`config.json`, processor configuration, and tokenizer files. This package does
not download checkpoints or deserialize pickle model files.

```bash
python examples/clip_readout.py \
  --clip-dir checkpoints/clip-vit-large-patch14-336 \
  --image example.jpg --question 'What is the person holding?' \
  --k 64 --device cuda --output outputs/readout_demo.npz
```

This standalone demonstration uses CLIP patch features for geometry. To match
the paper's VLM setup, use that VLM's **native projected features**, identical
preprocessing/view mapping, and its own multimodal sequence assembly. The
[integration guide](docs/integration.md) describes these boundaries.

## Method and reproducibility

- [Exact method contract](docs/method.md)
- [Integration and release scope](docs/integration.md)
- [Source provenance and validation](docs/reproducibility.md)
- [Third-party dependencies](THIRD_PARTY_NOTICES.md)

The readout response is a local CLIP intervention, not an answer-level causal
attribution. Qwen requires an auxiliary frozen CLIP branch. Include the entire
branch and token-selection cost in efficiency measurements; fewer retained
patches do not by themselves establish an end-to-end speedup.

## 中文说明

本仓库发布 ReadPrune（原 FRA-Q2）的核心参考实现：冻结 CLIP 读出响应、
`q²` 加权 DPP 选择，以及辅助 CLIP 到 Qwen 原生网格的空间映射组件。
所选 Token 保持原始表示与空间顺序，不包含 NDR、聚合或修复分支。

当前版本附带可运行示例和离线测试，不打包权重、数据集或服务器配置。
模型专用的全基准运行脚本、NeXT 执行图优化与视频运行接口尚未包含，
因此不能将这个 CPU 参考选择器直接视作论文全部表格与测速结果的一键复现。
