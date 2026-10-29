"""ReadPrune final-CLIP-readout connection ablation; no decoder or labels.

The native CLIPVisionModel is never cast or updated. H is observed at the actual
last-block pre-hook. FP32 shadow parameters originate from native values after
strict same-dtype checkpoint alignment. The text/projection heads are paired.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as F


FORMULA_VERSION = "fra-cls-edge-final-nonlinear-readout-v1"
FALLBACK_DENOMINATOR = 1e-4


@contextmanager
def fp32_scope():
    """Avoid inherited FP16 mask construction and TF32, restore even on error."""
    old_dtype = torch.get_default_dtype()
    old_mm = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.set_default_dtype(old_dtype)
        torch.backends.cuda.matmul.allow_tf32 = old_mm
        torch.backends.cudnn.allow_tf32 = old_cudnn


def tensor_hash(tensor):
    value = tensor.detach().contiguous().cpu()
    header = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)},
                        sort_keys=True).encode()
    return hashlib.sha256(header + value.numpy().tobytes()).hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(tensor_hash(state[key]).encode())
    return digest.hexdigest()


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _unit(value, name):
    if not torch.isfinite(value).all():
        raise FloatingPointError(name + ": nonfinite")
    norm = value.norm(dim=-1, keepdim=True)
    if bool((norm <= 0).any()):
        raise FloatingPointError(name + ": zero norm")
    return value / norm


def ablate_cls_connections(logits, values, threshold=FALLBACK_DENOMINATOR):
    """[heads, keys], [heads, keys, head_dim] -> base and all patch ablations.

    Key zero is CLS and stays in every normalization. Returns [heads, dim],
    [patches, heads, dim], and diagnostics. Bad denominators are never clamped.
    """
    if logits.ndim != 2 or values.ndim != 3:
        raise ValueError("Expected logits[heads,keys], values[heads,keys,dim]")
    if tuple(logits.shape) != tuple(values.shape[:2]) or logits.shape[1] < 2:
        raise ValueError("Invalid CLS+patch shapes")
    if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
        raise FloatingPointError("Nonfinite CLS inputs")
    attention = torch.softmax(logits, dim=-1)
    baseline = torch.bmm(attention.unsqueeze(1), values).squeeze(1)
    patch_attention = attention[:, 1:]
    denominator = 1.0 - patch_attention
    numerator = baseline[:, None, :] - patch_attention[:, :, None] * values[:, 1:, :]
    deleted = torch.empty_like(numerator)
    good = denominator >= threshold
    deleted[good] = numerator[good] / denominator[good].unsqueeze(-1)
    bad = (~good).nonzero(as_tuple=False)
    if bad.numel():
        head_ids, patch_offsets = bad.unbind(dim=1)
        masked_logits = logits[head_ids].clone()
        masked_logits[torch.arange(len(bad), device=logits.device), patch_offsets + 1] = -torch.inf
        # Explicit logsumexp-normalized softmax is the final reference.
        weights = torch.exp(masked_logits - torch.logsumexp(masked_logits, dim=-1, keepdim=True))
        deleted[head_ids, patch_offsets] = torch.bmm(
            weights.unsqueeze(1), values[head_ids]).squeeze(1)
    if not torch.isfinite(deleted).all():
        raise FloatingPointError("Connection ablation produced nonfinite values")
    diag = {
        "cls_key_included": True,
        "head_count": int(logits.shape[0]),
        "key_count": int(logits.shape[1]),
        "fallback_head_candidate_pairs": int(len(bad)),
        "fallback_threshold": float(threshold),
        "minimum_one_minus_patch_attention": float(denominator.min().item()),
        "cls_attention_mean": float(attention[:, 0].mean().item()),
        "attention_row_sum_max_abs_error": float((attention.sum(-1) - 1).abs().max().item()),
    }
    return baseline, deleted.permute(1, 0, 2).contiguous(), diag


def segment_content(content_ids, bos_token_id, eos_token_id, max_positions):
    """No truncation; legal BOS/content/EOT chunks and explicit EOT positions."""
    if max_positions < 3:
        raise ValueError("Text context must contain BOS, content, EOT")
    if any(token in (bos_token_id, eos_token_id) for token in content_ids):
        raise ValueError("Raw text tokenizer produced embedded special BOS/EOT")
    cap = max_positions - 2
    pieces = [content_ids[i:i + cap] for i in range(0, len(content_ids), cap)]
    if not pieces:
        pieces = [[]]
    return [
        {"input_ids": [bos_token_id] + piece + [eos_token_id],
         "content_tokens": len(piece), "eot_position": len(piece) + 1}
        for piece in pieces
    ]


class Readout:
    """Frozen image-local readout plus strictly paired raw-question CLIP text."""

    def __init__(self, native_vision_model, clip_path, device):
        from transformers import CLIPConfig, CLIPTextModelWithProjection, CLIPTokenizer
        self.device = torch.device(device)
        self.clip_path = str(Path(clip_path).resolve())
        self._native = native_vision_model
        self._latest_H = None
        self._hook_handle = None
        self.capture_count = 0
        checkpoint_path = Path(self.clip_path) / "model.safetensors"
        if not checkpoint_path.is_file():
            raise RuntimeError("ALIGNMENT_DEPENDENCY_BLOCKED: paired model.safetensors missing")
        with fp32_scope(), torch.inference_mode():
            from safetensors.torch import load_file
            state = load_file(str(checkpoint_path), device="cpu")
            mismatches = []
            compared = 0
            native_dtype_counts = {}
            native_hasher = hashlib.sha256()
            native_state = dict(native_vision_model.state_dict())
            # Older paired checkpoints persist position_ids; current HF does not.
            for key, buffer in native_vision_model.named_buffers():
                if key in state and key not in native_state:
                    native_state[key] = buffer
            for key, native_tensor in native_state.items():
                if key not in state:
                    mismatches.append(key + ": missing checkpoint tensor")
                    continue
                checkpoint_tensor = state[key]
                actual = native_tensor.detach().cpu()
                if tuple(actual.shape) != tuple(checkpoint_tensor.shape):
                    mismatches.append(key + ": shape mismatch")
                    continue
                expected = checkpoint_tensor.to(dtype=actual.dtype)
                if not torch.equal(actual, expected):
                    mismatches.append(key + ": native differs from checkpoint at native dtype")
                compared += 1
                native_dtype_counts[str(actual.dtype)] = native_dtype_counts.get(str(actual.dtype), 0) + 1
                native_hasher.update(key.encode())
                native_hasher.update(tensor_hash(actual).encode())
            if mismatches or not compared:
                raise RuntimeError("ALIGNMENT_DEPENDENCY_BLOCKED: " + json.dumps(mismatches[:20]))
            native_transformer = native_vision_model.vision_model
            self.block = deepcopy(native_transformer.encoder.layers[-1]).to(
                device=self.device, dtype=torch.float32).eval()
            self.post_ln = deepcopy(native_transformer.post_layernorm).to(
                device=self.device, dtype=torch.float32).eval()
            projection = state.get("visual_projection.weight")
            if projection is None or projection.ndim != 2:
                raise RuntimeError("ALIGNMENT_DEPENDENCY_BLOCKED: visual_projection missing")
            self.visual_projection = nn.Linear(projection.shape[1], projection.shape[0],
                                               bias=False, dtype=torch.float32)
            self.visual_projection.load_state_dict({"weight": projection}, strict=True)
            self.visual_projection.to(self.device).eval()
            config = CLIPConfig.from_pretrained(self.clip_path, local_files_only=True)
            self.text_model = CLIPTextModelWithProjection(config.text_config).float().eval()
            text_state = {key: val for key, val in state.items()
                          if key.startswith("text_model.") or key.startswith("text_projection.")}
            expected_text_keys = set(self.text_model.state_dict())
            extra_text_keys = set(text_state) - expected_text_keys
            allowed_buffer = "text_model.embeddings.position_ids"
            if extra_text_keys - {allowed_buffer}:
                raise RuntimeError("ALIGNMENT_DEPENDENCY_BLOCKED: unexpected text checkpoint keys")
            if allowed_buffer in extra_text_keys:
                actual_buffer = self.text_model.text_model.embeddings.position_ids.cpu()
                if not torch.equal(actual_buffer, text_state[allowed_buffer]):
                    raise RuntimeError("ALIGNMENT_DEPENDENCY_BLOCKED: text position_ids mismatch")
            self.text_model.load_state_dict(
                {key: value for key, value in text_state.items() if key in expected_text_keys},
                strict=True)
            self.text_model.to(self.device).eval()
            self.tokenizer = CLIPTokenizer.from_pretrained(self.clip_path, local_files_only=True)
            self.max_positions = int(config.text_config.max_position_embeddings)
            self.bos_token_id = int(self.tokenizer.bos_token_id)
            self.eos_token_id = int(self.tokenizer.eos_token_id)
            if projection.shape[0] != self.text_model.text_projection.weight.shape[0]:
                raise RuntimeError("ALIGNMENT_DEPENDENCY_BLOCKED: projected embedding dimensions differ")
            for module in (self.block, self.post_ln, self.visual_projection, self.text_model):
                module.requires_grad_(False)
            tokenizer_files = {}
            for name in ("tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"):
                path = Path(self.clip_path) / name
                if path.is_file():
                    tokenizer_files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.metadata = {
                "formula_version": FORMULA_VERSION,
                "shadow_dtype": "torch.float32",
                "paired_config_sha256": hashlib.sha256((Path(self.clip_path) / "config.json").read_bytes()).hexdigest(),
                "shadow_structure": {
                    "heads": int(self.block.self_attn.num_heads),
                    "head_dim": int(self.block.self_attn.head_dim),
                    "ln1_eps": float(self.block.layer_norm1.eps),
                    "ln2_eps": float(self.block.layer_norm2.eps),
                    "post_ln_eps": float(self.post_ln.eps),
                    "mlp_activation": str(config.vision_config.hidden_act),
                },
                "shadow_parameter_origin": "native checkpoint-aligned values converted to FP32",
                "native_checkpoint_comparison": "bitwise equality after checkpoint cast to each native dtype",
                "alignment_pass": True,
                "aligned_native_state_tensor_count": compared,
                "native_dtype_tensor_counts": native_dtype_counts,
                "native_vision_state_sha256": native_hasher.hexdigest(),
                "paired_checkpoint_path": str(checkpoint_path),
                "paired_checkpoint_size_bytes": checkpoint_path.stat().st_size,
                "block_sha256": state_hash(self.block.state_dict()),
                "post_ln_sha256": state_hash(self.post_ln.state_dict()),
                "visual_projection_sha256": state_hash(self.visual_projection.state_dict()),
                "text_model_sha256": state_hash(text_state),
                "tokenizer_file_sha256": tokenizer_files,
                "last_block_index": len(native_transformer.encoder.layers) - 1,
                "patch_count": int((config.vision_config.image_size // config.vision_config.patch_size) ** 2),
                "visual_hidden_size": int(projection.shape[1]),
                "embedding_size": int(projection.shape[0]),
                "text_max_positions": self.max_positions,
                "text_bos_token_id": self.bos_token_id,
                "text_eot_token_id": self.eos_token_id,
                "text_config_eos_token_id": int(config.text_config.eos_token_id),
                "text_pooling": "explicit known EOT index of each legal chunk; verify native pooled output",
                "long_text_rule": "contiguous content chunks, normalize each, content-length weighted normalized mean",
                "readout_batch_policy": "single 577-row final-CLS MLP batch",
                "tf32_in_shadow": False,
                "cache_owned_by_runtime": True,
                "auxiliary_llm_forward_calls": 0,
            }
            fingerprint_keys = ("formula_version", "shadow_dtype", "paired_config_sha256",
                                "shadow_structure", "native_vision_state_sha256",
                                "block_sha256", "post_ln_sha256", "visual_projection_sha256")
            self.metadata["image_response_fingerprint"] = hashlib.sha256(
                json.dumps({key: self.metadata[key] for key in fingerprint_keys},
                           sort_keys=True).encode()).hexdigest()
            del text_state, state
        self._hook_handle = native_transformer.encoder.layers[-1].register_forward_pre_hook(
            self._capture_last_block_input)

    def _capture_last_block_input(self, _module, args):
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("Cannot capture actual native last-block H")
        self._latest_H = args[0].detach().clone()
        self.capture_count += 1
        # Returning None preserves the complete original module input and execution.

    def pop_H(self):
        if self._latest_H is None:
            raise RuntimeError("No fresh native last-block input; run normal vision first")
        value, self._latest_H = self._latest_H, None
        return value

    def _prepare_H(self, H):
        if H.ndim == 2:
            H = H.unsqueeze(0)
        if H.ndim != 3 or H.shape[0] != 1:
            raise ValueError("FRA image readout requires one H[1,CLS+patches,hidden]")
        if not torch.isfinite(H).all():
            raise FloatingPointError("Nonfinite captured H")
        return H.detach().to(device=self.device, dtype=torch.float32)

    def _cls_attention(self, H):
        h = self.block.layer_norm1(H)
        attn = self.block.self_attn
        rows = H.shape[1]
        # Project all Q rows so the linear operation matches the native reference.
        # Equivalent to the legacy CLIPAttention._shape; newer Transformers
        # releases removed that private helper without changing this layout.
        def shape(value):
            return value.view(1, rows, attn.num_heads, attn.head_dim).transpose(1, 2).contiguous()[0]

        q = shape(attn.q_proj(h) * attn.scale)[:, :1, :]
        k = shape(attn.k_proj(h))
        v = shape(attn.v_proj(h))
        logits = torch.bmm(q, k.transpose(1, 2)).squeeze(1)
        return ablate_cls_connections(logits, v)

    def _complete_cls(self, cls_residual, head_outputs):
        out = self.block.self_attn.out_proj(head_outputs)
        z1 = cls_residual + out
        z2 = z1 + self.block.mlp(self.block.layer_norm2(z1))
        return _unit(self.visual_projection(self.post_ln(z2)), "image embedding")

    @torch.inference_mode()
    def image(self, H):
        _sync(self.device)
        begin = time.perf_counter()
        with fp32_scope(), torch.autocast(device_type=self.device.type, enabled=False):
            H32 = self._prepare_H(H)
            base, removed, diag = self._cls_attention(H32)
            all_heads = torch.cat((base.unsqueeze(0), removed), dim=0).flatten(1)
            _sync(self.device)
            before_tail = time.perf_counter()
            embeddings = self._complete_cls(H32[0, :1], all_heads)
            _sync(self.device)
            tail_seconds = time.perf_counter() - before_tail
            R = (embeddings[:1] - embeddings[1:]).contiguous()
            e_full = embeddings[0].contiguous()
            if not torch.isfinite(R).all():
                raise FloatingPointError("Nonfinite image response R")
            diag.update({
                "clip_readout_rows": int(len(embeddings)),
                "clip_readout_chunks": 1,
                "response_shape": list(R.shape),
                "response_bytes": R.numel() * R.element_size(),
                "captured_H_dtype": str(H.dtype),
                "shadow_H_dtype": str(H32.dtype),
                "readout_tail_seconds": tail_seconds,
                "auxiliary_llm_forward_calls": 0,
                "candidate_answer_probe_calls": 0,
            })
        _sync(self.device)
        seconds = time.perf_counter() - begin
        diag["readout_total_seconds"] = seconds
        diag["attention_ablation_and_overhead_seconds"] = seconds - tail_seconds
        return {"R": R, "e_full": e_full, "seconds": seconds, "diag": diag}

    @torch.inference_mode()
    def text(self, raw_question):
        if not isinstance(raw_question, str):
            raise TypeError("Only raw question text is accepted")
        _sync(self.device)
        begin = time.perf_counter()
        with fp32_scope(), torch.autocast(device_type=self.device.type, enabled=False):
            content = self.tokenizer.encode(raw_question, add_special_tokens=False,
                                            truncation=False)
            chunks = segment_content(content, self.bos_token_id, self.eos_token_id,
                                     self.max_positions)
            width = max(len(chunk["input_ids"]) for chunk in chunks)
            ids = torch.full((len(chunks), width), int(self.tokenizer.pad_token_id),
                             dtype=torch.long, device=self.device)
            mask = torch.zeros_like(ids)
            eot = torch.tensor([chunk["eot_position"] for chunk in chunks],
                               dtype=torch.long, device=self.device)
            for index, chunk in enumerate(chunks):
                length = len(chunk["input_ids"])
                ids[index, :length] = torch.tensor(chunk["input_ids"], dtype=torch.long,
                                                   device=self.device)
                mask[index, :length] = 1
            output = self.text_model.text_model(input_ids=ids, attention_mask=mask,
                                                return_dict=True)
            pooled = output.last_hidden_state[
                torch.arange(len(chunks), device=self.device), eot]
            pooled_error = float((pooled - output.pooler_output).abs().max().item())
            if not torch.equal(pooled, output.pooler_output):
                raise RuntimeError("ALIGNMENT_DEPENDENCY_BLOCKED: actual EOT and native pooling differ")
            per_chunk = _unit(self.text_model.text_projection(pooled), "text chunk")
            lengths = torch.tensor([chunk["content_tokens"] for chunk in chunks],
                                   dtype=torch.float32, device=self.device)
            # Empty raw text is the sole BOS/EOT chunk; avoid a zero weighted sum.
            if not len(content):
                lengths.fill_(1.0)
            t = _unit((per_chunk * lengths[:, None]).sum(0), "combined text embedding")
            if len(t) != self.visual_projection.weight.shape[0]:
                raise RuntimeError("Paired text/image dimension mismatch")
        _sync(self.device)
        return {
            "t": t.contiguous(),
            "segments": len(chunks),
            "seconds": time.perf_counter() - begin,
            "diag": {"content_token_count": len(content),
                     "segments": len(chunks),
                     "segment_content_lengths": [x["content_tokens"] for x in chunks],
                     "eot_positions": [x["eot_position"] for x in chunks],
                     "input_ids": [x["input_ids"] for x in chunks],
                     "eot_pooling_max_abs_error": pooled_error,
                     "raw_question_sha256": hashlib.sha256(raw_question.encode()).hexdigest(),
                     "truncated": False},
        }

    @torch.inference_mode()
    def verify(self, H, patch_ids=(0, 63, 287, 575)):
        with fp32_scope(), torch.autocast(device_type=self.device.type, enabled=False):
            H32 = self._prepare_H(H)
            result = self.image(H)
            reference_rows = self.block(H32, attention_mask=None,
                                        causal_attention_mask=None)[0]
            reference_full = _unit(self.visual_projection(self.post_ln(reference_rows[:, 0])),
                                   "reference full")[0]
            full_diff = (result["e_full"] - reference_full).abs()
            full_pass = bool(torch.allclose(result["e_full"], reference_full,
                                             atol=1e-5, rtol=1e-5))
            checks = []
            for patch in patch_ids:
                if not 0 <= int(patch) < H32.shape[1] - 1:
                    continue
                mask = torch.zeros((1, 1, H32.shape[1], H32.shape[1]),
                                   dtype=torch.float32, device=self.device)
                mask[0, 0, 0, int(patch) + 1] = -torch.inf
                rows = self.block(H32, attention_mask=mask,
                                  causal_attention_mask=None)[0]
                reference = _unit(self.visual_projection(self.post_ln(rows[:, 0])),
                                  "reference deletion")[0]
                actual = result["e_full"] - result["R"][int(patch)]
                checks.append({"patch_index": int(patch),
                               "max_abs_error": float((actual - reference).abs().max().item()),
                               "pass": bool(torch.allclose(actual, reference, atol=1e-5, rtol=1e-5))})
            passed = full_pass and bool(checks) and all(x["pass"] for x in checks)
            return {"pass": passed, "atol": 1e-5, "rtol": 1e-5,
                    "comparison": "same FP32 H and shadow parameters; not FP16 end-to-end bitwise",
                    "full_baseline_pass": full_pass,
                    "full_baseline_max_abs_error": float(full_diff.max().item()),
                    "masked_full_block_patch_checks": checks,
                    "clip_readout_rows": result["diag"]["clip_readout_rows"],
                    "captured_shape": list(H.shape), "captured_dtype": str(H.dtype),
                    "fallback_head_candidate_pairs": result["diag"]["fallback_head_candidate_pairs"]}

    def close(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._latest_H = None
