"""Single-image ReadPrune integration for dense Qwen2.5-VL and Qwen3-VL.

The native vision encoder runs in full. Selection happens after its merger.
The compact language-model input retains original mRoPE coordinates and, for
Qwen3, every DeepStack branch at the identical selected positions.
"""

import math

import numpy as np
import torch


FAMILIES = {"qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
            "qwen3_vl": "Qwen3VLForConditionalGeneration"}
TRANSFORMERS_VERSION = "4.57.6"


def token_budget(n, *, k=None, retain_ratio=None, rounding="floor"):
    """Choose exact K, or explicitly floor/ceil a fraction of native tokens."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    if rounding not in ("floor", "ceil"):
        raise ValueError("rounding must be floor or ceil")
    if (k is None) == (retain_ratio is None):
        raise ValueError("Specify exactly one of k and retain_ratio")
    if k is not None:
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= n:
            raise ValueError("k must be an integer in [1, N]")
        return int(k)
    if isinstance(retain_ratio, bool):
        raise ValueError("retain_ratio must be a number in (0, 1]")
    ratio = float(retain_ratio)
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("retain_ratio must be in (0, 1]")
    return max(1, min(int(n), getattr(math, rounding)(ratio * n)))


def _family(model):
    family = model.config.model_type
    if family not in FAMILIES:
        raise ValueError("Supported families are dense Qwen2.5-VL and Qwen3-VL")
    return family


def _validate_inputs(model, inputs):
    _family(model)
    ids, mask, grid = (inputs[key] for key in ("input_ids", "attention_mask", "image_grid_thw"))
    if ids.ndim != 2 or ids.shape[0] != 1 or tuple(grid.shape) != (1, 3) or int(grid[0, 0]) != 1:
        raise ValueError("Exactly one still image and batch size one are supported")
    if mask.shape != ids.shape or not bool((mask == 1).all()):
        raise ValueError("Use an unpadded single-sample attention mask")
    if any(inputs.get(key) is not None for key in ("pixel_values_videos", "video_grid_thw")):
        raise ValueError("Video inputs are not supported by this adapter")
    video_id = getattr(model.config, "video_token_id", None)
    if video_id is not None and bool((ids == video_id).any()):
        raise ValueError("Video placeholders are not supported")
    merge = int(model.model.visual.spatial_merge_size)
    h, w = (int(v) for v in grid[0, 1:])
    if min(h, w) <= 0 or h % merge or w % merge:
        raise ValueError("Invalid native merger grid")
    n = h * w // merge ** 2
    if int((ids == model.config.image_token_id).sum()) != n:
        raise ValueError("Image placeholders and native merger grid disagree")
    return n, (h // merge, w // merge)


@torch.inference_mode()
def visual_features(model, inputs):
    """Return native main/DeepStack tensors in original merger order."""
    n, grid_hw = _validate_inputs(model, inputs)
    visual = model.model.visual
    result = visual(inputs["pixel_values"].to(dtype=visual.dtype), grid_thw=inputs["image_grid_thw"])
    if _family(model) == "qwen3_vl":
        main, deep = result
        deep = list(deep)
        if len(deep) != len(visual.deepstack_visual_indexes):
            raise ValueError("Missing native Qwen3 DeepStack branches")
    else:
        main, deep = result, []
    if main.ndim != 2 or len(main) != n or any(branch.shape != main.shape for branch in deep):
        raise ValueError("Native main/DeepStack merger output shapes disagree")
    if not all(bool(torch.isfinite(branch).all()) for branch in (main, *deep)):
        raise ValueError("Nonfinite native visual features")
    return main, deep, grid_hw


@torch.inference_mode()
def pack_native(model, inputs, main, deep, support):
    """Gather unchanged vectors, prune placeholders, preserve original mRoPE."""
    n, _ = _validate_inputs(model, inputs)
    values = support.detach().cpu().tolist() if isinstance(support, torch.Tensor) else list(support)
    if (not values or any(type(value) is not int for value in values)
            or values != sorted(set(values)) or values[0] < 0 or values[-1] >= n):
        raise ValueError("support must be sorted, unique, in-range integer positions")
    if main.ndim != 2 or len(main) != n or any(branch.shape != main.shape for branch in deep):
        raise ValueError("Main/DeepStack features must align with the original image")
    family = _family(model)
    expected_deep = len(model.model.visual.deepstack_visual_indexes) if family == "qwen3_vl" else 0
    if len(deep) != expected_deep:
        raise ValueError("DeepStack branch count differs from the model")
    support = torch.tensor(values, device=main.device, dtype=torch.long)
    selected = main.index_select(0, support)
    selected_deep = [branch.index_select(0, support) for branch in deep]
    ids, mask = inputs["input_ids"], inputs["attention_mask"]
    positions, original_delta = model.model.get_rope_index(
        ids, image_grid_thw=inputs["image_grid_thw"], attention_mask=mask,
    )
    if positions.shape != (3, 1, ids.shape[1]):
        raise ValueError("Expected native three-axis mRoPE positions")
    image_positions = (ids[0] == model.config.image_token_id).nonzero().flatten()
    keep = torch.ones(ids.shape[1], device=ids.device, dtype=torch.bool)
    keep[image_positions] = False
    keep[image_positions.index_select(0, support.to(ids.device))] = True
    compact_ids = ids[:, keep]
    image_mask = compact_ids == model.config.image_token_id
    embeddings = model.model.get_input_embeddings()(compact_ids)
    if embeddings.dtype != main.dtype or embeddings.device != main.device:
        raise ValueError("Use a single device and matching native text/vision dtypes")
    embeddings[image_mask] = selected
    length = compact_ids.shape[1]
    delta = original_delta + n - len(values)
    packed = dict(inputs_embeds=embeddings, position_ids=positions[:, :, keep],
                  attention_mask=mask[:, keep], cache_position=torch.arange(length, device=ids.device))
    if family == "qwen3_vl":
        packed.update(visual_pos_masks=image_mask, deepstack_visual_embeds=selected_deep)
    audit = dict(model_family=family, original_tokens=n, retained_tokens=len(values),
                 indices=values, original_sequence_length=ids.shape[1],
                 pruned_sequence_length=length, original_mrope_preserved=True,
                 original_vectors_preserved=True, deepstack_lengths=[len(branch) for branch in selected_deep],
                 rope_delta=int(delta.item()))
    return packed, delta, audit


@torch.inference_mode()
def greedy_decode(model, packed, delta, *, max_new_tokens=128, min_new_tokens=0):
    """Greedy decoding with a compact KV cache; no sampling/beam/logits processors."""
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if isinstance(min_new_tokens, bool) or not isinstance(min_new_tokens, int) or not 0 <= min_new_tokens <= max_new_tokens:
        raise ValueError("min_new_tokens must be an integer in [0, max_new_tokens]")
    lm = model.model.language_model
    output = lm(**packed, use_cache=True, return_dict=True)
    logits = model.lm_head(output.last_hidden_state[:, -1])
    first_logits = logits.clone()
    cache = output.past_key_values
    length = packed["inputs_embeds"].shape[1]
    if cache.get_seq_length() != length:
        raise RuntimeError("Prefill cache length differs from the compact prompt")
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    generated, checks = [], []
    for step in range(max_new_tokens):
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("Nonfinite decoder logits")
        if step < min_new_tokens and eos:
            logits = logits.clone()
            logits[:, list(eos)] = -torch.inf
        token = logits.argmax(-1)
        generated.append(int(token.item()))
        if (step + 1 >= min_new_tokens and generated[-1] in eos) or step + 1 == max_new_tokens:
            break
        cache_position = torch.tensor([length + step], device=token.device)
        position = (cache_position.view(1, 1) + delta).unsqueeze(0).expand(3, 1, 1)
        output = lm(input_ids=token[:, None], position_ids=position,
                    attention_mask=torch.ones(1, length + step + 1, dtype=torch.long, device=token.device),
                    cache_position=cache_position, past_key_values=cache, use_cache=True, return_dict=True)
        cache = output.past_key_values
        if cache.get_seq_length() != length + step + 1:
            raise RuntimeError("Decode cache length differs from the compact sequence")
        checks.append(dict(cache_position=int(cache_position.item()),
                           position_ids=position.cpu().tolist(), cache_length=cache.get_seq_length()))
        logits = model.lm_head(output.last_hidden_state[:, -1])
    return {"token_ids": generated, "first_logits": first_logits, "decode_checks": checks}


@torch.inference_mode()
def generate_from_inputs(model, inputs, *, selector=None, image=None, raw_question=None,
                         k=None, retain_ratio=None, rounding="floor", max_new_tokens=128,
                         min_new_tokens=0):
    """Generate from native processor tensors and the identical observed RGB image.

    Full budget bypasses the auxiliary branch. Partial budgets require the
    paired CLIP selector; the question must not be inferred from answer options.
    """
    main, deep, grid_hw = visual_features(model, inputs)
    count = token_budget(len(main), k=k, retain_ratio=retain_ratio, rounding=rounding)
    selection = None
    if count == len(main):
        support = list(range(count))
    else:
        if selector is None or image is None or not isinstance(raw_question, str):
            raise ValueError("Pruning requires a selector, the same RGB image, and raw_question")
        prepared = selector.prepare_image(image, grid_hw)
        result = selector.select(main, raw_question, prepared, count)
        support = result["indices"].tolist()
        if len(support) != count or not torch.equal(result["tokens"], main[support]):
            raise RuntimeError("Selector changed native vectors or did not return exact K")
        selection = result["diagnostics"]
    packed, delta, audit = pack_native(model, inputs, main, deep, support)
    result = greedy_decode(model, packed, delta, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)
    audit.update(budget_rounding="exact_k" if k is not None else rounding,
                 selection=selection, auxiliary_clip_executed=selection is not None,
                 auxiliary_features_enter_llm=False, decode_checks=result.pop("decode_checks"))
    result["audit"] = audit
    return result


@torch.inference_mode()
def check_native_parity(model, inputs, max_new_tokens=4):
    """Check full-budget logits and greedy IDs against unmodified HF execution."""
    previous_delta = model.model.rope_deltas
    try:
        model.model.rope_deltas = None
        native_logits = model(**inputs, use_cache=True, logits_to_keep=1).logits[:, -1]
        model.model.rope_deltas = None
        ids = model.generate(**inputs, do_sample=False, num_beams=1, repetition_penalty=1.0,
                             max_new_tokens=max_new_tokens)[0, inputs["input_ids"].shape[1]:].tolist()
        result = generate_from_inputs(model, inputs, retain_ratio=1.0, max_new_tokens=max_new_tokens)
        error = float((native_logits.float() - result["first_logits"].float()).abs().max())
        passed = torch.allclose(native_logits, result["first_logits"], atol=1e-3, rtol=1e-3) and ids == result["token_ids"]
        if not passed:
            raise RuntimeError(f"Native full-budget parity failed: logits error={error}, token equality={ids == result['token_ids']}")
        return {"passed": True, "max_abs_logits_error": error, "token_ids_equal": True}
    finally:
        model.model.rope_deltas = previous_delta


class QwenReadPrune:
    """Convenience entry point for local checkpoints, one image, one device."""

    def __init__(self, model, processor, selector=None):
        import transformers
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise RuntimeError("Install readprune[qwen] in its own environment (Transformers 4.57.6)")
        _family(model)
        self.model, self.processor, self.selector = model.eval(), processor, selector

    @classmethod
    def from_pretrained(cls, model_dir, clip_dir=None, device="cuda:0", dtype=None,
                        min_pixels=200704, max_pixels=1605632, use_fast=False):
        import transformers
        from .qwen_selector import QwenSelector

        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise RuntimeError("Install readprune[qwen] with Transformers 4.57.6")
        config = transformers.AutoConfig.from_pretrained(model_dir, local_files_only=True)
        if config.model_type not in FAMILIES:
            raise ValueError("Expected a dense Qwen2.5-VL or Qwen3-VL checkpoint")
        device = torch.device(device)
        dtype = dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
        model = getattr(transformers, FAMILIES[config.model_type]).from_pretrained(
            model_dir, local_files_only=True, use_safetensors=True, dtype=dtype,
            attn_implementation="sdpa",
        ).to(device).eval().requires_grad_(False)
        processor = transformers.AutoProcessor.from_pretrained(
            model_dir, local_files_only=True, min_pixels=min_pixels, max_pixels=max_pixels,
            use_fast=use_fast,
        )
        selector = QwenSelector(clip_dir, device) if clip_dir is not None else None
        return cls(model, processor, selector)

    def prepare(self, image, question, prompt=None):
        from PIL import Image
        if not isinstance(image, Image.Image) or not isinstance(question, str):
            raise TypeError("Provide a PIL image and a raw question string")
        image = image.convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"},
                     {"type": "text", "text": question if prompt is None else prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=False,
                                return_tensors="pt").to(self.model.device)
        return inputs, image

    def generate(self, image, question, *, prompt=None, check_parity=False, **kwargs):
        inputs, observed_image = self.prepare(image, question, prompt)
        parity = check_native_parity(self.model, inputs) if check_parity else None
        result = generate_from_inputs(self.model, inputs, selector=self.selector,
                                      image=observed_image, raw_question=question, **kwargs)
        result.pop("first_logits")
        result["answer"] = self.processor.decode(result["token_ids"], skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False).strip()
        if parity is not None:
            result["native_parity"] = parity
        return result

    def close(self):
        if self.selector is not None:
            self.selector.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
