"""Single-image LLaVA-NeXT reference adapter with uniform per-crop budgets."""

from math import isqrt
from numbers import Integral

import torch
from transformers.models.llava_next.modeling_llava_next import get_anyres_image_grid_shape

from .readout import Readout
from .selection import select_tokens


def uniform_crop_budgets(k, crops, patches):
    """Split total patch budget evenly; assign the remainder in crop order."""
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in (k, crops, patches)):
        raise ValueError("k, crops and patches must be integers")
    if crops < 1 or patches < 1 or not crops <= k <= crops * patches:
        raise ValueError("Require crops <= k <= crops * patches, with positive dimensions")
    base, remainder = divmod(int(k), int(crops))
    return [base + int(i < remainder) for i in range(crops)]


def pack_selected_crops(features, supports, grid_shape):
    """Gather base view then spatially interleaved tiles, without changing values.

    The compressed sequence uses spatial packing without unpadding or row
    separators. Thus its length is exactly the sum of the patch budgets.
    """
    if features.ndim != 3:
        raise ValueError("Expected features[crops, patches, hidden]")
    crops, patches, _ = features.shape
    side = isqrt(patches)
    if side * side != patches or len(supports) != crops:
        raise ValueError("Require square patch grids and one support per crop")
    if len(grid_shape) != 2 or any(int(x) != x or x < 1 for x in grid_shape):
        raise ValueError("Invalid tile grid")
    gh, gw = map(int, grid_shape)
    if crops != 1 + gh * gw:
        raise ValueError("Crop count must match base view plus tile grid")
    mask = torch.zeros((crops, patches), dtype=torch.bool, device=features.device)
    for crop, support in enumerate(supports):
        indices = torch.as_tensor(support, device=features.device)
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Support must be a one-dimensional integer array")
        if not len(indices) or bool((indices < 0).any()) or bool((indices >= patches).any()):
            raise ValueError("Support must contain valid patch indices")
        if bool((indices[1:] <= indices[:-1]).any()):
            raise ValueError("Support must be sorted and unique")
        mask[crop, indices.long()] = True
    tiles = features[1:].reshape(gh, gw, side, side, -1).permute(0, 2, 1, 3, 4).reshape(-1, features.shape[-1])
    tile_mask = mask[1:].reshape(gh, gw, side, side).permute(0, 2, 1, 3).reshape(-1)
    return torch.cat((features[0, mask[0]], tiles[tile_mask]), dim=0)


class NextReadPrune:
    """Frozen CLIP/NeXT, one image and one prompt per request, greedy decoding.

    Supports HF LlavaNextForConditionalGeneration with CLIP, feature layer -2,
    and the default patch-only feature strategy. No model method is patched.
    """

    def __init__(self, model, processor, clip_dir=None):
        from transformers import LlavaNextForConditionalGeneration

        if not isinstance(model, LlavaNextForConditionalGeneration):
            raise TypeError("Expected LlavaNextForConditionalGeneration")
        if model.config.vision_feature_layer != -2 or model.config.vision_feature_select_strategy != "default":
            raise ValueError("Require vision_feature_layer=-2 and default patch selection")
        if model.config.vision_config.model_type != "clip_vision_model":
            raise ValueError("Require a paired CLIP vision tower")
        if len({p.device for p in model.parameters()}) != 1:
            raise ValueError("This reference adapter requires a single device")
        self.model = model.eval().requires_grad_(False)
        self.processor = processor
        self.device = next(model.parameters()).device
        self.dtype = next(model.model.vision_tower.parameters()).dtype
        self.reader = Readout(model.model.vision_tower, clip_dir, self.device) if clip_dir is not None else None

    @classmethod
    def from_pretrained(cls, model_dir, clip_dir=None, *, device="cuda:0", dtype=torch.bfloat16,
                        fixed_grid_size=None):
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=dtype, attn_implementation="eager").to(device)
        processor = LlavaNextProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
        processor.patch_size = model.config.vision_config.patch_size
        processor.vision_feature_select_strategy = "default"
        processor.num_additional_image_tokens = 1
        if fixed_grid_size is not None:
            size = model.config.vision_config.image_size
            if isinstance(fixed_grid_size, bool) or not isinstance(fixed_grid_size, Integral) or fixed_grid_size < size or fixed_grid_size % size:
                raise ValueError("fixed_grid_size must be a positive multiple of the vision crop size")
            points = [[int(fixed_grid_size), int(fixed_grid_size)]]
            model.config.image_grid_pinpoints = points
            processor.image_processor.image_grid_pinpoints = points
        return cls(model, processor, clip_dir)

    @torch.inference_mode()
    def prepare(self, inputs, raw_question, k):
        """Prepare a compact prompt from unpadded, processor-expanded inputs."""
        if self.reader is None:
            raise ValueError("A strictly paired CLIP checkpoint is required for pruning")
        ids, attention = inputs["input_ids"], inputs["attention_mask"]
        sizes, pixels = inputs["image_sizes"], inputs["pixel_values"]
        if ids.ndim != 2 or ids.shape[0] != 1 or attention.shape != ids.shape or not bool((attention == 1).all()):
            raise ValueError("Require one unpadded prompt")
        if sizes.shape != (1, 2) or pixels.ndim != 5 or pixels.shape[0] != 1:
            raise ValueError("Require exactly one image with processor crop tensors")
        grid = get_anyres_image_grid_shape(sizes[0], self.model.config.image_grid_pinpoints,
                                          self.model.config.vision_config.image_size)
        crops = 1 + grid[0] * grid[1]
        if pixels.shape[1] != crops:
            raise ValueError("Processor crops disagree with the model grid")
        patches = (self.model.config.vision_config.image_size // self.model.config.vision_config.patch_size) ** 2
        budgets = uniform_crop_budgets(k, crops, patches)
        native = self.model.model
        vision = native.vision_tower(pixels[0].to(self.device, self.dtype), output_hidden_states=True)
        features = native.multi_modal_projector(vision.hidden_states[-2][:, 1:])
        H = self.reader.pop_H()
        if features.shape[:2] != (crops, patches) or H.shape[:2] != (crops, patches + 1):
            raise RuntimeError("Native feature/readout crop alignment failed")
        # Validate the original processor expansion using the native packing path.
        full, _ = native.pack_image_features([features], sizes, "default", image_newline=native.image_newline)
        image_positions = (ids[0] == self.model.config.image_token_index).nonzero().flatten()
        if len(image_positions) != len(full[0]) or not len(image_positions) or bool((image_positions[1:] - image_positions[:-1] != 1).any()):
            raise ValueError("Require one complete contiguous processor-expanded image span")
        question = self.reader.text(raw_question)["t"].double().cpu().numpy()
        supports = []
        for crop, budget in enumerate(budgets):
            responses = self.reader.image(H[crop:crop + 1])["R"].double().cpu().numpy()
            supports.append(select_tokens(features[crop], responses @ question, budget)["indices"].tolist())
        selected = pack_selected_crops(features, supports, grid)
        start, stop = int(image_positions[0]), int(image_positions[-1]) + 1
        embedding = self.model.get_input_embeddings()
        packed = torch.cat((embedding(ids[:, :start].to(self.device)), selected[None],
                            embedding(ids[:, stop:].to(self.device))), dim=1)
        audit = {"allocation": "uniform", "crop_budgets": budgets, "supports": supports,
                 "grid_shape": list(grid), "K_actual": len(selected), "N_full_patches": crops * patches,
                 "native_visual_sequence_length": len(full[0]), "compressed_prompt_length": packed.shape[1],
                 "packing": "base view then spatial tile order; no unpadding or row separators",
                 "token_values_unchanged": True}
        return {"inputs_embeds": packed, "attention_mask": torch.ones(packed.shape[:2], dtype=torch.long, device=self.device)}, audit

    @torch.inference_mode()
    def generate(self, image, raw_question, *, prompt=None, k=None, max_new_tokens=128):
        """k=None uses native unpruned generation; k is a total patch budget."""
        messages = [{"role": "user", "content": [{"type": "image"},
                    {"type": "text", "text": raw_question if prompt is None else prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {name: value.to(self.device, self.dtype) if name == "pixel_values" else value.to(self.device)
                  for name, value in inputs.items()}
        if k is None:
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
            generated = output[:, inputs["input_ids"].shape[1]:]
            audit = {"allocation": "full", "packing": "native spatial_unpad with row separators"}
        else:
            compact, audit = self.prepare(inputs, raw_question, k)
            generated = self.model.generate(**compact, do_sample=False, max_new_tokens=max_new_tokens)
        return {"answer": self.processor.batch_decode(generated, skip_special_tokens=True)[0],
                "generated_ids": generated[0].tolist(), "audit": audit}

    def close(self):
        if self.reader is not None:
            self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
