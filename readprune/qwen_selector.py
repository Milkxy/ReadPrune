"""Frozen auxiliary CLIP scoring on the native Qwen merger grid."""

import numpy as np
import torch
from PIL import Image

from .readout import Readout
from .selection import select_tokens
from .spatial import map_letterbox_features


class QwenSelector:
    """Use CLIP for signed responses; retain original Qwen vectors for the LLM.

    The paper uses paired CLIP ViT-L/14-336. Other local CLIP configurations
    can be used for software tests but do not reproduce that checkpoint.
    """

    def __init__(self, clip_dir, device="cuda:0", dtype=None):
        from transformers import CLIPImageProcessor, CLIPVisionModel

        self.device = torch.device(device)
        dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.processor = CLIPImageProcessor.from_pretrained(clip_dir, local_files_only=True)
        self.vision = CLIPVisionModel.from_pretrained(
            clip_dir, use_safetensors=True, local_files_only=True,
            torch_dtype=dtype, attn_implementation="eager",
        ).to(self.device).eval().requires_grad_(False)
        self.readout = Readout(self.vision, clip_dir, self.device)
        size, patch = self.vision.config.image_size, self.vision.config.patch_size
        self.source_hw = (size // patch, size // patch)
        self._verified = False

    @torch.inference_mode()
    def prepare_image(self, image, grid_hw):
        if not isinstance(image, Image.Image):
            raise TypeError("image must be the same PIL image supplied to the Qwen processor")
        image = image.convert("RGB")
        side = max(image.size)
        background = tuple(int(channel * 255) for channel in self.processor.image_mean)
        padded = Image.new("RGB", (side, side), background)
        padded.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        pixels = self.processor(images=padded, return_tensors="pt")["pixel_values"]
        self.vision(pixels.to(device=self.device, dtype=self.vision.dtype))
        hidden = self.readout.pop_H()
        response = self.readout.image(hidden)
        verification = None
        if not self._verified:
            verification = self.readout.verify(hidden)
            if not verification["pass"]:
                raise RuntimeError("Auxiliary CLIP readout differs from explicit masked reference")
            self._verified = True
        mapped, mapping = map_letterbox_features(
            response["R"].double().cpu().numpy(), image.size, grid_hw, self.source_hw,
        )
        return {"response_vectors": mapped, "grid_hw": tuple(grid_hw),
                "mapping": mapping, "readout_verification": verification}

    @torch.inference_mode()
    def select(self, features, raw_question, prepared_image, k):
        if not isinstance(raw_question, str):
            raise TypeError("raw_question must be supplied separately from a formatted prompt")
        response = prepared_image["response_vectors"]
        if response.shape[0] != len(features) or np.prod(prepared_image["grid_hw"]) != len(features):
            raise ValueError("Auxiliary response and native Qwen merger grid disagree")
        text = self.readout.text(raw_question)
        # Transfer signed vectors before abs/deadzone; do the dot product in FP64.
        delta = response @ text["t"].double().cpu().numpy()
        result = select_tokens(features, delta, k)
        result["diagnostics"].update(
            auxiliary_clip=True, native_qwen_deletion_response=False,
            mapping=prepared_image["mapping"], question_segments=text["segments"],
        )
        return result

    def close(self):
        self.readout.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
