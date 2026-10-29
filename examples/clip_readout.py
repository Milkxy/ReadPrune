"""Standalone CLIP readout demonstration, not a VLM benchmark runner.

All model/tokenizer assets must already exist locally. This example uses CLIP
penultimate patch features as geometry; the paper instead uses the native
multimodal-projector features supplied to its decoder.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModel
    from readprune import select_tokens
    from readprune.readout import Readout

    native = CLIPVisionModel.from_pretrained(
        args.clip_dir, use_safetensors=True, local_files_only=True,
        attn_implementation="eager").eval().to(args.device)
    native.requires_grad_(False)
    processor = CLIPImageProcessor.from_pretrained(args.clip_dir, local_files_only=True)
    readout = Readout(native, args.clip_dir, args.device)
    try:
        with Image.open(args.image) as image:
            pixels = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
        with torch.inference_mode():
            encoded = native(pixels.to(args.device), output_hidden_states=True, return_dict=True)
            H = readout.pop_H()
            response = readout.image(H)
            text = readout.text(args.question)
            delta = (response["R"] @ text["t"]).double().cpu().numpy()
            features = encoded.hidden_states[-2][0, 1:]
            result = select_tokens(features, delta, args.k)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, indices=result["indices"],
                            tokens=result["tokens"].cpu().numpy(), quality=result["quality"],
                            response_deltas=delta, response_vectors=response["R"].cpu().numpy())
        print(json.dumps({"example": "standalone CLIP, not VLM benchmark",
                          "retained_tokens": len(result["indices"]),
                          "paired_alignment_pass": readout.metadata["alignment_pass"],
                          "question_segments": text["segments"]}, indent=2))
    finally:
        readout.close()


if __name__ == "__main__":
    main()
