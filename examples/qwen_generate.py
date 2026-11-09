"""Generate an answer with ReadPrune and a local Qwen2.5-VL/Qwen3-VL model."""

import argparse
import io
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True, help="Raw question used for selection")
    parser.add_argument("--prompt", help="Optional decoder prompt, e.g. with answer options")
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--retain-ratio", type=float)
    budget.add_argument("--k", type=int)
    budget.add_argument("--full", action="store_true", help="Unpruned baseline; no CLIP is loaded")
    parser.add_argument("--rounding", choices=("floor", "ceil"), default="floor")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=1605632)
    parser.add_argument("--use-fast-processor", action="store_true",
                        help="Explicitly select the fast image processor for a matching protocol")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, choices=range(1, 101), metavar="1..100",
                        help="Explicit JPEG transport for matching legacy evaluation protocols")
    parser.add_argument("--check-parity", action="store_true", help="Check native full-budget logits and greedy IDs first")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.full and args.clip_dir is None:
        parser.error("--clip-dir is required for ReadPrune")

    import torch
    from PIL import Image
    from readprune.qwen import QwenReadPrune

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    if args.jpeg_quality is not None:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=args.jpeg_quality)
        with Image.open(io.BytesIO(buffer.getvalue())) as transported:
            image = transported.convert("RGB")
    with QwenReadPrune.from_pretrained(
        args.model_dir, None if args.full else args.clip_dir, device=args.device,
        dtype=getattr(torch, args.dtype) if args.dtype else None,
        min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        use_fast=args.use_fast_processor,
    ) as runtime:
        result = runtime.generate(
            image, args.question, prompt=args.prompt, k=args.k,
            retain_ratio=1.0 if args.full else args.retain_ratio,
            rounding=args.rounding, max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens, check_parity=args.check_parity,
        )
    result["image_transport"] = "direct RGB" if args.jpeg_quality is None else f"JPEG quality={args.jpeg_quality}"
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
