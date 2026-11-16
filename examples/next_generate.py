"""Generate with a local HF LLaVA-NeXT checkpoint and uniform ReadPrune budgets."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--prompt", help="Optional decoder question/options; selection uses --question")
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--k", type=int, help="Total patch budget across all crops")
    budget.add_argument("--full", action="store_true")
    parser.add_argument("--fixed-grid-size", type=int, help="E.g. 672 for five views with a 336-pixel CLIP tower")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.full and args.clip_dir is None:
        parser.error("--clip-dir is required for ReadPrune")
    import torch
    from PIL import Image
    from readprune.next import NextReadPrune

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    with NextReadPrune.from_pretrained(
        args.model_dir, None if args.full else args.clip_dir, device=args.device,
        dtype=getattr(torch, args.dtype), fixed_grid_size=args.fixed_grid_size,
    ) as runtime:
        result = runtime.generate(image, args.question, prompt=args.prompt, k=args.k,
                                  max_new_tokens=args.max_new_tokens)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
