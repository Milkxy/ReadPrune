"""Select original tokens from stored features and signed readout responses.

Use --demo for synthetic input. No model, benchmark, or network access is used.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from readprune import select_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="NPZ with features[N,D] and response_deltas[N]")
    source.add_argument("--demo", action="store_true", help="Use seeded synthetic data")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.demo:
        rng = np.random.default_rng(42)
        features = rng.normal(size=(576, 128)).astype(np.float32)
        delta = rng.normal(scale=0.01, size=576)
    else:
        with np.load(args.input, allow_pickle=False) as data:
            features = data["features"]
            delta = data["response_deltas"]
    result = select_tokens(features, delta, args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, indices=result["indices"], tokens=result["tokens"],
                        quality=result["quality"], response_deltas=delta)
    summary = {"synthetic": args.demo, "input_tokens": len(features),
               "retained_tokens": len(result["indices"]),
               "original_vectors_preserved": bool(np.array_equal(
                   result["tokens"], features[result["indices"]])),
               "quality_exponent": result["diagnostics"]["quality_exponent"]}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
