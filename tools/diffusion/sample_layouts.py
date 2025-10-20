#!/usr/bin/env python3
"""
Sample layout patches using a trained diffusion model (skeleton).

Outputs raster PNGs into a target directory compatible with current training pipeline (no H pairing).

Current status: CLI skeleton and TODOs only.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample layout patches from diffusion model (skeleton)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained diffusion checkpoint or HF repo id")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to write sampled PNGs")
    parser.add_argument("--num", type=int, default=200)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cond_dir", type=str, default=None, help="Optional condition maps directory")
    parser.add_argument("--cond_types", type=str, nargs="*", default=None, help="e.g., edge skeleton dist")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # TODO: load pipeline from ckpt, set scheduler, handle conditions if provided,
    # sample args.num images, save as PNG files into out_dir.

    print("[TODO] Implement diffusion sampling and PNG saving.")


if __name__ == "__main__":
    main()
