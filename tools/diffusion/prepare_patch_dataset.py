#!/usr/bin/env python3
"""
Prepare raster patch dataset and optional condition maps for diffusion training.

Planned inputs:
- --src_dirs: one or more directories containing PNG layout images
- --out_dir: output root for images/ and conditions/
- --size: patch size (e.g., 256)
- --stride: sliding stride for patch extraction
- --min_fg_ratio: minimum foreground ratio to keep a patch (0-1)
- --make_conditions: flags to generate edge/skeleton/distance maps

Current status: CLI skeleton and TODOs only.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare patch dataset for diffusion training (skeleton)")
    parser.add_argument("--src_dirs", type=str, nargs="+", help="Source PNG dirs for layouts")
    parser.add_argument("--out_dir", type=str, required=True, help="Output root directory")
    parser.add_argument("--size", type=int, default=256, help="Patch size")
    parser.add_argument("--stride", type=int, default=256, help="Patch stride")
    parser.add_argument("--min_fg_ratio", type=float, default=0.02, help="Min foreground ratio to keep a patch")
    parser.add_argument("--make_edge", action="store_true", help="Generate edge map conditions (e.g., Sobel/Canny)")
    parser.add_argument("--make_skeleton", action="store_true", help="Generate morphological skeleton condition")
    parser.add_argument("--make_dist", action="store_true", help="Generate distance transform condition")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "images").mkdir(exist_ok=True)
    (out_root / "conditions").mkdir(exist_ok=True)

    # TODO: implement extraction loop over src_dirs, crop patches, filter by min_fg_ratio,
    # and save into images/; generate optional condition maps into conditions/ mirroring filenames.
    # Keep file naming consistent: images/xxx.png, conditions/xxx_edge.png, etc.

    print("[TODO] Implement patch extraction and condition map generation.")


if __name__ == "__main__":
    main()
