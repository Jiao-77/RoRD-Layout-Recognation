#!/usr/bin/env python3
"""
Programmatic synthetic IC layout generator using gdstk.
Generates GDS files with simple standard-cell-like patterns, wires, and vias.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random

import gdstk


def build_standard_cell(cell_name: str, rng: random.Random, layer: int = 1, datatype: int = 0) -> gdstk.Cell:
    cell = gdstk.Cell(cell_name)
    # Basic cell body
    w = rng.uniform(0.8, 2.0)
    h = rng.uniform(1.6, 4.0)
    rect = gdstk.rectangle((0, 0), (w, h), layer=layer, datatype=datatype)
    cell.add(rect)
    # Poly fingers
    nf = rng.randint(1, 4)
    pitch = w / (nf + 1)
    for i in range(1, nf + 1):
        x = i * pitch
        poly = gdstk.rectangle((x - 0.05, 0), (x + 0.05, h), layer=layer + 1, datatype=datatype)
        cell.add(poly)
    # Contact/vias
    for i in range(rng.randint(2, 6)):
        vx = rng.uniform(0.1, w - 0.1)
        vy = rng.uniform(0.1, h - 0.1)
        via = gdstk.rectangle((vx - 0.05, vy - 0.05), (vx + 0.05, vy + 0.05), layer=layer + 2, datatype=datatype)
        cell.add(via)
    return cell


def generate_layout(out_path: Path, width: float, height: float, seed: int, rows: int, cols: int, density: float):
    rng = random.Random(seed)
    lib = gdstk.Library()
    top = gdstk.Cell("TOP")

    # Create a few standard cell variants
    variants = [build_standard_cell(f"SC_{i}", rng, layer=1) for i in range(4)]

    # Place instances in a grid with random skips based on density
    x_pitch = width / cols
    y_pitch = height / rows
    for r in range(rows):
        for c in range(cols):
            if rng.random() > density:
                continue
            cell = rng.choice(variants)
            dx = c * x_pitch + rng.uniform(0.0, 0.1 * x_pitch)
            dy = r * y_pitch + rng.uniform(0.0, 0.1 * y_pitch)
            ref = gdstk.Reference(cell, (dx, dy))
            top.add(ref)

    lib.add(*variants)
    lib.add(top)
    lib.write_gds(str(out_path))


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic IC layouts (GDS)")
    parser.add_argument("--out-dir", type=str, default="data/synthetic/gds")
    parser.add_argument("--out_dir", dest="out_dir", type=str, help="Alias of --out-dir")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--num", dest="num_samples", type=int, help="Alias of --num-samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=float, default=200.0)
    parser.add_argument("--height", type=float, default=200.0)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--density", type=float, default=0.5)

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    for i in range(args.num_samples):
        sample_seed = rng.randint(0, 2**31 - 1)
        out_path = out_dir / f"chip_{i:06d}.gds"
        generate_layout(out_path, args.width, args.height, sample_seed, args.rows, args.cols, args.density)
        print(f"[OK] Generated {out_path}")


if __name__ == "__main__":
    main()
