#!/usr/bin/env python3
"""
Quickly preview training pairs (original, transformed, H) from ICLayoutTrainingDataset.
Saves a grid image for visual inspection.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid, save_image

from data.ic_dataset import ICLayoutTrainingDataset
from utils.data_utils import get_transform


def to_pil(t: torch.Tensor) -> Image.Image:
    # input normalized to [-1,1] for 3-channels; invert normalization
    x = t.clone()
    if x.dim() == 3 and x.size(0) == 3:
        x = (x * 0.5) + 0.5  # unnormalize
    x = (x * 255.0).clamp(0, 255).byte()
    if x.dim() == 3 and x.size(0) == 3:
        x = x
    elif x.dim() == 3 and x.size(0) == 1:
        x = x.repeat(3, 1, 1)
    else:
        raise ValueError("Unexpected tensor shape")
    np_img = x.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np_img)


def main():
    parser = argparse.ArgumentParser(description="Preview dataset samples")
    parser.add_argument("--dir", dest="image_dir", type=str, required=True, help="PNG images directory")
    parser.add_argument("--out", dest="out_path", type=str, default="preview.png")
    parser.add_argument("--n", dest="num", type=int, default=8)
    parser.add_argument("--patch", dest="patch_size", type=int, default=256)
    parser.add_argument("--elastic", dest="use_elastic", action="store_true")
    args = parser.parse_args()

    transform = get_transform()
    ds = ICLayoutTrainingDataset(
        args.image_dir,
        patch_size=args.patch_size,
        transform=transform,
        scale_range=(1.0, 1.0),
        use_albu=args.use_elastic,
        albu_params={"prob": 0.5},
    )

    images = []
    for i in range(min(args.num, len(ds))):
        orig, rot, H = ds[i]
        # Stack orig and rot side-by-side for each sample
        images.append(orig)
        images.append(rot)

    grid = make_grid(torch.stack(images, dim=0), nrow=2, padding=2)
    save_image(grid, args.out_path)
    print(f"Saved preview to {args.out_path}")


if __name__ == "__main__":
    main()
