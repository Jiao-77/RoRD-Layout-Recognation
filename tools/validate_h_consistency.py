#!/usr/bin/env python3
"""
Validate homography consistency produced by ICLayoutTrainingDataset.
For random samples, we check that cv2.warpPerspective(original, H) ≈ transformed.
Saves visual composites and prints basic metrics (MSE / PSNR).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.ic_dataset import ICLayoutTrainingDataset


def tensor_to_u8_img(t: torch.Tensor) -> np.ndarray:
    """Convert 1xHxW or 3xHxW float tensor in [0,1] to uint8 HxW or HxWx3."""
    if t.dim() != 3:
        raise ValueError(f"Expect 3D tensor, got {t.shape}")
    if t.size(0) == 1:
        arr = (t.squeeze(0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    elif t.size(0) == 3:
        arr = (t.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unexpected channels: {t.size(0)}")
    return arr


def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff * diff))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = mse(a, b)
    if m <= 1e-8:
        return float('inf')
    return 10.0 * np.log10((255.0 * 255.0) / m)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate homography consistency")
    parser.add_argument("--dir", dest="image_dir", type=str, required=True, help="PNG images directory")
    parser.add_argument("--out", dest="out_dir", type=str, default="validate_h_out", help="Output directory for composites")
    parser.add_argument("--n", dest="num", type=int, default=8, help="Number of samples to validate")
    parser.add_argument("--patch", dest="patch_size", type=int, default=256)
    parser.add_argument("--elastic", dest="use_elastic", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use no photometric/Sobel transform here to compare raw grayscale content
    ds = ICLayoutTrainingDataset(
        args.image_dir,
        patch_size=args.patch_size,
        transform=None,
        scale_range=(1.0, 1.0),
        use_albu=args.use_elastic,
        albu_params={"prob": 0.5},
    )

    n = min(args.num, len(ds))
    if n == 0:
        print("[WARN] Empty dataset.")
        return

    mses = []
    psnrs = []

    for i in range(n):
        patch_t, trans_t, H2x3_t = ds[i]
        # Convert to uint8 arrays
        patch_u8 = tensor_to_u8_img(patch_t)
        trans_u8 = tensor_to_u8_img(trans_t)
        if patch_u8.ndim == 3:
            patch_u8 = cv2.cvtColor(patch_u8, cv2.COLOR_BGR2GRAY)
        if trans_u8.ndim == 3:
            trans_u8 = cv2.cvtColor(trans_u8, cv2.COLOR_BGR2GRAY)

        # Reconstruct 3x3 H
        H2x3 = H2x3_t.numpy()
        H = np.vstack([H2x3, [0.0, 0.0, 1.0]]).astype(np.float32)

        # Warp original with H
        warped = cv2.warpPerspective(patch_u8, H, (patch_u8.shape[1], patch_u8.shape[0]))

        # Metrics
        m = mse(warped, trans_u8)
        p = psnr(warped, trans_u8)
        mses.append(m)
        psnrs.append(p)

        # Composite image: [orig | warped | transformed | absdiff]
        diff = cv2.absdiff(warped, trans_u8)
        comp = np.concatenate([
            patch_u8, warped, trans_u8, diff
        ], axis=1)
        out_path = out_dir / f"sample_{i:03d}.png"
        cv2.imwrite(out_path.as_posix(), comp)
        print(f"[OK] sample {i}: MSE={m:.2f}, PSNR={p:.2f} dB -> {out_path}")

    print(f"\nSummary: MSE avg={np.mean(mses):.2f} ± {np.std(mses):.2f}, PSNR avg={np.mean(psnrs):.2f} dB")


if __name__ == "__main__":
    main()
