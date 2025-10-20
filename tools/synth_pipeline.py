#!/usr/bin/env python3
"""
One-click synthetic data pipeline:
1) Generate synthetic GDS using tools/generate_synthetic_layouts.py
2) Rasterize GDS to PNG using tools/layout2png.py (KLayout preferred, fallback gdstk+SVG)
3) Preview random training pairs using tools/preview_dataset.py (optional)
4) Validate homography consistency using tools/validate_h_consistency.py (optional)
5) Optionally update a YAML config to enable synthetic mixing and elastic augmentation
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def run_cmd(cmd: list[str]) -> None:
    print("[RUN]", " ".join(str(c) for c in cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise SystemExit(f"Command failed with code {res.returncode}: {' '.join(map(str, cmd))}")


essential_scripts = {
    "gen": Path("tools/generate_synthetic_layouts.py"),
    "gds2png": Path("tools/layout2png.py"),
    "preview": Path("tools/preview_dataset.py"),
    "validate": Path("tools/validate_h_consistency.py"),
}


def ensure_scripts_exist() -> None:
    missing = [str(p) for p in essential_scripts.values() if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required scripts: {missing}")


def update_config(config_path: Path, png_dir: Path, ratio: float, enable_elastic: bool) -> None:
    cfg = OmegaConf.load(config_path)
    # Ensure nodes exist
    if "synthetic" not in cfg:
        cfg.synthetic = {}
    cfg.synthetic.enabled = True
    cfg.synthetic.png_dir = png_dir.as_posix()
    cfg.synthetic.ratio = float(ratio)

    if enable_elastic:
        if "augment" not in cfg:
            cfg.augment = {}
        if "elastic" not in cfg.augment:
            cfg.augment.elastic = {}
        cfg.augment.elastic.enabled = True
        # Don't override numeric params if already present
        if "alpha" not in cfg.augment.elastic:
            cfg.augment.elastic.alpha = 40
        if "sigma" not in cfg.augment.elastic:
            cfg.augment.elastic.sigma = 6
        if "alpha_affine" not in cfg.augment.elastic:
            cfg.augment.elastic.alpha_affine = 6
        if "prob" not in cfg.augment.elastic:
            cfg.augment.elastic.prob = 0.3
        # Photometric defaults
        if "photometric" not in cfg.augment:
            cfg.augment.photometric = {"brightness_contrast": True, "gauss_noise": True}

    OmegaConf.save(config=cfg, f=config_path)
    print(f"[OK] Config updated: {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click synthetic data pipeline")
    parser.add_argument("--out_root", type=str, default="data/synthetic", help="Root output dir for gds/png/preview")
    parser.add_argument("--num", type=int, default=200, help="Number of GDS samples to generate")
    parser.add_argument("--dpi", type=int, default=600, help="Rasterization DPI for PNG rendering")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=0.3, help="Mixing ratio for synthetic data in training")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="YAML config to update")
    parser.add_argument("--enable_elastic", action="store_true", help="Also enable elastic augmentation in config")
    parser.add_argument("--no_preview", action="store_true", help="Skip preview generation")
    parser.add_argument("--validate_h", action="store_true", help="Run homography consistency validation on rendered PNGs")
    parser.add_argument("--validate_n", type=int, default=6, help="Number of samples for H validation")
    parser.add_argument("--diffusion_dir", type=str, default=None, help="Directory of diffusion-generated PNGs to include")
    # Rendering style passthrough
    parser.add_argument("--layermap", type=str, default=None, help="Layer color map for KLayout, e.g. '1/0:#00FF00,2/0:#FF0000'")
    parser.add_argument("--line_width", type=int, default=None, help="Default draw line width for KLayout display")
    parser.add_argument("--bgcolor", type=str, default=None, help="Background color for KLayout display")

    args = parser.parse_args()
    ensure_scripts_exist()

    out_root = Path(args.out_root)
    gds_dir = out_root / "gds"
    png_dir = out_root / "png"
    gds_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    # 1) Generate GDS
    run_cmd([sys.executable, str(essential_scripts["gen"]), "--out_dir", gds_dir.as_posix(), "--num", str(args.num), "--seed", str(args.seed)])

    # 2) GDS -> PNG
    gds2png_cmd = [
        sys.executable, str(essential_scripts["gds2png"]),
        "--in", gds_dir.as_posix(),
        "--out", png_dir.as_posix(),
        "--dpi", str(args.dpi),
    ]
    if args.layermap:
        gds2png_cmd += ["--layermap", args.layermap]
    if args.line_width is not None:
        gds2png_cmd += ["--line_width", str(args.line_width)]
    if args.bgcolor:
        gds2png_cmd += ["--bgcolor", args.bgcolor]
    run_cmd(gds2png_cmd)

    # 3) Preview (optional)
    if not args.no_preview:
        preview_path = out_root / "preview.png"
        preview_cmd = [sys.executable, str(essential_scripts["preview"]), "--dir", png_dir.as_posix(), "--out", preview_path.as_posix(), "--n", "8"]
        if args.enable_elastic:
            preview_cmd.append("--elastic")
        run_cmd(preview_cmd)

    # 4) Validate homography consistency (optional)
    if args.validate_h:
        validate_dir = out_root / "validate_h"
        validate_cmd = [
            sys.executable, str(essential_scripts["validate"]),
            "--dir", png_dir.as_posix(),
            "--out", validate_dir.as_posix(),
            "--n", str(args.validate_n),
        ]
        if args.enable_elastic:
            validate_cmd.append("--elastic")
        run_cmd(validate_cmd)

    # 5) Update YAML config
    update_config(Path(args.config), png_dir, args.ratio, args.enable_elastic)
    # Include diffusion dir if provided (no automatic sampling here; integration only)
    if args.diffusion_dir:
        cfg = OmegaConf.load(args.config)
        if "synthetic" not in cfg:
            cfg.synthetic = {}
        if "diffusion" not in cfg.synthetic:
            cfg.synthetic.diffusion = {}
        cfg.synthetic.diffusion.enabled = True
        cfg.synthetic.diffusion.png_dir = Path(args.diffusion_dir).as_posix()
        # Keep ratio default at 0 unless user updates later; or reuse a small default like 0.1? Keep 0.0 for safety.
        if "ratio" not in cfg.synthetic.diffusion:
            cfg.synthetic.diffusion.ratio = 0.0
        OmegaConf.save(config=cfg, f=args.config)
        print(f"[OK] Config updated with diffusion_dir: {args.diffusion_dir}")

    print("\n[Done] Synthetic pipeline completed.")
    print(f"- GDS: {gds_dir}")
    print(f"- PNG: {png_dir}")
    if args.diffusion_dir:
        print(f"- Diffusion PNGs: {Path(args.diffusion_dir)}")
    if not args.no_preview:
        print(f"- Preview: {out_root / 'preview.png'}")
    if args.validate_h:
        print(f"- H validation: {out_root / 'validate_h'}")
    print(f"- Updated config: {args.config}")


if __name__ == "__main__":
    main()
