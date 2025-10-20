#!/usr/bin/env python3
"""
Minimal smoke test:
1) Generate a tiny synthetic set (num=8) and rasterize to PNG
2) Validate H consistency (n=4, with/without elastic)
3) Run a short training loop (epochs=1-2) to verify end-to-end pipeline
Prints PASS/FAIL with basic stats.
"""
from __future__ import annotations

import argparse
import subprocess
import os
import sys
from pathlib import Path


def run(cmd: list[str]) -> int:
    print("[RUN]", " ".join(cmd))
    env = os.environ.copy()
    # Ensure project root on PYTHONPATH for child processes
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = f"{root}:{env.get('PYTHONPATH','')}" if env.get("PYTHONPATH") else str(root)
    return subprocess.call(cmd, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal smoke test for E2E pipeline")
    parser.add_argument("--root", type=str, default="data/smoke", help="Root dir for smoke test outputs")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    args = parser.parse_args()

    root = Path(args.root)
    gds_dir = root / "gds"
    png_dir = root / "png"
    gds_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    rc = 0

    # 1) Generate a tiny set
    rc |= run([sys.executable, "tools/generate_synthetic_layouts.py", "--out_dir", gds_dir.as_posix(), "--num", "8", "--seed", "123"])
    if rc != 0:
        print("[FAIL] generate synthetic")
        sys.exit(2)

    # 2) Rasterize
    rc |= run([sys.executable, "tools/layout2png.py", "--in", gds_dir.as_posix(), "--out", png_dir.as_posix(), "--dpi", "600"])
    if rc != 0:
        print("[FAIL] layout2png")
        sys.exit(3)

    # 3) Validate H (n=4, both no-elastic and elastic)
    rc |= run([sys.executable, "tools/validate_h_consistency.py", "--dir", png_dir.as_posix(), "--out", (root/"validate_no_elastic").as_posix(), "--n", "4"])
    rc |= run([sys.executable, "tools/validate_h_consistency.py", "--dir", png_dir.as_posix(), "--out", (root/"validate_elastic").as_posix(), "--n", "4", "--elastic"])
    if rc != 0:
        print("[FAIL] validate H")
        sys.exit(4)

    # 4) Write back config via synth_pipeline and run short training (1 epoch)
    rc |= run([sys.executable, "tools/synth_pipeline.py", "--out_root", root.as_posix(), "--num", "0", "--dpi", "600", "--config", args.config, "--ratio", "0.3", "--enable_elastic", "--no_preview"])
    if rc != 0:
        print("[FAIL] synth_pipeline config update")
        sys.exit(5)

    # Train 1 epoch to smoke the loop
    rc |= run([sys.executable, "train.py", "--config", args.config, "--epochs", "1" ])
    if rc != 0:
        print("[FAIL] train 1 epoch")
        sys.exit(6)

    print("[PASS] Smoke test completed successfully.")


if __name__ == "__main__":
    main()
