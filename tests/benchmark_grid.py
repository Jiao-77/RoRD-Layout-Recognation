"""
三维基准对比：Backbone × Attention × (SingleMean / FPNMean)

示例：
  PYTHONPATH=. uv run python tests/benchmark_grid.py --device cpu --image-size 512 --runs 5 \
    --backbones vgg16 resnet34 efficientnet_b0 --attentions none se cbam --places backbone_high
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

import numpy as np
import torch

from models.rord import RoRD


def bench_once(model: torch.nn.Module, x: torch.Tensor, fpn: bool = False) -> float:
    if torch.cuda.is_available() and x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        _ = model(x, return_pyramid=fpn)
    if torch.cuda.is_available() and x.is_cuda:
        torch.cuda.synchronize()
    return (time.time() - t0) * 1000.0


def build_model(backbone: str, attention: str, places: List[str], device: torch.device) -> RoRD:
    cfg = type("cfg", (), {
        "model": type("m", (), {
            "backbone": type("b", (), {"name": backbone, "pretrained": False})(),
            "attention": type("a", (), {"enabled": attention != "none", "type": attention, "places": places})(),
        })()
    })()
    model = RoRD(cfg=cfg).to(device)
    model.eval()
    return model


def run_grid(backbones: List[str], attentions: List[str], places: List[str], device: torch.device, image_size: int, runs: int) -> List[Dict[str, float]]:
    x = torch.randn(1, 3, image_size, image_size, device=device)
    rows: List[Dict[str, float]] = []
    for bk in backbones:
        for attn in attentions:
            model = build_model(bk, attn, places, device)
            # warmup
            for _ in range(3):
                _ = model(x, return_pyramid=False)
                _ = model(x, return_pyramid=True)
            # bench
            t_single = [bench_once(model, x, fpn=False) for _ in range(runs)]
            t_fpn = [bench_once(model, x, fpn=True) for _ in range(runs)]
            rows.append({
                "backbone": bk,
                "attention": attn,
                "places": ",".join(places) if places else "-",
                "single_ms_mean": float(np.mean(t_single)),
                "single_ms_std": float(np.std(t_single)),
                "fpn_ms_mean": float(np.mean(t_fpn)),
                "fpn_ms_std": float(np.std(t_fpn)),
                "runs": int(runs),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="三维基准：Backbone × Attention × (Single/FPN)")
    parser.add_argument("--backbones", nargs="*", default=["vgg16","resnet34","efficientnet_b0"], help="骨干列表")
    parser.add_argument("--attentions", nargs="*", default=["none","se","cbam"], help="注意力列表")
    parser.add_argument("--places", nargs="*", default=["backbone_high"], help="插入位置")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json-out", type=str, default="benchmark_grid.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    rows = run_grid(args.backbones, args.attentions, args.places, device, args.image_size, args.runs)

    # 打印简表
    print("\n===== Grid Summary (Backbone × Attention) =====")
    for r in rows:
        print(f"{r['backbone']:<14} attn={r['attention']:<5} places={r['places']:<16} single {r['single_ms_mean']:.2f} | fpn {r['fpn_ms_mean']:.2f} ms")

    # 保存 JSON
    with open(args.json_out, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"Saved: {args.json_out}")


if __name__ == "__main__":
    main()
