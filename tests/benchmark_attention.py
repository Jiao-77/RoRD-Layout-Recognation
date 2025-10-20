"""
注意力模块 A/B 基准测试

目的：在相同骨干与输入下，对比注意力开/关（none/se/cbam）在单尺度与 FPN 前向的耗时差异；可选指定插入位置。

示例：
  PYTHONPATH=. uv run python tests/benchmark_attention.py --device cpu --image-size 512 --runs 10 --backbone resnet34 --places backbone_high desc_head
"""
from __future__ import annotations

import argparse
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


def build_model(backbone: str, attention_type: str, places: List[str], device: torch.device) -> RoRD:
    cfg = type("cfg", (), {
        "model": type("m", (), {
            "backbone": type("b", (), {"name": backbone, "pretrained": False})(),
            "attention": type("a", (), {"enabled": attention_type != "none", "type": attention_type, "places": places})(),
        })()
    })()
    model = RoRD(cfg=cfg).to(device)
    model.eval()
    return model


def run_suite(backbone: str, places: List[str], device: torch.device, image_size: int, runs: int) -> List[Dict[str, float]]:
    x = torch.randn(1, 3, image_size, image_size, device=device)
    results: List[Dict[str, float]] = []
    for attn in ["none", "se", "cbam"]:
        model = build_model(backbone, attn, places, device)
        # warmup
        for _ in range(3):
            _ = model(x, return_pyramid=False)
            _ = model(x, return_pyramid=True)
        # single
        t_list_single = [bench_once(model, x, fpn=False) for _ in range(runs)]
        # fpn
        t_list_fpn = [bench_once(model, x, fpn=True) for _ in range(runs)]
        results.append({
            "backbone": backbone,
            "attention": attn,
            "places": ",".join(places) if places else "-",
            "single_ms_mean": float(np.mean(t_list_single)),
            "single_ms_std": float(np.std(t_list_single)),
            "fpn_ms_mean": float(np.mean(t_list_fpn)),
            "fpn_ms_std": float(np.std(t_list_fpn)),
            "runs": int(runs),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="RoRD 注意力模块 A/B 基准")
    parser.add_argument("--backbone", type=str, default="resnet34", choices=["vgg16","resnet34","efficientnet_b0"], help="骨干")
    parser.add_argument("--places", nargs="*", default=["backbone_high"], help="插入位置：backbone_high det_head desc_head")
    parser.add_argument("--image-size", type=int, default=512, help="输入尺寸")
    parser.add_argument("--runs", type=int, default=10, help="重复次数")
    parser.add_argument("--device", type=str, default="cpu", help="cuda 或 cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    results = run_suite(args.backbone, args.places, device, args.image_size, args.runs)

    # 简要打印
    print("\n===== Attention A/B Summary =====")
    for r in results:
        print(f"{r['backbone']:<14} attn={r['attention']:<5} places={r['places']:<24} "
              f"single {r['single_ms_mean']:.2f}±{r['single_ms_std']:.2f} | "
              f"fpn {r['fpn_ms_mean']:.2f}±{r['fpn_ms_std']:.2f} ms")


if __name__ == "__main__":
    main()
