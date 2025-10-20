"""
Backbone A/B 基准测试脚本

目的：在相同输入与重复次数下，对比不同骨干（vgg16/resnet34/efficientnet_b0）
在单尺度与 FPN 前向推理的吞吐（毫秒）与显存占用（MB）。

示例：
  uv run python tests/benchmark_backbones.py --device cpu --image-size 512 --runs 5
  uv run python tests/benchmark_backbones.py --device cuda --runs 20 --backbones vgg16 resnet34 efficientnet_b0
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
import psutil
import torch

from models.rord import RoRD


def get_mem_mb() -> float:
    p = psutil.Process()
    return p.memory_info().rss / 1024 / 1024


def get_gpu_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def warmup(model: torch.nn.Module, x: torch.Tensor, steps: int = 3, fpn: bool = False) -> None:
    with torch.inference_mode():
        for _ in range(steps):
            _ = model(x, return_pyramid=fpn)


def bench_once(model: torch.nn.Module, x: torch.Tensor, fpn: bool = False) -> float:
    if torch.cuda.is_available() and x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        _ = model(x, return_pyramid=fpn)
    if torch.cuda.is_available() and x.is_cuda:
        torch.cuda.synchronize()
    return (time.time() - t0) * 1000.0


def run_benchmark(backbone: str, device: torch.device, image_size: int, runs: int) -> Dict[str, float]:
    cfg = type("cfg", (), {
        "model": type("m", (), {
            "backbone": type("b", (), {"name": backbone, "pretrained": False})(),
            "attention": type("a", (), {"enabled": False, "type": "none", "places": []})(),
        })()
    })()

    model = RoRD(cfg=cfg).to(device)
    model.eval()

    x = torch.randn(1, 3, image_size, image_size, device=device)

    # warmup
    warmup(model, x, steps=5, fpn=False)
    warmup(model, x, steps=5, fpn=True)

    # single-scale
    t_list_single: List[float] = []
    for _ in range(runs):
        t_list_single.append(bench_once(model, x, fpn=False))

    # FPN
    t_list_fpn: List[float] = []
    for _ in range(runs):
        t_list_fpn.append(bench_once(model, x, fpn=True))

    return {
        "backbone": backbone,
        "single_ms_mean": float(np.mean(t_list_single)),
        "single_ms_std": float(np.std(t_list_single)),
        "fpn_ms_mean": float(np.mean(t_list_fpn)),
        "fpn_ms_std": float(np.std(t_list_fpn)),
        "gpu_mem_mb": float(get_gpu_mem_mb()),
        "cpu_mem_mb": float(get_mem_mb()),
        "runs": int(runs),
    }


def main():
    parser = argparse.ArgumentParser(description="RoRD 骨干 A/B 基准测试")
    parser.add_argument("--backbones", nargs="*", default=["vgg16", "resnet34", "efficientnet_b0"],
                        help="要测试的骨干列表")
    parser.add_argument("--image-size", type=int, default=512, help="输入图像尺寸（正方形）")
    parser.add_argument("--runs", type=int, default=10, help="每个设置的重复次数")
    parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"使用设备: {device}")

    results: List[Dict[str, float]] = []
    for bk in args.backbones:
        print(f"\n=== Benchmark: {bk} ===")
        res = run_benchmark(bk, device, args.image_size, args.runs)
        print(f"single: {res['single_ms_mean']:.2f}±{res['single_ms_std']:.2f} ms  |  "
              f"fpn: {res['fpn_ms_mean']:.2f}±{res['fpn_ms_std']:.2f} ms  |  "
              f"gpu_mem: {res['gpu_mem_mb']:.1f} MB")
        results.append(res)

    # 简要对比打印
    print("\n===== 汇总 =====")
    for r in results:
        print(f"{r['backbone']:<16} single {r['single_ms_mean']:.2f} ms | fpn {r['fpn_ms_mean']:.2f} ms")


if __name__ == "__main__":
    main()
