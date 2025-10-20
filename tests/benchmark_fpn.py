"""
FPN vs 滑窗性能对标脚本

功能：比较 FPN 推理路径与传统图像金字塔滑窗路径的性能差异。

输出指标：
  - 推理时间（ms）
  - 内存占用（MB）
  - 检测到的关键点数
  - 检测精度（匹配内点数）

使用示例：
  uv run python tests/benchmark_fpn.py \
    --layout /path/to/layout.png \
    --template /path/to/template.png \
    --num-runs 5 \
    --output benchmark_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import psutil
import torch
from PIL import Image

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.rord import RoRD
from utils.config_loader import load_config, to_absolute_path
from utils.data_utils import get_transform


def get_memory_usage() -> float:
    """获取当前进程的内存占用（MB）"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def get_gpu_memory_usage() -> float:
    """获取 GPU 显存占用（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def benchmark_fpn(
    model: torch.nn.Module,
    layout_image: Image.Image,
    template_image: Image.Image,
    transform,
    matching_cfg,
    num_runs: int = 5,
) -> Dict[str, float]:
    """
    测试 FPN 性能
    
    Args:
        model: RoRD 模型
        layout_image: 大版图
        template_image: 模板
        transform: 图像预处理管道
        matching_cfg: 匹配配置
        num_runs: 运行次数
    
    Returns:
        性能指标字典
    """
    from match import extract_from_pyramid, extract_features_sliding_window, mutual_nearest_neighbor
    
    device = next(model.parameters()).device
    times = []
    keypoint_counts = []
    inlier_counts = []
    
    print(f"\n{'=' * 60}")
    print(f"性能测试：FPN 路径")
    print(f"{'=' * 60}")
    
    for run in range(num_runs):
        # 版图特征提取
        layout_tensor = transform(layout_image).unsqueeze(0).to(device)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        layout_kps, layout_descs = extract_from_pyramid(
            model, 
            layout_tensor, 
            float(matching_cfg.keypoint_threshold),
            getattr(matching_cfg, 'nms', {})
        )
        
        # 模板特征提取（单尺度，取 1.0）
        template_tensor = transform(template_image).unsqueeze(0).to(device)
        template_kps, template_descs = extract_from_pyramid(
            model,
            template_tensor,
            float(matching_cfg.keypoint_threshold),
            getattr(matching_cfg, 'nms', {})
        )
        
        # 匹配
        if len(layout_descs) > 0 and len(template_descs) > 0:
            matches = mutual_nearest_neighbor(template_descs, layout_descs)
            inlier_count = len(matches)
        else:
            inlier_count = 0
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.time() - start_time) * 1000  # 转换为 ms
        
        times.append(elapsed)
        keypoint_counts.append(len(layout_kps))
        inlier_counts.append(inlier_count)
        
        print(f"  Run {run + 1}/{num_runs}: {elapsed:.2f}ms, KPs: {len(layout_kps)}, Matches: {inlier_count}")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    mean_kps = np.mean(keypoint_counts)
    mean_inliers = np.mean(inlier_counts)
    gpu_mem = get_gpu_memory_usage()
    
    return {
        "method": "FPN",
        "mean_time_ms": float(mean_time),
        "std_time_ms": float(std_time),
        "min_time_ms": float(np.min(times)),
        "max_time_ms": float(np.max(times)),
        "all_times_ms": [float(t) for t in times],
        "mean_keypoints": float(mean_kps),
        "mean_matches": float(mean_inliers),
        "gpu_memory_mb": float(gpu_mem),
        "num_runs": num_runs,
    }


def benchmark_sliding_window(
    model: torch.nn.Module,
    layout_image: Image.Image,
    template_image: Image.Image,
    transform,
    matching_cfg,
    num_runs: int = 5,
) -> Dict[str, float]:
    """
    测试滑窗性能（图像金字塔路径）
    
    Args:
        model: RoRD 模型
        layout_image: 大版图
        template_image: 模板
        transform: 图像预处理管道
        matching_cfg: 匹配配置
        num_runs: 运行次数
    
    Returns:
        性能指标字典
    """
    from match import extract_features_sliding_window, extract_keypoints_and_descriptors, mutual_nearest_neighbor
    
    device = next(model.parameters()).device
    times = []
    keypoint_counts = []
    inlier_counts = []
    
    print(f"\n{'=' * 60}")
    print(f"性能测试：滑窗路径")
    print(f"{'=' * 60}")
    
    for run in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        # 版图滑窗特征提取
        layout_kps, layout_descs = extract_features_sliding_window(
            model, 
            layout_image, 
            transform, 
            matching_cfg
        )
        
        # 模板单尺度特征提取
        template_tensor = transform(template_image).unsqueeze(0).to(device)
        template_kps, template_descs = extract_keypoints_and_descriptors(
            model,
            template_tensor,
            float(matching_cfg.keypoint_threshold)
        )
        
        # 匹配
        if len(layout_descs) > 0 and len(template_descs) > 0:
            matches = mutual_nearest_neighbor(template_descs, layout_descs)
            inlier_count = len(matches)
        else:
            inlier_count = 0
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.time() - start_time) * 1000  # 转换为 ms
        
        times.append(elapsed)
        keypoint_counts.append(len(layout_kps))
        inlier_counts.append(inlier_count)
        
        print(f"  Run {run + 1}/{num_runs}: {elapsed:.2f}ms, KPs: {len(layout_kps)}, Matches: {inlier_count}")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    mean_kps = np.mean(keypoint_counts)
    mean_inliers = np.mean(inlier_counts)
    gpu_mem = get_gpu_memory_usage()
    
    return {
        "method": "Sliding Window",
        "mean_time_ms": float(mean_time),
        "std_time_ms": float(std_time),
        "min_time_ms": float(np.min(times)),
        "max_time_ms": float(np.max(times)),
        "all_times_ms": [float(t) for t in times],
        "mean_keypoints": float(mean_kps),
        "mean_matches": float(mean_inliers),
        "gpu_memory_mb": float(gpu_mem),
        "num_runs": num_runs,
    }


def compute_speedup(fpn_result: Dict, sw_result: Dict) -> Dict[str, float]:
    """计算 FPN 相对于滑窗的性能改进"""
    speedup = (sw_result["mean_time_ms"] - fpn_result["mean_time_ms"]) / sw_result["mean_time_ms"] * 100
    memory_saving = (sw_result["gpu_memory_mb"] - fpn_result["gpu_memory_mb"]) / sw_result["gpu_memory_mb"] * 100 if sw_result["gpu_memory_mb"] > 0 else 0
    
    return {
        "speedup_percent": float(speedup),
        "memory_saving_percent": float(memory_saving),
        "fpn_faster": speedup > 0,
        "meets_speedup_target": speedup >= 30,
        "meets_memory_target": memory_saving >= 20,
    }


def print_results(fpn_result: Dict, sw_result: Dict, comparison: Dict) -> None:
    """打印性能对比结果"""
    
    print(f"\n{'=' * 80}")
    print(f"{'性能基准测试结果':^80}")
    print(f"{'=' * 80}\n")
    
    print(f"{'指标':<30} {'FPN':<20} {'滑窗':<20}")
    print("-" * 70)
    
    print(f"{'平均推理时间 (ms)':<30} {fpn_result['mean_time_ms']:<20.2f} {sw_result['mean_time_ms']:<20.2f}")
    print(f"{'标准差 (ms)':<30} {fpn_result['std_time_ms']:<20.2f} {sw_result['std_time_ms']:<20.2f}")
    print(f"{'最小时间 (ms)':<30} {fpn_result['min_time_ms']:<20.2f} {sw_result['min_time_ms']:<20.2f}")
    print(f"{'最大时间 (ms)':<30} {fpn_result['max_time_ms']:<20.2f} {sw_result['max_time_ms']:<20.2f}")
    print()
    
    print(f"{'平均关键点数':<30} {fpn_result['mean_keypoints']:<20.0f} {sw_result['mean_keypoints']:<20.0f}")
    print(f"{'平均匹配数':<30} {fpn_result['mean_matches']:<20.0f} {sw_result['mean_matches']:<20.0f}")
    print()
    
    print(f"{'GPU 内存占用 (MB)':<30} {fpn_result['gpu_memory_mb']:<20.2f} {sw_result['gpu_memory_mb']:<20.2f}")
    print()
    
    print(f"{'=' * 80}")
    print(f"{'对标结果':^80}")
    print(f"{'=' * 80}\n")
    
    speedup = comparison["speedup_percent"]
    memory_saving = comparison["memory_saving_percent"]
    
    print(f"推理速度提升: {speedup:+.2f}% {'✅' if speedup >= 30 else '⚠️'}")
    print(f"  (目标: ≥30% | 达成: {'是' if comparison['meets_speedup_target'] else '否'})")
    print()
    
    print(f"内存节省: {memory_saving:+.2f}% {'✅' if memory_saving >= 20 else '⚠️'}")
    print(f"  (目标: ≥20% | 达成: {'是' if comparison['meets_memory_target'] else '否'})")
    print()
    
    if speedup > 0:
        print(f"🎉 FPN 相比滑窗快 {abs(speedup):.2f}%")
    elif speedup < 0:
        print(f"⚠️ FPN 相比滑窗慢 {abs(speedup):.2f}%")
    else:
        print(f"ℹ️ FPN 与滑窗性能相当")
    print()


def main():
    parser = argparse.ArgumentParser(description="RoRD FPN vs 滑窗性能对标测试")
    parser.add_argument('--config', type=str, default="configs/base_config.yaml", help="YAML 配置文件")
    parser.add_argument('--model_path', type=str, default=None, help="模型权重路径")
    parser.add_argument('--layout', type=str, required=True, help="版图路径")
    parser.add_argument('--template', type=str, required=True, help="模板路径")
    parser.add_argument('--num-runs', type=int, default=5, help="每个方法的运行次数")
    parser.add_argument('--output', type=str, default="benchmark_results.json", help="输出 JSON 文件路径")
    parser.add_argument('--device', type=str, default="cuda", help="使用设备: cuda 或 cpu")
    
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_config(args.config)
    config_dir = Path(args.config).resolve().parent
    matching_cfg = cfg.matching
    
    model_path = args.model_path or str(to_absolute_path(cfg.paths.model_path, config_dir))
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = RoRD().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 加载图像
    print(f"加载版图: {args.layout}")
    layout_image = Image.open(args.layout).convert('L')
    print(f"  尺寸: {layout_image.size}")
    
    print(f"加载模板: {args.template}")
    template_image = Image.open(args.template).convert('L')
    print(f"  尺寸: {template_image.size}")
    
    # 获取预处理管道
    transform = get_transform()
    
    # 运行基准测试
    print(f"\n{'=' * 80}")
    print(f"{'开始性能基准测试':^80}")
    print(f"{'=' * 80}")
    print(f"运行次数: {args.num_runs}")
    print(f"配置: {args.config}")
    
    with torch.no_grad():
        fpn_result = benchmark_fpn(
            model, layout_image, template_image, transform, matching_cfg, args.num_runs
        )
        
        # 临时禁用 FPN，启用滑窗
        original_use_fpn = getattr(matching_cfg, 'use_fpn', True)
        matching_cfg.use_fpn = False
        
        sw_result = benchmark_sliding_window(
            model, layout_image, template_image, transform, matching_cfg, args.num_runs
        )
        
        # 恢复配置
        matching_cfg.use_fpn = original_use_fpn
    
    # 计算对比指标
    comparison = compute_speedup(fpn_result, sw_result)
    
    # 打印结果
    print_results(fpn_result, sw_result, comparison)
    
    # 保存结果
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": str(args.config),
        "model_path": str(model_path),
        "layout_path": str(args.layout),
        "layout_size": list(layout_image.size),
        "template_path": str(args.template),
        "template_size": list(template_image.size),
        "device": str(device),
        "fpn": fpn_result,
        "sliding_window": sw_result,
        "comparison": comparison,
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 结果已保存至: {output_path}")
    print(f"{'=' * 80}\n")
    
    # 退出状态码
    if comparison["meets_speedup_target"] and comparison["meets_memory_target"]:
        print("🎉 所有性能指标均达到预期目标！")
        return 0
    elif comparison["fpn_faster"]:
        print("✅ FPN 性能优于滑窗，但未完全达到目标。")
        return 1
    else:
        print("⚠️ FPN 性能未优于滑窗，需要优化。")
        return 2


if __name__ == "__main__":
    sys.exit(main())
