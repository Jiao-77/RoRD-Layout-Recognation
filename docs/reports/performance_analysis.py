#!/usr/bin/env python3
"""
中期报告性能分析可视化脚本
生成各种图表用于中期报告展示
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_test_data():
    """加载测试数据"""
    data_dir = Path(__file__).parent.parent.parent / "tests" / "results"

    gpu_data = json.load(open(data_dir / "GPU_2048_ALL.json"))
    cpu_data = json.load(open(data_dir / "CPU_2048_ALL.json"))

    return gpu_data, cpu_data

def create_performance_comparison(gpu_data, cpu_data):
    """创建性能对比图表"""

    # 提取数据
    backbones = []
    single_gpu = []
    fpn_gpu = []
    single_cpu = []
    fpn_cpu = []

    for item in gpu_data:
        backbones.append(f"{item['backbone']}\n({item['attention']})")
        single_gpu.append(item['single_ms_mean'])
        fpn_gpu.append(item['fpn_ms_mean'])

    for item in cpu_data:
        single_cpu.append(item['single_ms_mean'])
        fpn_cpu.append(item['fpn_ms_mean'])

    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 图1: GPU单尺度性能
    bars1 = ax1.bar(backbones, single_gpu, color='skyblue', alpha=0.8)
    ax1.set_title('GPU单尺度推理性能 (ms)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('推理时间 (ms)')
    ax1.tick_params(axis='x', rotation=45)

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')

    # 图2: GPU FPN性能
    bars2 = ax2.bar(backbones, fpn_gpu, color='lightcoral', alpha=0.8)
    ax2.set_title('GPU FPN推理性能 (ms)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('推理时间 (ms)')
    ax2.tick_params(axis='x', rotation=45)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')

    # 图3: GPU vs CPU 单尺度对比
    x = np.arange(len(backbones))
    width = 0.35

    bars3 = ax3.bar(x - width/2, single_gpu, width, label='GPU', color='skyblue', alpha=0.8)
    bars4 = ax3.bar(x + width/2, single_cpu, width, label='CPU', color='orange', alpha=0.8)

    ax3.set_title('GPU vs CPU 单尺度性能对比', fontsize=14, fontweight='bold')
    ax3.set_ylabel('推理时间 (ms)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(backbones, rotation=45)
    ax3.legend()
    ax3.set_yscale('log')  # 使用对数坐标

    # 图4: 加速比分析
    speedup = [c/g for c, g in zip(single_cpu, single_gpu)]
    bars5 = ax4.bar(backbones, speedup, color='green', alpha=0.8)
    ax4.set_title('GPU加速比分析', fontsize=14, fontweight='bold')
    ax4.set_ylabel('加速比 (倍)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    for bar in bars5:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}x', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_attention_analysis(gpu_data):
    """创建注意力机制分析图表"""

    # 按骨干网络分组
    backbone_attention = {}
    for item in gpu_data:
        backbone = item['backbone']
        attention = item['attention']
        if backbone not in backbone_attention:
            backbone_attention[backbone] = {}
        backbone_attention[backbone][attention] = {
            'single': item['single_ms_mean'],
            'fpn': item['fpn_ms_mean']
        }

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 单尺度性能
    backbones = list(backbone_attention.keys())
    attentions = ['none', 'se', 'cbam']

    x = np.arange(len(backbones))
    width = 0.25

    for i, att in enumerate(attentions):
        single_times = [backbone_attention[bb].get(att, {}).get('single', 0) for bb in backbones]
        bars = ax1.bar(x + i*width, single_times, width,
                      label=f'{att.upper()}' if att != 'none' else 'None',
                      alpha=0.8)

    ax1.set_title('注意力机制对单尺度性能影响', fontsize=14, fontweight='bold')
    ax1.set_ylabel('推理时间 (ms)')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(backbones)
    ax1.legend()

    # FPN性能
    for i, att in enumerate(attentions):
        fpn_times = [backbone_attention[bb].get(att, {}).get('fpn', 0) for bb in backbones]
        bars = ax2.bar(x + i*width, fpn_times, width,
                      label=f'{att.upper()}' if att != 'none' else 'None',
                      alpha=0.8)

    ax2.set_title('注意力机制对FPN性能影响', fontsize=14, fontweight='bold')
    ax2.set_ylabel('推理时间 (ms)')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(backbones)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "attention_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_efficiency_analysis(gpu_data):
    """创建效率分析图表"""

    # 计算FPS和效率指标
    results = []
    for item in gpu_data:
        single_fps = 1000 / item['single_ms_mean']  # 单尺度FPS
        fpn_fps = 1000 / item['fpn_ms_mean']       # FPN FPS
        fpn_overhead = (item['fpn_ms_mean'] - item['single_ms_mean']) / item['single_ms_mean'] * 100

        results.append({
            'backbone': item['backbone'],
            'attention': item['attention'],
            'single_fps': single_fps,
            'fpn_fps': fpn_fps,
            'fpn_overhead': fpn_overhead
        })

    # 排序
    results.sort(key=lambda x: x['single_fps'], reverse=True)

    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 图1: FPS排名
    names = [f"{r['backbone']}\n({r['attention']})" for r in results]
    single_fps = [r['single_fps'] for r in results]

    bars1 = ax1.barh(names, single_fps, color='gold', alpha=0.8)
    ax1.set_title('模型推理速度排名 (FPS)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('每秒帧数 (FPS)')

    for bar in bars1:
        width = bar.get_width()
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{width:.1f}', ha='left', va='center')

    # 图2: FPN开销分析
    fpn_overhead = [r['fpn_overhead'] for r in results]
    bars2 = ax2.barh(names, fpn_overhead, color='lightgreen', alpha=0.8)
    ax2.set_title('FPN计算开销 (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('开销百分比 (%)')

    for bar in bars2:
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{width:.1f}%', ha='left', va='center')

    # 图3: 骨干网络性能对比
    backbone_fps = {}
    for r in results:
        bb = r['backbone']
        if bb not in backbone_fps:
            backbone_fps[bb] = []
        backbone_fps[bb].append(r['single_fps'])

    backbones = list(backbone_fps.keys())
    avg_fps = [np.mean(backbone_fps[bb]) for bb in backbones]
    std_fps = [np.std(backbone_fps[bb]) for bb in backbones]

    bars3 = ax3.bar(backbones, avg_fps, yerr=std_fps, capsize=5,
                   color='skyblue', alpha=0.8, edgecolor='navy')
    ax3.set_title('骨干网络平均性能对比', fontsize=14, fontweight='bold')
    ax3.set_ylabel('平均FPS')
    ax3.grid(True, alpha=0.3)

    # 图4: 性能分类
    performance_categories = {'优秀': [], '良好': [], '一般': []}
    for r in results:
        fps = r['single_fps']
        if fps >= 50:
            performance_categories['优秀'].append(r)
        elif fps >= 30:
            performance_categories['良好'].append(r)
        else:
            performance_categories['一般'].append(r)

    categories = list(performance_categories.keys())
    counts = [len(performance_categories[cat]) for cat in categories]
    colors = ['gold', 'silver', 'orange']

    wedges, texts, autotexts = ax4.pie(counts, labels=categories, colors=colors,
                                      autopct='%1.0f%%', startangle=90)
    ax4.set_title('模型性能分布', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "efficiency_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("正在生成中期报告可视化图表...")

    # 加载数据
    gpu_data, cpu_data = load_test_data()

    # 生成图表
    create_performance_comparison(gpu_data, cpu_data)
    create_attention_analysis(gpu_data)
    create_efficiency_analysis(gpu_data)

    print("图表生成完成！保存在 docs/reports/ 目录下")

if __name__ == "__main__":
    main()