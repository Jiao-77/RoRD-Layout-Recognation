"""
TensorBoard 实验数据导出工具

功能：从 TensorBoard event 文件中提取标量数据，并导出为多种格式。

支持的导出格式：
  - CSV: 便于电子表格和数据分析
  - JSON: 便于程序化处理
  - Markdown: 便于文档生成和报告

使用示例：
  # 导出为 CSV 格式
  python tools/export_tb_summary.py \
    --log-dir runs/train/baseline \
    --output-format csv \
    --output-file export_results.csv
  
  # 导出为 JSON 格式
  python tools/export_tb_summary.py \
    --log-dir runs/train/baseline \
    --output-format json \
    --output-file export_results.json
  
  # 导出为 Markdown 格式
  python tools/export_tb_summary.py \
    --log-dir runs/train/baseline \
    --output-format markdown \
    --output-file export_results.md
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def read_tensorboard_events(log_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
    """
    读取 TensorBoard event 文件，提取标量数据。
    
    Args:
        log_dir: TensorBoard 日志目录路径
    
    Returns:
        标量数据字典，格式为 {标量名: [(step, value), ...]}
    """
    try:
        from tensorboard.compat.proto import event_pb2
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("❌ 错误：需要安装 tensorboard。运行: pip install tensorboard")
        return {}
    
    print(f"读取 TensorBoard 日志: {log_dir}")
    
    if not log_dir.exists():
        print(f"❌ 日志目录不存在: {log_dir}")
        return {}
    
    # 使用 event_accumulator 加载数据
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()
    
    scalars_dict = defaultdict(list)
    
    # 遍历所有标量标签
    scalar_tags = ea.Tags().get('scalars', [])
    print(f"找到 {len(scalar_tags)} 个标量标签")
    
    for tag in scalar_tags:
        try:
            events = ea.Scalars(tag)
            for event in events:
                step = event.step
                value = event.value
                scalars_dict[tag].append((step, value))
            print(f"  ✓ {tag}: {len(events)} 个数据点")
        except Exception as e:
            print(f"  ⚠️ 读取 {tag} 失败: {e}")
    
    return dict(scalars_dict)


def export_to_csv(scalars_dict: Dict[str, List[Tuple[int, float]]], output_file: Path) -> None:
    """
    导出标量数据为 CSV 格式。
    
    格式：
      step,metric1,metric2,...
      0,1.234,5.678
      1,1.200,5.650
      ...
    """
    if not scalars_dict:
        print("❌ 没有标量数据可导出")
        return
    
    # 收集所有 step
    all_steps = set()
    for tag_data in scalars_dict.values():
        for step, _ in tag_data:
            all_steps.add(step)
    
    all_steps = sorted(all_steps)
    all_tags = sorted(scalars_dict.keys())
    
    # 建立 step -> {tag: value} 的映射
    step_data = defaultdict(dict)
    for tag, data in scalars_dict.items():
        for step, value in data:
            step_data[step][tag] = value
    
    # 写入 CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['step'] + all_tags)
        writer.writeheader()
        
        for step in all_steps:
            row = {'step': step}
            row.update(step_data.get(step, {}))
            writer.writerow(row)
    
    print(f"✅ CSV 文件已保存: {output_file}")
    print(f"   - 行数: {len(all_steps) + 1} (含表头)")
    print(f"   - 列数: {len(all_tags) + 1}")


def export_to_json(scalars_dict: Dict[str, List[Tuple[int, float]]], output_file: Path) -> None:
    """
    导出标量数据为 JSON 格式。
    
    格式：
      {
        "metric1": [[step, value], [step, value], ...],
        "metric2": [[step, value], [step, value], ...],
        ...
      }
    """
    if not scalars_dict:
        print("❌ 没有标量数据可导出")
        return
    
    # 转换为序列化格式
    json_data = {
        tag: [[step, float(value)] for step, value in data]
        for tag, data in scalars_dict.items()
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON 文件已保存: {output_file}")
    print(f"   - 标量数: {len(json_data)}")
    total_points = sum(len(v) for v in json_data.values())
    print(f"   - 数据点总数: {total_points}")


def export_to_markdown(scalars_dict: Dict[str, List[Tuple[int, float]]], output_file: Path) -> None:
    """
    导出标量数据为 Markdown 格式（包含表格摘要和详细数据）。
    """
    if not scalars_dict:
        print("❌ 没有标量数据可导出")
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# TensorBoard 实验数据导出\n\n")
        f.write(f"**导出时间**: {Path('').resolve().ctime()}\n\n")
        
        # 摘要表格
        f.write("## 📊 数据摘要\n\n")
        f.write("| 指标 | 最小值 | 最大值 | 平均值 | 标准差 | 数据点数 |\n")
        f.write("|------|--------|--------|--------|--------|----------|\n")
        
        for tag in sorted(scalars_dict.keys()):
            data = scalars_dict[tag]
            if not data:
                continue
            
            values = [v for _, v in data]
            min_val = float(np.min(values))
            max_val = float(np.max(values))
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            count = len(values)
            
            f.write(f"| {tag} | {min_val:.6g} | {max_val:.6g} | {mean_val:.6g} | {std_val:.6g} | {count} |\n")
        
        # 详细数据表格（仅保留前 20 个 step 作为示例）
        f.write("\n## 📈 详细数据（前 20 个 step）\n\n")
        
        # 收集所有 step
        all_steps = set()
        for tag_data in scalars_dict.values():
            for step, _ in tag_data:
                all_steps.add(step)
        
        all_steps = sorted(all_steps)[:20]
        all_tags = sorted(scalars_dict.keys())
        
        # 建立 step -> {tag: value} 的映射
        step_data = defaultdict(dict)
        for tag, data in scalars_dict.items():
            for step, value in data:
                step_data[step][tag] = value
        
        # 生成表格
        if all_steps:
            header = ['Step'] + all_tags
            f.write("| " + " | ".join(header) + " |\n")
            f.write("|" + "|".join(["---"] * len(header)) + "|\n")
            
            for step in all_steps:
                row = [str(step)]
                for tag in all_tags:
                    val = step_data.get(step, {}).get(tag, "-")
                    if isinstance(val, float):
                        row.append(f"{val:.6g}")
                    else:
                        row.append(str(val))
                f.write("| " + " | ".join(row) + " |\n")
        
        f.write(f"\n> **注**: 表格仅显示前 {len(all_steps)} 个 step 的数据。\n")
        f.write(f"> 完整数据包含 {len(sorted(set(s for tag_data in scalars_dict.values() for s, _ in tag_data)))} 个 step。\n")
    
    print(f"✅ Markdown 文件已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="TensorBoard 实验数据导出工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        required=True,
        help='TensorBoard 日志根目录（包含 event 文件）'
    )
    
    parser.add_argument(
        '--output-format',
        type=str,
        choices=['csv', 'json', 'markdown'],
        default='csv',
        help='导出格式（默认: csv）'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='输出文件路径'
    )
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir).expanduser()
    output_file = Path(args.output_file).expanduser()
    
    print(f"\n{'=' * 80}")
    print(f"{'TensorBoard 数据导出工具':^80}")
    print(f"{'=' * 80}\n")
    
    # 读取数据
    scalars_dict = read_tensorboard_events(log_dir)
    
    if not scalars_dict:
        print("❌ 未能读取任何数据")
        return 1
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 根据格式导出
    print(f"\n正在导出为 {args.output_format.upper()} 格式...\n")
    
    if args.output_format == 'csv':
        export_to_csv(scalars_dict, output_file)
    elif args.output_format == 'json':
        export_to_json(scalars_dict, output_file)
    elif args.output_format == 'markdown':
        export_to_markdown(scalars_dict, output_file)
    
    print(f"\n{'=' * 80}\n")
    print("✅ 导出完成！\n")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
