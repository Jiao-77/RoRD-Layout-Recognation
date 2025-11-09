#!/usr/bin/env python3
"""
IC版图匹配示例脚本

演示如何使用增强版的match.py进行版图匹配：
- 输入大版图和小版图
- 输出匹配区域的坐标、旋转角度、置信度等信息
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="IC版图匹配示例")
    parser.add_argument("--layout", type=str, help="大版图路径")
    parser.add_argument("--template", type=str, help="小版图（模板）路径")
    parser.add_argument("--model", type=str, help="模型路径")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="配置文件路径")
    parser.add_argument("--output_dir", type=str, default="matching_results", help="输出目录")

    args = parser.parse_args()

    # 检查必要参数
    if not args.layout or not args.template:
        print("❌ 请提供大版图和小版图路径")
        print("示例: python examples/layout_matching_example.py --layout data/large_layout.png --template data/small_template.png")
        return

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置输出文件路径
    viz_output = output_dir / "matching_visualization.png"
    json_output = output_dir / "matching_results.json"

    # 构建匹配命令
    cmd = [
        sys.executable, "match.py",
        "--layout", args.layout,
        "--template", args.template,
        "--config", args.config,
        "--output", str(viz_output),
        "--json_output", str(json_output)
    ]

    # 添加模型路径（如果提供）
    if args.model:
        cmd.extend(["--model_path", args.model])

    print("🚀 开始版图匹配...")
    print(f"📁 大版图: {args.layout}")
    print(f"📁 小版图: {args.template}")
    print(f"📁 输出目录: {output_dir}")
    print("-" * 50)

    # 执行匹配
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ 匹配完成！")
        print(f"📊 查看详细结果: {json_output}")
        print(f"🖼️ 查看可视化结果: {viz_output}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 匹配失败: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ 找不到match.py文件，请确保在项目根目录运行")
        sys.exit(1)


if __name__ == "__main__":
    main()