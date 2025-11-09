# IC版图匹配功能使用指南

本文档介绍如何使用增强版的`match.py`进行IC版图匹配，实现输入大版图和小版图，找到所有匹配区域并输出详细信息。

## 🎯 功能概述

### 输入
- **大版图**：待搜索的大型IC版图图像
- **小版图**：要查找的目标模板图像

### 输出
- **坐标信息**：每个匹配区域的边界框坐标 (x, y, width, height)
- **旋转角度**：检测到的旋转角度 (0°, 90°, 180°, 270°)
- **置信度**：匹配质量评分 (0-1)
- **相似度**：模板与区域的相似程度 (0-1)
- **差异描述**：文本化的差异说明
- **变换矩阵**：3x3单应性矩阵

## 🚀 快速开始

### 基本用法

```bash
python match.py \
    --layout data/large_layout.png \
    --template data/small_template.png \
    --output results/matching.png \
    --json_output results/matching.json
```

### 使用示例脚本

```bash
python examples/layout_matching_example.py \
    --layout data/large_layout.png \
    --template data/small_template.png \
    --model models/rord_model_best.pth
```

## 📋 命令行参数

### 必需参数
- `--layout`: 大版图图像路径
- `--template`: 小版图（模板）图像路径

### 可选参数
- `--config`: 配置文件路径 (默认: configs/base_config.yaml)
- `--model_path`: 模型权重路径
- `--output`: 可视化结果保存路径
- `--json_output`: JSON结果保存路径
- `--simple_format`: 使用简单输出格式（兼容旧版本）
- `--fpn_off`: 关闭FPN匹配路径
- `--no_nms`: 关闭关键点去重

## 📊 输出格式详解

### 详细格式 (默认)

```json
{
  "found_matches": true,
  "total_matches": 2,
  "matches": [
    {
      "bbox": {
        "x": 120,
        "y": 80,
        "width": 256,
        "height": 128
      },
      "rotation": 0,
      "confidence": 0.854,
      "similarity": 0.892,
      "inliers": 45,
      "scale": 1.0,
      "homography": [[1.0, 0.0, 120.0], [0.0, 1.0, 80.0], [0.0, 0.0, 1.0]],
      "description": "高度匹配, 无旋转"
    },
    {
      "bbox": {
        "x": 400,
        "y": 200,
        "width": 256,
        "height": 128
      },
      "rotation": 90,
      "confidence": 0.723,
      "similarity": 0.756,
      "inliers": 32,
      "scale": 0.8,
      "homography": [[0.0, -1.0, 528.0], [1.0, 0.0, 200.0], [0.0, 0.0, 1.0]],
      "description": "良好匹配, 旋转90度, 缩小1.25倍"
    }
  ]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `bbox.x` | int | 匹配区域左上角X坐标 |
| `bbox.y` | int | 匹配区域左上角Y坐标 |
| `bbox.width` | int | 匹配区域宽度 |
| `bbox.height` | int | 匹配区域高度 |
| `rotation` | int | 旋转角度 (0°, 90°, 180°, 270°) |
| `confidence` | float | 置信度 (0-1) |
| `similarity` | float | 相似度 (0-1) |
| `inliers` | int | 内点数量 |
| `scale` | float | 匹配尺度 |
| `homography` | array | 3x3变换矩阵 |
| `description` | string | 差异描述 |

## 🔧 技术原理

### 1. 特征提取
- 使用RoRD模型提取几何感知特征
- 支持FPN多尺度特征金字塔
- 旋转不变的关键点检测

### 2. 多尺度搜索
- 在不同尺度下搜索模板
- 支持模板缩放匹配
- 多实例检测算法

### 3. 几何验证
- RANSAC变换估计
- 单应性矩阵计算
- 旋转角度提取

### 4. 质量评估
- 内点比例计算
- 变换矩阵质量评估
- 综合置信度评分

## 📈 质量指标说明

### 置信度 (Confidence)
基于内点比例和变换质量计算：
- **0.8-1.0**: 高质量匹配
- **0.6-0.8**: 良好匹配
- **0.4-0.6**: 中等匹配
- **0.0-0.4**: 低质量匹配

### 相似度 (Similarity)
基于匹配率和覆盖率计算：
- 考虑模板关键点匹配率
- 考虑版图区域覆盖率
- 综合评估相似程度

### 差异描述
自动生成的文本描述：
- 匹配质量等级
- 旋转角度信息
- 缩放变换信息

## 🎨 可视化结果

匹配可视化包含：
- 绿色边界框标识匹配区域
- 匹配编号标签
- 置信度显示
- 旋转角度信息
- 差异描述摘要

## 🛠️ 高级配置

### 匹配参数调优

编辑`configs/base_config.yaml`中的匹配参数：

```yaml
matching:
  keypoint_threshold: 0.5        # 关键点阈值
  ransac_reproj_threshold: 5.0   # RANSAC重投影阈值
  min_inliers: 15                # 最小内点数量
  pyramid_scales: [0.75, 1.0, 1.5]  # 搜索尺度
  use_fpn: true                  # 使用FPN
  nms:
    enabled: true
    radius: 4                    # NMS半径
```

### 性能优化

1. **GPU加速**: 确保CUDA可用
2. **FPN优化**: 大图使用FPN，小图使用滑窗
3. **尺度调整**: 根据图像大小调整`pyramid_scales`
4. **阈值调优**: 根据应用场景调整`keypoint_threshold`

## 🔍 故障排除

### 常见问题

1. **未找到匹配**
   - 检查图像质量和分辨率
   - 降低`keypoint_threshold`
   - 减少`min_inliers`数量

2. **误匹配过多**
   - 提高`keypoint_threshold`
   - 增大`ransac_reproj_threshold`
   - 启用NMS去重

3. **性能较慢**
   - 使用FPN模式 (`use_fpn: true`)
   - 减少`pyramid_scales`数量
   - 调整滑窗口大小

4. **内存不足**
   - 减小图像尺寸
   - 降低批次大小
   - 使用CPU模式

### 调试技巧

1. **可视化检查**: 查看生成的可视化结果
2. **JSON分析**: 检查详细的匹配数据
3. **阈值调整**: 逐步调整参数找到最佳设置
4. **日志查看**: 启用TensorBoard日志记录

## 📝 API集成

### Python调用示例

```python
import subprocess
import json

# 执行匹配
result = subprocess.run([
    'python', 'match.py',
    '--layout', 'large.png',
    '--template', 'small.png',
    '--json_output', 'temp.json'
], capture_output=True, text=True)

# 解析结果
with open('temp.json') as f:
    data = json.load(f)

if data['found_matches']:
    for match in data['matches']:
        bbox = match['bbox']
        print(f"位置: ({bbox['x']}, {bbox['y']})")
        print(f"置信度: {match['confidence']}")
        print(f"旋转: {match['rotation']}°")
```

## 🎯 应用场景

1. **IC设计验证**: 检查设计是否符合规范
2. **IP保护**: 检测版图抄袭和侵权
3. **制造验证**: 确认制造结果与设计一致
4. **设计复用**: 在新设计中查找复用的模块
5. **质量检测**: 自动化版图质量检查

## 📚 更多资源

- [RoRD模型训练指南](diffusion_training.md)
- [配置文件说明](../configs/base_config.yaml)
- [项目架构文档](architecture.md)