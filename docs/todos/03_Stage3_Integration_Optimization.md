# 📋 第三阶段：集成与优化 (1-2 周)

**优先级**: 🟠 **中** (项目质量完善)  
**预计工时**: 1-2 周  
**目标**: 创建自动化脚本、补充测试框架、完善文档

---

## 📌 任务概览

本阶段专注于项目的工程实践完善，通过自动化脚本、测试框架和文档来提升开发效率。

---

## ✅ 任务清单

### 1. 自动化脚本 (Makefile / tasks.json)

**目标**: 一键启动常用操作

#### 1.1 创建 Makefile

- [ ] 创建项目根目录下的 `Makefile`
  - [ ] 添加 `make install` 目标: 运行 `uv sync`
  - [ ] 添加 `make train` 目标: 启动训练脚本
  - [ ] 添加 `make eval` 目标: 启动评估脚本
  - [ ] 添加 `make tensorboard` 目标: 启动 TensorBoard
  - [ ] 添加 `make benchmark` 目标: 运行性能测试
  - [ ] 添加 `make export` 目标: 导出 TensorBoard 数据
  - [ ] 添加 `make clean` 目标: 清理临时文件

**验收标准**:
- [ ] Makefile 语法正确，可正常执行
- [ ] 所有目标都有帮助文本说明
- [ ] 命令参数可配置

#### 1.2 创建 VS Code tasks.json

- [ ] 创建 `.vscode/tasks.json` 文件
  - [ ] 添加 "Install" 任务: `uv sync`
  - [ ] 添加 "Train" 任务: `train.py`
  - [ ] 添加 "Evaluate" 任务: `evaluate.py`
  - [ ] 添加 "TensorBoard" 任务（后台运行）
  - [ ] 添加 "Benchmark" 任务: `tests/benchmark_fpn.py`
  - [ ] 配置问题匹配器 (problemMatcher) 用于错误解析

**验收标准**:
- [ ] VS Code 可直接调用任务
- [ ] 输出能正确显示在问题面板中

---

### 2. 测试框架 (tests/)

**目标**: 建立单元测试、集成测试和端到端测试

#### 2.1 单元测试：NMS 函数

- [ ] 创建 `tests/test_nms.py`
  - [ ] 导入 `match.py` 中的 `radius_nms` 函数
  - [ ] 编写测试用例:
    - [ ] 空输入测试
    - [ ] 单个点测试
    - [ ] 重复点去重测试
    - [ ] 半径临界值测试
    - [ ] 大规模关键点测试（1000+ 点）
  - [ ] 验证输出维度和内容的正确性

**验收标准**:
- [ ] 所有测试用例通过
- [ ] 代码覆盖率 > 90%

#### 2.2 集成测试：FPN 推理

- [ ] 创建 `tests/test_fpn_inference.py`
  - [ ] 加载模型和配置
  - [ ] 编写测试用例:
    - [ ] 模型加载测试
    - [ ] 单尺度推理测试 (return_pyramid=False)
    - [ ] 多尺度推理测试 (return_pyramid=True)
    - [ ] 金字塔输出维度检查
    - [ ] 特征维度一致性检查
    - [ ] GPU/CPU 切换测试

**验收标准**:
- [ ] 所有测试用例通过
- [ ] 推理结果符合预期维度和范围

#### 2.3 端到端测试：完整匹配流程

- [ ] 创建 `tests/test_end_to_end.py`
  - [ ] 编写完整的匹配流程测试:
    - [ ] 加载版图和模板
    - [ ] 执行特征提取
    - [ ] 执行特征匹配
    - [ ] 验证输出实例数量和格式
    - [ ] FPN 路径 vs 滑窗路径对比

**验收标准**:
- [ ] 所有测试用例通过
- [ ] 两种路径输出结果一致

#### 2.4 配置 pytest 和测试运行

- [ ] 创建 `pytest.ini` 配置文件
  - [ ] 设置测试发现路径
  - [ ] 配置输出选项
  - [ ] 设置覆盖率报告

- [ ] 添加到 `pyproject.toml`:
  - [ ] 添加 pytest 和 pytest-cov 作为开发依赖
  - [ ] 配置测试脚本

**验收标准**:
- [ ] `pytest` 命令可正常运行所有测试
- [ ] 生成覆盖率报告

---

### 3. 文档完善

**目标**: 补充项目文档，降低新开发者学习成本

#### 3.1 完善 README.md

- [ ] 更新项目概述
  - [ ] 添加项目徽章（完成度、License 等）
  - [ ] 补充简要功能说明
  - [ ] 添加快速开始部分

- [ ] 添加安装说明
  - [ ] 系统要求（Python、CUDA 等）
  - [ ] 安装步骤（uv sync）
  - [ ] GPU 支持配置

- [ ] 添加使用教程
  - [ ] 基础使用：训练、评估、推理
  - [ ] 配置说明：YAML 参数详解
  - [ ] 高级用法：自定义骨干网络、损失函数等

- [ ] 添加故障排查部分
  - [ ] 常见问题和解决方案
  - [ ] 日志查看方法
  - [ ] GPU 内存不足处理

#### 3.2 编写配置参数文档

- [ ] 创建 `docs/CONFIG.md`
  - [ ] 详细说明 `configs/base_config.yaml` 的每个参数
  - [ ] 提供参数调整建议
  - [ ] 给出常用配置组合示例

**验收标准**:
- [ ] 文档清晰、示例完整
- [ ] 新开发者可按文档快速上手

#### 3.3 编写 API 文档

- [ ] 为核心模块生成文档
  - [ ] `models/rord.py`: RoRD 模型 API
  - [ ] `match.py`: 匹配流程 API
  - [ ] `utils/`: 工具函数 API

- [ ] 添加代码示例和最佳实践

**验收标准**:
- [ ] API 文档完整、易于查阅

---

## 📊 完成进度

| 子任务 | 完成度 | 状态 |
|--------|--------|------|
| Makefile | 0% | ⏳ 未开始 |
| tasks.json | 0% | ⏳ 未开始 |
| 单元测试 (NMS) | 0% | ⏳ 未开始 |
| 集成测试 (FPN) | 0% | ⏳ 未开始 |
| 端到端测试 | 0% | ⏳ 未开始 |
| README 补充 | 0% | ⏳ 未开始 |
| 配置文档 | 0% | ⏳ 未开始 |
| API 文档 | 0% | ⏳ 未开始 |

---

## 📝 开发指南

### 步骤 1: 创建 Makefile

```bash
# 新建 Makefile
touch Makefile

# 添加基础内容，参考 docs/description/README.md 中的常用命令
```

### 步骤 2: 设置测试框架

```bash
# 安装 pytest
uv pip install pytest pytest-cov

# 创建测试文件
touch tests/test_nms.py
touch tests/test_fpn_inference.py
touch tests/test_end_to_end.py

# 运行测试
pytest tests/ -v --cov=
```

### 步骤 3: 完善文档

```bash
# 更新 README.md
nano README.md

# 创建配置文档
touch docs/CONFIG.md

# 生成 API 文档（如使用 Sphinx）
# sphinx-quickstart docs/_build
```

---

## 🔗 相关资源

- [Pytest 官方文档](https://docs.pytest.org/)
- [Makefile 教程](https://www.gnu.org/software/make/manual/)
- [VS Code tasks 文档](https://code.visualstudio.com/docs/editor/tasks)
- [Markdown 最佳实践](https://www.markdownguide.org/)

---

## ✅ 验收标准

本阶段完成的标准：

- [ ] Makefile 包含所有关键命令并可正常运行
- [ ] VS Code tasks.json 配置完整
- [ ] 所有核心函数都有单元测试
- [ ] 关键流程都有集成和端到端测试
- [ ] 测试覆盖率 > 80%
- [ ] README 包含快速开始、配置和故障排查
- [ ] API 文档清晰、示例完整
- [ ] 新开发者可按文档快速上手

---

**预计完成时间**: 1-2 周  
**下一阶段**: 高级功能集成（第四阶段）
