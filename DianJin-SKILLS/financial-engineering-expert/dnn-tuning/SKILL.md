---
name: dnn-tuning
category: 调参优化
description: 【DNN调参】当用户说"DNN调参"、"深度学习调参"、"调整网络结构"、"调整dropout"、"DNN过拟合"时使用。基于 Optuna TPE 贝叶斯优化，搜索网络架构（层数、宽度、dropout）和训练参数（learning_rate、weight_decay、batch_size），诊断驱动约束搜索空间。前置条件：需先用 dnn-modeling 训练出基线模型。
next_steps:
  - skill: model-comparison
    text: 多算法对比评估
    prompt: 与XGBoost和LR对比一下效果
---

# DNN 深度学习参数调优 (portable)

DNN 调参的**唯一入口**，基于 `_vendor/tuning/dnn_engine.DNNTuningEngine`。

核心设计：搜索期间缩减 epochs（加速），诊断驱动动态约束搜索空间。

---

## 调优流程

```
基线 DNN 模型 → 诊断分析(过拟合/欠拟合) → 约束空间构造 → Optuna 搜索(30 epochs) → 最优参数 → 迭代
```

---

## 执行模式

| 模式 | 触发条件 | 行为 |
|------|---------|------|
| **交互式**（默认） | 用户说"调参"/"帮我调一下DNN" | 每轮暂停等待用户反馈 |
| **AUTO** | 用户说"自动调优"/"帮我调到最优" | Agent 自动迭代直到收敛 |

**默认模式**: 交互式（更安全，用户可控）

---

## 参数说明

| 参数 | 必选 | 默认值 | 说明 |
|------|:----:|--------|------|
| `--data_path` / `-d` | ✅ | - | 数据文件路径（parquet/csv） |
| `--target` / `-t` | ✅ | - | 目标变量列名（0/1 二分类） |
| `--features` | | 自动推断 | 特征列表，逗号分隔 |
| `--time_col` | | `busi_dt` | 时间列名 |
| `--train_filter` | | 自动切分 | 训练集筛选条件 |
| `--val_filter` | | `val_ratio` 切出 | 验证集筛选条件 |
| `--oot_filter` | | 按时间切出 | OOT 条件 |
| `--oot_ratio` | | `0.20` | OOT 占比 |
| `--val_ratio` | | `0.25` | Val 占比 |
| `--random_seed` | | `42` | 随机种子 |
| `--exclude_cols` | | - | 排除列，逗号分隔 |
| `--n_layers` | | `3` | 隐藏层数 |
| `--layer_width` | | `128` | 首层宽度 |
| `--dropout` | | `0.3` | Dropout 率 |
| `--learning_rate` | | `0.001` | 学习率 |
| `--weight_decay` | | `1e-4` | L2 正则化 |
| `--batch_size` | | `512` | 批次大小 |
| `--epochs` | | `100` | 完整训练 epochs |
| `--search_epochs` | | `30` | 搜索期间 epochs（加速） |
| `--round` / `-r` | | `0` | 当前轮次 |
| `--max_rounds` | | `5` | 最大调优轮数 |
| `--auto` | | - | 自动调优模式（flag） |
| `--metric` | | `auc` | 评估指标 |
| `--model_name` | | 自动生成 | 模型名称 |
| `--output_dir` | | `./outputs/<ts>` | 产物输出目录 |
| `--config` | | - | JSON 配置路径 |

---

## 搜索空间

| 参数 | 类型 | 范围 | 说明 |
|------|------|------|------|
| n_layers | int | 2-4 | 隐藏层数 |
| layer_width | int | 32-256 | 首层宽度（递减结构） |
| dropout | float | 0.1-0.5 | Dropout 比率 |
| learning_rate | float(log) | 1e-4 ~ 0.01 | Adam 学习率 |
| weight_decay | float(log) | 1e-5 ~ 1e-3 | L2 正则化 |
| batch_size | categorical | 128/256/512/1024 | 批次大小 |

---

## 诊断驱动策略

| 诊断 | dropout | weight_decay | n_layers | layer_width |
|------|---------|-------------|----------|-------------|
| 过拟合 | ↑ 抬高 | ↑ 增强 | ↓ 减少 | ↓ 缩小 |
| 欠拟合 | ↓ 降低 | ↓ 减弱 | ↑ 增加 | ↑ 增大 |
| 拟合良好 | ±微调 | ±微调 | ±微调 | ±微调 |

---

## 执行方式

### 交互式模式（单轮调优）

```bash
python scripts/tuner.py \
  --data_path ./data.parquet --target y_label \
  --round 1 --output_dir ./outputs/dnn_tuning
```

### AUTO 模式（自动调优循环）

```bash
python scripts/tuner.py \
  --data_path ./data.parquet --target y_label \
  --auto --max_rounds 5 --output_dir ./outputs/dnn_tuning
```

---

## 调优策略

### 策略1：抗过拟合

**适用条件**：Train-Val Gap > 0.05

**调整方向**：
- `dropout`: 当前值 + 0.1（上限 0.5）
- `weight_decay`: 当前值 × 2
- `n_layers`: 当前值 - 1（最小为 2）
- `layer_width`: 当前值 - 32（最小为 32）

### 策略2：增强拟合

**适用条件**：Val AUC < 0.58 且 Gap < 0.03

**调整方向**：
- `n_layers`: 当前值 + 1（最大为 4）
- `layer_width`: 当前值 + 32（最大为 256）
- `weight_decay`: 当前值 × 0.5
- `learning_rate`: 当前值 × 1.2

### 策略3：精细微调

**适用条件**：Gap ∈ [0.03, 0.05]，模型状态良好

**调整方向**：
- `learning_rate`: 小幅调整 ±20%
- `dropout`: 小幅调整 ±0.05
- 其他参数保持不变

### 策略4：收敛判定

**条件**：连续2轮 Val 指标提升 < 0.001

**行为**：停止调优，输出最终结果

---

## 输出格式规范

与 xgb-tuning 保持一致的逐轮诊断报告格式。

每轮调优结束后，**必须**输出以下结构化信息：

```markdown
### 第 N 轮 DNN 调优结果

**参数变化**:
| 参数 | 上一轮 | 本轮 | 调整原因 |
|------|-------|------|----------|
| n_layers | 3 | 3 | 不变 |
| layer_width | 128 | 96 | 降低过拟合 |
| dropout | 0.3 | 0.4 | 增强正则化 |

**效果对比**:
| 指标 | 上一轮 | 本轮 | 变化 |
|------|-------|------|------|
| Val AUC | 0.72 | 0.73 | +0.01 ✓ |
| OOT AUC | 0.70 | 0.71 | +0.01 ✓ |
| Gap | 0.06 | 0.04 | -0.02 ✓ |

**诊断结论**: 轻微过拟合（Gap 下降但仍 > 0.03）

**下一步建议**: 可继续微调 dropout，或接受当前结果
```

---

## 与其他技能的关系

| 技能 | 职责 | 关系 |
|------|------|------|
| `dnn-modeling` | 基线建模 | 前置：需先用其训练出基线模型 |
| `model-comparison` | 多算法对比 | 后续：可与 XGB/LR 做公平对比 |
| `xgb-tuning` | XGBoost 调参 | 平行：同数据不同算法的调参 |

---

## 注意事项

1. **数据要求**：目标变量必须为 0/1 二分类
2. **搜索加速**：搜索期间使用 `search_epochs=30`，最终模型使用 `epochs=100`
3. **收敛判定**：连续2轮提升不足 0.001 自动停止
4. **最大轮数**：默认最多 5 轮
5. **产物位置**：模型和报告保存到 `<output_dir>/models/` 和 `<output_dir>/`
