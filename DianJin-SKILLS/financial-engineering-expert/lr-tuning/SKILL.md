---
name: lr-tuning
category: 调参优化
description: 【LR评分卡调参】当用户说"LR调参"、"评分卡调优"、"调整分箱数"、"调整C值"、"LR过拟合"时使用。基于 Optuna TPE 贝叶斯优化，联合搜索 WoE 分箱参数（max_n_bins、iv_threshold）和 LR 模型参数（C、regularization），诊断驱动约束搜索空间。前置条件：需先用 lr-modeling 训练出基线模型。
next_steps:
  - skill: model-comparison
    text: 多算法对比评估
    prompt: 与XGBoost和DNN对比一下效果
---

# LR 评分卡参数调优 (portable)

LR 评分卡调参的**唯一入口**，基于 `_vendor/tuning/lr_engine.LRTuningEngine`。

核心设计：WoE 分箱参数与 LR 正则化参数**联合搜索**，确保最优组合。

---

## 调优流程

```
基线 LR 模型 → 诊断分析(过拟合/欠拟合) → 约束空间构造 → Optuna 搜索 → 最优参数 → 迭代
```

---

## 执行模式

| 模式 | 触发条件 | 行为 |
|------|---------|------|
| **交互式**（默认） | 用户说"调参"/"帮我调一下LR" | 每轮暂停等待用户反馈 |
| **AUTO** | 用户说"自动调优"/"帮我调到最优" | Agent 自动迭代直到收敛 |

---

## 参数说明

| 参数 | 必选 | 默认值 | 说明 |
|------|:----:|--------|------|
| `--data_path` / `-d` | ✅ | - | 数据文件路径 |
| `--target` / `-t` | ✅ | - | 目标变量列名 |
| `--features` | | 自动推断 | 特征列表，逗号分隔 |
| `--time_col` | | `busi_dt` | 时间列名 |
| `--train_filter` | | 自动切分 | 训练集筛选条件 |
| `--val_filter` | | `val_ratio` 切出 | 验证集筛选条件 |
| `--oot_filter` | | 按时间切出 | OOT 条件 |
| `--oot_ratio` | | `0.20` | OOT 占比 |
| `--val_ratio` | | `0.25` | Val 占比 |
| `--random_seed` | | `42` | 随机种子 |
| `--exclude_cols` | | - | 排除列 |
| `--max_n_bins` | | `8` | 当前 WoE 分箱数 |
| `--iv_threshold` | | `0.02` | 当前 IV 阈值 |
| `--C` | | `1.0` | 当前正则化强度倒数 |
| `--regularization` | | `l2` | 正则化类型 |
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
| max_n_bins | int | 3-15 | WoE 分箱数（越大越精细） |
| iv_threshold | float(log) | 0.005-0.10 | IV 筛选阈值（越低入模特征越多） |
| C | float(log) | 0.01-100 | 正则化强度倒数（越大正则化越弱） |
| regularization | categorical | l1/l2/elasticnet | 正则化类型 |

---

## 诊断驱动策略

| 诊断 | C | iv_threshold | max_n_bins |
|------|---|--------------|------------|
| 过拟合 | ↓ 收紧 | ↑ 抬高（减少特征） | ↓ 减少 |
| 欠拟合 | ↑ 放松 | ↓ 降低（更多特征） | ↑ 增大 |
| 拟合良好 | ±微调 | ±微调 | ±微调 |

---

## 执行方式

### 交互式模式（单轮）

```bash
python scripts/tuner.py \
  --data_path ./data.parquet --target y_label \
  --round 1 --output_dir ./outputs/lr_tuning
```

### AUTO 模式

```bash
python scripts/tuner.py \
  --data_path ./data.parquet --target y_label \
  --auto --max_rounds 5 --output_dir ./outputs/lr_tuning
```

---

## 调优策略

### 策略1：抗过拟合

**适用条件**：Train-OOT Gap > 0.05

**调整方向**：
- `C`: 当前值 × 0.5（收紧正则化）
- `iv_threshold`: 当前值 × 1.5（减少入模特征）
- `max_n_bins`: 当前值 - 1（降低分箱精细度）

### 策略2：增强拟合

**适用条件**：OOT AUC < 0.58 且 Gap < 0.03

**调整方向**：
- `C`: 当前值 × 2（放松正则化）
- `iv_threshold`: 当前值 × 0.5（增加入模特征）
- `max_n_bins`: 当前值 + 2（提升分箱精细度）

### 策略3：精细微调

**适用条件**：Gap ∈ [0.03, 0.05]，模型状态良好

**调整方向**：
- `C`: 小幅调整 ±20%
- `max_n_bins`: 微调 ±1
- 其他参数保持不变

### 策略4：收敛判定

**条件**：连续2轮 OOT 指标提升 < 0.001

**行为**：停止调优，输出最终结果

---

## 输出格式规范

每轮调优结束后，**必须**输出以下结构化信息：

```markdown
### 第 N 轮 LR 调优结果

**参数变化**:
| 参数 | 上一轮 | 本轮 | 调整原因 |
|------|-------|------|----------|
| C | 1.0 | 0.5 | 收紧正则化 |
| iv_threshold | 0.02 | 0.03 | 减少入模特征 |
| max_n_bins | 8 | 6 | 降低过拟合 |

**效果对比**:
| 指标 | 上一轮 | 本轮 | 变化 |
|------|-------|------|------|
| OOT AUC | 0.72 | 0.73 | +0.01 ✓ |
| OOT KS | 0.17 | 0.18 | +0.01 ✓ |
| Gap | 0.06 | 0.04 | -0.02 ✓ |

**诊断结论**: 轻微过拟合（Gap 下降但仍 > 0.03）

**下一步建议**: 可继续微调 C 值，或接受当前结果
```

---

## 与其他技能的关系

| 技能 | 职责 | 关系 |
|------|------|------|
| `lr-modeling` | 基线建模 | 前置：需先用其训练出基线模型 |
| `model-comparison` | 多算法对比 | 后续：可与 XGB/DNN 做公平对比 |
| `xgb-tuning` | XGBoost 调参 | 平行：同数据不同算法的调参 |

---

## 注意事项

1. **数据要求**：目标变量必须为 0/1 二分类
2. **联合搜索**：WoE 分箱与 LR 参数联合优化，确保最优组合
3. **收敛判定**：连续2轮提升不足 0.001 自动停止
4. **最大轮数**：默认最多 5 轮
5. **产物位置**：模型和报告保存到 `<output_dir>/models/` 和 `<output_dir>/`
