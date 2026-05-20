---
name: xgb-deepmodel
category: 建模训练
description: 【DeepModel深度建模】当用户需要"分群集成建模"、"分群+Stacking融合"、"子模型训练+集成"时使用。核心能力：按业务分群条件训练多个 XGBoost 子模型，通过 Stacking 融合成主模型，并与单模型基线对比。适用于"分群建模"、"按客群分别建模"、"子模型融合"、"集成学习"等高级场景。包含三个脚本：sub_trainer（分群子模型训练）、stacker（XGBoost Stacking融合）、comparator（集成vs基线对比）。与 xgb-modeling 的区别：本 Skill 拆分客群分别建模再融合，xgb-modeling 全量样本统一建模。与 segment-modeling 的区别：本 Skill 专注分群+Stacking融合，segment-modeling 专注分群策略探索。
next_steps:
  - skill: model-explanation
    text: SHAP解释融合模型
    prompt: 解释 Stacking 融合后模型的特征重要性
---

# XGBoost DeepModel — 分群集成建模 (portable)

分群训练多个 XGBoost 子模型，通过 OOF Stacking 融合成主模型，评估集成收益。

---

## 阶段1：训练分群子模型

### 参数说明

> 参数 spec：`_vendor/xgb_cli.py` 中 `domain=deepmodel-sub`。复杂嵌套 JSON（segments / features_per_segment / pos_weight_per_segment）**必须**走 `--config`。

| 参数 | 必选 | 默认值 | 说明 |
|------|:----:|--------|------|
| `--data_path` / `-d` | ✅ | - | 数据文件路径 |
| `--target` / `-t` | ✅ | - | 目标变量列名 |
| `--segments` | ✅* | - | 分群条件 JSON（**推荐通过 `--config` 的 `segments` 字段**） |
| `--segment_col` | ✅* | - | 按列唯一值自动分群（与 segments 二选一） |
| `--time_col` | | `busi_dt` | 时间列名 |
| `--train_filter` | | 自动切分 | 全局训练集筛选条件 |
| `--val_filter` | | `val_ratio` 切出 | 全局验证集筛选条件 |
| `--oot_filter` | | 按时间切出 | OOT 测试集条件 |
| `--exclude_cols` | | - | 排除列，逗号分隔 |
| `--features` | | 自动筛选 | 全局特征列表，逗号分隔 |
| `--features_per_segment` | | - | 分群差异化特征 JSON（**走 `--config`**） |
| `--sample_weight_col` | | - | 样本权重列名 |
| `--pos_weight_per_segment` | | - | 各分群正样本权重 JSON（**走 `--config`**） |
| `--auto` | | `false` | flag。对 OOT Gap > 0.05 的分群自动调参 |
| `--output_dir` | | `./outputs/<ts>` | 子模型保存目录 |
| `--config` | | - | JSON 配置路径 |

### 执行方式

```bash
python scripts/sub_trainer.py \
  --data_path ./data.parquet --target y_label \
  --time_col busi_dt --exclude_cols "cust_code,busi_dt" \
  --auto --output_dir ./outputs/deepmodel
```

配合 `--config` 传入结构化 JSON：

```json
{
  "segments": {"高风险": "risk_score > 500", "低风险": "risk_score <= 500"},
  "pos_weight_per_segment": {"高风险": 2.0, "低风险": 1.0}
}
```

```bash
python scripts/sub_trainer.py \
  --data_path ./data.parquet --target y_label \
  --auto --config ./config.json --output_dir ./outputs/deepmodel
```

### 输出

- 每个分群的模型文件：`{output_dir}/{segment_name}.json`
- Markdown 报告：各分群 Train/Val/OOT AUC/KS/Gap 对比表、质量门控检查结果

---

## 阶段2：Stacking 融合

### 参数说明

> 参数 spec：`domain=deepmodel-stack`。`submodel_paths` / `segments` 等结构化 JSON **走 `--config`**。

| 参数 | 必选 | 默认值 | 说明 |
|------|:----:|--------|------|
| `--data_path` / `-d` | ✅ | - | 数据文件路径 |
| `--target` / `-t` | ✅ | - | 目标变量列名 |
| `--submodel_paths` | ✅ | - | 子模型路径 JSON 列表（**走 `--config`**） |
| `--segments` | ✅* | - | 与 sub_trainer 一致的分群条件 |
| `--segment_col` | ✅* | - | 与 sub_trainer 一致的分群列 |
| `--time_col` | | `busi_dt` | 时间列名 |
| `--train_filter` | | 自动切分 | 训练集条件（与 sub_trainer 一致） |
| `--val_filter` | | `val_ratio` 切出 | 验证集条件 |
| `--oot_filter` | | 按时间切出 | OOT 条件 |
| `--cv_folds` | | `5` | OOF 交叉验证折数 |
| `--output_dir` | | `./outputs/<ts>` | Meta-learner 保存目录 |
| `--config` | | - | JSON 配置路径 |

### 执行方式

```bash
python scripts/stacker.py \
  --data_path ./data.parquet --target y_label \
  --cv_folds 5 --output_dir ./outputs/deepmodel
```

配合 `--config`：

```json
{
  "submodel_paths": ["./outputs/deepmodel/高风险.json", "./outputs/deepmodel/低风险.json"],
  "segments": {"高风险": "risk_score > 500", "低风险": "risk_score <= 500"}
}
```

```bash
python scripts/stacker.py \
  --data_path ./data.parquet --target y_label \
  --config ./config.json --output_dir ./outputs/deepmodel
```

### 输出

- Meta-learner 模型文件：`{output_dir}/stack_meta.json`
- OOF 预测分分布图（各子模型 OOF AUC vs Meta-learner）

---

## 阶段3：集成对比报告

### 参数说明

> 参数 spec：`domain=deepmodel-compare`。

| 参数 | 必选 | 默认值 | 说明 |
|------|:----:|--------|------|
| `--data_path` / `-d` | ✅ | - | 数据文件路径 |
| `--target` / `-t` | ✅ | - | 目标变量列名 |
| `--stack_model_path` | | - | Stacking 模型路径（可选） |
| `--submodel_paths` | ✅ | - | 子模型路径 JSON 列表（**走 `--config`**） |
| `--segments` | ✅* | - | 分群条件 |
| `--segment_col` | ✅* | - | 分群列 |
| `--time_col` | | `busi_dt` | 时间列名 |
| `--train_filter` | | 自动切分 | 训练集条件 |
| `--val_filter` | | `val_ratio` 切出 | 验证集条件 |
| `--oot_filter` | | 按时间切出 | OOT 条件 |
| `--baseline_model_path` | | - | 单模型基线路径（不传则自动训练一个全量基线） |
| `--extra_baseline_algos` | | - | 额外单模型基线算法，逗号分隔（可选值：`lr,dnn`） |
| `--output_dir` | | `./outputs/<ts>` | 产物保存目录 |
| `--config` | | - | JSON 配置路径 |

### 执行方式

```bash
python scripts/comparator.py \
  --data_path ./data.parquet --target y_label \
  --stack_model_path ./outputs/deepmodel/stack_meta.json \
  --output_dir ./outputs/deepmodel
```

配合 `--config`：

```json
{
  "submodel_paths": ["./outputs/deepmodel/高风险.json", "./outputs/deepmodel/低风险.json"],
  "segments": {"高风险": "risk_score > 500", "低风险": "risk_score <= 500"}
}
```

```bash
python scripts/comparator.py \
  --data_path ./data.parquet --target y_label \
  --stack_model_path ./outputs/deepmodel/stack_meta.json \
  --config ./config.json --output_dir ./outputs/deepmodel
```

### 输出

- 对比表：**单模型 XGB 基线 / （可选 LR 基线 / DNN 基线）/ 最优子模型 / Stacking 集成**（Train/Val/OOT AUC/KS/Gap）
- 每个分群在各模型下的 OOT KS 对比
- 最终推荐：是否 Stacking / 最优子模型 / 基线

> 提示：如需**严格的多算法公平对比（含调参）**，请切换回 Agent 模式调用 `xgb-tuning → lr-tuning → dnn-tuning → model-comparison --use_tuned`。本技能的 `--extra_baseline_algos` 仅走默认超参，定位为"低成本初筛对照"。

---

## 完整工作流示例

```bash
# 阶段1：训练分群子模型
python scripts/sub_trainer.py \
  --data_path ./data.parquet --target y_label \
  --config ./config.json --auto --output_dir ./outputs/deepmodel

# 阶段2：Stacking 融合
python scripts/stacker.py \
  --data_path ./data.parquet --target y_label \
  --config ./config.json --output_dir ./outputs/deepmodel

# 阶段3：集成对比报告
python scripts/comparator.py \
  --data_path ./data.parquet --target y_label \
  --stack_model_path ./outputs/deepmodel/stack_meta.json \
  --config ./config.json --output_dir ./outputs/deepmodel
```

---

## 质量门控标准

| 检查项 | 标准 | 不达标处理 |
|--------|------|-----------|
| 各分群样本量 | ≥ 500 | 建议合并该分群 |
| 各分群 OOT Gap | < 0.05 | 启用 `--auto` |
| Stacking OOT AUC | ≥ max(子模型 OOT AUC) - 0.01 | 说明集成无明显增益 |
| 集成 vs 基线 OOT AUC | 集成更高 | 建议放弃集成，用最优子模型或基线 |

---

## 与其他技能的关系

| 技能 | 职责 | 关系 |
|------|------|------|
| `xgb-modeling` | 单模型训练 | 被复用训练子模型和基线 |
| `segment-modeling` | 分群策略探索 | 区别：segment-modeling 探索最优分群，本 Skill 按指定分群做 Stacking |
| `model-comparison` | 多算法公平对比 | 后续：严格对比请用 model-comparison（含调参） |

---

## 注意事项

1. **分群互斥**：各分群条件应互斥（一个样本只属于一个分群），否则 Stacking 会有信息泄漏
2. **样本量**：每个分群的训练样本应 ≥ 500，过少的分群建议合并
3. **OOF 生成**：Stacking 的 OOF 预测在各分群内独立做 k-fold，保证无标签泄漏
4. **Meta-learner**：固定使用保守参数（max_depth=2），不做超参搜索，避免过拟合
5. **子模型路径传递**：stacker 和 comparator 支持从 `--config` 读取子模型路径
6. **三阶段顺序**：必须先 sub_trainer → stacker → comparator，阶段间通过文件系统传递产物
