---
name: model-explanation
category: 评估解释
description: 【模型解释】当用户问"为什么模型这样预测"、"哪些特征最重要"、"解释一下这个样本"、"模型怎么做的决策"、"feature importance"、"为什么给这个分数"时使用。基于SHAP的模型解释，支持全局特征重要性、单样本预测解释、特征交互分析。前提条件：需要已训练好的XGBoost模型。不做模型训练，不做特征筛选。与 feature-analysis 的区别：本 Skill 解释"已训练模型的决策逻辑"，feature-analysis 分析"建模前的特征质量"。
next_steps:
  - skill: model-comparison
    text: 多算法对比
    prompt: 与LR和DNN对比一下效果
---

# 模型解释报告 (portable)

基于 SHAP (SHapley Additive exPlanations) 对 XGBoost 模型进行可解释性分析，生成包含可视化图表的 Markdown 报告。

---

## 参数说明

| 参数 | 必选 | 默认值 | 说明 |
|------|:----:|--------|------|
| `--model_path` | ✅ | - | XGBoost 模型文件路径 (.json) |
| `--data_path` | ✅ | - | 数据文件路径（parquet/csv） |
| `--target` | ✅ | - | 目标变量列名 |
| `--features` | | *auto* | 特征列表，逗号分隔。不传时从 `<model>_meta.json` 自动读取 |
| `--sample_id` | | - | 单样本解释：样本索引或ID |
| `--sample_filter` | | - | 单样本解释：pandas query 条件 |
| `--top_n` | | `20` | 全局解释显示 Top N 特征 |
| `--interaction_features` | | - | 交互分析特征对，如 `f1,f2` |
| `--output_dir` | | `./outputs/<ts>` | 产物输出目录 |
| `--output_name` | | `model_explanation_report` | 报告基名 |
| `--config` | | - | JSON 配置文件路径 |

---

## 执行方式

```bash
python scripts/explainer.py \
  --model_path ./models/my_model.json \
  --data_path ./examples/toy.parquet \
  --target y_label \
  --output_dir ./outputs/explain_run
```

单样本解释：

```bash
python scripts/explainer.py \
  --model_path ./models/my_model.json \
  --data_path ./examples/toy.parquet \
  --target y_label \
  --sample_id 1001 \
  --output_dir ./outputs/explain_run
```

特征交互分析：

```bash
python scripts/explainer.py \
  --model_path ./models/my_model.json \
  --data_path ./examples/toy.parquet \
  --target y_label \
  --interaction_features "age,income" \
  --output_dir ./outputs/explain_run
```

---

## 常用场景

### 场景一：全局特征重要性分析

解释模型整体决策逻辑：

```bash
python scripts/explainer.py --model_path ./models/my_model.json \
  --data_path ./examples/toy.parquet --target y_label \
  --output_dir ./outputs/explain_run
```

输出：
- SHAP Summary Plot（特征重要性排序）
- SHAP Bar Plot（平均绝对 SHAP 值）
- 特征重要性表格

### 场景二：单样本预测解释

解释特定样本的预测原因：

```bash
python scripts/explainer.py --model_path ./models/my_model.json \
  --data_path ./examples/toy.parquet --target y_label \
  --sample_id 1001 --output_dir ./outputs/explain_run
```

输出：
- SHAP Force Plot（推动预测的正负特征）
- 瀑布图（特征贡献分解）
- 该样本特征值与分布对比

### 场景三：特征交互分析

分析两个特征的交互效应：

```bash
python scripts/explainer.py --model_path ./models/my_model.json \
  --data_path ./examples/toy.parquet --target y_label \
  --interaction_features "age,income" --output_dir ./outputs/explain_run
```

输出：
- SHAP Dependence Plot（特征值 vs SHAP 值）
- 交互热力图

---

## 跨 skill 串联（推荐流程）

`xgb-modeling` 产物的 `result.json` 中 `role=model` 的 path 可直接喂给 `model-explanation`：

```bash
# 1) 训练
python ../xgb-modeling/scripts/modeling.py \
  --data_path ./examples/toy.parquet --target y_label \
  --exclude_cols "cust_code" --output_dir ./outputs/mdl

# 2) 从 result.json 提模型路径
MODEL=$(jq -r '.files[] | select(.role=="model") | .path' ./outputs/mdl/result.json)

# 3) 解释（特征自动从 <model>_meta.json 读取）
python scripts/explainer.py \
  --model_path "$MODEL" \
  --data_path ./examples/toy.parquet --target y_label \
  --output_dir ./outputs/xpl
```

---

## 特征列表来源

1. `--features "f1,f2,..."` 显式传入
2. `config.json` 中 `"features": [...]`
3. **自动发现** `<model>_meta.json`（xgb-modeling 产物自带）

---

## 报告输出结构

生成的 `model_explanation_report.md` 包含以下章节：

| 章节 | 内容 |
|------|------|
| 1. 模型概览 | 模型路径、特征数量、样本规模 |
| 2. 全局特征解释 | SHAP Summary Plot、Top N 特征重要性表 |
| 3. 单样本解释 | Force Plot、瀑布图、特征值对比（如指定样本） |
| 4. 特征交互分析 | Dependence Plot、交互效应说明（如指定） |
| 5. 可视化附件 | 生成的 PNG 图表文件列表 |

---

## 产物目录

```
<output_dir>/
├── model_explanation_report.md
└── assets/
    ├── shap_summary.png
    ├── shap_bar.png
    ├── force_plot_sample_1001.png
    └── dependence_age_income.png
```

---

## 与其他技能的关系

| 技能 | 职责 | 关系 |
|------|------|------|
| `xgb-modeling` | 训练模型 | 前置：需先用其训练并保存模型 |
| `model-comparison` | 多算法对比 | 后续：可与 LR/DNN 模型做对比解释 |
| `feature-analysis` | 特征分析 | 区别：feature-analysis 分析建模前特征质量 |

---

## 依赖

- `shap`、`matplotlib`（图表）
- `xgboost`、`pandas`、`numpy`

---

## 注意事项

1. **模型格式**：仅支持 XGBoost JSON 格式模型文件
2. **数据一致性**：`--data_path` 需与训练时使用的数据字段一致
3. **样本定位**：`--sample_id` 为数据框的整数索引，非业务ID
4. **内存占用**：大数据集的全局 SHAP 计算可能耗时较长
5. **图表依赖**：需要 matplotlib 和 shap 库支持
6. **base_score 兼容**：XGBoost >= 1.7 的 `base_score` 为数组，脚本会自动修复为标量以兼容 SHAP
