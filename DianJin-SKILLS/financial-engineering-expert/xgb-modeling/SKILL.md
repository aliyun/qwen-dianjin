---
name: xgb-modeling
category: 建模训练
description: 【XGBoost建模】当用户需要"训练模型"、"跑一个基线"、"建个模型试试"、"对比特征方案效果"、"预测一下"、"跑个xgb"、"看AUC/KS/BCR"时使用。支持多特征方案对比、AUC/KS/BCR/Brier评估、Bootstrap置信区间、概率校准曲线、稳定性分析（Mann-Kendall趋势检验）。前提条件：需要已完成特征分析并确定特征方案。**不做超参数调优，调参请用 xgb-tuning**；不做特征工程。与 auto-experiment 的区别：本 Skill 按用户指定方案训练，auto-experiment 自主探索特征组合。
next_steps:
  - skill: xgb-tuning
    text: 调参优化模型效果
    prompt: 帮我调参，优化模型效果
  - skill: model-explanation
    text: SHAP解释模型决策
    prompt: 哪些特征最重要？解释模型决策逻辑
---

# XGBoost 建模 (portable)

基于 XGBoost 进行二分类建模，支持自动特征筛选、多方案对比、稳定性分析。可运行在任何 Python 环境，无平台耦合。调参请用 [`xgb-tuning`](./xgb-tuning/SKILL.md)。

**核心流程**：数据加载 → 特征筛选（四大算子）→ 多方案对比 → 最优方案训练 → 三段式评估 → 稳定性分析 → 报告生成

## 参数

> 通用参数 spec 定义在 `_vendor/xgb_cli.py`（domain=`modeling`）。

| 参数 | 必选 | 默认值 | 说明 |
|------|:----:|--------|------|
| `--data_path` / `-d` | ✅ | - | 数据文件路径（parquet/csv） |
| `--target` / `-t` | ✅ | - | 目标变量列名（0/1 二分类） |
| `--time_col` | | `busi_dt` | 时间列名 |
| `--train_filter` | | 自动切分 | 训练集筛选条件（pandas query） |
| `--val_filter` | | `val_ratio` 切出 | 验证集筛选条件 |
| `--oot_filter` | | 按时间切出 | OOT 跨时间测试集条件 |
| `--oot_ratio` | | `0.20` | 未传 `--oot_filter` 时按时间切 OOT 的比例 |
| `--val_ratio` | | `0.25` | 未传 `--val_filter` 时从 train_full 切 val 的比例 |
| `--random_seed` | | `42` | 随机种子 |
| `--exclude_cols` | | - | 排除列，逗号分隔 |
| `--features` | | - | 指定单套特征，逗号分隔 |
| `--feature_sets` | | - | JSON 多方案；**复杂嵌套优先放到 `--config` 的 `feature_sets` 字段** |
| `--feature_scheme` | | - | 指定单套自动筛选方案：`full`/`decorr`/`high_iv`/`stable`；不传或传空则走 `--auto_select` 全量对比 |
| `--auto_select` | | `true` | 自动特征筛选（四大算子全量对比；指定 `--feature_scheme` 时忽略本参数） |
| `--baseline_filter` | | 同 train | PSI 基准条件 |
| `--comparison_filter` | | 同 oot | PSI 对比条件 |
| `--sample_strategy` | | `auto_weight` | `auto_weight` / `undersample` / `none` |
| `--model_name` | | 自动生成 | 模型名称（不含扩展名） |
| `--report_output` | | `xgb_modeling_report` | 报告基名 |
| `--output_dir` | | `./outputs/<ts>` | **portable 独有**：产物输出目录 |
| `--config` | | - | JSON 配置文件路径（命令行 > config > 默认） |

> **复杂嵌套参数（如 `feature_sets`）不要用命令行拼转义 JSON**，改走 `--config config.json` 结构化通道。

---

## 执行方式

```bash
python scripts/modeling.py \
  --data_path ./examples/toy.parquet \
  --target y_label \
  --time_col busi_dt \
  --output_dir ./outputs/run1
```

复杂 `feature_sets` 通过 `--config` 传入：

```bash
python scripts/modeling.py \
  --data_path ./examples/toy.parquet \
  --target y_label \
  --config ./config.json \
  --output_dir ./outputs/run1
```

执行结束后 `<output_dir>/` 下生成：

- `xgb_modeling_report.md` — 建模报告（数据概览、方案对比、最优方案、稳定性分析…）
- `models/<model_name>.json` — XGBoost 模型文件
- `models/<model_name>_meta.json` — 模型元数据（特征列表等）
- `models/<model_name>_card.json` — ModelCard
- `result.json` — 结构化产物清单（见 PROTOCOL.md）

stdout 末行打印 `result.json` 绝对路径，Agent 读这个文件即可获取全部产物路径与 metrics。

## 产物示例（result.json）

```json
{
  "skill": "xgb-modeling",
  "status": "success",
  "files": [
    {"path": ".../xgb_modeling_report.md", "role": "report"},
    {"path": ".../models/xgb_model_20260512_143000.json", "role": "model",
     "meta": {"feature_count": 8, "scheme_name": "去共线性", "model_type": "xgboost"}},
    {"path": ".../models/xgb_model_20260512_143000_meta.json", "role": "meta"}
  ],
  "metrics": {
    "oot_auc": 0.7823, "oot_ks": 0.4215,
    "auc_gap": 0.0214, "ks_gap": 0.0312,
    "n_features": 8, "overall_score_psi": 0.0423
  },
  "summary": "xgb-modeling 完成：最优方案「去共线性」，OOT AUC=0.7823，OOT KS=0.4215，特征数=8，Gap=0.0214"
}
```

## 跨 skill 串联

下游（如 `model-explanation`）通过 `--model_path` 传入上游 result.json 中 `role=model` 的文件路径：

```bash
# 从上游 result.json 提取模型路径
MODEL_PATH=$(jq -r '.files[] | select(.role=="model") | .path' ./outputs/run1/result.json)

python ../model-explanation/scripts/explain.py --model_path "$MODEL_PATH" --data_path ./examples/toy.parquet
```

---

## 常用场景

### 场景一：自动特征筛选建模

不指定特征，自动执行四大算子生成四套方案对比：

```bash
python scripts/modeling.py --data_path ./examples/toy.parquet --target y_label
```

### 场景二：指定特征建模

使用指定的特征列表：

```bash
python scripts/modeling.py --data_path ./examples/toy.parquet --target y_label --features "feat1,feat2,feat3"
```

### 场景二-b：指定单套自动筛选方案

只跑一种筛选方案（如"去共线性"）：

```bash
python scripts/modeling.py --data_path ./examples/toy.parquet --target y_label --feature_scheme decorr
```

可选值：`full`(全量入模) / `decorr`(去共线性) / `high_iv`(高预测力) / `stable`(稳定性优先)

### 场景三：多方案对比

传入多套特征方案（**优先用 `--config`**，避免命令行拼 JSON 转义）：

```json
{
  "feature_sets": {
    "方案A": ["f1", "f2"],
    "方案B": ["f1", "f3"]
  }
}
```

```bash
python scripts/modeling.py --data_path ./examples/toy.parquet --target y_label --config ./config.json
```

### 场景四：保存模型

训练完成后保存最优模型，供后续 model-explanation 使用：

```bash
python scripts/modeling.py --data_path ./examples/toy.parquet --target y_label --model_name my_best_model
```

模型自动保存，输出：
- 模型文件：`<output_dir>/models/my_best_model.json`
- 元数据文件：`<output_dir>/models/my_best_model_meta.json`（含特征列表、参数等）

---

## 报告结构

| 章节 | 内容 |
|------|------|
| 执行摘要 | 4 行业务结论：最优方案、Gap 状态、PSI 状态、头部特征风险 |
| 1. 数据概览 | 样本量、正样本率（Train/Test/OOT） |
| 2. 特征方案对比 | 各方案 AUC/KS/Gap 横向对比 |
| 3. 最优方案详情 | AUC/KS/Gini 指标、模型分数 IV 最优分箱、Lift 表、特征重要性、BCR @ Top5/10/20/30%、Brier Score + 校准曲线 |
| 4. 稳定性分析 | 按月 AUC/KS（含 95% Bootstrap 置信区间）、Mann-Kendall 趋势检验、分数 PSI |
| 5. 最终模型总结 | 特征列表、综合表现 |

---

## 与其他建模 Skill 的对比

| 维度 | xgb-modeling | lr-modeling | dnn-modeling |
|------|-------------|-------------|--------------|
| 算法 | XGBoost | Logistic Regression + WoE | MLP (PyTorch) |
| 特征处理 | 原始值直接输入 | WoE 分箱编码 | StandardScaler |
| 非线性能力 | 强（树结构） | 弱（仅通过分箱引入） | 强（多层激活） |
| 可解释性 | 中（需 SHAP） | 强（系数 × WoE） | 弱 |
| 训练速度 | 快 | 很快 | 慢 |
| 适用数据量 | 任意 | 任意 | >10k |
| 评估体系 | AUC/KS/BCR/PSI | AUC/KS/BCR/PSI | AUC/KS/BCR/PSI |

---

## 上下游关系

- **前置**：`data-profiling` → `feature-analysis`（特征筛选）
- **后续**：`xgb-tuning`（调参优化）、`model-explanation`（SHAP 解释）
- **平行**：与 `lr-modeling` / `dnn-modeling` 可做横向对比（同数据不同算法）

---

## 依赖

见 `requirements.txt`（核心：`xgboost`, `scikit-learn`, `optbinning`）。

---

## 注意事项

1. **目标变量**：必须为 0/1 二分类
2. **时间切分**：建议按时间切分，确保 OOT 为未来数据
3. **自动特征筛选**：需要数据同时满足 IV、PSI、缺失率条件
4. **样本策略**：强不均衡场景（正样本率 < 2%）建议 `--sample_strategy undersample`，一般场景用默认 `auto_weight`
5. **不提供调参**：需要调参请切 xgb-tuning（基于 Optuna TPE + 诊断驱动的约束搜索）
6. **模型保存**：每次训练自动保存最优模型至 `<output_dir>/models/`，无需传 `--save_model`；可用 `--model_name` 自定义文件名
7. **稳定性分析**：基准月份取自训练集时间段；Bootstrap CI 需样本量 >= 100 才计算
8. **BCR/校准曲线**：BCR（Bad Capture Rate）反映拒绝 Top K% 人群能捕获多少坏客户；校准曲线反映模型概率输出可信度
