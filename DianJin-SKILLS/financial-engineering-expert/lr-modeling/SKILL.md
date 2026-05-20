---
name: lr-modeling
category: 建模训练
description: 【LR评分卡建模】当用户需要"逻辑回归"、"LR建模"、"评分卡"、"scorecard"、"WoE建模"、"跑个LR"时使用。基于 WoE 编码 + Logistic Regression 训练标准评分卡模型，输出系数表、分箱WoE映射、评分卡分数转换。复用平台已有的三段式数据切分、AUC/KS/BCR评估体系、稳定性分析。前提条件：需要已有数据集。**不做超参数调优**；不做特征工程。与 xgb-modeling 的区别：本 Skill 使用逻辑回归+WoE编码，输出可解释的评分卡模型；xgb-modeling 使用 XGBoost 树模型。
next_steps:
  - skill: lr-tuning
    text: 调参优化评分卡
    prompt: 帮我调整WoE分箱和正则化参数
  - skill: model-comparison
    text: 多算法对比
    prompt: 与XGBoost和DNN对比一下效果
---

# LR 评分卡建模 (portable)

基于 WoE (Weight of Evidence) 编码 + Logistic Regression 进行二分类评分卡建模。

**核心流程**：原始特征 → WoE 最优分箱编码 → LR 训练 → 评分卡转换

**适用场景**：
- 风控评分卡开发（标准 A/B/C 卡）
- 需要强可解释性的业务场景
- 监管合规要求模型白盒化

---

## 参数说明

| 参数 | 必选 | 默认值 | 说明 |
|------|:----:|--------|------|
| `--data_path` / `-d` | ✅ | - | 数据文件路径（parquet/csv） |
| `--target` / `-t` | ✅ | - | 目标变量列名（0/1 二分类） |
| `--time_col` | | `busi_dt` | 时间列名 |
| `--train_filter` | | 自动切分 | 训练集筛选条件（pandas query） |
| `--oot_filter` | | 按时间切出 | OOT 跨时间测试集条件 |
| `--oot_ratio` | | `0.20` | 未传 `--oot_filter` 时按时间切 OOT 的比例 |
| `--val_ratio` | | `0.25` | 从 train_full 切 val 的比例 |
| `--random_seed` | | `42` | 随机种子 |
| `--exclude_cols` | | - | 排除列，逗号分隔 |
| `--features` | | - | 指定特征列表，逗号分隔；不传则自动按 IV 筛选 |
| `--max_n_bins` | | `8` | WoE 分箱最大箱数 |
| `--min_bin_size` | | `0.05` | 最小分箱比例 |
| `--iv_threshold` | | `0.02` | IV 筛选阈值（低于此值的特征排除） |
| `--regularization` | | `l2` | 正则化类型：`l1` / `l2` / `elasticnet` |
| `--C` | | `1.0` | 正则化强度（越小正则化越强） |
| `--max_iter` | | `1000` | LR 最大迭代次数 |
| `--base_score` | | `600` | 评分卡基础分 |
| `--pdo` | | `50` | 评分卡 PDO（分数翻倍点） |
| `--base_odds` | | `50.0` | 基础 Odds（好坏比） |
| `--model_name` | | 自动生成 | 模型名称 |
| `--report_output` | | 自动生成 | 报告输出路径 |
| `--output_dir` | | `./outputs/<ts>` | 产物输出目录 |
| `--config` | | - | JSON 配置文件路径 |

---

## 执行方式

```bash
python scripts/modeling.py \
  --data_path ./data.parquet --target y_label \
  --time_col busi_dt \
  --exclude_cols "cust_code,busi_dt" \
  --output_dir ./outputs/lr_run
```

指定特征建模：

```bash
python scripts/modeling.py \
  --data_path ./data.parquet --target y_label \
  --features "feat1,feat2,feat3,feat4" \
  --regularization l1 --C 0.5 \
  --output_dir ./outputs/lr_run
```

---

## 常用场景

### 场景一：默认参数快速建模

```bash
python scripts/modeling.py --data_path ./data.parquet --target y_label --output_dir ./outputs/lr_run
```

### 场景二：指定特征建模

```bash
python scripts/modeling.py --data_path ./data.parquet --target y_label \
  --features "feat1,feat2,feat3" --output_dir ./outputs/lr_run
```

### 场景三：自定义评分卡参数

```bash
python scripts/modeling.py --data_path ./data.parquet --target y_label \
  --base_score 650 --pdo 40 --output_dir ./outputs/lr_run
```

---

## 输出产物

1. **建模报告**（Markdown）— 含数据切分、WoE分箱表、LR系数表、评分卡转换表、三段式评估指标、稳定性分析
2. **模型文件**（joblib）— LR 模型 + WoE 编码器序列化
3. **评分卡表**（JSON）— 可直接用于部署的分箱-分数映射
4. **result.json** — 结构化产物清单

---

## 与其他建模 Skill 的对比

| 维度 | lr-modeling | xgb-modeling | dnn-modeling |
|------|-------------|--------------|-------------|
| 算法 | Logistic Regression | XGBoost | MLP (PyTorch) |
| 特征编码 | WoE 分箱编码 | 原始特征直接输入 | StandardScaler |
| 可解释性 | 白盒（系数 × WoE = 贡献） | 黑盒（需 SHAP 解释） | 弱 |
| 非线性能力 | 弱（仅通过分箱引入） | 强（树结构天然支持） | 强（多层激活） |
| 适用场景 | 评分卡 / 合规 / 白盒 | 高精度 / 特征交互 | 高维复杂交互 |
| 评估体系 | AUC/KS/BCR/PSI | AUC/KS/BCR/PSI | AUC/KS/BCR/PSI |

---

## 上下游关系

- **前置**：`data-profiling` → `feature-analysis` / `univariate-analysis`（特征筛选）
- **后续**：`lr-tuning`（调参优化）、`model-comparison`（多算法对比）
- **平行**：与 `xgb-modeling` / `dnn-modeling` 可做横向对比（同数据不同算法）

---

## 注意事项

1. **目标变量**：必须为 0/1 二分类
2. **时间切分**：建议按时间切分，确保 OOT 为未来数据
3. **特征筛选**：不传 `--features` 时自动按 IV 筛选（阈值 `--iv_threshold`）
4. **WoE 分箱**：使用 `optbinning` 做最优分箱，每特征最多 `--max_n_bins` 箱
5. **评分卡公式**：`Score = base_score - factor × ln(odds)`，其中 `factor = pdo / ln(2)`
6. **不提供调参**：需要调参请切 `lr-tuning`（搜索 WoE 分箱 + LR 正则化参数）
7. **模型保存**：模型和评分卡保存到 `<output_dir>/models/`
