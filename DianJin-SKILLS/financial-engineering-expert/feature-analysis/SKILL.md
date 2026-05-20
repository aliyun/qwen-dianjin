---
name: feature-analysis
category: 数据探索
description: 【特征深度分析】当用户需要"全面分析特征"、"生成特征报告"、"看相关性"、"推荐建模方案"、"跑一遍特征"、"特征全景"、"哪些特征有用"时使用。包含四大维度：基础统计、IV值、PSI稳定性、相关性分析，输出四套建模特征方案。前提条件：必须提供目标变量。不做单特征快速探索，不做模型训练。与 univariate-analysis 的区别：本 Skill 是全量特征深度分析+方案推荐，univariate-analysis 聚焦少量特征快速分析。
next_steps:
  - skill: xgb-modeling
    text: 用推荐方案跑基线模型
    prompt: 跑一个基线模型，看看AUC和KS
---

# 建模特征分析报告 (portable)

基于 `scripts/analyzer.py` 主脚本，对数据集特征进行全面分析并生成 Markdown 报告。

---

## 参数说明

| 参数 | 必选 | 默认值 | 说明 |
|------|:----:|--------|------|
| `--data_path` | ✅ | - | 数据文件路径（parquet/csv） |
| `--target` | ✅ | - | 目标变量列名（0/1 二分类） |
| `--exclude_cols` | | - | 排除列，逗号分隔 |
| `--baseline_filter` | | - | PSI 基准数据条件（pandas query） |
| `--comparison_filter` | | - | PSI 对比数据条件（pandas query） |
| `--top_n` | | `10` | 输出 IV 排名前 N 的特征分箱明细（默认 10） |
| `--specified_features` | | - | 指定特征分箱明细，逗号分隔 |
| `--output` | | `feature_analysis_report.md` | 报告输出名称 |
| `--output_dir` | | `./outputs/<ts>` | 产物输出目录 |
| `--config` | | - | JSON 配置文件路径 |

---

## 执行方式

```bash
python scripts/analyzer.py \
  --data_path ./data.parquet --target y_label \
  --exclude_cols "cust_code,busi_dt" \
  --baseline_filter "busi_dt <= '20250501'" \
  --comparison_filter "busi_dt > '20250701'" \
  --output_dir ./outputs/fa_run
```

---

## 常用场景

### 场景一：基础分析（不含 PSI）

适用于无时间维度的数据集：

```bash
python scripts/analyzer.py --data_path ./data.parquet --target y_label --output_dir ./outputs/fa_run
```

### 场景二：完整分析（含 PSI）

适用于有时间切分条件的数据集：

```bash
python scripts/analyzer.py --data_path ./data.parquet --target y_label \
  --baseline_filter "busi_dt <= '20250501'" \
  --comparison_filter "busi_dt > '20250701'" \
  --output_dir ./outputs/fa_run
```

### 场景三：单变量深度分析

输出 IV Top N 或指定特征的完整分箱明细表：

```bash
python scripts/analyzer.py --data_path ./data.parquet --target y_label --top_n 10 --output_dir ./outputs/fa_run
python scripts/analyzer.py --data_path ./data.parquet --target y_label --specified_features "umeng_ALL,bscore" --output_dir ./outputs/fa_run
```

---

## 报告输出结构

生成的 `feature_analysis_report.md` 包含以下章节：

| 章节 | 内容 |
|------|------|
| 1. 数据概览 | 样本量、特征数、正样本率 |
| 2. 基础统计分析 | 各特征的均值、标准差、缺失率、偏度、峰度等 |
| 3. IV值分析 | IV 排名 Top 20、IV 分布统计 |
| 4. PSI稳定性分析 | PSI 排名、不稳定特征清单 |
| 5. 相关性分析 | 高相关特征对、共线性处理建议 |
| 6. 综合建议 | 推荐保留/移除特征 |
| 7. 建模特征方案 | 四套特征筛选方案（全量/去共线性/高IV/稳定性优先） |
| 8. 单变量深度分析 | 默认输出 Top 10 特征分箱明细，可通过 `--top_n` 调整 |

---

## 特征筛选方案说明

报告自动生成四套建模特征方案：

| 方案 | 筛选条件 | 适用场景 |
|------|----------|----------|
| 方案一：全量入模 | 基础合格池（IV>=0.02, PSI<0.25, 缺失率<50%） | XGBoost/LightGBM 等树模型 |
| 方案二：去共线性标准 | 基础池 + 贪心去 \|r\|>=0.7 | 逻辑回归、评分卡 |
| 方案三：高预测力精选 | IV>=0.1 + 去共线性 | 特征受限、可解释性要求高 |
| 方案四：稳定性优先 | PSI<0.1 + 去共线性 | 线上部署、高稳定性要求 |

---

## 与其他 Skill 的关系

| Skill | 用途区别 |
|-------|----------|
| **feature-analysis** | 全量特征深度分析+方案推荐，需要目标变量 |
| data-profiling | 快速了解数据轮廓，不做深度分析 |
| univariate-analysis | 少量特征快速分析，不含全量相关性和方案推荐 |

**建议流程**：`data-profiling`（快速扫描）→ `feature-analysis`（深度分析+方案推荐）→ `xgb-modeling`（建模）

---

## 注意事项

1. **目标变量**：必须为数值型
2. **PSI 分析**：需同时提供 `baseline_filter` 和 `comparison_filter`，否则跳过
3. **相关性**：仅对数值型特征有效，非数值列自动跳过
4. **大数据集**：报告默认展示 IV Top 20，完整数据在文件中
5. **产物位置**：报告保存到 `<output_dir>/`
