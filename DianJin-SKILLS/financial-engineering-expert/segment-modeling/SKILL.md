---
name: segment-modeling
category: 建模训练
description: 【分群自主探索】当用户需要"分群建模"、"客群拆分训练"、"探索最优分群方案"、"按xxx分组建模"、"不同客群分别训练"、"分群策略"时使用。支持规则分群（人类先验）、聚类分群（无监督发现）、决策树分群（有监督优化），自动探索最优分群策略并汇总子模型。前提条件：需要已完成基线建模（xgb-modeling）。不做全量样本统一建模，不做超参数调优。与 xgb-modeling 的区别：本 Skill 拆分客群分别建模，xgb-modeling 全量样本统一建模。
next_steps:
  - skill: model-explanation
    text: 解释各子模型
    prompt: 哪些特征最重要？解释模型决策逻辑
  - skill: model-comparison
    text: 多算法对比
    prompt: 与统一模型对比一下效果
---

# 分群自主探索 (portable)

在用户指定或 AI 自主发现的分群策略下，拆分客群训练子模型，自动探索最优分群方案。

**核心理念**：

```
探索空间（策略 × 参数 × 组合）
        │
        ▼
Try → Measure → Keep/Discard → Repeat
        │
        ▼
    最优分群方案 + 子模型
```

---

## 三种分群策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| **规则分群** | 用户指定规则（如 `age < 30`） | 有业务先验知识 |
| **聚类分群** | K-Means 等无监督自动发现 | 探索数据内在结构 |
| **决策树分群** | 有监督找最优分割点 | 直接优化目标变量 |

---

## 参数说明

| 参数 | 必选 | 默认值 | 说明 |
|------|:----:|--------|------|
| `--data_path` | ✅ | - | 数据文件路径（parquet/csv） |
| `--target` | ✅ | - | 目标变量列名（0/1 二分类） |
| `--mode` | | `auto` | 模式: `auto`(自主探索) / `manual`(指定策略) |
| `--max_rounds` | | `5` | 自主探索最大轮数 |
| `--segment_rules` | | - | 规则分群，JSON格式 |
| `--segment_col` | | - | 直接指定分群列名 |
| `--n_clusters` | | `3` | 聚类分群数 |
| `--tree_depth` | | `2` | 决策树分群深度 |
| `--tree_features` | | - | 决策树使用的特征，逗号分隔 |
| `--min_segment_ratio` | | `0.05` | 最小分群占比（<5%会警告） |
| `--merge_strategy` | | `route` | 汇总策略: `route` / `stacking` |
| `--metric` | | `ks` | 优化指标（auc/ks） |
| `--significance` | | `2.0` | 显著性阈值（MAD倍数） |
| `--time_col` | | `busi_dt` | 时间列名 |
| `--train_filter` | | - | 训练集筛选条件 |
| `--val_filter` | | - | 验证集筛选条件 |
| `--output_dir` | | `./outputs/<ts>` | 产物输出目录 |
| `--config` | | - | JSON 配置文件 |

---

## 执行方式

### 自主探索最优分群

```bash
python scripts/run_segment.py \
  --data_path ./data.parquet --target y_label \
  --mode auto --max_rounds 5 \
  --output_dir ./outputs/seg
```

### 用户规则分群

```bash
python scripts/run_segment.py \
  --data_path ./data.parquet --target y_label \
  --mode manual \
  --segment_rules '{"年轻": "age < 30", "中年": "age >= 30 and age < 50", "高龄": "age >= 50"}' \
  --output_dir ./outputs/seg
```

### 聚类自动分群

```bash
python scripts/run_segment.py \
  --data_path ./data.parquet --target y_label \
  --mode manual --n_clusters 4 \
  --output_dir ./outputs/seg
```

### 决策树分群

```bash
python scripts/run_segment.py \
  --data_path ./data.parquet --target y_label \
  --mode manual \
  --tree_depth 3 --tree_features "age,income,credit_score" \
  --output_dir ./outputs/seg
```

---

## 输出内容

### 实时进度（流式输出）

```
━━━ Round 1/5: 基线（不分群）━━━
单一模型 AUC: 0.742

━━━ Round 2/5: 用户规则分群 ━━━
规则: age < 30 | 30-50 | > 50
分群样本: [12,340 | 18,560 | 14,288]
子模型 AUC: [0.71 | 0.75 | 0.78]
汇总 AUC: 0.758 (+2.2%)
决策: ✅ KEEP

━━━ Round 3/5: 决策树分群 (depth=2) ━━━
自动分割: income < 5000 → ...
汇总 AUC: 0.772 (+1.8%)
决策: ✅ KEEP (新最佳!)
```

### 最终报告

- 最优分群方案详情
- 各策略对比表
- 子模型列表及指标
- vs 基线提升幅度
- 分群稳定性分析（PSI）

---

## 评估标准

| 指标 | 说明 | 阈值 |
|------|------|------|
| 整体 AUC | 分群模型汇总后效果 | 越高越好 |
| vs 基线 | 相比不分群的提升 | > 0 |
| 分群稳定性 | OOT分群比例变化 | PSI < 0.1 |
| 最小覆盖率 | 最小群占比 | > 5% |

---

## 与其他技能的关系

| 技能 | 职责 | 关系 |
|------|------|------|
| `xgb-modeling` | 单模型训练 | 被复用训练子模型 |
| `auto-experiment` | 特征探索 | 可在分群后对各群独立优化 |
| `feature-analysis` | 特征分析 | 辅助选择分群特征 |

---

## 注意事项

1. **分群数量**：建议 2-5 群，太多易过拟合
2. **最小样本量**：每群建议 > 5% 总样本
3. **稳定性**：关注 OOT 分群比例是否稳定
4. **业务可解释**：决策树分群规则更易解释
5. **组合使用**：可先分群再对各群做特征探索
