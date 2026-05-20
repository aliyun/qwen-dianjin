---
name: auto-experiment
category: 调参优化
description: 【自主实验循环·特征组级探索】当用户说"自动探索哪些特征组有用"、"批量对比不同特征组贡献"、"帮我自主试多种方案"、"自动跑一下"、"试各种特征组合"、"自动特征选择"、"消融分析"时使用。核心能力：自动发现数据中的特征分组，按组执行 独立评估→增量叠加→消融分析→精细筛选 四阶段渐进探索。不做超参数调优，不做单次固定方案训练。与 xgb-tuning 的区别：本 Skill 负责"自主探索特征和方案"，xgb-tuning 负责"调整模型超参数"。与 xgb-modeling 的区别：本 Skill 自主多轮探索，xgb-modeling 按指定方案单次训练。
next_steps:
  - skill: xgb-tuning
    text: 调参优化最优方案
    prompt: 帮我调参，优化模型效果
---

# 自主实验循环 (portable)

自动发现数据中的特征分组，执行四阶段渐进式探索实验，每轮完整展示探索逻辑和量化结果。

---

## 核心理念

借鉴 autoresearch 的自主循环思想，升级为**组级探索**：

```
发现特征组 → 组独立评估 → 组间叠加 → 组级消融 → 精细筛选
```

---

## 四阶段探索流程

### Phase 1: 特征组独立评估
- 自动发现数据中的特征分组（按前缀聚合: `firefly_*`, `mob3_*`, `umeng_*` 等）
- 逐组单独建模，评估每组的独立预测能力
- 输出各组的独立 AUC/KS 排名

### Phase 2: 组间增量叠加
- 从 Phase 1 表现最佳的组开始
- 逐步叠加下一个最强组，观察边际增量
- 找到最优组合点

### Phase 3: 组级消融分析
- 从全量特征出发，逐组移除
- 量化每个组的不可替代性
- 如果移除后指标不降，说明该组信息已被其他组覆盖

### Phase 4: 组内精细筛选
- 在最优组合内，逐个尝试移除特征
- 剔除冗余特征，精简模型

---

## 参数说明

| 参数 | 必选 | 默认值 | 说明 |
|------|:----:|--------|------|
| `--data_path` | ✅ | - | 数据文件路径（parquet/csv） |
| `--target` | ✅ | - | 目标变量列名（0/1 二分类） |
| `--exploration` | ✅ | - | 探索方向描述，如 "不同特征组对mob3逾期的贡献" |
| `--max_rounds` | | `5` | 最大实验轮数 |
| `--metric` | | `ks` | 优化指标（auc/ks） |
| `--direction` | | `maximize` | 优化方向（maximize/minimize） |
| `--significance` | | `2.0` | 显著性阈值（MAD倍数） |
| `--baseline_features` | | - | 基线特征列表，JSON格式（默认使用全量特征） |
| `--time_col` | | `busi_dt` | 时间列名 |
| `--train_filter` | | - | 训练集筛选条件 |
| `--val_filter` | | - | 验证集筛选条件 |
| `--test_filter` | | - | [Deprecated] 等价 `--val_filter`，仅作向后兼容 |
| `--oot_filter` | | - | OOT测试集条件 |
| `--val_ratio` | | `0.25` | val 在 train_full 内占比 |
| `--oot_ratio` | | `0.20` | OOT 在全量内占比 |
| `--output_dir` | | `./outputs/<ts>` | 产物输出目录 |
| `--config` | | - | JSON 配置文件 |

---

## 执行方式

```bash
python scripts/run_experiment.py \
  --data_path ./data.parquet --target y_label \
  --exploration "不同特征组(firefly/子模型/友盟)对mob3逾期的贡献" \
  --max_rounds 10 \
  --output_dir ./outputs/exp
```

---

## 常用场景

### 场景一：探索各特征组贡献

```bash
python scripts/run_experiment.py \
  --data_path ./data.parquet --target y_label \
  --exploration "不同特征组(firefly模型/子模型/友盟数据)对mob3逾期预测的贡献" \
  --max_rounds 10 --output_dir ./outputs/exp
```

### 场景二：探索特定时间窗口特征

```bash
python scripts/run_experiment.py \
  --data_path ./data.parquet --target y_label \
  --exploration "mob3相关的逾期、还款特征" \
  --max_rounds 8 --output_dir ./outputs/exp
```

### 场景三：在已有基线上探索新组

```bash
python scripts/run_experiment.py \
  --data_path ./data.parquet --target y_label \
  --baseline_features '["feat1","feat2","feat3"]' \
  --exploration "添加mob6时间窗口特征组" \
  --max_rounds 5 --output_dir ./outputs/exp
```

---

## 输出内容

> **核心原则：每轮必须完整输出探索逻辑和结果**
> 禁止只输出最终汇总。每一轮实验完成后，**必须**立即输出：探索策略、推理逻辑、假设描述、新增/移除特征、训练指标详情、与基线的对比、显著性检验、决策理由。

### 数据概览（实验开始前）

- 数据规模、特征数、正样本率
- 自动发现的特征分组及其统计

### 单轮输出格式（每轮必须使用）

```markdown
### Round N/总轮数

**Phase X: 探索策略名称**

**探索逻辑**:
[Phase 1/3/5] 特征组独立评估
目标: 单独测试「firefly」组的 12 个特征
逻辑: 先让每个特征组独立上场，获得各组的独立贡献排名

**本轮假设**: 独立评估特征组「firefly」(12个特征)

**模型评估** (共 12 个特征):
| 指标 | Train | Test | 基线Test | 变化 |
|------|-------|------|----------|------|
| AUC | 0.812 | 0.735 | 0.726 | +0.009 |

**Top-5 特征重要性**:
| # | 特征 | 重要性 |
|---|------|--------|
| 1 | firefly_score [NEW] | 0.2341 |

**显著性检验**: [PASS] 2.3x MAD → 改进显著
**决策**: [+] KEEP - 改进显著
```

### 最终汇总（所有轮次后输出）

- 实验总览表（Round/阶段/假设/指标/决策）
- 最终 vs 基线对比
- 保留和丢弃的特征组/特征列表
- 下一步建议

---

## 统计显著性检验

使用 MAD (Median Absolute Deviation) 判断改进是否显著：

| 置信度 | 标记 | 含义 |
|--------|------|------|
| ≥ 2.0× | [PASS] | 改进可能是真实的 |
| 1.0-2.0× | [EDGE] | 高于噪声但边缘 |
| < 1.0× | [FAIL] | 在噪声范围内 |

---

## 与其他技能的关系

| 技能 | 职责 | 关系 |
|------|------|------|
| `xgb-modeling` | 单次训练和评估 | 被本技能复用 |
| `xgb-tuning` | 超参调优 | 可在循环后使用 |
| `feature-analysis` | 特征分析 | 辅助假设生成 |

---

## 注意事项

1. **探索方向**：可以是宽泛的（如"探索各特征组贡献"），系统会自动发现分组
2. **轮数建议**：设置为特征组数的2-3倍（如6个组建议10-15轮），以覆盖多个探索阶段
3. **基线策略**：默认使用全量特征作为基线，也可指定特定特征集
4. **显著性阈值**：默认2.0×，数据量小时可适当降低
5. **可打断**：用户可随时停止，已有结果会保留
6. **产物位置**：报告和模型保存到 `<output_dir>/`
