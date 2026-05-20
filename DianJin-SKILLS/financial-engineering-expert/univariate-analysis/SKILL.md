---
name: univariate-analysis
category: 数据探索
description: 【单变量分析】当用户提到"分析某个/某些特征"、"看看xxx特征"、"xxx特征分布"、"帮我分箱"、"交叉分布"、"筛选特征"、"分析一下xxx"等涉及少量特征分析的请求时使用。支持两种模式：(1) 无目标变量：等频/等距分箱分布、交叉分布表；(2) 有目标变量：IV值和分箱明细。适用场景关键词：分析特征、看特征、特征分布、分箱、交叉分析、IV值、单个特征、几个特征。与 feature-analysis 的区别：聚焦少量特征的快速分析，不含全量相关性和方案推荐。
next_steps:
  - skill: feature-analysis
    text: 全量特征分析，看全局视角
    prompt: 全面分析所有特征，推荐建模方案
---

# 单变量分析 (portable)

基于 `scripts/analyzer.py` 主脚本，对指定特征进行单变量级别的分布分析或预测力评估。

---

## 功能定位

| 模式 | 目标变量 | 核心功能 | 使用场景 |
|------|:--------:|----------|----------|
| 数据探索模式 | ❌ 无 | 分布分析、交叉分布 | 刚上传数据、初步了解特征分布 |
| 特征筛选模式 | ✅ 有 | IV值、分箱表、筛选建议 | 建模前快速筛选特征 |

---

## 参数说明

| 参数 | 必选 | 默认值 | 说明 |
|------|:----:|--------|------|
| `--data_path` | ✅ | - | 数据文件路径（parquet/csv） |
| `--features` | ✅ | - | 待分析特征列名，逗号分隔，或 `"all"` |
| `--target` | | - | 目标变量列名（可选，有则进入筛选模式） |
| `--exclude_cols` | | - | 排除的列，逗号分隔 |
| `--binning_method` | | `quantile` | 分箱方式：`quantile`(等频) / `distance`(等距) |
| `--n_bins` | | `10` | 分箱数量 |
| `--cross` | | - | 交叉分布的两个特征，逗号分隔 |
| `--output` | | 自动生成 | 报告输出名称 |
| `--output_dir` | | `./outputs/<ts>` | 产物输出目录 |
| `--config` | | - | JSON 配置文件路径 |

---

## 执行方式

### 数据探索模式（无目标变量）

```bash
python scripts/analyzer.py \
  --data_path ./data.parquet --features "age,income,score" \
  --output_dir ./outputs/uni_run
```

### 特征筛选模式（有目标变量）

```bash
python scripts/analyzer.py \
  --data_path ./data.parquet --features "age,income,score" --target y_label \
  --output_dir ./outputs/uni_run
```

### 交叉分布分析

```bash
python scripts/analyzer.py \
  --data_path ./data.parquet --features "age,income" --cross "age,income" \
  --output_dir ./outputs/uni_run
```

---

## 常用场景

### 场景一：数据探索（无目标变量）

查看单特征分布：

```bash
python scripts/analyzer.py --data_path ./data.parquet --features "age,income,score" --output_dir ./outputs/uni_run
```

指定等距分箱：

```bash
python scripts/analyzer.py --data_path ./data.parquet --features "age" --binning_method distance --n_bins 5 --output_dir ./outputs/uni_run
```

### 场景二：交叉分布分析

查看两个特征的联合分布：

```bash
python scripts/analyzer.py --data_path ./data.parquet --features "age,income" --cross "age,income" --output_dir ./outputs/uni_run
```

### 场景三：特征筛选（有目标变量）

计算 IV 值，评估特征预测力：

```bash
python scripts/analyzer.py --data_path ./data.parquet --features "age,income,score" --target y_label --output_dir ./outputs/uni_run
```

---

## 报告输出结构

### 数据探索模式（无目标变量）

| 章节 | 内容 |
|------|------|
| 1. 数据概览 | 样本量、分析特征数 |
| 2. 基础统计 | 各特征的均值、中位数、缺失率、基数等 |
| 3. 分布分析 | 各特征的分箱分布表（区间、样本数、占比）+ 柱状分布图 |
| 4. 交叉分布 | 两特征联合分布表（可选） |
| 5. 数据质量标记 | 缺失率过高、低基数等问题标记 |

### 特征筛选模式（有目标变量）

| 章节 | 内容 |
|------|------|
| 1. 数据概览 | 样本量、正样本率、分析特征数 |
| 2. 基础统计 | 各特征的均值、中位数、缺失率、基数等 |
| 3. IV值分析 | 各特征 IV 值排名及分箱明细表 |
| 4. 数据质量标记 | 缺失率过高、低基数等问题标记 |
| 5. 筛选建议 | 建议保留/剔除的特征清单 |

---

## 分箱说明

### 等频分箱（quantile）

按数据分位数划分，每箱样本量大致相等。适合分布不均匀的数据。

### 等距分箱（distance）

按数值区间等间隔划分。适合分布均匀的数据。

### 特殊值处理

- **缺失值**：单独一箱，标记为 `Missing`
- **零值**：当零值占比 > 5% 时单独一箱
- **负数**：当存在负数时单独处理

---

## 输出示例

### 分布分析表

```
特征: age | 分箱方式: 等频 | 分箱数: 10

| 区间          | 样本数  | 占比    | 累计占比 |
|---------------|---------|---------|----------|
| Missing       | 156     | 1.56%   | 1.56%    |
| [18, 23)      | 984     | 9.84%   | 11.40%   |
| [23, 28)      | 1,012   | 10.12%  | 21.52%   |
| ...           | ...     | ...     | ...      |
```

### 交叉分布表

```
age × income 交叉分布

|              | 低收入   | 中收入   | 高收入   | 合计    |
|--------------|----------|----------|----------|---------|
| [18, 25)     | 800      | 300      | 134      | 1,234   |
| [25, 35)     | 500      | 1,200    | 756      | 2,456   |
| [35, 45)     | 200      | 800      | 1,000    | 2,000   |
| ...          | ...      | ...      | ...      | ...     |
```

### IV 分箱明细表（有目标变量）

```
特征: age | IV = 0.1523

| 区间          | 样本数  | 正样本数 | 正样本率 | WoE     | IV      |
|---------------|---------|----------|----------|---------|---------|
| [18, 23)      | 984     | 156      | 15.85%   | 0.32    | 0.0234  |
| [23, 28)      | 1,012   | 98       | 9.68%    | -0.15   | 0.0089  |
| ...           | ...     | ...      | ...      | ...     | ...     |
```

---

## 与 feature-analysis 的关系

| 维度 | univariate-analysis | feature-analysis |
|------|---------------------|------------------|
| 定位 | 单特征快速分析 | 全量特征深度分析 |
| 目标变量 | 可选 | 必须 |
| 相关性分析 | ❌ | ✅ |
| PSI稳定性 | ❌ | ✅ |
| 方案推荐 | ❌ | ✅ 四套方案 |
| 典型用法 | 建模前快速筛选 | 特征工程完整报告 |

**推荐工作流**：`univariate-analysis`（快速筛选）→ `feature-analysis`（深度分析）

---

## 注意事项

1. **特征数量**：建议单次分析不超过 50 个特征，大量特征请分批处理
2. **分箱数量**：默认 10 箱，可根据数据量调整（数据量少时建议 5 箱）
3. **交叉分布**：仅支持两个特征的交叉，建议选择离散或已分箱的特征
4. **IV 计算**：需要目标变量为 0/1 二分类
5. **产物位置**：报告保存到 `<output_dir>/`
