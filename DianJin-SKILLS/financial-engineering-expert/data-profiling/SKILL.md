---
name: data-profiling
category: 数据探索
description: 【数据轮廓速览】当用户刚上传数据、问"数据有多少行"、"有哪些字段"、"数据长什么样"时使用。描述数据结构（行列数、字段类型、缺失率、分布特征、数据质量），不计算建模指标。无需目标变量。与 feature-analysis 的区别：只做"描述"不做"分析"，不计算IV/PSI/相关性等建模指标。
next_steps:
  - skill: feature-analysis
    text: 全量特征分析，推荐建模方案
    prompt: 全面分析所有特征，推荐建模方案
---

# 数据洞察报告 (portable)

基于 `scripts/profiler.py` 主脚本，对数据集进行轮廓扫描，生成数据概况报告。

---

## 参数说明

| 参数 | 必选 | 默认值 | 说明 |
|------|:----:|--------|------|
| `--data_path` | ✅ | - | 数据文件路径（parquet/csv） |
| `--output` | | `data_profiling_report.md` | 报告输出路径 |
| `--output_dir` | | `./outputs/<ts>` | 产物输出目录 |
| `--output_name` | | `data_profiling_report` | 报告基名（不含扩展名） |
| `--config` | | - | JSON 配置文件路径 |

---

## 执行方式

```bash
python scripts/profiler.py \
  --data_path ./examples/toy.parquet \
  --output_dir ./outputs/profile_run
```

执行结束后：
- 产物目录 `<output_dir>/` 下生成：
  - `report.md` — 数据概况报告
  - `result.json` — 结构化产物清单（见 PROTOCOL.md）
- stdout 末行打印 `result.json` 绝对路径，Agent 读这个文件即可

---

## 产物示例

```json
{
  "skill": "data-profiling",
  "status": "success",
  "files": [
    {"path": ".../report.md", "role": "report"}
  ],
  "metrics": {"n_rows": 10000, "n_cols": 28, "n_missing_cols": 5},
  "summary": "中等规模数据集（10,000行），28 个字段，数值型为主，发现 2 个高缺失字段"
}
```

---

## 报告输出结构

| 章节 | 内容 |
|------|------|
| 1. 数据概览 | 文件名、格式、大小、行列数、字段类型分布饼图 |
| 2. 字段详情清单 | 每个字段的类型、缺失率、唯一值数、示例值 |
| 3. 数值特征分析 | 描述性统计（均值/标准差/分位数/偏度/峰度）+ histogram 分布图 |
| 4. 类别特征分析 | 唯一值数、Top 值占比、集中度 + bar chart |
| 5. 缺失值分析 | 缺失率排名表格 + 柱状图 |
| 6. 数据质量 | 重复行、空列、常量列、高缺失列 |
| 7. 样本预览 | 前 5 行数据展示 |

---

## 自适应展示

当数值/类别特征数量超过 20 个时，统计表格和图表只展示前 20 个，避免报告过长。

---

## 与其他 Skill 的关系

| Skill | 用途区别 |
|-------|----------|
| **data-profiling** | 快速了解数据轮廓，不做深度分析 |
| feature-analysis | 深度特征分析（IV、PSI、相关性等），需要目标变量 |
| xgb-modeling | 建模全流程，需要目标变量 |

**建议流程**：
1. 先用 `data-profiling` 了解数据基本情况
2. 根据洞察结果，决定是否需要 `feature-analysis` 或 `xgb-modeling`

---

## 注意事项

1. **无需目标变量**：本 Skill 不需要提供目标变量，纯数据描述
2. **快速轻量**：执行速度快，适合大数据集的快速扫描
3. **自动类型推断**：识别数值型、类别型、布尔型、时间型、文本型字段
