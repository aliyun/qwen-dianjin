---
name: portfolio-health-check
description: >-
  对客户全部持仓进行穿透式体检，从资产配置、基金相关性、回测表现、风险暴露四个维度
  输出诊断报告和可视化图表。
  当用户提到持仓分析、组合诊断、持仓体检、配置检视、帮我看看持仓时触发。
  不用于单只基金分析（由fund-deep-research处理），
  不用于专项风险排查（由portfolio-risk-radar处理）。
---

# 持仓健康诊断

## 可用工具

本技能可调用以下 MCP 数据服务，执行流程中按需选用：

**盈米金融数据（qieman）**
- 服务地址：`https://dashscope.aliyuncs.com/api/v1/mcps/Qieman/sse`
- 核心能力：基金搜索/诊断、组合分析/回测、资产配置方案、CFP 工具链、图表渲染
- 本技能主要工具：`DiagnoseFundPortfolio`, `GetFundAssetClassAnalysis`, `BatchGetFundsDetail`, `GetBatchFundPerformance`, `AnalyzePortfolioRisk`, `GetFundsCorrelation`, `GetFundsBackTest`, `GuessFundCode`, `RenderEchart`

**恒生聚源金融数据（gildata-aidata）**
- 服务地址：开通恒生聚源 MCP 服务后获取，格式为 `https://dashscope.aliyuncs.com/api/v1/mcps/<your-mcp-id>/mcp`
- 核心能力：个股研究(A/H/US)、财务报表、资金流向、研报舆情、理财产品、宏观数据
- 本技能可选工具：`IndustryValuation`, `ProductBasicInfoList`

## 核心原则

**图表优先，文字精简。** 所有定量数据必须通过 `RenderEchart` 生成可视化图表来呈现，文字仅用于解读关键洞察，不要大段重复图表中已展示的数据。整份报告力求直观、简洁、可操作。

## 输入要求

### 必填信息
- 持仓产品列表：产品名称 + 基金代码 + 持有金额

### 可选信息
- 客户风险等级、投资目标
- 需要重点关注的维度

如果用户仅提供基金名称未提供代码，先调用 `GuessFundCode` 匹配。如果用户未提供任何持仓信息，主动追问。

## 执行流程

### 第一步：信息收集
- 解析持仓列表，提取基金代码，计算各基金持有权重（金额占比）
- 如持仓中包含理财产品（非基金），可调用 `ProductBasicInfoList`（gildata-aidata）获取产品基本信息

### 第二步：多维度数据采集（尽量并行调用）
- `DiagnoseFundPortfolio`：传入基金代码和持有金额，获取资产配置/相关性/回测三维评分及诊断建议
- `GetFundAssetClassAnalysis`：穿透分析资产大类分布（股票/债券/现金/另类各占比）
- `BatchGetFundsDetail`：各基金基本信息（类型、风险等级）
- `GetBatchFundPerformance`：各基金近期业绩（近1月/3月/1年收益、同类排名）
- `AnalyzePortfolioRisk`：组合整体风险指标
- `GetFundsCorrelation`：基金间相关性系数
- `GetFundsBackTest`：组合回测（年化收益、最大回撤、夏普比率）

**行业估值参考（gildata-aidata，可选）：**
- `IndustryValuation`：获取持仓重点行业的估值水平（PE/PB 百分位），辅助判断持仓行业是否处于高估区间

### 第三步：生成可视化图表（必须执行，不可跳过）

数据采集完成后，**必须**依次调用 `RenderEchart` 生成以下图表。每个图表调用一次 `RenderEchart`，将返回的图片URL用 markdown 图片语法嵌入报告。

#### 图表1：资产配置饼图（必须生成）

用穿透后的大类资产比例数据，调用 `RenderEchart` 生成饼图。ECharts option 参考：

```json
{
  "title": { "text": "资产配置分布", "left": "center" },
  "tooltip": { "trigger": "item", "formatter": "{b}: {d}%" },
  "legend": { "orient": "vertical", "left": "left", "top": "middle" },
  "color": ["#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de"],
  "series": [{
    "type": "pie",
    "radius": ["40%", "70%"],
    "center": ["55%", "50%"],
    "itemStyle": { "borderRadius": 8 },
    "label": { "formatter": "{b}\n{d}%" },
    "data": [
      { "name": "股票", "value": 45.2 },
      { "name": "债券", "value": 30.1 },
      { "name": "现金", "value": 15.5 },
      { "name": "另类", "value": 9.2 }
    ]
  }]
}
```

将 `data` 中的 name 和 value 替换为实际穿透数据。

#### 图表2：各基金业绩对比柱状图（必须生成）

用各基金的近1年收益率数据，调用 `RenderEchart` 生成水平柱状图，直观展示谁强谁弱。ECharts option 参考：

```json
{
  "title": { "text": "各基金近1年收益对比", "left": "center" },
  "tooltip": { "trigger": "axis" },
  "grid": { "left": "25%", "right": "10%", "containLabel": false },
  "xAxis": { "type": "value", "axisLabel": { "formatter": "{value}%" } },
  "yAxis": { "type": "category", "data": ["基金A", "基金B", "基金C"] },
  "series": [{
    "type": "bar",
    "data": [12.5, 8.3, -2.1],
    "itemStyle": { "borderRadius": [0, 4, 4, 0] },
    "label": { "show": true, "position": "right", "formatter": "{c}%" }
  }]
}
```

将 yAxis.data 替换为基金简称，series.data 替换为实际收益率。收益为负的柱子自动显示为红色（可通过 visualMap 或逐项设置 itemStyle.color 实现）。

#### 图表3：诊断评分雷达图（必须生成）

用 DiagnoseFundPortfolio 返回的评分数据，调用 `RenderEchart` 生成雷达图。ECharts option 参考：

```json
{
  "title": { "text": "组合健康评分", "left": "center" },
  "radar": {
    "indicator": [
      { "name": "资产配置", "max": 5 },
      { "name": "分散程度", "max": 5 },
      { "name": "回测表现", "max": 5 },
      { "name": "风险控制", "max": 5 }
    ],
    "shape": "circle"
  },
  "series": [{
    "type": "radar",
    "data": [{
      "value": [4, 2, 3.5, 3],
      "name": "当前组合",
      "areaStyle": { "opacity": 0.3 }
    }]
  }]
}
```

将 value 替换为实际评分。

#### 图表4：相关性热力图（如基金≥3只则必须生成）

用 GetFundsCorrelation 返回的相关系数矩阵，调用 `RenderEchart` 生成热力图。ECharts option 参考：

```json
{
  "title": { "text": "基金相关性矩阵", "left": "center" },
  "tooltip": { "formatter": "{c}" },
  "xAxis": { "type": "category", "data": ["基金A", "基金B", "基金C"], "axisLabel": { "rotate": 30 } },
  "yAxis": { "type": "category", "data": ["基金A", "基金B", "基金C"] },
  "visualMap": { "min": -1, "max": 1, "orient": "horizontal", "left": "center", "bottom": 0, "inRange": { "color": ["#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"] } },
  "series": [{
    "type": "heatmap",
    "data": [[0,0,1],[0,1,0.85],[0,2,0.32],[1,0,0.85],[1,1,1],[1,2,0.45],[2,0,0.32],[2,1,0.45],[2,2,1]],
    "label": { "show": true, "formatter": "{c}" }
  }]
}
```

data 格式为 [x索引, y索引, 相关系数]，替换为实际数据。相关系数>0.8的高亮标注。

### 第四步：输出诊断报告

## 输出模板

按以下结构输出，**图表嵌入对应章节，文字控制在每个章节2-4句话**：

```markdown
## 持仓健康诊断报告

### 一、综合评分

{诊断评分雷达图}

综合得分 **{X}/5**，{一句话总评，如"配置均衡但分散度不足"}。

| 维度 | 评分 | 简评 |
|------|------|------|
| 资产配置 | {X}/5 | {一句话} |
| 分散程度 | {X}/5 | {一句话} |
| 回测表现 | {X}/5 | {一句话} |
| 风险控制 | {X}/5 | {一句话} |

### 二、资产配置

{资产配置饼图}

{2-3句解读：当前配置特征、与风险等级匹配度、是否存在集中风险}

### 三、业绩表现

{各基金业绩对比柱状图}

{2-3句解读：组合整体表现（年化收益X%、最大回撤X%、夏普X）、哪些基金拖累/贡献突出}

### 四、持仓相关性

{相关性热力图}

{2-3句解读：分散化程度、是否有高度重叠持仓（相关系数>0.8）}

### 五、关键发现与建议

1. **{发现1}**：{一句话描述} → 建议：{具体操作}
2. **{发现2}**：{一句话描述} → 建议：{具体操作}
3. **{发现3}**：{一句话描述} → 建议：{具体操作}

---
*诊断基于历史数据和量化模型，仅供参考，不构成投资建议。*
```

## 注意事项

- **图表为必选项**：资产配置饼图、业绩对比柱状图、诊断雷达图为必须生成项，不得用文字表格替代
- 合规要求：不得包含收益承诺，建议用"建议考虑"而非"应该买"
- 措辞温和：如亏损较大，避免"亏损严重""配置混乱"等刺激性表述
- 文字精简：全文控制在800-1200字（不含图表），每个章节不超过4句话
- 数据完整性：如部分基金代码无法识别，标注并说明
