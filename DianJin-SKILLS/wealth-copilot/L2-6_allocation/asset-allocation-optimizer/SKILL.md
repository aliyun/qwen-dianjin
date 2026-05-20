---
name: asset-allocation-optimizer
description: >-
  基于客户风险偏好和当前持仓，结合市场环境和量化模型，
  生成个性化资产配置方案并模拟预期收益分布。
  当用户提到资产配置、配置优化、怎么调仓、投资方案设计、配置方案时触发。
  不用于单只基金分析（由fund-deep-research处理），
  不用于具体产品推荐（由smart-product-matching处理）。
---

# 资产配置优化

## 可用工具

本技能可调用以下 MCP 数据服务，执行流程中按需选用：

**盈米金融数据（qieman）**
- 服务地址：`https://dashscope.aliyuncs.com/api/v1/mcps/Qieman/sse`
- 核心能力：基金搜索/诊断、组合分析/回测、资产配置方案、CFP 工具链、图表渲染
- 本技能主要工具：`GetAssetAllocationPlan`, `GetCompositeModel`, `GetFundAssetClassAnalysis`, `MonteCarloSimulate`, `GetLatestQuotations`, `RenderEchart`

**恒生聚源金融数据（gildata-aidata）**
- 服务地址：开通恒生聚源 MCP 服务后获取，格式为 `https://dashscope.aliyuncs.com/api/v1/mcps/<your-mcp-id>/mcp`
- 核心能力：个股研究(A/H/US)、财务报表、资金流向、研报舆情、理财产品、宏观数据
- 本技能可选工具：`MacroIndustryEDB`, `IndustryValuation`

## 核心原则

**图表优先，文字精简。** 资产配置比例、调仓缺口、收益模拟等定量数据必须通过 `RenderEchart` 生成可视化图表呈现，文字仅用于解读配置逻辑和实施建议。

## 输入要求

### 必填信息
- 客户风险等级：R1-R5
- 可投资金额或AUM

### 可选信息
- 当前持仓概况（用于分析配置缺口）
- 投资期限偏好
- 预期年化收益率或可承受最大回撤
- 特殊需求（如"不要权益""想加黄金"）

如果用户仅说"帮我做个配置方案"，追问风险等级和可投资金额。

## 执行流程

### 第一步：信息收集与需求确认
- 解析客户上下文，提取风险等级、AUM、当前持仓
- 确定配置约束条件

### 第二步：获取基准配置方案
- 调用 `GetAssetAllocationPlan` 传入投资三性参数（预期收益率 / 最大回撤 / 投资期限，至少一个）

### 第三步：落地到具体产品
- 调用 `GetCompositeModel` 通过方案ID获取复合模型

### 第四步：分析当前持仓缺口（如有持仓数据）
- 调用 `GetFundAssetClassAnalysis` 穿透分析当前持仓的资产大类分布
- 对比目标方案和当前配置，计算各大类缺口

### 第五步：收益风险模拟
- 调用 `MonteCarloSimulate` 对配置方案做蒙特卡洛模拟

### 第六步：市场环境参考
- 调用 `GetLatestQuotations`（qieman）了解当前市场环境
- 可选：调用 `MacroIndustryEDB`（gildata-aidata）获取最新宏观指标（GDP增速、CPI、利率等），为配置决策提供宏观背景
- 可选：调用 `IndustryValuation`（gildata-aidata）查看主要行业估值百分位，辅助判断权益配置时机

### 第七步：生成可视化图表（必须执行，不可跳过）

**必须**依次调用 `RenderEchart` 生成以下图表：

#### 图表1：目标资产配置饼图（必须生成）

用目标方案的各大类资产权重生成饼图。ECharts option 参考：

```json
{
  "title": { "text": "目标资产配置", "left": "center" },
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
      { "name": "权益类", "value": 30 },
      { "name": "固收类", "value": 40 },
      { "name": "固收+", "value": 15 },
      { "name": "现金管理", "value": 10 },
      { "name": "另类/商品", "value": 5 }
    ]
  }]
}
```

替换为实际方案权重。

#### 图表2：当前 vs 目标对比柱状图（如有当前持仓则必须生成）

用当前配置和目标配置的各大类占比生成分组柱状图，直观显示缺口。ECharts option 参考：

```json
{
  "title": { "text": "当前配置 vs 目标配置", "left": "center" },
  "tooltip": { "trigger": "axis" },
  "legend": { "top": "bottom" },
  "xAxis": { "type": "category", "data": ["权益类", "固收类", "固收+", "现金管理", "另类"] },
  "yAxis": { "type": "value", "axisLabel": { "formatter": "{value}%" } },
  "series": [
    { "name": "当前配置", "type": "bar", "data": [45, 20, 10, 20, 5], "itemStyle": { "borderRadius": [4,4,0,0] } },
    { "name": "目标配置", "type": "bar", "data": [30, 40, 15, 10, 5], "itemStyle": { "borderRadius": [4,4,0,0] } }
  ]
}
```

替换为实际数据。

#### 图表3：收益模拟区间图（必须生成）

用蒙特卡洛模拟的三情景收益率数据生成柱状图。ECharts option 参考：

```json
{
  "title": { "text": "预期收益模拟（蒙特卡洛）", "left": "center" },
  "tooltip": { "trigger": "axis" },
  "legend": { "top": "bottom" },
  "xAxis": { "type": "category", "data": ["1年", "3年", "5年"] },
  "yAxis": { "type": "value", "axisLabel": { "formatter": "{value}%" } },
  "series": [
    { "name": "悲观(25%)", "type": "bar", "stack": "range", "data": [-3, 2, 8], "itemStyle": { "color": "#ee6666", "borderRadius": [0,0,0,0] } },
    { "name": "中性(50%)", "type": "bar", "stack": "range", "data": [5, 15, 28], "itemStyle": { "color": "#fac858" } },
    { "name": "乐观(75%)", "type": "bar", "stack": "range", "data": [12, 30, 52], "itemStyle": { "color": "#91cc75", "borderRadius": [4,4,0,0] } }
  ]
}
```

替换为实际模拟数据。注意：堆叠柱状图中各层 data 为累加关系，需计算差值。如不方便用堆叠图，也可改为分组柱状图。

## 输出模板

按以下结构输出，**图表嵌入对应章节，文字精简**：

```markdown
## 资产配置优化方案

### 一、配置需求
| 项目 | 内容 |
|------|------|
| 风险等级 | {R等级} |
| 可投资金额 | {X}万 |
| 投资期限 | {X} |
| 配置目标 | {目标描述} |

### 二、推荐配置

{目标资产配置饼图}

| 资产大类 | 目标占比 | 建议金额 | 作用 |
|----------|---------|---------|------|
| 权益类 | {X}% | {X}万 | 收益增强 |
| 固收类 | {X}% | {X}万 | 底仓稳健 |
| ... | ... | ... | ... |

### 三、调仓建议（如有当前持仓）

{当前vs目标对比柱状图}

{2-3句解读：主要缺口在哪、需要增减配的方向}

### 四、收益模拟

{收益模拟区间图}

{2-3句通俗解读：持有X年大概率收益区间，亏损概率}

### 五、具体落地产品
| 资产类别 | 建议产品 | 配置金额 |
|----------|---------|---------|
| {类别} | {产品名}({代码}) | {X}万 |

### 六、实施建议
1. {分步实施建议}
2. {定期再平衡建议}

---
*配置方案基于量化模型和历史数据，收益模拟不代表实际收益。投资有风险，配置需谨慎。*
```

## 注意事项

- **图表为必选项**：配置饼图和收益模拟图为必须生成项
- 合规要求：收益模拟明确标注"模拟"性质，不等于收益承诺
- 适当性匹配：方案风险等级不得超过客户风险等级
- 流动性保障：现金管理类资产占比不低于5-10%
- 文字精简：全文控制在800-1200字（不含图表）
