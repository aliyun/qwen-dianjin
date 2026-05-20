---
name: investment-simulation
description: >-
  运用蒙特卡洛模拟对拟投资组合进行多周期收益概率测算，
  可视化乐观/中性/悲观三种情景，帮助客户理解预期风险收益。
  当用户提到收益模拟、能赚多少、预期收益、蒙特卡洛、
  投资回报、投多少能赚多少时触发。
  不用于资产配置方案设计（由asset-allocation-optimizer处理）。
---

# 投资收益模拟

## 可用工具

本技能可调用以下 MCP 数据服务，执行流程中按需选用：

**盈米金融数据（qieman）**
- 服务地址：`https://dashscope.aliyuncs.com/api/v1/mcps/Qieman/sse`
- 核心能力：基金搜索/诊断、组合分析/回测、资产配置方案、CFP 工具链、图表渲染
- 本技能主要工具：`MonteCarloSimulate`, `GetFundsBackTest`, `AnalyzePortfolioRisk`, `RenderEchart`

**恒生聚源金融数据（gildata-aidata）**
- 服务地址：开通恒生聚源 MCP 服务后获取，格式为 `https://dashscope.aliyuncs.com/api/v1/mcps/<your-mcp-id>/mcp`
- 核心能力：个股研究(A/H/US)、财务报表、资金流向、研报舆情、理财产品、宏观数据
- 本技能可选工具：`IndexDailyQuote`

## 核心原则

**图表优先，文字精简。** 收益模拟的核心价值在于可视化——概率分布、三情景对比等必须通过 `RenderEchart` 生成图表呈现，文字仅用于通俗解读。

## 输入要求

### 必填信息
- 投资组合或资产配置方案（基金代码+权重 或 大类资产+权重）
- 投资金额

### 可选信息
- 投资期限（默认1年/3年/5年三档）
- 定投模式（每月追加金额）
- 目标收益率

## 执行流程

### 第一步：解析投资方案
- 提取资产配置权重
- 如传入基金代码，映射到大类资产

### 第二步：蒙特卡洛模拟
- 调用 `MonteCarloSimulate` 传入资产权重配置
- 获取不同周期（1年/3年/5年）的收益分布
- 提取关键分位数：5%/25%/50%/75%/95%

### 第三步：补充历史回测
- 调用 `GetFundsBackTest` 对组合做历史回测
- 调用 `AnalyzePortfolioRisk` 获取组合风险指标
- 可选：调用 `IndexDailyQuote`（gildata-aidata）获取基准指数历史行情，用于回测对比参考

### 第四步：生成可视化图表（必须执行，不可跳过）

**必须**依次调用 `RenderEchart` 生成以下图表：

#### 图表1：三情景收益对比柱状图（必须生成）

用1年/3年/5年的三情景收益率数据生成分组柱状图。ECharts option 参考：

```json
{
  "title": { "text": "预期收益模拟", "left": "center", "subtext": "基于蒙特卡洛模拟" },
  "tooltip": { "trigger": "axis" },
  "legend": { "top": "bottom" },
  "xAxis": { "type": "category", "data": ["1年", "3年", "5年"] },
  "yAxis": { "type": "value", "axisLabel": { "formatter": "{value}%" } },
  "series": [
    { "name": "悲观(25%分位)", "type": "bar", "data": [-5.2, 1.8, 8.5], "itemStyle": { "color": "#ee6666", "borderRadius": [4,4,0,0] } },
    { "name": "中性(50%分位)", "type": "bar", "data": [4.8, 16.2, 30.1], "itemStyle": { "color": "#fac858", "borderRadius": [4,4,0,0] } },
    { "name": "乐观(75%分位)", "type": "bar", "data": [15.3, 35.8, 58.2], "itemStyle": { "color": "#91cc75", "borderRadius": [4,4,0,0] } }
  ]
}
```

替换为实际模拟数据。

#### 图表2：资产配置饼图（必须生成）

用输入的资产配置权重生成饼图，让客户直观看到钱怎么分配。ECharts option 参考：

```json
{
  "title": { "text": "投资组合配置", "left": "center" },
  "tooltip": { "trigger": "item", "formatter": "{b}: {d}%" },
  "color": ["#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de"],
  "series": [{
    "type": "pie",
    "radius": ["40%", "70%"],
    "center": ["50%", "55%"],
    "itemStyle": { "borderRadius": 8 },
    "label": { "formatter": "{b}\n{d}%" },
    "data": [
      { "name": "权益类", "value": 40 },
      { "name": "固收类", "value": 35 },
      { "name": "现金类", "value": 15 },
      { "name": "另类", "value": 10 }
    ]
  }]
}
```

替换为实际配置数据。

#### 图表3：盈亏概率柱状图（必须生成）

用不同期限的盈利概率、年化>5%概率、亏损>10%概率生成柱状图。ECharts option 参考：

```json
{
  "title": { "text": "盈亏概率分析", "left": "center" },
  "tooltip": { "trigger": "axis", "formatter": "{b}<br/>{a}: {c}%" },
  "legend": { "top": "bottom" },
  "xAxis": { "type": "category", "data": ["1年", "3年", "5年"] },
  "yAxis": { "type": "value", "max": 100, "axisLabel": { "formatter": "{value}%" } },
  "series": [
    { "name": "盈利概率", "type": "bar", "data": [62, 78, 89], "itemStyle": { "color": "#91cc75", "borderRadius": [4,4,0,0] } },
    { "name": "年化>5%概率", "type": "bar", "data": [45, 60, 72], "itemStyle": { "color": "#5470c6", "borderRadius": [4,4,0,0] } },
    { "name": "亏损>10%概率", "type": "bar", "data": [15, 8, 3], "itemStyle": { "color": "#ee6666", "borderRadius": [4,4,0,0] } }
  ]
}
```

替换为实际概率数据。

## 输出模板

按以下结构输出，**图表嵌入对应章节，文字通俗精简**：

```markdown
## 投资收益模拟 | {投资金额}万

### 一、投资方案

{资产配置饼图}

### 二、收益模拟

{三情景收益对比柱状图}

| 投资期限 | 悲观 | 中性 | 乐观 |
|----------|------|------|------|
| 1年 | {X}万({Y}%) | {X}万({Y}%) | {X}万({Y}%) |
| 3年 | {X}万({Y}%) | {X}万({Y}%) | {X}万({Y}%) |
| 5年 | {X}万({Y}%) | {X}万({Y}%) | {X}万({Y}%) |

### 三、盈亏概率

{盈亏概率柱状图}

### 四、历史回测参考
- **年化收益**：{X}%
- **最大回撤**：{X}%
- **夏普比率**：{X}

### 五、通俗解读
> {2-3句客户听得懂的解读，如"投10万持有3年，大概率能赚1-3万，亏损超过1万的概率不到10%"}

---
*模拟基于历史数据和统计模型，不代表实际收益承诺。市场有风险，投资需谨慎。*
```

## 注意事项

- **图表为必选项**：收益模拟柱状图、配置饼图、盈亏概率图为必须生成项
- 不是收益承诺：反复强调"模拟""概率""参考"
- 通俗解读：将概率数据翻译成客户能理解的具体金额和生活化语言
- 展示风险面：不仅展示收益，也展示亏损概率
- 文字精简：全文控制在600-1000字（不含图表）
