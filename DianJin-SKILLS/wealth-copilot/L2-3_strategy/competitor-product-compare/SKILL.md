---
name: competitor-product-compare
description: >-
  对比同类基金产品，从收益、风险、费率、流动性、基金经理能力五个维度
  生成对比报告，帮助客户经理回应"别家产品更好"的异议。
  当用户提到跟XX比怎么样、竞品对比、同类比较、为什么选我们、
  两只基金比较时触发。
  不用于单只基金分析（由fund-deep-research处理）。
---

# 竞品对比分析

## 可用工具

本技能可调用以下 MCP 数据服务，执行流程中按需选用：

**盈米金融数据（qieman）**
- 服务地址：`https://dashscope.aliyuncs.com/api/v1/mcps/Qieman/sse`
- 核心能力：基金搜索/诊断、组合分析/回测、资产配置方案、CFP 工具链、图表渲染
- 本技能主要工具：`SearchFunds`, `GuessFundCode`, `BatchGetFundsDetail`, `GetBatchFundPerformance`, `AnalyzeFundRisk`, `BatchGetFundTradeRules`, `RenderEchart`

**恒生聚源金融数据（gildata-aidata）**
- 服务地址：开通恒生聚源 MCP 服务后获取，格式为 `https://dashscope.aliyuncs.com/api/v1/mcps/<your-mcp-id>/mcp`
- 核心能力：个股研究(A/H/US)、财务报表、资金流向、研报舆情、理财产品、宏观数据
- 本技能可选工具：`FundManagerInfoReport`, `FundIncomeRiskReport`, `ProductBasicInfoList`

## 核心原则

**图表优先，文字精简。** 产品对比的核心是一眼看出差异——五维评分雷达图和业绩对比柱状图必须通过 `RenderEchart` 生成，文字仅用于点出关键差异和话术建议。

## 输入要求

### 必填信息
- 至少2只产品的名称或代码（支持基金 vs 基金，也支持基金 vs 理财产品的跨品类对比）

### 可选信息
- 对比侧重点（收益/风险/费率）
- 客户风险等级（影响适配性评价）

## 执行流程

### 第一步：确认产品代码
- 对每只基金调用 `SearchFunds` 或 `GuessFundCode` 确认代码
- 如涉及理财产品，调用 `ProductBasicInfoList`（gildata-aidata）获取产品信息

### 第二步：并行采集数据

**基金类（qieman）：**
- `BatchGetFundsDetail`：基本信息
- `GetBatchFundPerformance`：业绩对比
- `AnalyzeFundRisk`：风险评估
- `BatchGetFundTradeRules`：交易规则和费率

**经理能力增强（gildata-aidata，可选）：**
- `FundManagerInfoReport`：经理详细从业背景、管理规模、获奖情况
- `FundIncomeRiskReport`：基金收益风险详细报告

**理财产品（gildata-aidata，如有跨品类对比）：**
- `ProductBasicInfoList`：理财产品详情（发行方、期限、业绩基准、费率）

### 第三步：五维度对比
- **收益维度**：各阶段收益对比、同类排名
- **风险维度**：最大回撤、波动率、夏普比率
- **费率维度**：申购费、管理费、托管费
- **流动性**：赎回到账时间、大额赎回限制
- **经理能力**：任职时间、管理规模、历史业绩、投资风格

### 第四步：生成可视化图表（必须执行，不可跳过）

**必须**依次调用 `RenderEchart` 生成以下图表：

#### 图表1：五维对比雷达图（必须生成）

将每个维度评分归一化为1-5分，生成多系列雷达图。ECharts option 参考：

```json
{
  "title": { "text": "五维对比", "left": "center" },
  "legend": { "top": "bottom" },
  "radar": {
    "indicator": [
      { "name": "收益", "max": 5 },
      { "name": "风险控制", "max": 5 },
      { "name": "费率", "max": 5 },
      { "name": "流动性", "max": 5 },
      { "name": "经理能力", "max": 5 }
    ],
    "shape": "circle"
  },
  "series": [{
    "type": "radar",
    "data": [
      { "value": [4, 3, 4, 5, 4], "name": "易方达蓝筹精选", "areaStyle": { "opacity": 0.2 } },
      { "value": [3, 4, 3, 5, 3], "name": "景顺长城新兴成长", "areaStyle": { "opacity": 0.2 } }
    ]
  }]
}
```

替换基金名称和各维度评分。评分规则：同类排名前20%=5分、前40%=4分、前60%=3分、前80%=2分、后20%=1分。

#### 图表2：各阶段业绩对比柱状图（必须生成）

用两只/多只产品的各阶段收益率生成分组柱状图。ECharts option 参考：

```json
{
  "title": { "text": "业绩对比", "left": "center" },
  "tooltip": { "trigger": "axis" },
  "legend": { "top": "bottom" },
  "xAxis": { "type": "category", "data": ["近1月", "近3月", "近6月", "近1年", "近3年"] },
  "yAxis": { "type": "value", "axisLabel": { "formatter": "{value}%" } },
  "series": [
    { "name": "产品A", "type": "bar", "data": [2.1, 5.3, 8.7, 15.2, 42.1], "itemStyle": { "borderRadius": [4,4,0,0] } },
    { "name": "产品B", "type": "bar", "data": [1.8, 4.1, 7.2, 12.8, 35.6], "itemStyle": { "borderRadius": [4,4,0,0] } }
  ]
}
```

替换为实际产品名称和收益率数据。如有3只以上产品，增加对应 series。

### 第五步：生成对比报告和话术

## 输出模板

按以下结构输出，**图表嵌入对应章节，文字精简聚焦差异**：

```markdown
## 产品对比分析 | {产品A} vs {产品B}

### 一、五维评分

{五维对比雷达图}

| 维度 | {产品A} | {产品B} | 优势方 |
|------|---------|---------|--------|
| 收益 | {X}/5 | {X}/5 | {标注} |
| 风险控制 | {X}/5 | {X}/5 | {标注} |
| 费率 | {X}/5 | {X}/5 | {标注} |
| 流动性 | {X}/5 | {X}/5 | {标注} |
| 经理能力 | {X}/5 | {X}/5 | {标注} |

### 二、业绩对比

{业绩对比柱状图}

{2句解读：短中长期各自优势}

### 三、关键差异
1. **{差异1}**：{一句话对比}
2. **{差异2}**：{一句话对比}
3. **{差异3}**：{一句话对比}

### 四、话术参考
> 当客户说"{竞品更好的异议}"时，可以这样回应：
> "{基于数据的客观对比话术}"

---
*对比基于历史数据，仅供参考。不同产品适合不同投资需求，无绝对优劣。*
```

## 注意事项

- **图表为必选项**：五维雷达图和业绩对比柱状图为必须生成项
- 客观中立：不刻意贬低竞品或美化自家产品
- 数据说话：对比基于客观数据，不做主观评判
- 合规要求：不诋毁竞争对手，不误导客户
- 文字精简：全文控制在600-1000字（不含图表）
- 跨品类对比：如涉及基金 vs 理财产品，需说明两者的本质差异（如开放/封闭、净值/预期收益等）
