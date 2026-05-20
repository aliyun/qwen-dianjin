---
name: market-insight-qa
description: >-
  快速回答市场行情类问题，包括指数表现、板块走势、
  市场温度、北向资金等，返回简洁精准的数据和简评。
  当用户提到大盘怎么样、哪些板块表现好、A股走势、
  黄金价格、北向资金时触发。
  不用于生成完整早报（由market-morning-brief处理）。
---

# 市场行情问答

## 可用工具

本技能可调用以下 MCP 数据服务，根据问题类型智能路由：

**盈米金融数据（qieman）**
- 服务地址：`https://dashscope.aliyuncs.com/api/v1/mcps/Qieman/sse`
- 核心能力：基金搜索/诊断、组合分析/回测、资产配置方案、CFP 工具链、图表渲染
- 本技能主要工具：`GetLatestQuotations`, `SearchFinancialNews`, `SearchHotTopic`, `searchRealtimeAiAnalysis`, `SearchFunds`, `RenderEchart`

**恒生聚源金融数据（gildata-aidata）**
- 服务地址：开通恒生聚源 MCP 服务后获取，格式为 `https://dashscope.aliyuncs.com/api/v1/mcps/<your-mcp-id>/mcp`
- 核心能力：个股研究(A/H/US)、财务报表、资金流向、研报舆情、理财产品、宏观数据
- 本技能主要工具：`AShareLiveQuote`, `IndexDailyQuote`, `SectorRank`, `HShareLiveQuote`, `USStockDailyQuotes`, `MacroIndustryEDB`

## 输入要求

### 必填信息
- 行情问题（可以是一句自然语言提问）

### 可选信息
- 具体关注的指数/板块/品种
- 时间范围

## 执行流程

### 第一步：解析问题并路由数据源

识别用户关心的市场维度，按问题类型选择最佳数据源：

| 问题类型 | 优先数据源 | 工具选择 |
|---------|-----------|---------|
| A股个股行情（如"贵州茅台今天怎么样"） | gildata-aidata | `AShareLiveQuote` |
| 港股行情（如"腾讯今天涨了吗"） | gildata-aidata | `HShareLiveQuote` |
| 美股行情（如"英伟达收盘价"） | gildata-aidata | `USStockDailyQuotes` |
| 板块/行业排行（如"哪些板块涨得好"） | gildata-aidata | `SectorRank` |
| 宏观经济数据（如"CPI多少""PMI走势"） | gildata-aidata | `MacroIndustryEDB` |
| 主要指数概览（如"大盘怎么样"） | qieman 为主 | `GetLatestQuotations`，可用 `IndexDailyQuote` 补充 |
| 基金表现（如"某基金今天净值"） | qieman | `SearchFunds` + 基金相关工具 |
| 市场资讯/热点 | qieman | `SearchFinancialNews`, `SearchHotTopic` |

### 第二步：数据获取

根据路由结果调用对应工具：
- **大盘/指数概览**：先调用 `GetLatestQuotations`，如需更详细数据调用 `IndexDailyQuote`
- **个股行情**：根据市场调用 `AShareLiveQuote` / `HShareLiveQuote` / `USStockDailyQuotes`
- **板块排行**：调用 `SectorRank` 获取行业涨跌排名
- **宏观数据**：调用 `MacroIndustryEDB` 获取指定宏观指标
- 如涉及资讯：调用 `SearchFinancialNews` 获取相关新闻
- 如涉及热点：调用 `SearchHotTopic` 获取热点话题
- 如需深度分析：调用 `searchRealtimeAiAnalysis` 获取AI解读

### 第三步：简洁回答
- 先给数据（精确数值）
- 再给简评（1-2句原因分析）
- 不展开成完整报告

## 输出模板

```markdown
### 市场行情速答

{直接回答数据}

| 指数/品种 | 最新值 | 涨跌幅 |
|----------|--------|--------|
| {名称} | {值} | {X}% |

**简评**：{1-2句原因分析}

---
*数据可能存在延迟，仅供参考。*
```

## 注意事项

- 快速简洁：区别于早报的完整报告，这里是快问快答
- 数据优先：先给数据，再给分析
- 不预测：不做涨跌预测
- 输出控制：控制在200-500字
- 数据源选择：个股/板块/港美股/宏观优先用 gildata-aidata，基金/指数概览优先用 qieman
