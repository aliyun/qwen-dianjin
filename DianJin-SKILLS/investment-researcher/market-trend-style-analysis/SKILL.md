---
name: market-trend-style-analysis
description: 证券策略市场大势与风格研判技能。适用于策略研究员进行市场方向判断和风格配置。基于量价、资金、情绪及宏观数据，输出大势研判（牛/熊/震荡）、风格强弱排序（成长/价值/大盘/小盘）及仓位建议，生成标准化 Markdown 策略报告。当用户要求分析市场大势、判断市场方向、进行风格配置、评估仓位时使用，包括"市场大势分析"、"当前市场怎么样"、"风格怎么配"、"现在该买大盘还是小盘"、"成长和价值哪个强"、"市场趋势研判"、"仓位建议"等表述。即使用户只说"看看大盘"、"市场怎么样"、"现在什么风格"等简短表述也要触发。本技能自动通过 gildata-aidata 专业金融数据服务获取核心数据，进行量价与逻辑推演，生成标准化策略研判报告。
---

# 市场大势与风格研判 (Market Trend & Style Analysis)

## 概述

本技能专为**证券策略研究员**设计，旨在提供系统化的**市场大势研判**与**风格配置建议**。通过综合分析量价趋势、流动性环境、市场情绪及宏观因子，量化评估市场状态，输出明确的指数方向判断、风格强弱排序及动态仓位建议，辅助投资决策。

**所有数据查询必须使用 `gildata-aidata` 服务**。

## 核心分析框架

### 1. 大势研判模型 (Trend & Direction)
*   **量价趋势**: 均线系统（多头/空头排列）、关键支撑/阻力位、量价配合（放量上涨/缩量下跌）。
*   **流动性**: 宏观流动性（M1/M2 剪刀差、社融增速）、微观流动性（北向资金、ETF 申赎、两融余额）。
*   **情绪周期**: 换手率、新增开户数、股票型公募基金发行量、期权隐含波动率 (IV)。
*   **研判结论**: 🟢 牛市/上涨趋势，🟡 震荡市/结构性行情，🔴 熊市/下跌趋势。

### 2. 风格强弱排序 (Style Rotation)
*   **大盘 vs 小盘**:
    *   **逻辑**: 信用扩张期利好小盘（流动性充裕），信用收缩期利好大盘（确定性溢价）。
    *   **指标**: 信用脉冲、利率走势、微盘股流动性风险。
*   **成长 vs 价值**:
    *   **逻辑**: 利率下行/盈利复苏利好成长，利率上行/滞胀利好价值（红利）。
    *   **指标**: 10 年期国债收益率、盈利增速差（成长 - 价值）、风险偏好。
*   **风格矩阵**:
    *   大盘价值 (红利/金融)
    *   大盘成长 (茅指数/核心资产)
    *   小盘价值 (周期/资源)
    *   小盘成长 (科创/微盘)

### 3. 仓位管理模型 (Position Sizing)
*   **评分系统**: 综合大势、风格、风险指标得出 0-100 分。
*   **仓位映射**:
    *   **80-100 分 (高仓位)**: 趋势向上 + 流动性宽松 + 情绪回暖 -> 建议仓位 **70-100%**
    *   **50-80 分 (中仓位)**: 震荡市 + 结构性机会 -> 建议仓位 **40-70%**
    *   **0-50 分 (低仓位)**: 趋势向下 + 流动性收紧 + 情绪冰点 -> 建议仓位 **0-40%**

## 数据采集指引 (gildata-aidata 工具映射)

**所有数据必须通过 `gildata-aidata` 服务获取**，具体工具调用如下：

### 指数行情与技术面
```
finx gildata-aidata IndexLiveQuote --body '{"indexCode": "指数代码"}'
```
获取 A 股指数实时行情，包括价量、涨跌幅、委比等指标。

```
finx gildata-aidata IndexDailyQuote --body '{"indexCode": "指数代码", "tradeDate": "交易日期"}'
```
获取 A 股指数日行情，包括价量、涨跌幅、换手率等指标。

```
finx gildata-aidata IndexRangeQuotation --body '{"indexCode": "指数代码", "startDate": "开始日期", "endDate": "结束日期"}'
```
获取 A 股指数在指定区间内的行情数据，如区间涨跌幅、成交额等。

```
finx gildata-aidata IndexQuoteTecIndicators --body '{"indexCode": "指数代码", "tradeDate": "交易日期"}'
```
获取指数技术指标，如 MA、MACD、KDJ、BOLL 等分析数据。

### 指数估值分析
```
finx gildata-aidata IndexValueAnalysis --body '{"indexCode": "指数代码", "tradeDate": "交易日期"}'
```
获取指数估值指标，如 PE、PB、股息率、PE 历史百分位等。

### 指数资金流向
```
finx gildata-aidata IndexCapitalFlow --body '{"indexCode": "指数代码", "tradeDate": "交易日期"}'
```
获取指数每日资金流入流出情况。

### 市场整体交易统计
```
finx gildata-aidata StockMarketTradeStats --body '{"tradeDate": "交易日期"}'
```
获取沪深北市场成交量、成交额、成交笔数等交易统计数据。

```
finx gildata-aidata StockMarketSizeStats --body '{"tradeDate": "交易日期"}'
```
获取沪深北市场板块的上市公司数量、股本、市值等规模数据。

```
finx gildata-aidata StockMarketValueAnalysis --body '{"tradeDate": "交易日期"}'
```
获取股票市场估值指标，如 PE、PB、PS、PCF 等。

### 市场资金流向
```
finx gildata-aidata StockMarketCapitalFlow --body '{"tradeDate": "交易日期"}'
```
获取沪深京市场每日资金流入流出的统计数据。

```
finx gildata-aidata MarketFundFlowRank --body '{"tradeDate": "交易日期"}'
```
按资金流向指标对沪深北市场成分股进行排序，支持多维度分析。

### 涨跌停家数统计
```
finx gildata-aidata MarketLimitUpDownCount --body '{"tradeDate": "交易日期"}'
```
获取沪深京市场实时上涨、下跌、涨停、跌停家数统计。

### 两融市场统计
```
finx gildata-aidata MarginTradeStats --body '{"tradeDate": "交易日期"}'
```
获取沪深北市场融资融券余额、买入额、卖出量等统计数据。

### 北向资金（沪深港通）
```
finx gildata-aidata HSGTTradeStats --body '{"tradeDate": "交易日期"}'
```
获取沪深北市场的北向、南向资金成交额、成交笔数等跨境交易统计数据。

### 行业/板块排序与资金流
```
finx gildata-aidata SectorRank --body '{"sectorType": "行业板块", "tradeDate": "交易日期"}'
```
获取板块列表并按涨跌幅、换手率等指标排序，支持多板块类型。

```
finx gildata-aidata SectorFundFlowRank --body '{"sectorType": "行业板块", "tradeDate": "交易日期"}'
```
按资金流向指标对板块进行排序，支持主力、散户资金分析。

### 行业估值
```
finx gildata-aidata IndustryValuation --body '{"industryCode": "行业代码", "tradeDate": "交易日期"}'
```
查询行业估值类指标数据，包含总市值、流通市值、PB、PE 等。

### 宏观与行业经济数据
```
finx gildata-aidata MacroIndustryEDB --body '{"indicatorCode": "指标代码", "startDate": "开始日期", "endDate": "结束日期"}'
```
查询中国宏观、行业经济、国际宏观等经济指标数据（如 M1、M2、社融、PMI 等）。

### 宏观/行业分析观点
```
finx gildata-aidata MacroeconomicAnalysisViewpoints --body '{"limit": 5}'
```
获取研报中对 GDP、CPI、PMI、政策等宏观维度的分析观点。

```
finx gildata-aidata IndustryAnalysisViewpoints --body '{"industry": "行业名称", "limit": 5}'
```
获取研报中对行业趋势、竞争格局、政策等的分析观点。

## 执行流程

1.  **数据采集**:
    *   使用 `IndexDailyQuote` / `IndexRangeQuotation` 获取主要指数（上证、深成、创业板、沪深 300、中证 500、中证 1000）行情数据
    *   使用 `IndexQuoteTecIndicators` 获取指数技术指标（均线、MACD 等）
    *   使用 `IndexValueAnalysis` 获取主要指数估值及历史分位
    *   使用 `HSGTTradeStats` 获取北向资金流向
    *   使用 `MarginTradeStats` 获取两融余额变化
    *   使用 `StockMarketTradeStats` 获取全市场成交额与量能
    *   使用 `MarketLimitUpDownCount` 获取涨跌停家数
    *   使用 `SectorRank` / `SectorFundFlowRank` 获取行业板块涨跌幅与资金流向排序
    *   使用 `MacroIndustryEDB` 获取宏观流动性数据（M1、M2、社融等）
    *   使用 `MacroeconomicAnalysisViewpoints` 获取宏观分析观点

2.  **模型分析与打分**:
    *   运用上述框架对市场进行多维诊断
    *   计算大势评分与风格偏好得分

3.  **逻辑推演**:
    *   结合宏观背景（如美林时钟位置、政策周期）解释当前市场状态
    *   预判未来 1-3 个月的市场走势及风格切换可能性

4.  **报告生成**:
    *   生成 Markdown 格式的策略研判报告，数据需注明来源和截止日期

## 报告模板 (Markdown)

```markdown
# 📈 市场大势与风格研判报告 | [日期]

> **数据来源**: gildata-aidata 专业金融数据服务
> **数据截止**: [具体日期]
> **大势研判**: [🟢 震荡偏强 / 🟡 震荡市 / 🔴 调整]
> **建议仓位**: [XX%]
> **核心风格**: [如：大盘价值占优，成长风格筑底]

---

## 一、 核心观点 (Executive Summary)

*   **大势判断**: [简述当前市场处于什么阶段，核心驱动力或阻力是什么]
*   **风格展望**: [明确当前及未来一段时间的最强风格，如"哑铃策略：红利打底，科技进攻"]
*   **操作建议**: [简述仓位管理和交易节奏，如"逢低布局，切忌追高"]

---

## 二、 大势研判 (Market Trend Analysis)

### 2.1 量价与技术面
*   **指数状态**: [描述主要指数的技术形态，基于 IndexDailyQuote 和 IndexQuoteTecIndicators 数据]
*   **成交量**: [描述量能变化，基于 StockMarketTradeStats 数据，是否支持指数突破]

### 2.2 资金与流动性
*   **宏观流动性**: [基于 MacroIndustryEDB 数据，解读社融、M1 等]
*   **微观资金**:
    *   北向资金：[基于 HSGTTradeStats 数据分析流向]
    *   两融数据：[基于 MarginTradeStats 分析杠杆情绪]
    *   主力资金：[基于 StockMarketCapitalFlow 分析]

### 2.3 情绪与风险
*   **情绪指标**: [基于 MarketLimitUpDownCount 分析涨跌停家数、炸板率]
*   **估值水平**: [基于 IndexValueAnalysis 和 StockMarketValueAnalysis 分析市场整体估值分位]
*   **风险预警**: [提示潜在风险点，如"高位股拥挤度过高"]

---

## 三、 风格强弱排序 (Style Ranking)

| 风格维度 | 强弱排序 (强 -> 弱) | 核心逻辑 |
|----------|---------------------|----------|
| **市值** | 大盘 > 小盘 | [如：流动性边际收敛，确定性溢价上升] |
| **属性** | 价值 > 成长 | [如：利率下行空间有限，红利资产性价比仍存] |
| **综合** | **大盘价值** > 小盘价值 > 大盘成长 > 小盘成长 | [综合排序结论] |

### 3.1 重点风格解析
*   **[最强风格]**: [详细分析为何该风格占优，结合 SectorFundFlowRank 资金数据]
*   **[最弱风格]**: [详细分析为何该风格承压，资金流出，利空因素]

---

## 四、 策略建议与仓位管理 (Strategy & Allocation)

*   **仓位建议**: **[建议仓位区间，如 50%-70%]**
    *   **依据**: [基于大势评分和风险提示]
*   **配置主线**:
    *   **主线 1**: [如：高股息红利（底仓配置）]
    *   **主线 2**: [如：出海/涨价（弹性进攻）]
*   **交易节奏**: [如：急跌买入，冲高减仓]

---

*免责声明：本报告基于 gildata-aidata 公开数据和量化模型生成，仅供参考，不构成投资建议。*
```

## 注意事项

*   **数据源唯一性**: 所有市场大势与风格相关数据必须通过 `gildata-aidata` 服务获取，确保数据权威性和一致性
*   **数据时效性**: 必须使用最新的市场数据进行分析，并在报告中标注数据日期
*   **逻辑自洽**: 大势判断与仓位建议必须匹配（例如：判断为熊市趋势，仓位建议不应过高）
*   **动态调整**: 风格研判需结合宏观环境变化（如降息、政策转向）及时调整
*   **风险提示**: 明确指出可能导致研判失效的极端情况（黑天鹅）
*   **工具调用规范**: 调用 gildata-aidata 工具时，指数代码、行业代码需使用标准代码，日期格式需符合 API 要求
