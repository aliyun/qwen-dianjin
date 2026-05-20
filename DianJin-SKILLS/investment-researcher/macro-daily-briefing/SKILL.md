---
name: macro-daily-briefing
description: 证券宏观研究员日报生成技能。当用户要求生成宏观日报、宏观经济简报、每日宏观速评时使用。本技能自动抓取最新宏观数据、政策动态、市场舆情，生成包含核心观点、数据跟踪、政策解读、资产表现及策略展望的标准化 Markdown 宏观日报。
---

# 证券宏观研究员日报生成技能

## 概述

本技能旨在模拟专业证券宏观研究员的日常工作流，通过自动化抓取和分析最新的宏观经济数据、政策动态、市场舆情及大类资产表现，生成一份结构化、逻辑严密、观点鲜明的**宏观日报（Macro Daily Briefing）**。报告风格贴近卖方宏观首席分析师，注重“边际变化”、“预期差”及“投资启示”。

## 数据获取策略

### 核心数据源：gildata-aidata

所有国内宏观数据、政策、舆情、资金面及行情数据均优先通过 `gildata-aidata` 服务获取。

| 数据需求 | 推荐工具 | 说明 |
|---------|---------|------|
| **宏观数据** | `MacroIndustryEDB` | 查询中国宏观、行业经济、国际宏观等经济指标（PMI, CPI, PPI, 社融等） |
| **政策会议** | `PolicyMeetingsList` | 国内经济金融领域政策会议动态及内容 |
| **官员讲话** | `OfficialSpeechEventList` | 国内外重要官员讲话及活动信息 |
| **法律法规** | `LawsRegulations` | 法律法规信息，按类型、关键词筛选 |
| **宏观舆情** | `MacroNewslist` | 宏观舆情，按区域、情感类型筛选经济预测及事件 |
| **宏观分析观点** | `MacroeconomicAnalysisViewpoints` | 研报中对 GDP、CPI、PMI、政策等宏观维度的分析观点 |
| **资金流向** | `StockMarketCapitalFlow` | 股票市场资金流入流出统计数据 |
| **北向资金** | `HSGTTradeStats` | 沪深港通市场交易统计（北向/南向资金） |
| **两融数据** | `MarginTradeStats` | 融资融券余额、买入额、卖出量等统计数据 |
| **指数行情** | `IndexDailyQuote` | A 股指数日行情（上证、创业板等） |
| **债券行情** | `MostActiveRateBondQuotation` | 利率债最活跃券收益率及涨跌情况 |
| **海外市场** | `web_search` | 补充查询美联储、海外通胀、美债等（Gildata 覆盖不足时） |

### 数据获取优先级

1.  **国内宏观与政策**：优先 `MacroIndustryEDB` + `PolicyMeetingsList` + `OfficialSpeechEventList`
2.  **舆情与观点**：优先 `MacroNewslist` + `MacroeconomicAnalysisViewpoints`
3.  **资金与行情**：优先 `StockMarketCapitalFlow` + `HSGTTradeStats` + `IndexDailyQuote` + `MostActiveRateBondQuotation`
4.  **海外数据**：使用 `web_search` 补充查询（如美联储最新决议、美债收益率等）

## 执行流程

### 第一阶段：数据采集与监控

1.  **宏观数据抓取**
    *   **工具**：`gildata-aidata MacroIndustryEDB`
    *   **查询示例**：
        ```
        调用 gildata-aidata MacroIndustryEDB
        query: "PMI CPI PPI 社融 信贷 M1 M2 进出口 工业增加值 社零 固投"
        ```
    *   **重点关注**：PMI（官方/财新）、CPI/PPI、进出口、社融/信贷、M1/M2、工业增加值、社零、固投。
    *   **高频数据**：高炉开工率、商品房成交面积、猪肉/蔬菜价格、螺纹钢/水泥价格（若 EDB 无高频，可用 `web_search` 补充）。

2.  **政策动态追踪**
    *   **工具**：`gildata-aidata PolicyMeetingsList` + `OfficialSpeechEventList` + `LawsRegulations`
    *   **查询示例**：
        ```
        调用 gildata-aidata PolicyMeetingsList
        查询最新政策会议动态
        
        调用 gildata-aidata OfficialSpeechEventList
        查询央行行长、发改委官员等讲话
        
        调用 gildata-aidata LawsRegulations
        query: "央行 货币政策 OMO MLF LPR 专项债 减税降费"
        ```
    *   **重点关注**：央行（OMO、MLF、LPR、降准降息）、财政部（专项债、特别国债）、国务院常务会议、发改委、证监会。
    *   **海外政策**：使用 `web_search` 查询美联储（FOMC 声明、点阵图、官员讲话）、欧央行等。

3.  **市场舆情与资金面**
    *   **工具**：`gildata-aidata MacroNewslist` + `MacroeconomicAnalysisViewpoints` + `StockMarketCapitalFlow` + `HSGTTradeStats` + `MarginTradeStats`
    *   **查询示例**：
        ```
        调用 gildata-aidata MacroNewslist
        query: "宏观经济 政策 最新舆情"
        
        调用 gildata-aidata MacroeconomicAnalysisViewpoints
        query: "宏观分析观点 最新"
        
        调用 gildata-aidata StockMarketCapitalFlow
        获取今日资金流向统计
        
        调用 gildata-aidata HSGTTradeStats
        获取北向资金数据
        
        调用 gildata-aidata MarginTradeStats
        获取融资融券数据
        ```
    *   **重点关注**：资金流向（北向/南向、ETF、融资余额）、市场情绪（成交额、换手率）、舆情热点。

4.  **大类资产行情**
    *   **工具**：`gildata-aidata IndexDailyQuote` + `MostActiveRateBondQuotation`
    *   **查询示例**：
        ```
        调用 gildata-aidata IndexDailyQuote
        获取上证、创业板、沪深 300 等指数日行情
        
        调用 gildata-aidata MostActiveRateBondQuotation
        获取 10Y 国债等最活跃券收益率
        ```
    *   **补充**：汇率、商品（原油、黄金）行情可用 `web_search` 补充。

### 第二阶段：逻辑分析与研判

4.  **预期差分析**
    *   对比实际数据与 Wind/彭博一致预期，识别超预期（Beat）或不及预期（Miss）的数据点。
    *   分析数据背后的驱动因素（季节性、基数效应、结构性因素）。

5.  **政策意图解读**
    *   解读政策措辞变化（如从“合理充裕”到“精准有力”）。
    *   评估政策落地的可行性及对市场流动性的实际影响。

6.  **资产定价映射**
    *   **股**：分子端（盈利预期）与分母端（无风险利率、风险溢价）的博弈。
    *   **债**：基本面（弱现实）与资金面（宽货币）的共振或背离。
    *   **汇**：中美利差、贸易收支与央行态度的综合反映。
    *   **商**：供需错配、金融属性（美元/利率）与地缘溢价的叠加。

### 第三阶段：报告生成

7.  **生成 Markdown 格式报告**

   使用以下标准模板生成最终报告：

```markdown
# 宏观日报 | [YYYY-MM-DD] [一句话核心观点]

> **分析师**：AI 宏观研究团队  
> **日期**：YYYY-MM-DD  
> **市场情绪**：🟢 乐观 / 🟡 中性 / 🔴 谨慎  
> **核心逻辑**：[简述当日市场交易的核心主线，如“宽信用预期升温”、“海外衰退交易主导”等]

---

## 一、 核心观点与策略建议

**1.1 宏观全景**
（总结当日宏观数据、政策及市场表现的核心特征，指出主要矛盾。例如：“今日公布的 3 月 CPI 低于预期，核心通胀依然疲软，印证了内需修复尚需时日……"）

**1.2 策略展望**
（基于宏观判断，给出大类资产配置建议。例如：“短期维持‘股强债弱’判断，建议超配上游资源品，低配长久期利率债……"）

---

## 二、 宏观数据高频追踪

| 指标 | 最新值 | 前值 | 预期值 | 边际变化 | 信号解读 |
|------|--------|------|--------|----------|----------|
| 官方制造业 PMI | | | | | |
| 财新服务业 PMI | | | | | |
| 30 城商品房成交 | | | | | |
| 猪肉批发价 | | | | | |
| 螺纹钢价格 | | | | | |

**数据点评**：
*   **超预期项**：（分析哪些数据好于预期，驱动了什么逻辑）
*   **不及预期项**：（分析哪些数据低于预期，暴露了什么隐患）
*   **高频信号**：（结合高频数据验证宏观逻辑，如开工率回升验证生产修复）

---

## 三、 政策动态与解读

**3.1 国内政策**
*   **[政策名称/事件]**：（简述政策内容）
    *   **解读**：（分析政策意图、力度、受益方向及潜在风险。例如：“央行开展 5000 亿 MLF 操作，利率维持不变，符合预期，但缩量续作显示央行对资金空转的警惕……"）

**3.2 海外政策**
*   **[美联储/欧央行等]**：（简述最新动态）
    *   **解读**：（分析对美元指数、美债收益率及国内货币政策的溢出效应）

---

## 四、 市场舆情与资金面

**4.1 资金流向**
*   **央行操作**：今日净投放/回笼 [金额] 亿元，资金面 [宽松/平衡/收敛]。
*   **北向资金**：净买入/卖出 [金额] 亿元，主要流向 [板块]。
*   **融资余额**：环比增加/减少 [金额] 亿元，杠杆资金情绪 [高涨/低迷]。

**4.2 舆情热点**
*   **热点一**：[事件描述] -> **市场反应**：[板块异动/情绪变化]
*   **热点二**：[事件描述] -> **机构观点**：[主流券商看法汇总]

---

## 五、 大类资产表现

| 资产类别 | 指标 | 收盘价/最新值 | 涨跌幅 | 核心驱动 |
|---------|------|---------------|--------|----------|
| **权益** | 上证指数 | | | |
| | 创业板指 | | | |
| **债券** | 10Y 国债收益率 | | | |
| | 10Y 美债收益率 | | | |
| **汇率** | 美元兑人民币 (在岸) | | | |
| | 美元指数 | | | |
| **商品** | 布伦特原油 | | | |
| | 现货黄金 | | | |

**资产轮动分析**：
（分析股债商汇的联动关系，如“股债跷跷板效应显现”、“商品领涨反映通胀预期”等）

---

## 六、 风险提示

*   [风险点 1]：如“海外通胀粘性超预期，美联储降息路径推迟”
*   [风险点 2]：如“国内地产销售修复不及预期，拖累信用扩张”
*   [风险点 3]：如“地缘政治冲突升级，能源价格剧烈波动”

---

*免责声明：本报告由 AI 自动生成，基于公开数据与宏观模型分析，仅供参考，不构成任何投资建议。*
```

## 注意事项

*   **时效性优先**：日报必须基于**当日**或**最近 24 小时**的最新数据与新闻。
*   **观点鲜明**：避免模棱两可的陈述，必须给出明确的逻辑推演和方向性判断（看多/看空/震荡）。
*   **预期差思维**：不仅罗列数据，更要强调实际值与市场预期的偏差（Expectation Gap），这是市场交易的核心。
*   **联动视角**：分析单一指标时，必须结合其他资产或宏观背景进行交叉验证（如分析汇率时结合利差和贸易数据）。
*   **专业术语**：使用规范的宏观与金融市场术语（如“宽信用”、“期限利差”、“风险溢价”、“剪刀差”等）。
*   **gildata-aidata 优先**：所有国内宏观数据、政策、资金面、行情查询优先使用 gildata-aidata 服务，web_search 仅用于补充海外政策及高频细节数据。

## 示例调用

用户输入："生成今日宏观经济简报"

技能执行：
1. 获取宏观数据：gildata-aidata MacroIndustryEDB
2. 获取政策会议：gildata-aidata PolicyMeetingsList
3. 获取官员讲话：gildata-aidata OfficialSpeechEventList
4. 获取法律法规：gildata-aidata LawsRegulations
5. 获取宏观舆情：gildata-aidata MacroNewslist + MacroeconomicAnalysisViewpoints
6. 获取资金流向：gildata-aidata StockMarketCapitalFlow + HSGTTradeStats + MarginTradeStats
7. 获取行情数据：gildata-aidata IndexDailyQuote + MostActiveRateBondQuotation
8. 补充海外数据：web_search 查询美联储、美债等
9. 撰写报告：按模板生成 Markdown 格式宏观日报
10. 输出报告：直接在对话中显示完整报告内容

---

## 技能更新日志

**2026-05-11 更新**：
- **核心数据源变更**：将宏观数据查询改为优先使用 gildata-aidata 服务
- **新增工具矩阵**：为宏观日报建立 gildata-aidata 工具查询对照表（MacroIndustryEDB, PolicyMeetingsList, OfficialSpeechEventList, LawsRegulations 等）
- **优化数据获取顺序**：明确 gildata-aidata 优先获取国内宏观、政策、资金、行情，web_search 仅用于补充海外政策细节
- **移除单一 web_search 依赖**：原技能主要依赖 web_search，现全面接入 gildata-aidata 专业宏观与舆情数据库

---

> **技能版本**：v2.0  
> **最后更新**：2026-05-11  
> **维护者**：点金智能助理