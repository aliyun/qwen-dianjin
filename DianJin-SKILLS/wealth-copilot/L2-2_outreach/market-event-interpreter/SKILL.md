---
name: market-event-interpreter
description: >-
  将市场突发事件翻译成客户听得懂的语言，生成事件简析、
  对持仓影响评估和建议行动话术。
  当用户提到市场大跌怎么说、政策怎么解读、波动话术、热点事件影响、
  突发事件怎么跟客户解释时触发。
  不用于常规行情查询（由market-insight-qa处理），
  不用于生成完整早报（由market-morning-brief处理）。
---

# 市场事件解读

## 可用工具

本技能可调用以下 MCP 数据服务，根据事件类型智能路由：

**盈米金融数据（qieman）**
- 服务地址：`https://dashscope.aliyuncs.com/api/v1/mcps/Qieman/sse`
- 核心能力：基金搜索/诊断、组合分析/回测、资产配置方案、CFP 工具链、图表渲染
- 本技能主要工具：`SearchFinancialNews`, `searchRealtimeAiAnalysis`, `GetLatestQuotations`, `SearchManagerViewpoint`, `RenderEchart`

**恒生聚源金融数据（gildata-aidata）**
- 服务地址：开通恒生聚源 MCP 服务后获取，格式为 `https://dashscope.aliyuncs.com/api/v1/mcps/<your-mcp-id>/mcp`
- 核心能力：个股研究(A/H/US)、财务报表、资金流向、研报舆情、理财产品、宏观数据
- 本技能主要工具：`NewsInfoList`, `PolicyMeetingsList`, `MacroeconomicAnalysisViewpoints`, `SectorRank`, `MarketFundFlowRank`

## 输入要求

### 必填信息
- 事件描述（如"市场大跌""降息""AI板块暴涨"）

### 可选信息（通过上下文注入）
- 客户持仓信息（用于分析对客户的具体影响）
- 客户风险等级

## 执行流程

### 第一步：事件信息采集（按事件类型路由）

**通用采集（qieman）：**
- `SearchFinancialNews`：搜索事件相关资讯
- `searchRealtimeAiAnalysis`：获取AI对事件的实时解读
- `GetLatestQuotations`：获取受影响市场的行情数据
- `SearchManagerViewpoint`：获取机构观点

**按事件类型追加（gildata-aidata）：**

| 事件类型 | 追加工具 | 获取信息 |
|---------|---------|---------|
| 政策/会议类（如"降息""两会""美联储"） | `PolicyMeetingsList`, `MacroeconomicAnalysisViewpoints` | 政策会议决议、宏观经济分析观点 |
| 市场波动类（如"大跌""暴涨""熔断"） | `SectorRank`, `MarketFundFlowRank` | 板块涨跌排行、主力资金流向 |
| 行业/板块类（如"AI板块""新能源"） | `SectorRank`, `MarketFundFlowRank`, `NewsInfoList` | 板块排名、资金动向、行业新闻 |
| 国际事件类（如"关税""地缘冲突"） | `MacroeconomicAnalysisViewpoints`, `NewsInfoList` | 宏观影响分析、国际新闻 |

### 第二步：事件分析
- 事件本身：发生了什么、多大影响
- 数据支撑：受影响板块排名、资金流向变化
- 历史类比：历史上类似事件的市场反应
- 对不同资产的影响：股票/债券/商品

### 第三步：话术生成
- 面向客户的通俗解读（去专业术语）
- 对不同类型客户的差异化话术

## 输出模板

```markdown
## 市场事件解读 | {事件标题}

### 一、发生了什么
{2-3句话简述事件}

### 二、为什么会这样
{原因分析，通俗易懂}

### 三、市场反应
**板块表现**：{涨幅/跌幅前列板块}
**资金流向**：{主力资金动向}

| 资产类别 | 短期影响 | 中期展望 |
|----------|---------|---------|
| A股 | {影响} | {展望} |
| 债市 | {影响} | {展望} |
| 商品 | {影响} | {展望} |

### 四、对我们客户的影响
- **权益类持仓客户**：{影响和建议}
- **固收类持仓客户**：{影响和建议}

### 五、客户沟通话术
> **面对焦虑客户**：
> "{安抚+理性分析话术}"
>
> **面对观望客户**：
> "{机会分析话术}"
>
> **朋友圈版本**：
> "{简短解读，200字以内}"

### 六、建议动作
- 立即：{紧急行动}
- 本周：{跟进计划}

---
*以上解读基于公开信息，仅供参考，不构成投资建议。*
```

## 注意事项

- 时效性：突发事件需快速响应，话术要简洁有力
- 不制造恐慌：大跌时不用"崩盘""暴跌"等情绪化用词
- 不预测底部/顶部：不说"已经见底""还会继续跌"
- 差异化：对不同风险等级客户的话术要有区分
- 数据源路由：政策类优先用 gildata-aidata 的宏观/政策工具，市场波动类补充板块排行和资金流向
