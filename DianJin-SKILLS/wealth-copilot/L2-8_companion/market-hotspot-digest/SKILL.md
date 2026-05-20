---
name: market-hotspot-digest
description: >-
  将当日市场热点翻译成通俗语言，生成可直接转发给客户的简短解读卡片。
  面向"给客户发内容"场景。
  当用户提到今天热点、帮我解读热点、给客户发点内容、
  市场热点速递、最近值得关注的事时触发。
  不用于生成完整早报（由market-morning-brief处理），
  不用于突发事件应对话术（由market-event-interpreter处理）。
---

# 市场热点速递

## 可用工具

本技能可调用以下 MCP 数据服务，执行流程中按需选用：

**盈米金融数据（qieman）**
- 服务地址：`https://dashscope.aliyuncs.com/api/v1/mcps/Qieman/sse`
- 核心能力：基金搜索/诊断、组合分析/回测、资产配置方案、CFP 工具链、图表渲染
- 本技能主要工具：`SearchHotTopic`, `SearchFinancialNews`, `searchRealtimeAiAnalysis`, `GetLatestQuotations`

**恒生聚源金融数据（gildata-aidata）**
- 服务地址：开通恒生聚源 MCP 服务后获取，格式为 `https://dashscope.aliyuncs.com/api/v1/mcps/<your-mcp-id>/mcp`
- 核心能力：个股研究(A/H/US)、财务报表、资金流向、研报舆情、理财产品、宏观数据
- 本技能主要工具：`NewsInfoList`, `IndustryNewsFlash`, `ConceptIndexLiveQuote`, `SectorRank`

## 输入要求

### 必填信息
- 无特殊必填，一句话触发即可

### 可选信息
- 关注的特定领域
- 输出数量（默认3条）

## 执行流程

### 第一步：热点采集

**qieman 数据源：**
- `SearchHotTopic`：获取当日热门话题
- `SearchFinancialNews`：获取最新财经资讯
- `searchRealtimeAiAnalysis`：获取AI实时解读
- `GetLatestQuotations`：获取相关行情数据

**gildata-aidata 数据源（扩大覆盖面）：**
- `NewsInfoList`：获取最新金融新闻资讯列表
- `IndustryNewsFlash`：获取行业快讯，捕捉细分行业动态
- `ConceptIndexLiveQuote`：获取概念板块实时行情（如AI、新能源、芯片等热门概念）
- `SectorRank`：获取行业板块涨跌排行，辅助判断热点方向

### 第二步：筛选和翻译
- 综合两个数据源的信息，筛选3-5条最有价值的热点
- 将专业内容翻译成客户听得懂的语言
- 每条控制在100字以内

### 第三步：生成可转发卡片

## 输出模板

```markdown
## 市场热点速递 | {日期}

### 热点一：{标题}
{通俗解读，100字以内}
> 跟客户聊的时候可以说："{一句话版本}"

---

### 热点二：{标题}
{通俗解读}
> 跟客户聊的时候可以说："{一句话版本}"

---

### 热点三：{标题}
{通俗解读}
> 跟客户聊的时候可以说："{一句话版本}"

---
*热点速递基于公开资讯整理，仅供交流参考。*
```

## 注意事项

- 通俗易懂：目标是"客户能看懂"，去除所有行业黑话
- 简短精炼：每条热点100字以内
- 可转发：内容格式适合直接发给客户
- 区分于早报：早报面向客户经理自己，热点速递面向客户
