---
name: market-morning-brief
description: >-
  自动整合行情数据、热点资讯、AI解读和机构观点，生成专业且易读的每日市场早报，
  回答"昨天发生了什么、为什么发生、对我有什么影响"三个核心问题。
  当用户提到今天市场怎么样、生成早报、晨会内容、市场快报、昨天行情、每日简报时触发。
  不用于回答具体单一行情数据问题（如"沪深300现在多少点"），该场景由market-insight-qa处理。
---

# 晨会市场早报

## 可用工具

本技能可调用以下 MCP 数据服务，执行流程中按需选用：

**盈米金融数据（qieman）**
- 服务地址：`https://dashscope.aliyuncs.com/api/v1/mcps/Qieman/sse`
- 核心能力：基金搜索/诊断、组合分析/回测、资产配置方案、CFP 工具链、图表渲染
- 本技能主要工具：`GetCurrentTime`, `GetLatestQuotations`, `SearchFinancialNews`, `SearchHotTopic`, `searchRealtimeAiAnalysis`, `SearchManagerViewpoint`, `RenderEchart`

**恒生聚源金融数据（gildata-aidata）**
- 服务地址：开通恒生聚源 MCP 服务后获取，格式为 `https://dashscope.aliyuncs.com/api/v1/mcps/<your-mcp-id>/mcp`
- 核心能力：个股研究(A/H/US)、财务报表、资金流向、研报舆情、理财产品、宏观数据
- 本技能主要工具：`IndexDailyQuote`, `SectorRank`, `ConceptIndexLiveQuote`, `MarketFundFlowRank`, `MacroeconomicAnalysisViewpoints`

## 核心原则

**图表优先，文字精简。** 行情涨跌数据必须通过 `RenderEchart` 生成柱状图直观呈现，让客户经理一眼看懂市场全貌，文字聚焦解读和影响分析。

## 输入要求

### 必填信息
- 无特殊必填字段，一句话指令即可触发

### 可选信息
- 关注的特定板块或主题
- 输出用途（晨会分享 / 发给客户 / 自己看）

## 执行流程

### 第一步：获取当前时间
- 调用 `GetCurrentTime` 确定当前日期和最近交易日

### 第二步：多维度数据采集（并行调用）

**qieman 数据源：**
- `GetLatestQuotations`：主要指数行情（A股、债市、商品、汇率、海外）
- `SearchFinancialNews`：最新财经资讯
- `SearchHotTopic`：市场热点话题
- `searchRealtimeAiAnalysis`：AI实时解读
- `SearchManagerViewpoint`：基金经理/机构最新观点

**gildata-aidata 数据源：**
- `IndexDailyQuote`：指数详细行情数据（补充qieman覆盖不到的指数）
- `SectorRank`：行业板块涨跌排行（涨幅/跌幅前5板块）
- `ConceptIndexLiveQuote`：概念板块实时行情（热门概念表现）
- `MarketFundFlowRank`：市场资金流向排行（主力资金净流入/流出板块）
- `MacroeconomicAnalysisViewpoints`：宏观经济分析观点

### 第三步：分析与加工
- 将行情数据按涨跌幅排序，标注显著变动（涨跌超2%）
- 从资讯和热点中提取3-5条核心事件
- 整合AI解读与机构观点
- 汇总板块排行与资金流向，识别当日市场主线

### 第四步：生成可视化图表（必须执行，不可跳过）

**必须**调用 `RenderEchart` 生成以下图表：

#### 图表1：主要指数涨跌柱状图（必须生成）

用主要指数的涨跌幅数据生成水平柱状图，红涨绿跌。ECharts option 参考：

```json
{
  "title": { "text": "主要指数涨跌幅", "left": "center", "subtext": "2026-04-25" },
  "tooltip": { "trigger": "axis" },
  "grid": { "left": "28%", "right": "10%" },
  "xAxis": { "type": "value", "axisLabel": { "formatter": "{value}%" } },
  "yAxis": { "type": "category", "data": ["纳斯达克", "标普500", "恒生指数", "黄金", "创业板指", "深证成指", "上证指数", "10Y国债"] },
  "series": [{
    "type": "bar",
    "data": [
      { "value": 1.52, "itemStyle": { "color": "#ee6666" } },
      { "value": 0.74, "itemStyle": { "color": "#ee6666" } },
      { "value": -0.35, "itemStyle": { "color": "#91cc75" } },
      { "value": 0.82, "itemStyle": { "color": "#ee6666" } },
      { "value": 1.21, "itemStyle": { "color": "#ee6666" } },
      { "value": 0.56, "itemStyle": { "color": "#ee6666" } },
      { "value": 0.33, "itemStyle": { "color": "#ee6666" } },
      { "value": -0.02, "itemStyle": { "color": "#91cc75" } }
    ],
    "label": { "show": true, "position": "right", "formatter": "{c}%" },
    "itemStyle": { "borderRadius": [0, 4, 4, 0] }
  }]
}
```

替换为实际行情数据。涨为红色(#ee6666)，跌为绿色(#91cc75)，符合A股涨红跌绿习惯。按涨跌幅从大到小排列。将 subtext 替换为实际日期。

### 第五步：按模板生成早报

## 输出模板

按以下结构输出，**行情图表在最前面，文字精简**：

```markdown
## 每日市场早报 | {日期}

### 一、市场全景

{主要指数涨跌柱状图}

{1-2句总结：整体是涨是跌、幅度如何、哪个最突出}

### 二、板块与资金

**涨幅前5板块**：{板块名称及涨幅}
**跌幅前5板块**：{板块名称及跌幅}
**主力资金流向**：{净流入/流出最大的板块}

### 三、核心要闻
1. **{标题}**：{一句话摘要}
2. **{标题}**：{一句话摘要}
3. **{标题}**：{一句话摘要}

### 四、市场解读
{综合AI解读和机构观点，分析"为什么发生"，2-3段，每段2-3句}

### 五、对我们客户的影响
- **权益类客户**：{影响和建议动作}
- **固收类客户**：{影响和建议动作}
- **待配置客户**：{是否有配置窗口}

### 六、今日话术建议
> {可直接用于朋友圈或客户沟通的1-2句话}

---
*以上内容基于公开市场数据整理，仅供内部参考，不构成投资建议。*
```

## 注意事项

- **图表为必选项**：主要指数涨跌柱状图为必须生成项，不得用文字表格替代
- 数据时效性：明确标注数据截止时间
- 语言风格：专业但不晦涩，客户经理能直接引用
- 合规要求：不做收益预测，不推荐具体产品
- 文字精简：全文控制在600-1000字（不含图表），适合5分钟快速阅读
- 异常处理：如遇非交易日，提示并提供上一交易日数据
