---
name: fund-deep-research
description: >-
  对单只基金进行全面尽调分析，按基金类型（股票型/债券型/混合型/指数型）自动选择分析路径，
  输出业绩、风险、持仓、基金经理能力四维诊断报告。
  当用户提到分析基金、基金怎么样、能不能买、基金诊断、基金尽调、这个基金经理靠谱吗时触发。
  不用于多只基金的组合持仓分析（由portfolio-health-check处理）。
---

# 基金深度尽调

## 可用工具

本技能可调用以下 MCP 数据服务，执行流程中按需选用：

**盈米金融数据（qieman）**
- 服务地址：`https://dashscope.aliyuncs.com/api/v1/mcps/Qieman/sse`
- 核心能力：基金搜索/诊断、组合分析/回测、资产配置方案、CFP 工具链、图表渲染
- 本技能主要工具：`SearchFunds`, `GuessFundCode`, `BatchGetFundsDetail`, `GetBatchFundPerformance`, `AnalyzeFundRisk`, `GetFundDiagnosis`, `BatchGetFundsHolding`, `getFundIndustryAllocation`, `getFundIndustryConcentration`, `getFundTurnoverRate`, `getFundBrinsonIndicator`, `getMarketTimingIndicator`, `getBondAllocationByFundCode`, `getBondFundCreditRatingLevel`, `getBondIndicator`, `getFundCampisiIndicator`, `RenderEchart`

**恒生聚源金融数据（gildata-aidata）**
- 服务地址：开通恒生聚源 MCP 服务后获取，格式为 `https://dashscope.aliyuncs.com/api/v1/mcps/<your-mcp-id>/mcp`
- 核心能力：个股研究(A/H/US)、财务报表、资金流向、研报舆情、理财产品、宏观数据
- 本技能主要工具：`FundManagerInfoReport`, `FundManagerImageReport`, `ManagerProductsIncome`, `CompanyBasicInfo`, `ConsensusExpectation`, `FundAnnouncement`

## 核心原则

**图表优先，文字精简。** 业绩对比、行业配置等定量数据必须通过 `RenderEchart` 生成可视化图表呈现，文字仅用于解读关键洞察。

## 输入要求

### 必填信息
- 基金代码或基金名称（至少提供一个）

### 可选信息
- 分析侧重点（如"重点看风险""关注经理能力"）
- 对比基准或同类基金
- 客户风险等级（影响适配性评价）

如果用户提供基金名称但未提供代码，调用 `GuessFundCode` 或 `SearchFunds` 匹配。

## 执行流程

### 第一步：确定基金代码与类型
- 通过 `SearchFunds` 或 `GuessFundCode` 确认基金代码
- 调用 `BatchGetFundsDetail` 获取基金基本信息，确定基金类型

### 第二步：全维度数据采集（根据基金类型调整）

**通用数据（所有类型，qieman）：**
- `BatchGetFundsDetail`：基本概况、规模、基准、风险等级、经理信息
- `GetBatchFundPerformance`：各阶段收益、业绩分析指标
- `AnalyzeFundRisk`：风险评分、R方、残差方差
- `GetFundDiagnosis`：综合诊断、估值、盈利概率

**股票型/混合型追加（qieman）：**
- `BatchGetFundsHolding`：十大重仓股
- `getFundIndustryAllocation`：行业配置分布
- `getFundIndustryConcentration`：行业集中度
- `getFundTurnoverRate`：换手率
- `getFundBrinsonIndicator`：Brinson收益归因（配置/选股/交互）
- `getMarketTimingIndicator`：择时能力

**债券型追加（qieman）：**
- `getBondAllocationByFundCode`：券种配置和风格
- `getBondFundCreditRatingLevel`：信用评级分布
- `getBondIndicator`：久期、杠杆、集中度
- `getFundCampisiIndicator`：Campisi收益归因

### 第三步：基金经理深度画像（gildata-aidata）

- `FundManagerInfoReport`：经理从业背景、管理规模、任职年限等详细信息
- `FundManagerImageReport`：经理投资风格画像（成长/价值/均衡、大盘/小盘偏好）
- `ManagerProductsIncome`：经理管理的全部产品业绩一览，评估其跨产品一致性

### 第四步：重仓股基本面增强（股票型/混合型，gildata-aidata）

对十大重仓股中排名靠前的个股（选取前3-5只），补充：
- `CompanyBasicInfo`：公司基本面信息（行业地位、主营业务）
- `ConsensusExpectation`：市场一致预期（盈利预测、目标价、评级分布）

### 第五步：近期公告检索（gildata-aidata）

- `FundAnnouncement`：获取该基金近期重要公告（分红、拆分、经理变更、限购等）

### 第六步：四维诊断分析

- **业绩维度**：各阶段收益、同类排名、业绩持续性
- **风险维度**：最大回撤、波动率、夏普/卡玛比率、风险评分
- **持仓维度**（股票型/混合型）：重仓股集中度、行业分布、换手率、重仓股基本面质量
- **经理能力维度**：选股能力（Brinson）、择时胜率、任职年限、投资风格画像、跨产品管理能力

### 第七步：生成可视化图表（必须执行，不可跳过）

数据采集完成后，**必须**调用 `RenderEchart` 生成以下图表：

#### 图表1：各阶段业绩对比柱状图（必须生成）

用该基金和同类平均/基准的各阶段收益率，生成分组柱状图。ECharts option 参考：

```json
{
  "title": { "text": "业绩表现 vs 同类平均", "left": "center" },
  "tooltip": { "trigger": "axis" },
  "legend": { "top": "bottom" },
  "xAxis": { "type": "category", "data": ["近1月", "近3月", "近6月", "近1年", "近3年"] },
  "yAxis": { "type": "value", "axisLabel": { "formatter": "{value}%" } },
  "series": [
    { "name": "该基金", "type": "bar", "data": [2.1, 5.3, 8.7, 15.2, 42.1], "itemStyle": { "borderRadius": [4,4,0,0] } },
    { "name": "同类平均", "type": "bar", "data": [1.5, 3.8, 6.2, 10.1, 28.5], "itemStyle": { "borderRadius": [4,4,0,0] } }
  ]
}
```

将 data 替换为实际收益率数据。

#### 图表2：行业/券种配置饼图（必须生成）

- **股票型/混合型**：用 `getFundIndustryAllocation` 的行业配置数据生成饼图
- **债券型**：用 `getBondAllocationByFundCode` 的券种配置数据生成饼图

ECharts option 参考：

```json
{
  "title": { "text": "行业配置分布", "left": "center" },
  "tooltip": { "trigger": "item", "formatter": "{b}: {d}%" },
  "legend": { "orient": "vertical", "left": "left", "top": "middle" },
  "series": [{
    "type": "pie",
    "radius": ["35%", "65%"],
    "center": ["58%", "50%"],
    "itemStyle": { "borderRadius": 6 },
    "label": { "formatter": "{b}\n{d}%" },
    "data": [
      { "name": "食品饮料", "value": 25.3 },
      { "name": "医药生物", "value": 18.1 },
      { "name": "电子", "value": 12.5 },
      { "name": "银行", "value": 10.2 },
      { "name": "其他", "value": 33.9 }
    ]
  }]
}
```

将 data 替换为实际行业/券种数据。债券型基金改 title 为"券种配置分布"。

#### 图表3：风险收益四维雷达图（必须生成）

用四维诊断评分数据生成雷达图。ECharts option 参考：

```json
{
  "title": { "text": "四维诊断评分", "left": "center" },
  "radar": {
    "indicator": [
      { "name": "收益能力", "max": 100 },
      { "name": "风险控制", "max": 100 },
      { "name": "持仓质量", "max": 100 },
      { "name": "经理能力", "max": 100 }
    ],
    "shape": "circle"
  },
  "series": [{
    "type": "radar",
    "data": [{
      "value": [78, 65, 72, 80],
      "name": "综合评分",
      "areaStyle": { "opacity": 0.3 }
    }]
  }]
}
```

根据业绩排名、风险评分、持仓集中度/行业分散度、经理选股择时能力等数据，将各维度归一化为0-100分后填入。

### 第八步：适配性评估
- 如果用户提供了客户风险等级，评估该基金是否适配
- 给出"适合什么类型的投资者"结论

## 输出模板

按以下结构输出，**图表嵌入对应章节，文字每章节控制在2-4句话**：

```markdown
## 基金深度尽调 | {基金名称}({基金代码})

### 一、基本信息
| 项目 | 详情 |
|------|------|
| 基金类型 | {类型} |
| 成立日期 | {日期} |
| 最新规模 | {X}亿 |
| 基金经理 | {姓名}（任职{X}年） |
| 风险等级 | {R等级} |

### 二、四维诊断

{四维诊断雷达图}

### 三、业绩表现

{业绩对比柱状图}

{2-3句解读：各阶段表现特征、同类排名水平}

### 四、持仓分析

{行业/券种配置饼图}

{2-3句解读：集中度、行业偏好、风格特征}

**重仓股基本面速览**（股票型/混合型）：
| 重仓股 | 行业 | 市场一致预期 |
|--------|------|------------|
| {股票名} | {行业} | {盈利预测/评级} |

### 五、基金经理画像
- **投资风格**：{成长/价值/均衡}，{大盘/小盘偏好}
- **跨产品表现**：管理{X}只产品，整体业绩{评价}
- **选股能力**：Brinson 归因 {评价}
- **择时胜率**：{X}%

### 六、风险评估
- **最大回撤**：{X}%（{评价}）
- **夏普比率**：{X}（{评价}）
- **风险评分**：{X}/100

### 七、近期动态
{近期重要公告摘要，如无重大公告可省略}

### 八、综合结论
**尽调评级：{推荐/中性/谨慎}**
- 优势：{1-2点}
- 风险点：{1-2点}
- 适合：{适配客户类型}

---
*尽调报告基于历史数据分析，不构成投资建议。基金有风险，投资需谨慎。*
```

## 注意事项

- **图表为必选项**：业绩对比柱状图、配置饼图、四维雷达图为必须生成项
- 合规底线：不得出现"推荐买入""建议加仓"等直接投资建议用语
- 客观中立：优势和风险点都要提及，不做单方面美化
- 类型适配：根据基金类型（股/债/混合/QDII）自动调整分析框架
- 文字精简：全文控制在800-1200字（不含图表），每章节不超过4句话
- 数据源分工：基金维度数据用 qieman，经理画像和重仓股基本面用 gildata-aidata
