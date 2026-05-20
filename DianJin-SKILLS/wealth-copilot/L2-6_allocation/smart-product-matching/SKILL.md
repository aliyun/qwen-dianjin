---
name: smart-product-matching
description: >-
  从客户配置缺口出发筛选匹配产品，经尽调验证后输出推荐方案和
  "为什么适合您"的销售话术。支持基金、理财产品、债券等多品类匹配。
  当用户提到推荐产品、该买什么、产品匹配、有什么好产品、推荐基金、理财产品推荐时触发。
  不用于已持有产品的诊断分析（由portfolio-health-check处理），
  不用于资产配置方案设计（由asset-allocation-optimizer处理）。
---

# 产品智能匹配

## 可用工具

本技能可调用以下 MCP 数据服务，执行流程中按需选用：

**盈米金融数据（qieman）**
- 服务地址：`https://dashscope.aliyuncs.com/api/v1/mcps/Qieman/sse`
- 核心能力：基金搜索/诊断、组合分析/回测、资产配置方案、CFP 工具链、图表渲染
- 本技能主要工具：`SearchFunds`, `GetPopularFund`, `BatchGetFundsDetail`, `GetBatchFundPerformance`, `AnalyzeFundRisk`, `GetFundDiagnosis`, `RenderEchart`

**恒生聚源金融数据（gildata-aidata）**
- 服务地址：开通恒生聚源 MCP 服务后获取，格式为 `https://dashscope.aliyuncs.com/api/v1/mcps/<your-mcp-id>/mcp`
- 核心能力：个股研究(A/H/US)、财务报表、资金流向、研报舆情、理财产品、宏观数据
- 本技能主要工具：`FundMultipleFactorFilter`, `FinancialProductFilter`, `ProductBasicInfoList`, `CreditBondBaseInfo`

## 核心原则

**图表优先，文字精简。** 推荐产品的业绩对比数据必须通过 `RenderEchart` 生成可视化图表，让客户经理一眼看出产品优劣，文字聚焦匹配理由和话术。

## 输入要求

### 必填信息
- 客户风险等级（R1-R5）
- 需求描述（产品类型/配置缺口/投资金额，至少一项）

### 可选信息
- 当前持仓概况（避免推荐重复或高相关的产品）
- AUM、客户层级
- 排除条件（如"不要封闭式""不要新基金"）
- 产品品类偏好（基金/理财产品/债券）

如果用户仅说"推荐几只基金"，追问风险等级和产品类型需求。

## 执行流程

### 第一步：需求解析
- 确定筛选维度：产品类型、风险约束、规模要求、流动性需求
- 确定产品品类：基金 / 理财产品 / 债券 / 综合匹配

### 第二步：产品筛选（按品类路由）

**基金类需求（qieman + gildata-aidata）：**
- `SearchFunds`（qieman）：按条件搜索基金
- `GetPopularFund`（qieman）：获取热门基金作为补充候选
- `FundMultipleFactorFilter`（gildata-aidata）：多因子筛选，从收益、风险、风格等多维度精准筛选基金

**理财产品需求（gildata-aidata）：**
- `FinancialProductFilter`：按风险等级、期限、收益率等条件筛选银行理财产品
- `ProductBasicInfoList`：获取理财产品基本信息（发行机构、期限、业绩比较基准等）

**债券类需求（gildata-aidata）：**
- `CreditBondBaseInfo`：获取信用债基本信息（评级、票面利率、到期日等）

### 第三步：候选产品尽调验证（3-5只）

**基金类验证（qieman）：**
- `BatchGetFundsDetail`：详细信息
- `GetBatchFundPerformance`：业绩数据
- `AnalyzeFundRisk`：风险评估
- `GetFundDiagnosis`：诊断信息

**理财产品类验证（gildata-aidata）：**
- `ProductBasicInfoList`：产品详情、历史业绩、费率结构

### 第四步：与当前持仓去重（如有持仓数据）
- 检查推荐产品与当前持仓的重叠度

### 第五步：生成可视化图表（必须执行，不可跳过）

**必须**调用 `RenderEchart` 生成以下图表：

#### 图表1：推荐产品业绩对比柱状图（必须生成）

用候选产品的近1年收益率生成柱状图，直观展示各产品表现。ECharts option 参考：

```json
{
  "title": { "text": "推荐产品业绩对比", "left": "center" },
  "tooltip": { "trigger": "axis" },
  "grid": { "left": "30%", "right": "10%" },
  "xAxis": { "type": "value", "axisLabel": { "formatter": "{value}%" } },
  "yAxis": { "type": "category", "data": ["广发纯债A", "招商中证白酒", "易方达蓝筹精选"] },
  "series": [{
    "type": "bar",
    "data": [3.5, 12.8, 18.2],
    "label": { "show": true, "position": "right", "formatter": "{c}%" },
    "itemStyle": { "borderRadius": [0, 4, 4, 0], "color": "#5470c6" }
  }]
}
```

替换为实际推荐产品名称和收益率。按收益率从小到大排列（最高的在顶部）。

#### 图表2：推荐产品多维对比雷达图（推荐生成，如有≥2只产品）

将推荐的产品在收益、风险、规模、费率等维度做1-5分评分，生成雷达图。ECharts option 参考：

```json
{
  "title": { "text": "推荐产品综合评分", "left": "center" },
  "legend": { "top": "bottom" },
  "radar": {
    "indicator": [
      { "name": "收益", "max": 5 },
      { "name": "风险控制", "max": 5 },
      { "name": "规模稳定", "max": 5 },
      { "name": "费率", "max": 5 },
      { "name": "经理能力", "max": 5 }
    ],
    "shape": "circle"
  },
  "series": [{
    "type": "radar",
    "data": [
      { "value": [4, 3, 4, 3, 5], "name": "推荐一", "areaStyle": { "opacity": 0.2 } },
      { "value": [3, 5, 4, 4, 3], "name": "推荐二", "areaStyle": { "opacity": 0.2 } },
      { "value": [5, 2, 3, 3, 4], "name": "推荐三", "areaStyle": { "opacity": 0.2 } }
    ]
  }]
}
```

替换为实际产品名称和评分。

### 第六步：生成推荐方案和话术

## 输出模板

按以下结构输出，**图表嵌入对应章节，每只产品的推荐理由精简**：

```markdown
## 产品智能匹配

### 匹配条件
- **风险等级**：{R等级}
- **产品类型**：{类型}
- **配置目的**：{描述}

### 业绩一览

{推荐产品业绩对比柱状图}

### 综合评分

{推荐产品多维对比雷达图}

### 推荐详情

#### 推荐一：{产品名称}({代码})
| 类型 | 规模/期限 | 经理/发行方 | 近1年/业绩基准 | 最大回撤 | 风险等级 |
|------|----------|-----------|--------------|---------|---------|
| {类型} | {X}亿/{X}月 | {姓名/机构} | {X}% | {X}% | {R等级} |

**为什么适合**：{1-2句}
> **话术**："{推荐话术}"

#### 推荐二：{产品名称}({代码})
{同上格式}

#### 推荐三：{产品名称}({代码})
{同上格式}

---
*产品推荐基于量化筛选和历史数据，不构成投资建议。基金有风险，投资需谨慎。*
```

## 注意事项

- **图表为必选项**：推荐产品业绩对比柱状图为必须生成项
- 合规红线：不使用"推荐买入"，使用"供参考""可以关注"
- 适当性匹配：推荐产品风险等级不得高于客户风险等级
- 客观呈现：每只产品都呈现优势和风险点
- 推荐数量：一般推荐3只（给选择空间但不造成选择困难）
- 文字精简：全文控制在800-1200字（不含图表）
- 品类标注：明确标注每只推荐产品的品类（基金/理财产品/债券），方便客户经理区分
