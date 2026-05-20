---
name: product-knowledge-qa
description: >-
  快速解答产品费率、交易规则、分红规则、申赎限制、业绩基准等
  日常产品知识问题，支持基金、理财产品、债券等多品类。
  当用户提到费率多少、什么时候到账、分红方式、怎么买、
  交易规则、起购金额、能不能定投时触发。
  不用于生成完整的产品卖点话术（由hot-product-briefing处理）。
---

# 产品知识问答

## 可用工具

本技能可调用以下 MCP 数据服务，根据产品类型智能路由：

**盈米金融数据（qieman）**
- 服务地址：`https://dashscope.aliyuncs.com/api/v1/mcps/Qieman/sse`
- 核心能力：基金搜索/诊断、组合分析/回测、资产配置方案、CFP 工具链、图表渲染
- 本技能主要工具：`SearchFunds`, `GuessFundCode`, `BatchGetFundsDetail`, `BatchGetFundTradeRules`, `BatchGetFundTradeLimit`, `BatchGetFundsDividendRecord`, `getFundBenchmarkInfo`

**恒生聚源金融数据（gildata-aidata）**
- 服务地址：开通恒生聚源 MCP 服务后获取，格式为 `https://dashscope.aliyuncs.com/api/v1/mcps/<your-mcp-id>/mcp`
- 核心能力：个股研究(A/H/US)、财务报表、资金流向、研报舆情、理财产品、宏观数据
- 本技能可选工具：`ProductBasicInfoList`, `CreditBondBaseInfo`, `FundManagerInfoReport`

## 输入要求

### 必填信息
- 产品名称或代码（基金/理财产品/债券）
- 具体问题（费率/交易规则/分红/限制等）

如果用户仅问"这个产品怎么买"但未指定产品，追问产品名称或代码。

## 执行流程

### 第一步：确认产品并识别问题类型
- 确认产品代码和品类（基金/理财产品/债券）
- 判断问题类型：费率/交易规则/分红/限制/业绩基准/经理信息/其他

### 第二步：精准数据获取

**基金类问题（qieman）：**
- 费率问题：`BatchGetFundTradeRules` 获取费率详情
- 交易限制：`BatchGetFundTradeLimit` 获取申赎限制
- 分红记录：`BatchGetFundsDividendRecord` 获取历史分红
- 业绩基准：`getFundBenchmarkInfo` 获取基准信息
- 综合问题：`BatchGetFundsDetail` 获取完整信息

**理财产品问题（gildata-aidata）：**
- `ProductBasicInfoList`：获取理财产品详情（发行机构、期限、收益类型、风险等级、费率等）

**债券问题（gildata-aidata）：**
- `CreditBondBaseInfo`：获取信用债基本信息（评级、票面利率、到期日、发行主体等）

**经理相关问题（gildata-aidata）：**
- `FundManagerInfoReport`：获取基金经理详细信息（背景、任职记录、管理规模）

### 第三步：精炼回答
- 直接回答问题，不展开不相关的信息
- 数据精确到具体数值
- 如有特殊注意事项，附带提醒

## 输出模板

```markdown
### {产品名称}({代码}) - {问题类别}

{精确回答，3-5行}

| 项目 | 详情 |
|------|------|
| {相关字段1} | {值} |
| {相关字段2} | {值} |

> 提示：{需要注意的事项}

---
*数据来源于产品公开信息，具体以产品合同/说明书为准。*
```

## 注意事项

- 快速精准：问什么答什么，不做过多延伸
- 数据准确：费率等数据必须精确
- 补充提醒：对可能遗漏的注意事项主动提示
- 输出控制：简短回答控制在200-500字
- 品类识别：自动判断产品是基金、理财产品还是债券，调用对应数据源
