---
name: portfolio-alert-narrator
description: >-
  监测持仓基金异动信号（净值大幅波动、分红公告、基金经理变更），
  生成客户可理解的解读和应对建议。
  当用户提到基金跌了怎么办、净值异动、基金分红了、
  基金经理换了、基金有什么公告时触发。
  不用于系统性市场事件解读（由market-event-interpreter处理）。
---

# 持仓异动解读

## 可用工具

本技能可调用以下 MCP 数据服务，执行流程中按需选用：

**盈米金融数据（qieman）**
- 服务地址：`https://dashscope.aliyuncs.com/api/v1/mcps/Qieman/sse`
- 核心能力：基金搜索/诊断、组合分析/回测、资产配置方案、CFP 工具链、图表渲染
- 本技能主要工具：`BatchGetFundNavHistory`, `GetFundAnnouncements`, `GetAnnouncementContent`, `BatchGetFundsDetail`, `getBondFundWithAlertRecord`, `getFundDiveCount`

**恒生聚源金融数据（gildata-aidata）**
- 服务地址：开通恒生聚源 MCP 服务后获取，格式为 `https://dashscope.aliyuncs.com/api/v1/mcps/<your-mcp-id>/mcp`
- 核心能力：个股研究(A/H/US)、财务报表、资金流向、研报舆情、理财产品、宏观数据
- 本技能可选工具：`FundAnnouncement`, `StockNewslist`

## 输入要求

### 必填信息
- 基金名称或代码
- 异动类型或用户关注的情况

### 可选信息（通过上下文注入）
- 客户持有金额
- 客户风险等级

## 执行流程

### 第一步：异动信息采集

**qieman 数据源：**
- 调用 `BatchGetFundNavHistory` 获取近期净值走势，识别异常波动
- 调用 `GetFundAnnouncements` 查询近期公告
- 如有异动公告，调用 `GetAnnouncementContent` 获取公告详情
- 调用 `BatchGetFundsDetail` 获取基金最新信息
- 调用 `getBondFundWithAlertRecord` 检查债券基金异动告警（如为债基）
- 调用 `getFundDiveCount` 获取跳水/异动次数

**gildata-aidata 数据源（异动原因补充，可选）：**
- 调用 `FundAnnouncement` 获取更全面的基金公告信息
- 调用 `StockNewslist` 获取重仓股相关新闻，辅助分析净值异动是否与重仓股事件有关

### 第二步：异动分析
- 判断异动类型：
  - 净值大跌（>2%单日跌幅）
  - 净值大涨（>3%单日涨幅）
  - 分红公告
  - 基金经理变更
  - 规模大幅变动
  - 暂停申购/赎回
- 分析原因：市场系统性 or 基金个别因素（结合重仓股新闻判断）

### 第三步：生成解读和话术

## 输出模板

```markdown
## 持仓异动解读 | {基金名称}({代码})

### 异动摘要
| 项目 | 内容 |
|------|------|
| 异动类型 | {类型} |
| 发生日期 | {日期} |
| 异动详情 | {描述} |

### 原因分析
{分析异动原因，区分系统性和个别性}

### 对客户的影响
- **持仓影响**：{金额变动估算}
- **后续预判**：{短期和中期展望}

### 客户沟通话术
> **如客户主动问起**：
> "{被动应答话术}"
>
> **如需主动告知**：
> "{主动告知话术}"

### 建议行动
- {建议1}
- {建议2}

---
*解读基于公开信息，仅供参考，不构成投资建议。*
```

## 注意事项

- 及时性：异动解读需要快速响应
- 不恐慌：即使是负面异动，也要理性分析
- 区分系统性和个别性：市场整体下跌 vs 基金个股问题
- 有预案：针对客户可能的反应准备话术
