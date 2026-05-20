---
name: wechat-moments-creator
description: >-
  为理财经理创作专业且有温度的企微/朋友圈营销内容，
  支持市场点评、产品解读、投教科普、节日关怀四种风格。
  当用户提到朋友圈文案、企微内容、私域运营、营销内容、发条朋友圈时触发。
  不用于电话触达话术（由outreach-script-generator处理）。
---

# 朋友圈内容创作

## 可用工具

本技能可调用以下 MCP 数据服务，执行流程中按需选用：

**盈米金融数据（qieman）**
- 服务地址：`https://dashscope.aliyuncs.com/api/v1/mcps/Qieman/sse`
- 核心能力：基金搜索/诊断、组合分析/回测、资产配置方案、CFP 工具链、图表渲染
- 本技能主要工具：`GetLatestQuotations`, `SearchFinancialNews`, `SearchHotTopic`, `BatchGetFundsDetail`, `GetBatchFundPerformance`

**恒生聚源金融数据（gildata-aidata）**
- 服务地址：开通恒生聚源 MCP 服务后获取，格式为 `https://dashscope.aliyuncs.com/api/v1/mcps/<your-mcp-id>/mcp`
- 核心能力：个股研究(A/H/US)、财务报表、资金流向、研报舆情、理财产品、宏观数据
- 本技能可选工具：`NewsInfoList`, `ConceptIndexLiveQuote`

## 输入要求

### 必填信息
- 内容方向（以下至少一种）：
  - 市场点评：基于今日行情
  - 产品解读：指定产品或类型
  - 投教科普：指定话题
  - 节日关怀：指定节日/场景

### 可选信息
- 风格偏好（专业/轻松/温暖）
- 目标客群（年轻白领/中老年/高净值）
- 是否需要配图建议

## 执行流程

### 第一步：确定内容类型和素材
- 识别用户需要的内容方向

### 第二步：数据采集

**qieman 数据源：**
- 市场点评类：调用 `GetLatestQuotations` + `SearchFinancialNews`
- 产品解读类：调用 `BatchGetFundsDetail` + `GetBatchFundPerformance`
- 补充热点：调用 `SearchHotTopic` 获取热门话题

**gildata-aidata 数据源（文案素材补充，可选）：**
- 调用 `NewsInfoList` 获取最新金融新闻，提供文案时效性素材
- 调用 `ConceptIndexLiveQuote` 获取热门概念板块行情（如AI、新能源），丰富市场点评类文案

- 投教科普类：基于内置金融知识
- 节日关怀类：不需要MCP数据

### 第三步：内容创作
- 控制在200字以内（朋友圈阅读习惯）
- 首句吸引注意力
- 专业但不晦涩
- 留互动钩子（引导客户私聊）
- 生成3条备选

### 第四步：输出内容

## 输出模板

```markdown
## 朋友圈文案 | {内容类型}

### 文案一（{风格标签}）
> {文案内容，200字以内}

适合发布时间：{建议时间}

---

### 文案二（{风格标签}）
> {文案内容}

适合发布时间：{建议时间}

---

### 文案三（{风格标签}）
> {文案内容}

适合发布时间：{建议时间}

### 配图建议
- {配图描述或方向}

---
*内容仅供参考，请根据个人风格调整。发布前请确认符合机构合规要求。*
```

## 注意事项

- 合规要求：不在朋友圈中推荐具体产品、不承诺收益
- 长度控制：每条200字以内
- 人格化：避免AI味过重，要有"人"的温度
- 多样性：3条文案风格要有差异，不要雷同
