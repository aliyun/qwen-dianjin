---
name: portfolio-risk-radar
description: >-
  对投资组合进行专项风险扫描，识别集中度、相关性、流动性、
  风格漂移等风险因子，输出风险预警和对冲建议。
  当用户提到风险评估、组合风险、风险预警、集中度过高、
  相关性分析、最大回撤时触发。
  不用于全面持仓体检（由portfolio-health-check处理）。
---

# 组合风险雷达

## 可用工具

本技能可调用以下 MCP 数据服务，执行流程中按需选用：

**盈米金融数据（qieman）**
- 服务地址：`https://dashscope.aliyuncs.com/api/v1/mcps/Qieman/sse`
- 核心能力：基金搜索/诊断、组合分析/回测、资产配置方案、CFP 工具链、图表渲染
- 本技能主要工具：`AnalyzePortfolioRisk`, `AnalyzeFundRisk`, `GetFundsCorrelation`, `GetAssetAllocation`, `fund-equity-position`, `fund-recovery-ability`, `RenderEchart`

**恒生聚源金融数据（gildata-aidata）**
- 服务地址：开通恒生聚源 MCP 服务后获取，格式为 `https://dashscope.aliyuncs.com/api/v1/mcps/<your-mcp-id>/mcp`
- 核心能力：个股研究(A/H/US)、财务报表、资金流向、研报舆情、理财产品、宏观数据
- 本技能可选工具：`MarketFundFlowRank`, `IndustryValuation`

## 核心原则

**图表优先，文字精简。** 风险评分、相关性矩阵等定量数据必须通过 `RenderEchart` 生成可视化图表呈现，文字仅用于解读预警要点和给出对冲建议。

## 输入要求

### 必填信息
- 持仓基金列表：基金代码 + 持有金额/权重

### 可选信息
- 客户风险等级（评估风险是否超标）
- 关注的具体风险维度

## 执行流程

### 第一步：解析持仓
- 提取基金代码和权重
- 如仅有金额，计算相对权重

### 第二步：风险维度扫描（尽量并行）

**qieman 数据源：**
- `AnalyzePortfolioRisk`：组合风险指标（风险评分、R方、残差方差）
- `AnalyzeFundRisk`：各基金风险评分
- `GetFundsCorrelation`：基金间相关性
- `GetAssetAllocation`：资产配置分析（含雷达图评分）
- `fund-equity-position`：权益仓位
- `fund-recovery-ability`：回撤修复能力

**gildata-aidata 数据源（估值风险补充，可选）：**
- `IndustryValuation`：获取持仓重点行业的估值百分位（PE/PB），判断是否处于高估区间
- `MarketFundFlowRank`：查看持仓相关行业的资金流向，判断是否存在资金撤离风险

### 第三步：风险诊断
- **集中度风险**：单一基金/行业/风格占比是否过高
- **相关性风险**：基金间相关系数是否过高（>0.8预警）
- **流动性风险**：是否存在大额赎回限制
- **风格漂移风险**：基金实际风格是否偏离声明
- **回撤风险**：组合最大回撤水平与客户承受力匹配度
- **估值风险**（如有 gildata-aidata 数据）：重点行业估值是否处于历史高位

### 第四步：生成可视化图表（必须执行，不可跳过）

**必须**依次调用 `RenderEchart` 生成以下图表：

#### 图表1：风险五维雷达图（必须生成）

用五个风险维度的等级评分（高=1/中=3/低=5）生成雷达图。ECharts option 参考：

```json
{
  "title": { "text": "组合风险雷达", "left": "center" },
  "radar": {
    "indicator": [
      { "name": "集中度", "max": 5 },
      { "name": "相关性", "max": 5 },
      { "name": "流动性", "max": 5 },
      { "name": "回撤控制", "max": 5 },
      { "name": "风格稳定", "max": 5 }
    ],
    "shape": "circle"
  },
  "series": [{
    "type": "radar",
    "data": [{
      "value": [2, 3, 4, 2, 5],
      "name": "风险评估",
      "areaStyle": { "opacity": 0.3 },
      "lineStyle": { "color": "#ee6666" }
    }]
  }]
}
```

分值含义：5=安全（低风险），3=关注（中风险），1=预警（高风险）。替换为实际评分。

#### 图表2：各基金风险评分对比柱状图（必须生成）

用 `AnalyzeFundRisk` 返回的各基金风险评分，生成柱状图。ECharts option 参考：

```json
{
  "title": { "text": "各基金风险评分", "left": "center" },
  "tooltip": { "trigger": "axis" },
  "grid": { "left": "25%", "right": "10%" },
  "xAxis": { "type": "value", "max": 100 },
  "yAxis": { "type": "category", "data": ["基金A", "基金B", "基金C"] },
  "visualMap": { "show": false, "min": 0, "max": 100, "inRange": { "color": ["#91cc75", "#fac858", "#ee6666"] } },
  "series": [{
    "type": "bar",
    "data": [35, 58, 82],
    "label": { "show": true, "position": "right", "formatter": "{c}分" },
    "itemStyle": { "borderRadius": [0, 4, 4, 0] }
  }]
}
```

风险评分越高越危险，通过 visualMap 颜色渐变（绿→黄→红）直观显示。

#### 图表3：相关性热力图（如基金≥3只则必须生成）

用 `GetFundsCorrelation` 返回的相关系数矩阵生成热力图。ECharts option 参考：

```json
{
  "title": { "text": "基金相关性矩阵", "left": "center" },
  "tooltip": { "formatter": "{c}" },
  "xAxis": { "type": "category", "data": ["基金A", "基金B", "基金C"], "axisLabel": { "rotate": 30 } },
  "yAxis": { "type": "category", "data": ["基金A", "基金B", "基金C"] },
  "visualMap": { "min": -1, "max": 1, "orient": "horizontal", "left": "center", "bottom": 0, "inRange": { "color": ["#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"] } },
  "series": [{
    "type": "heatmap",
    "data": [[0,0,1],[0,1,0.85],[0,2,0.32],[1,0,0.85],[1,1,1],[1,2,0.45],[2,0,0.32],[2,1,0.45],[2,2,1]],
    "label": { "show": true, "formatter": "{c}" }
  }]
}
```

data 格式为 [x索引, y索引, 相关系数]。相关系数>0.8标红预警。

### 第五步：输出风险报告

## 输出模板

按以下结构输出，**图表嵌入对应章节，文字精简聚焦预警和建议**：

```markdown
## 组合风险雷达

### 一、风险总览

{风险五维雷达图}

综合风险等级：**{高/中/低}**，{一句话总评}。

| 风险维度 | 等级 | 简评 |
|----------|------|------|
| 集中度 | {高/中/低} | {一句话} |
| 相关性 | {高/中/低} | {一句话} |
| 流动性 | {高/中/低} | {一句话} |
| 回撤控制 | {高/中/低} | {一句话} |
| 风格稳定 | {高/中/低} | {一句话} |

### 二、个基风险

{各基金风险评分对比柱状图}

{1-2句解读：哪只基金风险最高、是否拉高组合整体风险}

### 三、持仓相关性

{相关性热力图}

{1-2句解读：是否存在高度重叠、分散化程度}

### 四、风险预警与对冲建议

1. **{风险1}**：{一句话描述} → 建议：{具体操作}
2. **{风险2}**：{一句话描述} → 建议：{具体操作}
3. **{风险3}**：{一句话描述} → 建议：{具体操作}

---
*风险评估基于历史数据和量化模型，不构成投资建议。市场有风险，投资需谨慎。*
```

## 注意事项

- **图表为必选项**：风险雷达图和基金风险对比柱状图为必须生成项
- 区分于全面诊断：本Skill聚焦风险维度，不做收益评价
- 不制造恐慌：客观陈述风险，不使用耸人听闻的表达
- 可操作性：每个风险点都要给出具体的对冲建议
- 文字精简：全文控制在600-1000字（不含图表）
