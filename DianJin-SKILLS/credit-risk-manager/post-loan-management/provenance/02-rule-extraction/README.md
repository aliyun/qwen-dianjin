# 步骤 2：规则提取与结构化

## 输入
- 步骤 1 提取的监管要求清单
- 原 SKILL.md 中的业务规则（检查频率矩阵、资金用途核查、财务指标阈值、担保核查标准、预警指标体系、P1-P6 一票否决条件）

## 过程
1. 从原 SKILL.md 中提取所有业务硬规则，列入"核心逻辑保护清单"（共 14 项）
2. 将规则按业务领域分类：
   - 检查频率规则 → references/check-frequency-policy.md
   - 资金用途规则 → references/fund-usage-policy.md
   - 财务预警规则 → references/financial-warning-thresholds.md
   - 担保有效性规则 → references/collateral-policy.md
   - 五级分类规则 → references/five-classification-policy.md
   - 预警指标规则 → references/early-warning-indicators.md
   - 一票否决规则 → references/p1-p6-veto-conditions.md
   - 处置上报规则 → references/disposal-escalation-policy.md
3. 每条规则标注数据有效期和监管文件编号
4. 确保规则提取过程中不修改原有阈值和判断标准

## 输出
- 8 份结构化 policy 文件（存于 references/）
- 核心逻辑保护清单核对报告（14 项全部保留）
- 详见：[规则提取记录](rule-extraction-log.md)（待创建）

## 日期
2026-05-06

## 负责人
AI 架构师 / 风控能力中心
