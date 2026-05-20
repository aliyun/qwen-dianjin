# 步骤 2：方法论研究与资料萃取

## 输入
- 步骤 1 的分析报告
- METHODOLOGY.md 金融级标准
- 原 SKILL.md 中的业务规则

## 过程
1. 对照 METHODOLOGY.md 十二部分标准，识别原 SKILL.md 缺失项：
   - 缺失：Frontmatter 扩展字段（target_role/business_domain/risk_level/version/data_sources/upstream/downstream）
   - 缺失：Target Role 结构化章节
   - 缺失：Data Sources 矩阵（含脱敏规则和降级策略）
   - 缺失：Terminology 术语消歧
   - 缺失：Workflow 门控路径（✅/❌/⚠️）
   - 缺失：Output Format 结构化字段规范
   - 缺失：Constraints 编号红线（原为散列描述）
   - 缺失：Audit Trail 审计日志
   - 缺失：Gotchas 踩坑记录
   - 缺失：Out of Scope 非功能范围
2. 研究 CoT 推理领域的监管文件和行业基准：
   - 发改委《产业结构调整指导目录》
   - 银保监会产能过剩行业信贷风险通知
   - Wind 金融终端行业基准数据
   - CAMELS 框架（金融企业专用）
3. 设计 4 份 references/ 文件：
   - cot-signal-library.md：推理信号类型库
   - industry-benchmarks.md：行业基准数据
   - policy-regulatory-framework.md：政策监管框架
   - financial-stress-test-params.md：压力测试参数

## 输出
- 现状差异分析报告（10 项缺失）
- 4 份 references/ 文件
- 知识源清单（4 个来源）

## 日期
2026-05-06

## 负责人
AI 架构师
