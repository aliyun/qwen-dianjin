# 步骤 3：数据模型设计

## 输入
- 步骤 2 提取的结构化规则
- METHODOLOGY.md 中的金融级 Skill 标准
- skill-refactor 技能的操作指南

## 过程
1. 判定管线类型：**混合型**（制度合规型为主 + 数据驱动型为辅）
   - 制度合规型特征：核心逻辑源于监管文件，强调红线定义和审批流程
   - 数据驱动型特征：包含大量财务指标计算和数据分析
2. 判定交互模式：**模式 B（步骤门控型）**
   - 6 步流程，每步需验证，存在通过/不通过的分支路径
   - 每个门控步骤标注完整的分支路径（通过/不通过/部分异常）
3. 编排定位分析：
   - 上游依赖：无强制上游 Skill，可独立调用
   - 下游消费：credit-risk-classification、early-warning-disposal
4. 设计 SKILL.md 的金融级结构：
   - YAML Frontmatter：添加 target_role、risk_level、version、data_sources 等扩展字段
   - Target Role：明确角色、场景、输出用途、决策层级
   - Data Sources：构建数据接入矩阵（含脱敏规则、降级策略）
   - Terminology：消歧易混淆术语（逾期天数、流动性、重组贷款等）
   - Workflow：6 步流程改造为门控型，增加先读后写、指令式动词、防偷懒指令
   - Output Format：结构化输出，标注字段类型和下游可解析性
   - Constraints：8 条不可逾越的红线
   - Audit Trail：标准 JSON 审计日志格式
   - Gotchas：4 条实战踩坑记录
   - Examples：2 个端到端案例
   - Out of Scope：5 项非功能范围
5. 设计验证脚本（scripts/）：
   - validate_post_loan_data.py：数据完整性和勾稽关系验证
   - check_fund_usage.py：资金用途合规性检查
   - calculate_financial_ratios.py：财务指标计算与基准对比
   - evaluate_risk_classification.py：风险分类评估
6. 设计测试用例（tests/test_cases.yaml）：9 个用例覆盖 4 种类型

## 输出
- 金融级 SKILL.md 设计方案
- 4 个验证脚本（scripts/）
- 9 个测试用例定义（tests/test_cases.yaml）
- 详见：[数据模型设计记录](data-model-design.md)（待创建）

## 日期
2026-05-06

## 负责人
AI 架构师
