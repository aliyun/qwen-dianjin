# 步骤4：Skill 合成与迭代

## 输入
- 步骤1输出的分析框架设计
- 步骤2输出的专家访谈与经验萃取结果
- 步骤3输出的监管政策分析结果
- 现有 credit-industry-analysis SKILL.md（v1.0.0）
- METHODOLOGY.md 金融级标准
- skill-refactor 改造操作指南

## 过程
1. 对照 METHODOLOGY.md 的12项核心标准，评估现有 SKILL.md 的差异
2. 重构 Frontmatter（version: 1.0.0 → 2.0.0，增加 status: draft）
3. 将 Workflow 从 8 步扩展为 10 步（增加步骤0数据确认和步骤9报告输出）
4. 为每个步骤添加门控条件（✅ 通过 / ❌ 不通过 / ⚠️ 部分异常）
5. 增强 Constraints（从7条增加至9条，增加禁止跳过步骤、红线执行强制）
6. 优化 Output Format，增加下游兼容性说明和免责声明要求
7. 创建验证脚本 scripts/validate_industry.py
8. 设计测试用例覆盖正例、反例、边界案例（tests/test_cases.yaml，12个用例）
9. 校验 SKILL.md 行数（确保 < 500行）

## 输出
- 新版 SKILL.md（v2.0.0，~490行）
- 验证脚本 scripts/validate_industry.py
- 测试用例集 tests/test_cases.yaml（12个用例，覆盖4种类型）
- 输出模板 assets/industry-analysis-template.md（已优化）

## 日期
2026-05-05

## 负责人
AI架构师团队

## 核心逻辑保护清单
改造过程中严格保留以下原有业务规则：
1. ✅ 6条行业红线（I1-I6）完整保留
2. ✅ 波特五力定量评估框架完整保留
3. ✅ 行业生命周期判定标准（导入期/成长期/成熟期/衰退期）完整保留
4. ✅ 市场规模测算方法（TAM/SAM/SOM）完整保留
5. ✅ 信贷场景专属关注点（6个维度）完整保留
6. ✅ 5条踩坑记录（Gotchas）完整保留并优化
7. ✅ 数据脱敏规则（5条）完整保留
8. ✅ 降级策略（5条）完整保留

## 关键变更
1. Workflow 增加步骤0"先读后写"机制
2. 所有步骤增加门控条件（✅/❌/⚠️）
3. 增加防偷懒指令（"不得跳过任何测算步骤"、"必须给出 CR3/CR5 及 HHI 量化指标"）
4. Constraints 增加2条（禁止跳过步骤、红线执行强制）
5. Output Format 增加下游兼容性说明和免责声明要求
6. Frontmatter 增加 status 字段和完善 data_sources
