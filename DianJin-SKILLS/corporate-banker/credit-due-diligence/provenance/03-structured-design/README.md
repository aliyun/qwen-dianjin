# 步骤3：结构化设计

## 输入
- 步骤2输出的规则清单和阈值表
- 现有 credit-due-diligence SKILL.md（v1.0.0）
- METHODOLOGY.md 金融级标准
- skill-refactor 改造操作指南

## 过程
1. 对照 METHODOLOGY.md 的12项核心标准，评估现有 SKILL.md 的差异
2. 重构 Frontmatter，增加 upstream_skills/downstream_skills 编排字段
3. 将 Workflow 从9步扩展为10步（增加步骤0：数据确认与验证）
4. 为每个步骤添加门控条件（✅ 通过 / ❌ 不通过 / ⚠️ 部分异常）
5. 增强 Output Format 的结构化程度，确保下游 Skill 可解析
6. 创建验证脚本 scripts/validate_due_diligence.py
7. 设计测试用例覆盖正例、反例、边界案例（tests/test_cases.yaml）

## 输出
- 新版 SKILL.md（v2.0.0，符合金融级标准）
- 验证脚本 scripts/validate_due_diligence.py
- 测试用例集 tests/test_cases.yaml（12个用例）
- 输出模板 assets/due-diligence-report-template.md（优化版）

## 日期
2026-05-05

## 负责人
AI架构师团队

## 设计原则
1. **步骤门控型交互模式**：每个关键步骤必须标注门控路径
2. **先读后写**：Workflow 开头增加数据确认步骤（步骤0）
3. **对抗模型偷懒**：核心步骤增加防偷懒指令
4. **编排兼容性**：输出格式结构化，下游 Skill 可可靠解析
5. **渐进式披露**：SKILL.md 控制在500行以内，详细内容移至 L3 资源

## 关键变更
1. 版本号从 1.0.0 升级为 2.0.0（Major版本，流程重构）
2. 增加步骤0（数据确认与验证），确保"先读后写"
3. 所有9个步骤均添加完整的门控条件
4. Output Format 从自然语言段落改为表格/结构化格式
5. Constraints 从9条扩展至11条（增加"禁止跳过步骤"和"一票否决执行"）
6. 增加编排字段（upstream_skills/downstream_skills）
