# 步骤4:Skill 合成与迭代

## 输入
- 步骤1输出的需求分析结果
- 步骤2输出的专家访谈与经验萃取结果
- 步骤3输出的监管政策分析结果
- 现有 pre-visit-plan SKILL.md(v1.0.0)
- METHODOLOGY.md 金融级标准
- skill-refactor 改造操作指南

## 过程
1. 对照 METHODOLOGY.md 的12项核心标准,评估现有 SKILL.md 的差异
2. 重构 Frontmatter(version: 1.0.0 → 2.0.0,增加 status: draft)
3. 完善 description,增加反触发条件("个人信贷访前规划")
4. 将 Workflow 从 7 步扩展为 10 步(增加步骤0数据确认、步骤8报告格式化、步骤9最终验证)
5. 为每个步骤添加门控条件(✅ 通过 / ❌ 不通过 / ⚠️ 部分异常)
6. 增强 Constraints(从7条增加至9条,增加禁止跳过步骤、红线执行强制)
7. 优化 Output Format,增加下游兼容性说明和免责声明要求
8. 更新 Audit Trail 格式(JSON格式,记录操作步骤和数据源)
9. 校验 SKILL.md 行数(确保 < 500行)
10. 创建 provenance/ 目录结构和 PROVENANCE.md
11. 更新测试用例,确保覆盖4种类型(positive/edge_case/negative/rejection)

## 输出
- 新版 SKILL.md(v2.0.0,行数:~470行)
- PROVENANCE.md(生产溯源文件,~150行)
- provenance/ 目录(4个子目录,每个包含 README.md)
- 更新后的 tests/test_cases.yaml(10个用例,覆盖4种类型)

## 日期
2026-05-05

## 负责人
AI架构师团队

## 核心业务逻辑保护清单
以下原有业务规则在改造过程中完整保留:
1. ✅ 客户基本面4维度分析框架
2. ✅ 5种拜访类型分类体系
3. ✅ 拜访策略差异化模板(5种类型 × 4个维度)
4. ✅ 风险前置原则(风险信号必须在报告开头显著标注)
5. ✅ 目标明确原则(每次拜访必须设定1-3个主目标)
6. ✅ 行业分类精确至中类或小类(GB/T 4754-2017)
7. ✅ 历史拜访记录读取要求(最近3次)
8. ✅ 数据脱敏规则(身份证号、银行账号、联系方式)
9. ✅ 3条金融合规红线(R1-R3)
10. ✅ 降级策略(6种场景)

## 关键变更点
1. **增加步骤0**:数据确认与验证,运行 validate_visit_plan.py
2. **增加步骤8**:报告格式化输出,确保模板一致性
3. **增加步骤9**:最终验证与交付,运行验证脚本
4. **增强门控**:所有步骤增加 ✅/❌/⚠️ 门控条件
5. **增强防偷懒**:核心步骤增加"不得跳过"、"必须展示"等指令
6. **增强下游兼容**:Output Format 标注可被 visit-memo 和 submit-credit-application 解析
7. **增强免责声明**:引用 shared/disclaimer-template.md

## 验证结果
- ✅ SKILL.md 行数:~470行(< 500行要求)
- ✅ 目录名使用 kebab-case:pre-visit-plan
- ✅ 主文件名为 SKILL.md
- ✅ 配套目录完整(references/、assets/、scripts/、tests/、provenance/)
- ✅ 10个测试用例覆盖4种类型
