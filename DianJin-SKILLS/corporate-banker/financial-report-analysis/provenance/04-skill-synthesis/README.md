# 步骤4:Skill 合成与迭代

## 输入
- 步骤1输出的方法论设计
- 步骤2输出的工具和验证脚本
- 步骤3输出的监管政策分析结果
- 现有 financial-report-analysis SKILL.md(v1.0.0)
- METHODOLOGY.md 金融级标准
- skill-refactor 改造操作指南

## 过程
1. 对照 METHODOLOGY.md 的12项核心标准,评估现有 SKILL.md 的差异
2. 重构 Frontmatter(version: 1.0.0 → 2.0.0,增加 status: draft)
3. 完善 description,增加反触发条件("个人信贷财务分析")
4. 将 Workflow 从 8 步扩展为 10 步(增加步骤0数据确认和步骤8勾稽验证)
5. 为每个步骤添加门控条件(✅ 通过 / ❌ 不通过 / ⚠️ 部分异常)
6. 增强 Constraints(从7条增加至9条,增加禁止跳过步骤、红线执行强制)
7. 优化 Output Format,增加下游兼容性说明和免责声明要求
8. 更新 Audit Trail 格式(JSON格式,记录操作步骤和数据源)
9. 校验 SKILL.md 行数(确保 < 500行)
10. 创建 provenance/ 目录结构和 PROVENANCE.md
11. 更新测试用例,增加 rejection 类型

## 输出
- 新版 SKILL.md(v2.0.0,行数:~480行)
- PROVENANCE.md(生产溯源文件,~180行)
- provenance/ 目录(4个子目录,每个包含 README.md)
- 更新后的 tests/test_cases.yaml(12个用例,覆盖4种类型)

## 日期
2026-05-05

## 负责人
AI架构师团队

## 核心逻辑保护清单
以下业务规则在改造过程中保持不变:
1. 8条金融合规红线(R1-R8)定义和触发条件
2. 财务指标计算公式(ROE、毛利率、净利率、收现比、净现比等)
3. 杜邦分析三因素/五因素拆解方法
4. 信贷场景专属关注点(6个维度)
5. 数据脱敏规则(5条)
6. 降级策略(5种场景)
7. 分析原则(6条)
8. 财务粉饰识别手段(收入端、成本端、利润端、现金流端)
