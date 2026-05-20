# 步骤4: 技能综合

## 输入
- 监管合规审查成果
- 规则提取成果
- 结构化设计成果

## 过程
- 整合5大步骤到SKILL.md(步骤0-5)
- 重构Workflow为步骤门控型(B模式),标注执行模态
- 强化8条Constraints(含4条合规红线)
- 添加Audit Trail机制、4条Gotchas、2个Examples、Out of Scope
- 生成PROVENANCE.md生产溯源文件(193行)

## 输出
- 最终版SKILL.md(331行)
- PROVENANCE.md(193行)
- 完整L3资源目录结构
  - references/5个文件(exposure-calculation-guide.md, limit-matrix.md, group-identification-guide.md, mitigation-measures.md, concentration-analysis-guide.md)
  - scripts/2个文件(validate_exposure.py, requirements.txt)
  - tests/1个文件(test_cases.yaml, 13个用例)
  - provenance/4个子目录(每个含README.md)

## 关键变更
1. Frontmatter重构: 添加target_role/business_domain/risk_level/version/status/upstream_skills/downstream_skills/data_sources
2. Data Sources完善: 定义5类数据源、脱敏规则、4种降级策略
3. Workflow标准化: 从"执行步骤"改为6步Workflow(步骤0-5),每步标注执行模态,步骤0含先读后写机制
4. Constraints强化: 从5条扩展到8条,新增4条合规红线
5. Output Format更新: 添加JSON输出格式,包含免责声明引用
6. 新增章节: Audit Trail、Gotchas(4条)、Examples(2个)、Out of Scope

## 质量验证
- SKILL.md行数:331行(<500行要求)
- 测试用例:13个(覆盖4种类型)
- 验证脚本:✅ 通过(4个测试用例运行正常)
- PROVENANCE.md:193行(≥100行要求)
