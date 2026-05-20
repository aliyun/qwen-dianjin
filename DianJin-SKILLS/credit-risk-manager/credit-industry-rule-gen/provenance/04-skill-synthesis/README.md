# 步骤4: 技能综合

## 输入
- 框架设计成果
- 专家访谈与案例萃取成果
- 监管合规分析成果

## 过程
- 整合5大分析维度到SKILL.md
- 重构Workflow为7步(步骤0-6),标注执行模态
- 强化8条Constraints(含4条合规红线)
- 添加Audit Trail机制、4条Gotchas、2个Examples、Out of Scope
- 生成PROVENANCE.md生产溯源文件(161行)

## 输出
- 最终版SKILL.md(290行)
- PROVENANCE.md(161行)
- 完整L3资源目录结构
  - references/3个文件(industry-lifecycle-guide.md, fraud-pattern-catalog.md, regulatory-requirements.md)
  - scripts/2个文件(validate_rule_output.py, requirements.txt)
  - tests/1个文件(test_cases.yaml, 13个用例)
  - provenance/4个子目录(每个含README.md)

## 关键变更
1. Frontmatter重构: 添加target_role/business_domain/risk_level/version/status/upstream_skills/downstream_skills/data_sources
2. Data Sources完善: 定义5类数据源、脱敏规则、4种降级策略
3. Workflow标准化: 从"执行步骤"改为7步Workflow,步骤0含先读后写机制
4. Constraints强化: 从6条扩展到8条,新增4条合规红线
5. 新增章节: Audit Trail、Gotchas(4条)、Examples(2个)、Out of Scope
6. 创建验证脚本scripts/validate_rule_output.py(346行)
7. 创建测试用例tests/test_cases.yaml(13个用例,覆盖4种类型)

## 验证结果
- SKILL.md行数: 290行(<500行要求) ✅
- 验证脚本测试: 4个测试用例全部通过 ✅
- 测试用例覆盖: 13个用例(positive 3/edge_case 3/negative 3/rejection 4) ✅
- PROVENANCE.md行数: 161行(≥100行要求) ✅
