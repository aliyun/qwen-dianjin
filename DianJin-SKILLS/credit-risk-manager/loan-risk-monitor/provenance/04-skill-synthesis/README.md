# 步骤4:技能综合

## 输入
- 框架设计成果
- 专家访谈与案例萃取成果
- 监管合规分析成果

## 过程
- 整合五大监测维度到SKILL.md
- 重构Workflow为7步(步骤0-6),标注执行模态
- 强化8条Constraints(含2条合规红线)
- 添加Audit Trail机制、4条Gotchas、2个Examples、Out of Scope
- 生成PROVENANCE.md生产溯源文件

## 输出
- 最终版SKILL.md(256行)
- PROVENANCE.md(139行)
- 完整L3资源目录结构
  - references/4个文件(risk-rating-matrix.md, data-sources-priority.md, search-keywords-template.md, mitigation-measures.md)
  - scripts/2个文件(validate_monitor_report.py, requirements.txt)
  - tests/1个文件(test_cases.yaml, 13个用例)
  - provenance/4个子目录(每个含README.md)

## 关键变更
1. Frontmatter重构:添加target_role/business_domain/risk_level/version/status/upstream_skills/downstream_skills/data_sources
2. Data Sources完善:定义6类数据源、脱敏规则、4种降级策略
3. Workflow标准化:从"执行步骤"改为7步Workflow(步骤0-6),每步标注执行模态,步骤0含先读后写机制
4. Constraints强化:从6条扩展到8条,新增2条合规红线(禁止数据猜测、禁止越权建议)
5. 新增章节:Audit Trail、Gotchas(4条)、Examples(2个)、Out of Scope
6. L3资源创建:references 4个文件、scripts验证脚本、tests 13个用例

## 验证结果
- SKILL.md行数:256行(<500行要求)
- 验证脚本运行通过:3个测试用例全部符合预期
- 管线类型适配:混合型(01-framework-design/02-expert-interview/03-regulatory-analysis/04-skill-synthesis)
