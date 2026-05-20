# 阶段4：技能综合

## 输入
- 阶段1-3的所有中间产物
- METHODOLOGY.md 金融级Skill标准
- skill-refactor 改造操作指南

## 过程
1. 按照14步流程对原有SKILL.md进行金融级标准化改造
2. 重构Frontmatter（从2字段扩展到10字段）
3. 补充Target Role、Data Sources章节
4. 重构Workflow为7步标准化流程（步骤0-6），含先读后写机制
5. 将长内容下沉至references/目录（欺诈模式、交叉验证矩阵、行业基准、置信度规则）
6. 创建验证脚本（validate_fraud_report.py）和测试用例（12个，覆盖4种类型）
7. 生成PROVENANCE.md生产溯源文件
8. 执行最终校验（SKILL.md行数、目录结构、合规检查）

## 输出
- 新版 `SKILL.md`（符合METHODOLOGY.md金融级标准）
- L3资源目录结构完整搭建：
  - `references/`：4个参考文件（欺诈模式、交叉验证矩阵、数据源优先级、置信度规则）
  - `assets/`：1个报告模板文件
  - `scripts/`：1个验证脚本 + requirements.txt
  - `tests/`：test_cases.yaml（12个测试用例）+ golden/（1个黄金标准文件）
  - `provenance/`：4个阶段子目录，每个含README.md
- `PROVENANCE.md`：完整生产溯源文件

## 质量验证
- **SKILL.md行数**：控制在500行以内（渐进式披露原则）
- **测试覆盖**：12个测试用例，覆盖positive(2)/edge_case(3)/negative(3)/rejection(4)
- **合规检查**：8条Constraints（含禁止收益承诺、禁止数据猜测、禁止越权建议等）
- **管线类型适配**：provenance/目录命名符合混合型管线模板（01-framework-design/02-expert-interview/03-regulatory-analysis/04-skill-synthesis/）
- **数据保鲜**：所有references/文件标注data_valid_until
