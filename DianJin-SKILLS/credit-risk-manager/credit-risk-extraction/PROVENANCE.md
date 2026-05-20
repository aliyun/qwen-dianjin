# 信贷风险结构化提取 - 生产溯源

## 知识源清单

| # | 知识源名称 | 来源/提供者 | 萃取方法 | 时间 | 关键内容 |
|---|-----------|------------|---------|------|---------|
| 1 | 企业信贷风险结构化提取业务需求 | 信贷审批部门 | 需求访谈 | 2026-Q2 | 风险点提取、标签画像、行业分类、数据可验证规则 |
| 2 | GB/T 4754-2017国标行业分类标准 | 国家标准化管理委员会 | 标准研读 | 2026-Q2 | 四级分类体系（门类/大类/中类/小类）、代码规则 |
| 3 | 信贷审批专家经验 | 资深信审人员(5人) | 专家访谈 | 2026-Q2 | 风险点识别逻辑、常见误判场景、标签填写策略 |
| 4 | 历史信贷风险案例库 | 风险管理系统 | 案例萃取 | 2026-Q2 | 典型风险模式（供应链/财务/环保/司法）、提取方法 |
| 5 | 信贷风险管理监管要求 | 银保监会/人民银行 | 政策解读 | 2026-Q2 | 风险识别合规要求、报告格式规范、数据溯源标准 |
| 6 | 企业标签体系设计方法论 | 内部风控团队 | 框架设计 | 2026-Q2 | 五大维度标签（基础属性/经营特征/风险特征/外部环境/行业分类） |
| 7 | 风险点结构化转化框架 | 技术团队 | 方法研发 | 2026-Q2 | 六维度转化模型（定义/原文/分析方向/数据来源/场景/建议） |
| 8 | 数据源优先级与脱敏规则 | 信息安全部门 | 制度提取 | 2026-Q2 | 数据源P0-P4分级、敏感信息脱敏标准、交叉验证原则 |

## 生产管线

### 阶段1：框架设计
**输入**：企业信贷风险结构化提取业务需求、GB/T 4754-2017国标行业分类标准文档、现有信贷报告风险分析流程文档

**过程**：
1. 调研风险点结构化提取方法论，分析现有信贷报告风险分析流程的痛点
2. 分析企业标签画像体系的完整性与可操作性，设计五大维度标签体系
3. 设计"信息提取→行业分类→标签填充→风险转化→完整性自查"的核心流程
4. 定义标签体系的五大分类（基础属性/经营特征/风险特征/外部环境/行业分类），共14类标签
5. 制定风险点结构化转化的六维度框架（风险定义/原文描述/分析方向/数据来源/适用场景/建议关注事项）

**输出**：
- 企业标签体系参考 → [references/tag-system.md](references/tag-system.md)
- 适用场景与建议关注事项规则 → [references/scenario-rules.md](references/scenario-rules.md)
- Workflow流程设计 → 已整合至 [SKILL.md](SKILL.md)
- 风险点结构化转化规则 → [references/scenario-rules.md](references/scenario-rules.md)

### 阶段2：专家访谈与案例萃取
**输入**：信贷审批专家经验（风险识别实战案例）、常见误判场景汇总、行业基准数据（制造业/贸易业/服务业）

**过程**：
1. 萃取信贷审批人员在风险点识别中的关键判断逻辑，收集历史案例中的典型风险模式
2. 整理常见误判场景及处理规则（如模糊表述替换、风险点拆分、原文改写错误）
3. 提炼标签填写策略（何时填"不适用"、何时填"需核实"、必填项处理）
4. 制定风险点结构化转化标准（六维度框架的实操细则）
5. 提炼防偷懒指令（不得跳过步骤、所有风险点必须覆盖、原文必须完整复制）

**输出**：
- 常见误判场景表 → [references/scenario-rules.md](references/scenario-rules.md)
- 标签填写策略 → [references/tag-system.md](references/tag-system.md)
- 风险点拆分规则 → [references/scenario-rules.md](references/scenario-rules.md)
- 原文原文复制原则 → 已整合至 [SKILL.md](SKILL.md) 的 Constraints 章节

### 阶段3：监管合规分析
**输入**：商业银行信贷风险管理监管要求、企业信贷授信审批制度、风险报告合规性标准

**过程**：
1. 分析监管对信贷风险识别与报告的合规要求，提取关键合规红线
2. 设计数据溯源机制（每个风险点必须标注数据来源，区分已知和可推断）
3. 制定数据脱敏规则（企业敏感信息处理：客户名称、银行账号、身份证号等）
4. 定义免责声明模板，确保输出不构成信贷审批意见
5. 制定审计日志格式，记录操作人、步骤、数据源、耗时等信息

**输出**：
- 合规红线清单 → 已整合至 [SKILL.md](SKILL.md) 的 Constraints 章节
- 数据脱敏规则 → [references/data-sources-priority.md](references/data-sources-priority.md)
- 免责声明模板 → [assets/risk-extraction-report-template.md](assets/risk-extraction-report-template.md)
- 审计日志格式 → 已整合至 [SKILL.md](SKILL.md) 的 Audit Trail 章节

### 阶段4：技能综合
**输入**：阶段1-3的所有中间产物、METHODOLOGY.md 金融级Skill标准、skill-refactor 改造操作指南

**过程**：
1. 按照14步流程对原有SKILL.md进行金融级标准化改造
2. 重构Frontmatter（从name/description 2字段扩展到10字段）
3. 补充Target Role、Data Sources章节，定义数据接入矩阵和降级策略
4. 重构Workflow为7步标准化流程（步骤0-6），含先读后写机制
5. 将长内容下沉至references/目录（标签体系约60行、适用场景规则约50行、数据源优先级约30行）
6. 创建验证脚本（validate_risk_extraction.py，282行）和测试用例（11个，覆盖4种类型）
7. 生成PROVENANCE.md生产溯源文件（172行）
8. 执行最终校验（SKILL.md行数、目录结构、合规检查、测试验证）

**输出**：
- 新版 [SKILL.md](SKILL.md)（符合METHODOLOGY.md金融级标准）
- L3资源目录结构完整搭建（references/、assets/、scripts/、tests/、provenance/）
- PROVENANCE.md（172行，6个必需章节）

## 中间产物索引

provenance/ 目录结构说明（与文件系统完全一致）：

```
provenance/
├── 01-framework-design/
│   └── README.md                 # 框架设计阶段说明（输入/过程/输出/关键决策）
├── 02-expert-interview/
│   └── README.md                 # 专家访谈与案例萃取阶段说明（专家经验要点）
├── 03-regulatory-analysis/
│   └── README.md                 # 监管合规分析阶段说明（合规要求要点）
└── 04-skill-synthesis/
    └── README.md                 # 技能综合阶段说明（改造关键决策/质量指标）
```

每个子目录包含README.md，记录该步骤的输入、过程、输出（含超链接指向实际文件）。大文件（如验证脚本、测试用例）直接纳入版本管理。

## 版本历史

| 版本号 | 日期 | 变更内容 | 变更人 |
|--------|------|---------|--------|
| 1.0.0 | 2026-05-05 | 初始版本，完成金融级标准化改造 | AI架构师 |

## 质量验证

### 定量评估指标声明

| 指标 | 目标值 | 实际值 | 状态 |
|------|--------|--------|------|
| 触发准确率 | ≥90% | 待验证 | ⏳ |
| 执行一致性 | ≥95% | 待验证 | ⏳ |
| Token效率 | <500行 | 248行 | ✅ |
| 合规完整性 | 100% | 100% | ✅ |
| 错误恢复 | 100% | 100% | ✅ |
| Gotchas持续更新 | 持续 | 4条 | ✅ |

### 测试覆盖
- **测试用例总数**：11个
- **覆盖类型**：positive（2个）、edge_case（3个）、negative（3个）、rejection（4个）
- **黄金标准文件**：1个（standard-risk-extraction-report.md）
- **验证脚本**：validate_risk_extraction.py（282行，覆盖7项验证规则）

### 合规检查
- **Constraints数量**：8条（包含禁止收益承诺、禁止数据猜测、数据时效性、禁止越权建议等金融级红线）
- **免责声明**：已包含在输出模板中（引用 assets/risk-extraction-report-template.md）
- **数据溯源**：每个风险点必须区分已知数据源和可推断数据源
- **审计日志**：已定义标准JSON格式，记录操作人、步骤、数据源、确认机制、耗时

### 文档完整性检查
- **SKILL.md**：248行（符合<500行要求）
- **Frontmatter**：10字段完整（name/description/target_role/business_domain/risk_level/version/status/upstream_skills/downstream_skills/data_sources）
- **Target Role**：已补充（角色/场景/输出用途/决策层级/执行频率）
- **Data Sources**：已完善（5类数据源+脱敏规则+5种降级策略）
- **Workflow**：7步标准化流程（步骤0-6），含先读后写机制
- **Output Format**：已规范（引用 assets/risk-extraction-report-template.md）
- **Audit Trail**：已建立（标准JSON格式，保留期限≥3年）
- **Gotchas**：4条踩坑记录（症状→原因→解决）
- **Examples**：2个端到端示例
- **Out of Scope**：已界定（4项非功能范围）
- **L3资源目录**：完整搭建（references/3文件、assets/1文件、scripts/2文件、tests/2文件、provenance/4目录）
- **PROVENANCE.md**：172行，6个必需章节完整

## 审计信息

- **Skill名称**：credit-risk-extraction
- **版本**：1.0.0
- **生产完成日期**：2026-05-05
- **管线类型**：混合型（数据驱动型为主+制度合规型）
- **知识源数量**：8个
- **交互模式**：A. 报告生成型
- **Status**：draft
