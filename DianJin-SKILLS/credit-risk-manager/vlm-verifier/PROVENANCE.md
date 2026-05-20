# 企业信贷反欺诈多模态交叉验证 - 生产溯源

## 知识源清单

| # | 知识源名称 | 来源/提供者 | 萃取方法 | 时间 | 关键内容 |
|---|-----------|------------|---------|------|---------|
| 1 | 企业信贷反欺诈业务需求 | 信贷审批部门 | 需求访谈 | 2026-Q2 | 多模态材料核验、交叉验证规则、欺诈模式识别 |
| 2 | 跨模态数据交叉验证方法论 | 技术团队/VLM研究 | 技术调研 | 2026-Q2 | VLM图像解析能力、LLM文本分析、置信度计算模型 |
| 3 | 信贷审批专家经验 | 资深信审人员(5人) | 专家访谈 | 2026-Q2 | 反欺诈实战案例、常见误判场景、关键判断逻辑 |
| 4 | 历史反欺诈案例库 | 风险管理系统 | 案例萃取 | 2026-Q2 | 典型欺诈模式(财务造假/经营虚假/身份伪造/抵押物异常) |
| 5 | 行业基准数据 | Wind金融终端/内部数据 | 数据收集 | 2026-Q2 | 餐饮/零售/制造业经营指标、装修成本、收入基准 |
| 6 | 商业银行信贷反欺诈监管要求 | 银保监会/人民银行 | 法规研读 | 2026-Q2 | 材料真实性核验标准、证据链完整性要求、合规红线 |
| 7 | 企业贷款材料真实性核验制度 | 内部信贷政策 | 制度分析 | 2026-Q2 | 数据脱敏规则、审计追踪要求、免责声明规范 |
| 8 | METHODOLOGY.md金融级Skill标准 | 架构团队 | 标准对标 | 2026-05 | 12项核心标准、L3资源目录、生产溯源体系 |

## 生产管线

### 管线类型：混合型（数据驱动型为主+制度合规型）

**判定依据**：
- 数据驱动特征：多模态数据分析（VLM视觉识别+LLM文本分析）、交叉验证规则、检测点构建、置信度计算、验证脚本
- 制度合规特征：反欺诈合规审查、证据链要求、监管红线遵循、免责声明、审计追踪

### 步骤1：框架设计 → `provenance/01-framework-design/`

**输入**：企业信贷反欺诈业务需求、多模态技术可行性分析、现有信贷材料核验流程文档

**过程**：
1. 调研跨模态数据交叉验证方法论
2. 分析VLM在图像证据解析中的适用场景与局限性
3. 设计"检测点构建→迭代推理→结论输出"的核心流程
4. 定义欺诈模式识别库的四大分类
5. 制定交叉验证矩阵，明确各验证维度的数据源组合与判断标准

**输出**：
- [欺诈模式识别库初稿](provenance/01-framework-design/README.md) → 已下沉至 `references/fraud-patterns.md`
- [交叉验证矩阵设计](provenance/01-framework-design/README.md) → 已下沉至 `references/cross-validation-matrix.md`
- [Workflow流程设计](provenance/01-framework-design/README.md) → 已整合至 `SKILL.md`
- [置信度评估规则](provenance/01-framework-design/README.md) → 已下沉至 `references/confidence-rules.md`

**关键决策**：
- 交互模式：A. 报告生成型（输入明确，输出完整反欺诈评估报告，单次调用）
- 核心原则：用不同模态的数据验证同一个事实，每个检测点至少涉及2个独立数据源

### 步骤2：专家访谈与案例萃取 → `provenance/02-expert-interview/`

**输入**：信贷审批专家经验、常见误判场景汇总、行业基准数据

**过程**：
1. 萃取信贷审批人员在材料核验中的关键判断逻辑
2. 收集历史案例中的典型欺诈模式与识别方法
3. 整理常见误判场景及处理规则
4. 提炼VLM调用策略（何时用LLM描述判断，何时调用VLM分析原图）
5. 制定置信度三因子计算模型

**输出**：
- [常见误判场景表](provenance/02-expert-interview/README.md) → 已整合至 `references/cross-validation-matrix.md`
- [VLM调用策略](provenance/02-expert-interview/README.md) → 已整合至 `references/confidence-rules.md`
- [行业基准参考值](provenance/02-expert-interview/README.md) → 已整合至 `references/cross-validation-matrix.md`
- [迭代推理机制](provenance/02-expert-interview/README.md) → 已整合至 `references/confidence-rules.md`

### 步骤3：监管合规分析 → `provenance/03-regulatory-analysis/`

**输入**：商业银行信贷反欺诈监管要求、企业贷款材料真实性核验制度、证据链完整性标准

**过程**：
1. 分析监管对信贷材料真实性核验的合规要求
2. 提取反欺诈审查中的关键合规红线
3. 设计证据链可追溯机制
4. 制定数据脱敏规则
5. 定义免责声明模板

**输出**：
- [合规红线清单](provenance/03-regulatory-analysis/README.md) → 已整合至 `SKILL.md` Constraints章节
- [数据脱敏规则](provenance/03-regulatory-analysis/README.md) → 已整合至 `SKILL.md` Data Sources章节
- [免责声明模板](provenance/03-regulatory-analysis/README.md) → 已整合至 `assets/verification-report-template.md`
- [审计日志格式](provenance/03-regulatory-analysis/README.md) → 已整合至 `SKILL.md` Audit Trail章节

### 步骤4：技能综合 → `provenance/04-skill-synthesis/`

**输入**：阶段1-3的所有中间产物、METHODOLOGY.md、skill-refactor改造操作指南

**过程**：
1. 按照14步流程对原有SKILL.md进行金融级标准化改造
2. 重构Frontmatter（从2字段扩展到10字段）
3. 补充Target Role、Data Sources章节
4. 重构Workflow为7步标准化流程（步骤0-6），含先读后写机制
5. 将长内容下沉至references/目录
6. 创建验证脚本和测试用例
7. 生成PROVENANCE.md生产溯源文件
8. 执行最终校验

**输出**：
- 新版 `SKILL.md`（符合METHODOLOGY.md金融级标准）
- 完整L3资源目录结构（references/、assets/、scripts/、tests/、provenance/）
- `PROVENANCE.md`（本文件）

## 中间产物索引

```
provenance/
├── 01-framework-design/
│   └── README.md              # 框架设计阶段：管线类型判定、核心流程设计、欺诈模式分类
├── 02-expert-interview/
│   └── README.md              # 专家访谈阶段：误判场景萃取、VLM策略、置信度模型
├── 03-regulatory-analysis/
│   └── README.md              # 监管合规阶段：合规红线、脱敏规则、免责声明、审计日志
└── 04-skill-synthesis/
    └── README.md              # 技能综合阶段：14步改造、L3目录构建、质量验证
```

**说明**：
- 每个子目录的README.md记录该阶段的输入、过程、输出及关键决策
- 所有中间产物均已整合至最终交付物（SKILL.md、references/、assets/等）
- 大文件（如专家访谈录音、原始案例数据）作为外部对象存储，未纳入版本管理

## 版本历史

| 版本号 | 日期 | 变更内容 | 变更人 |
|--------|------|---------|--------|
| 1.0.0 | 2026-05-05 | 首次生产：按14步流程完成金融级标准化改造，建立完整L3资源目录 | AI架构师 |

## 质量验证

### 定量评估指标

| 维度 | 指标 | 达标标准 | 实际值 |
|------|------|---------|--------|
| 触发准确率 | 相关查询触发率 / 无关查询误触发率 | ≥ 90% / ≤ 5% | 待验证 |
| 执行一致性 | 同一输入多次运行，输出结构稳定率 | ≥ 95% | 待验证 |
| Token 效率 | 主文件行数 / L3资源按需加载率 | < 500 行 / 100% | 待验证（目标<500行） |
| 合规完整性 | 免责声明 + 数据溯源 + 审计日志出现率 | 100% | 100%（已包含） |
| 错误恢复 | 异常输入有明确处理路径（非崩溃）的比例 | 100% | 100%（降级策略已定义） |
| Gotchas 覆盖 | 已知坑的文档覆盖率 | 持续更新 | 4条（初始版本） |

### 测试覆盖

- **测试用例总数**：12个
- **用例类型分布**：
  - positive（正例）：2个（标准餐饮企业、制造业固定资产贷款）
  - edge_case（边界）：3个（材料不完整、常见误判场景、流水账户非主要）
  - negative（反例）：3个（单一数据源、数据超期、矛盾数据）
  - rejection（拒绝）：4个（个人信贷、非反欺诈场景、要求审批意见、要求违约概率）
- **黄金标准文件**：1个（standard-fraud-verification-report.md）
- **验证脚本**：validate_fraud_report.py（验证报告结构、必填字段、免责声明）

### 合规检查

- [x] 禁止收益承诺：Constraints第1条
- [x] 禁止数据猜测：Constraints第3条
- [x] 数据时效性标注：Constraints第4条
- [x] 禁止越权建议：Constraints第5、6条
- [x] 证据链完整：Constraints第2条
- [x] 免责声明：Output Format引用assets/verification-report-template.md
- [x] 审计日志格式：Audit Trail章节定义JSON格式

### 文档完整性检查

- [x] SKILL.md包含全部必需章节（Target Role、Data Sources、Workflow、Output Format、Constraints、Audit Trail、Gotchas、Examples、Out of Scope）
- [x] references/所有文件标注data_valid_until（4个文件）
- [x] provenance/每个子目录含README.md（4个目录）
- [x] PROVENANCE.md包含全部6个必需章节（知识源清单、生产管线、中间产物索引、版本历史、质量验证、审计信息）
- [x] test_cases.yaml覆盖4种用例类型（12个测试用例）

## 审计信息

- **Skill名称**：vlm-verifier
- **Skill描述**：企业信贷跨模态材料核验与分析技能
- **版本号**：1.0.0
- **状态**：draft
- **生产完成日期**：2026-05-05
- **管线类型**：混合型（数据驱动型为主+制度合规型）
- **交互模式**：A. 报告生成型
- **知识源数量**：8个
- **改造步骤**：14步（skill-refactor标准流程）
- **L3资源文件数**：references/(4) + assets/(1) + scripts/(2) + tests/(2) + provenance/(4) = 13个文件
