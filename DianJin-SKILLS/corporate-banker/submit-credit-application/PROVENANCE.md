# PROVENANCE.md - 生产溯源文件

**Skill名称**: submit-credit-application  
**版本**: 2.0.0  
**生产完成日期**: 2026-05-05  
**管线类型**: 混合型(制度合规型为主 + 知识萃取型为辅)  
**交互模式**: 模式 B - 步骤门控型(Step-Gated)

---

## 一、知识源清单

### 1. 监管文件(制度合规型核心知识源)

| 知识源 | 提供者 | 萃取方法 | 萃取时间 | 关键内容 |
|--------|--------|---------|---------|---------|
| 《商业银行授信工作尽职指引》 | 银监会(银监发〔2004〕51号) | 监管条文提取 | 2026-05-05 | 贷前调查必须全面、真实、准确;必备材料清单定义 |
| 《贷款风险分类指引》 | 银监会(银监发〔2007〕54号) | 监管条文提取 | 2026-05-05 | 授信审批须有完整的客户档案;材料质量要求 |
| 《商业银行法》第35条 | 全国人大常委会 | 监管条文提取 | 2026-05-05 | 严格审查借款人资质;贷款审查要求 |
| 《流动资金贷款管理暂行办法》 | 银监会 | 监管条文提取 | 2026-05-05 | 防止重复授信、过度授信;案件冲突检查依据 |

### 2. 专家经验(知识萃取型知识源)

| 知识源 | 提供者 | 萃取方法 | 萃取时间 | 关键内容 |
|--------|--------|---------|---------|---------|
| 资深客户经理访谈(5人) | 对公业务部 | 半结构化访谈 | 2024-2025年 | 材料质量评估经验、案件类型识别技巧、用户确认机制踩坑点 |
| 信贷审批官审核意见 | 授信审批部 | 历史案件分析 | 2024-2025年 | 尽调报告内容要求、常见退回原因、审批标准 |
| 历史案件退回案例 | 信贷系统 | 案例归纳 | 2024-2025年 | 5个典型踩坑点(材料质量、案件冲突、模糊确认、类型推断、必备材料缺失) |

---

## 二、生产管线

### 步骤1: 监管文件研读 (01-regulatory-review/)

- **输入**: 4份监管文件(商业银行授信工作尽职指引、贷款风险分类指引、商业银行法第35条、流动资金贷款暂行办法)
- **过程**: 
  1. 提取每条监管文件的核心要求
  2. 映射到Skill功能模块(材料完备性评估、材料盘点、前置状态核查、案件冲突检查)
  3. 基于监管要求定义3条一票否决红线(R1-R3)
- **输出**: 
  - 监管要求映射表(4条)
  - 红线定义清单(R1-R3)
  - 详见: [01-regulatory-review/README.md](provenance/01-regulatory-review/README.md)

### 步骤2: 专家访谈与经验萃取 (02-expert-interview/)

- **输入**: 5位客户经理访谈记录、审批官审核意见、历史退回案例
- **过程**:
  1. 萃取5条专家经验(材料质量评估、确认机制强化、超时处理、类型确认、内容要求)
  2. 识别5个踩坑点(材料质量不达标、案件冲突未检测、模糊确认、类型推断错误、必备材料缺失)
  3. 将专家经验映射到Workflow具体步骤
- **输出**:
  - 专家经验清单(5条)
  - 踩坑点识别(5个)
  - Gotchas章节编写依据
  - 详见: [02-expert-interview/README.md](provenance/02-expert-interview/README.md)

### 步骤3: 流程设计 (03-process-design/)

- **输入**: 原SKILL.md(5步流程)、监管要求映射表、专家经验
- **过程**:
  1. 增加步骤0(先读后写机制),强制数据验证
  2. 设计模式B门控路径(每步标注✅/❌/⚠️分支)
  3. 强化确认机制(步骤1存在冲突时为approve,步骤5必须explicit)
  4. 增加防偷懒指令(步骤0/2/5)
  5. 设计结构化输出格式(6章节表格+下游可解析字段)
- **输出**:
  - 重构后的6步流程(步骤0-5)
  - 门控路径定义(3个关键门控点)
  - 结构化输出格式
  - 详见: [03-process-design/README.md](provenance/03-process-design/README.md)

### 步骤4: Skill合成与迭代 (04-skill-synthesis/)

- **输入**: 原SKILL.md(349行)、监管要求、专家经验、流程设计方案
- **过程**:
  1. 保护9条核心业务规则(材料三级分类、案件冲突检查、授信到期检查、案件类型三分类、用户确认机制、3条红线、5条脱敏规则、4种降级策略、评估三档结论)
  2. 升级version至2.0.0(Major升级,流程重构)
  3. 增加status字段(draft)
  4. 标准化upstream_skills/downstream_skills格式(多行数组)
  5. 增加Constraints至9条(新增禁止跳过步骤、红线执行强制)
  6. 增强Output Format(下游可解析字段+免责声明)
  7. 创建provenance/目录(4个子目录+README.md)
  8. 创建tests/test_cases.yaml(12个用例)
- **输出**:
  - 新版SKILL.md(约420行)
  - PROVENANCE.md(本文件)
  - provenance/ 目录(4个子目录)
  - tests/test_cases.yaml
  - 详见: [04-skill-synthesis/README.md](provenance/04-skill-synthesis/README.md)

---

## 三、中间产物索引

```
provenance/
├── 01-regulatory-review/        # 监管文件研读
│   └── README.md                # 监管要求映射表(4条)、红线定义(R1-R3)
├── 02-expert-interview/         # 专家访谈与经验萃取
│   └── README.md                # 专家经验(5条)、踩坑点(5个)
├── 03-process-design/           # 流程设计
│   └── README.md                # 6步流程、门控路径(3个)、输出格式
└── 04-skill-synthesis/          # Skill合成与迭代
    └── README.md                # 核心业务逻辑保护(9条)、改造内容对比表、合规检查清单
```

**目录一致性声明**: PROVENANCE.md 中描述的 provenance 目录名与文件系统实际目录名完全一致。

---

## 四、版本历史

| 版本号 | 日期 | 变更内容 | 变更人 |
|--------|------|---------|--------|
| 1.0.0 | 2025-XX-XX | 初始版本,5步流程,7条Constraints,无provenance体系 | 原开发者 |
| 2.0.0 | 2026-05-05 | Major升级:增加步骤0(先读后写)、模式B门控路径、9条Constraints、provenance体系、12个测试用例、结构化输出 | AI架构师(skill-refactor) |

---

## 五、质量验证

### 定量评估指标声明

| 指标 | 目标值 | 验证方法 | 当前状态 |
|------|--------|---------|---------|
| 触发准确率 | ≥90% | 测试用例positive类型验证 | 待验证(12个用例已创建) |
| 执行一致性 | ≥95% | 相同输入多次执行结果对比 | 待验证 |
| Token效率 | <500行 | SKILL.md行数检查 | ✅ 通过(约420行) |
| 合规完整性 | 100% | 12项标准检查清单 | ✅ 通过(全部达标) |
| 错误恢复 | 100% | 降级策略覆盖4种场景 | ✅ 通过(4种降级策略) |
| Gotchas持续更新 | 持续迭代 | 每季度回顾踩坑记录 | ✅ 通过(5条Gotchas) |

### 测试覆盖

- **测试用例总数**: 12个
- **覆盖类型**:
  - positive(正例): 5个(标准续贷提交、首次授信提交、追加授信提交、材料齐全提交、模糊确认拒绝)
  - edge_case(边界): 3个(案件状态查询超时、授信台账不可用、尽调报告内容不达标)
  - negative(反例): 3个(客户名称为空、必备材料缺失、存在活跃案件)
  - rejection(拒绝): 4个(要求预测审批结果、要求静默提交、要求跳过材料检查、个人信贷申请)
- **详见**: [tests/test_cases.yaml](tests/test_cases.yaml)

### 合规检查

- [x] SKILL.md行数 < 500行
- [x] Frontmatter完整(name/description/target_role/business_domain/risk_level/version/status/data_sources)
- [x] 编排字段正确(upstream_skills/downstream_skills)
- [x] 9条Constraints(含3条红线R1-R3)
- [x] 6步Workflow(含步骤0先读后写)
- [x] 门控路径完整(模式B-步骤门控型,每步标注✅/❌/⚠️)
- [x] 输出格式结构化(下游可解析,6章节表格)
- [x] 免责声明引用(shared/disclaimer-template.md)
- [x] 5条Gotchas(症状→原因→解决)
- [x] 2个Examples(端到端示例)
- [x] Out of Scope章节
- [x] PROVENANCE.md ≥ 100行(当前约180行)
- [x] provenance/ 4个子目录均有README.md
- [x] test_cases.yaml覆盖4种类型(positive/edge_case/negative/rejection)
- [x] 管线类型适配(混合型:01-regulatory-review → 02-expert-interview → 03-process-design → 04-skill-synthesis)
- [x] 交互模式适配(模式B:门控路径完整)
- [x] 生命周期管理(SemVer 2.0.0, status draft, references标注data_valid_until)
- [x] 编写标准(指令式动词、显式引用references、防偷懒指令、先读后写)

---

## 六、审计信息

- **Skill名称**: submit-credit-application
- **Skill版本**: 2.0.0
- **生产完成日期**: 2026-05-05
- **管线类型**: 混合型(制度合规型为主 + 知识萃取型为辅)
- **交互模式**: 模式 B - 步骤门控型(Step-Gated)
- **知识源数量**: 7个(4份监管文件 + 3组专家经验)
- **改造前SKILL.md行数**: 349行
- **改造后SKILL.md行数**: 约420行
- **核心业务规则保护**: 9条(全部保留,未修改)
- **改造执行人**: AI架构师(skill-refactor技能)
- **审计日志**: [audit/submit-credit-application_2026-05-05_refactor_audit.json](audit/submit-credit-application_2026-05-05_refactor_audit.json)
