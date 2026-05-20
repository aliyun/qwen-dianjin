# 生产溯源：credit-industry-analysis Skill

本文件记录 credit-industry-analysis（信贷行业深度分析）Skill 的完整生产过程，用于审计追踪和 demo 展示。

## Skill 基本信息

- **Skill名称**：credit-industry-analysis（信贷行业深度分析）
- **业务域**：对公业务 > 信贷管理 > 贷前调查 > 行业分析
- **版本**：2.0.0
- **风险等级**：medium
- **生产完成日期**：2026-05-05
- **管线类型**：混合型（数据驱动型 + 制度合规型）
- **交互模式**：模式 A - 报告生成型（Report Generation）
- **状态**：draft（待审核）

---

## 知识源清单

### 知识源1：经典行业分析框架
- **来源**：PEST 宏观环境分析框架、波特五力竞争模型、产业链价值链拆解方法论、TAM/SAM/SOM 市场规模测算框架、情景分析法
- **提供者**：行业研究员 + 投研报告方法论
- **萃取方法**：框架研究与信贷场景适配
- **萃取日期**：2026-05-05
- **萃取人**：AI架构师团队 + 行业研究员
- **关键内容**：不同行业类型需要不同的分析框架组合（传统制造业/科技行业/消费行业/周期性行业）

### 知识源2：专家访谈与实战经验
- **来源**：3位资深对公客户经理访谈记录、5位信贷审批官行业准入评估经验、10份历史行业分析报告
- **提供者**：对公客户经理 + 信贷审批官
- **萃取方法**：深度访谈 + 案例分析
- **萃取日期**：2026-05-05
- **萃取人**：AI架构师团队
- **关键内容**：信贷场景专属关注点（6个维度）、行业分析踩坑记录（5条 Gotchas）、数据脱敏规则（5条）

### 知识源3：监管政策与制度文件
- **来源**：《商业银行授信工作尽职指引》、《贷款风险分类指引》、《产业结构调整指导目录》（现行版）、工信部《重点行业领域产能过剩预警机制》、国家统计局《国民经济行业分类》（GB/T 4754-2017）
- **提供者**：风险管理部
- **萃取方法**：法规研读与规则提取
- **萃取日期**：2026-05-05
- **萃取人**：AI架构师团队 + 风险管理部
- **关键内容**：6条行业红线（I1-I6）、产能过剩预警指标体系、行业分类标准

### 知识源4：现有 Skill 业务逻辑
- **来源**：credit-industry-analysis SKILL.md（v1.0.0）
- **提供者**：原 Skill 开发团队
- **萃取方法**：代码审查与语义提取
- **萃取日期**：2026-05-05
- **萃取人**：AI架构师团队
- **关键内容**：波特五力定量评估框架、行业生命周期判定标准、市场规模测算方法、信贷场景专属关注点

---

## 生产管线

### 步骤 1：分析框架设计（01-framework-design/）

**输入**：
- PEST 宏观环境分析框架
- 波特五力竞争模型
- 产业链价值链拆解方法论
- TAM/SAM/SOM 市场规模测算框架
- 情景分析法（乐观/中性/悲观）

**过程**：
1. 研究经典行业分析框架在银行信贷场景的适用性
2. 设计分析框架选择指南（传统制造业/科技行业/消费行业/周期性行业）
3. 定义行业生命周期判定标准（导入期/成长期/成熟期/衰退期）
4. 建立竞争格局量化指标体系（CR3/CR5/CR10、HHI指数）
5. 设计供需平衡表推演方法

**输出**：
- [行业分析框架选择指南](references/industry-analysis-guide.md)
- [行业生命周期判定标准](SKILL.md)（步骤2）
- [竞争格局量化指标体系](SKILL.md)（步骤4）
- [供需平衡表推演模板](assets/industry-analysis-template.md)

详见：[01-framework-design/README.md](provenance/01-framework-design/README.md)

### 步骤 2：专家访谈与经验萃取（02-expert-interview/）

**输入**：
- 3位资深对公客户经理访谈记录
- 5位信贷审批官行业准入评估经验
- 10份历史行业分析报告

**过程**：
1. 访谈对公客户经理，了解贷前行业准入评估的实际痛点
2. 萃取信贷审批官关注的行业风险信号（产能利用率、政策敏感度、竞争格局）
3. 分析历史行业分析报告的优劣势，提炼最佳实践
4. 识别常见踩坑点（行业边界过泛、市场规模测算不透明、红线信号被淹没）
5. 建立信贷场景专属关注点清单（6个维度）

**输出**：
- [信贷场景专属关注点](SKILL.md)（6个维度）
- [行业分析踩坑记录](SKILL.md)（5条 Gotchas）
- [行业红线判断标准](SKILL.md)（6条红线 I1-I6）
- [数据脱敏规则](SKILL.md)（5条）

详见：[02-expert-interview/README.md](provenance/02-expert-interview/README.md)

### 步骤 3：监管政策分析（03-regulatory-analysis/）

**输入**：
- 《商业银行授信工作尽职指引》
- 《贷款风险分类指引》
- 《产业结构调整指导目录》（现行版）
- 工信部《重点行业领域产能过剩预警机制》
- 国家统计局《国民经济行业分类》（GB/T 4754-2017）

**过程**：
1. 研读监管文件，提取与行业分析相关的核心要求
2. 识别行业准入的监管红线和禁止性行为
3. 分析《产业结构调整指导目录》的鼓励类/限制类/淘汰类分类标准
4. 建立产能过剩预警指标体系（产能利用率 < 60% 且连续2年下降）
5. 设计行业红线核查流程（6条红线 I1-I6）

**输出**：
- [行业准入监管要求清单](SKILL.md)（Constraints）
- [6条行业红线定义](SKILL.md)（I1-I6）
- [产能过剩预警指标体系](SKILL.md)（步骤6）
- [行业分类标准](references/industry-analysis-guide.md)（GB/T 4754-2017）

详见：[03-regulatory-analysis/README.md](provenance/03-regulatory-analysis/README.md)

### 步骤 4：Skill 合成与迭代（04-skill-synthesis/）

**输入**：
- 步骤1-3输出的所有中间产物
- 现有 credit-industry-analysis SKILL.md（v1.0.0）
- METHODOLOGY.md 金融级标准
- skill-refactor 改造操作指南

**过程**：
1. 对照 METHODOLOGY.md 的12项核心标准，评估现有 SKILL.md 的差异
2. 重构 Frontmatter（version: 1.0.0 → 2.0.0，增加 status: draft）
3. 将 Workflow 从 8 步扩展为 10 步（增加步骤0数据确认和步骤9报告输出）
4. 为每个步骤添加门控条件（✅ 通过 / ❌ 不通过 / ⚠️ 部分异常）
5. 增强 Constraints（从7条增加至9条）
6. 优化 Output Format，增加下游兼容性说明和免责声明要求
7. 校验 SKILL.md 行数（确保 < 500行）

**输出**：
- [新版 SKILL.md](SKILL.md)（v2.0.0，~490行）
- [验证脚本](scripts/validate_industry.py)
- [测试用例集](tests/test_cases.yaml)（12个用例）
- [输出模板](assets/industry-analysis-template.md)（已优化）

详见：[04-skill-synthesis/README.md](provenance/04-skill-synthesis/README.md)

---

## 中间产物索引

所有生产过程的原始数据和中间产物存储在 provenance/ 目录中：

```
provenance/
├── 01-framework-design/          # 步骤1：分析框架设计
│   └── README.md                 # 输入、过程、输出、关键发现
├── 02-expert-interview/          # 步骤2：专家访谈与经验萃取
│   └── README.md                 # 访谈记录摘要、经验萃取结果
├── 03-regulatory-analysis/       # 步骤3：监管政策分析
│   └── README.md                 # 监管文件研读记录、红线提取过程
└── 04-skill-synthesis/           # 步骤4：Skill 合成与迭代
    └── README.md                 # 改造过程记录、核心逻辑保护清单
```

每个子目录包含 README.md，记录该步骤的完整信息（输入、过程、输出、关键发现）。

---

## 版本历史

| 版本 | 日期 | 变更内容 | 变更人 |
|------|------|---------|--------|
| 2.0.0 | 2026-05-05 | 金融级改造：Workflow扩展至10步、增加门控条件、Constraints增强、provenance体系建立 | AI架构师团队 |
| 1.0.0 | 未知 | 初始版本 | 原开发团队 |

---

## 质量验证

### 定量评估指标声明

| 指标 | 目标值 | 当前状态 |
|------|--------|---------|
| 触发准确率 | ≥ 90% | 待验证 |
| 执行一致性 | ≥ 95% | 待验证 |
| Token 效率 | < 500 行 | ✅ 490行 |
| 合规完整性 | 100% | 待验证 |
| 错误恢复 | 100% | 待验证 |
| Gotchas 持续更新 | 持续更新 | ✅ 5条 |

### 测试覆盖

- ✅ 正例测试：5个测试用例（制造业、钢铁、新能源汽车、基础化工、房地产）
- ✅ 反例测试：3个测试用例（个人信贷、企业财务数据、股权穿透）
- ✅ 边界案例：3个测试用例（行业边界过泛、数据不完整、触发多条红线）
- ✅ Golden file：1个标准输出参照文件

### 合规检查

- ✅ Constraints 章节包含9条红线（含6条行业红线 I1-I6）
- ✅ Audit Trail 格式完整（JSON格式）
- ✅ 数据溯源标注清晰（每个步骤标注数据来源）
- ✅ 免责声明模板已引用（shared/disclaimer-template.md）
- ✅ 无收益承诺/投资建议
- ✅ 降级策略覆盖5种场景

### 文档完整性检查

- ✅ SKILL.md < 500 行（~490行）
- ✅ Frontmatter 完整（name、description、target_role、business_domain、risk_level、version、status、data_sources、upstream_skills、downstream_skills）
- ✅ Workflow 10步，每步标注门控条件
- ✅ references/ 包含1个文件（industry-analysis-guide.md，标注 data_valid_until）
- ✅ assets/ 包含1个文件（industry-analysis-template.md）
- ✅ scripts/ 包含验证脚本和 requirements.txt
- ✅ tests/ 包含 test_cases.yaml（12个用例）和 golden/ 目录
- ✅ provenance/ 包含4个子目录，每个包含 README.md

---

## 审计信息

- **Skill名称**：credit-industry-analysis（信贷行业深度分析）
- **Skill版本**：2.0.0
- **生产完成日期**：2026-05-05
- **管线类型**：混合型（数据驱动型 + 制度合规型）
- **交互模式**：模式 A - 报告生成型
- **知识源数量**：4个
- **审核状态**：待审核（draft）
- **下次数据巡检日期**：2026-08-01
