# 04 - Skill合成与迭代

## 输入

- 原SKILL.md(349行)
- 监管要求映射表(来自01-regulatory-review)
- 专家经验(来自02-expert-interview)
- 流程设计方案(来自03-process-design)

## 过程

### 核心业务逻辑保护清单

改造过程中严格保护以下9条核心业务规则,未做任何修改:

1. **材料三级分类**:必备/建议/补充,缺失则建议暂缓
2. **案件冲突检查**:同一客户存在活跃案件时禁止重复提交
3. **授信到期检查**:到期前90天内判定为续贷
4. **案件类型三分类**:续贷(renewal)/新增(first_credit)/追加(additional)
5. **用户确认机制**:必须明确说"提交"或"确认"后才可发起案件创建
6. **3条金融合规红线**:R1(必备材料缺失)、R2(活跃案件)、R3(未明确确认)
7. **5条数据脱敏规则**:身份证号、银行账号、案件编号、授信金额、商业机密
8. **4种降级策略**:影像档案不可用、案件状态超时、授信台账不可用、尽调报告缺失
9. **评估三档结论**:可以提交/建议补充后提交/暂缓提交

### 改造内容

| 改造项 | 改造前 | 改造后 | 变更类型 |
|--------|--------|--------|----------|
| version | 1.0.0 | 2.0.0 | Major升级(流程重构) |
| status | 无 | draft | 新增生命周期字段 |
| upstream_skills | 单行逗号分隔 | 多行数组 | 格式标准化 |
| downstream_skills | 单行 | 多行数组 | 格式标准化 |
| Workflow步骤 | 5步 | 6步(增加步骤0) | 新增先读后写机制 |
| 门控机制 | 无 | 每步标注✅/❌/⚠️ | 新增模式B门控路径 |
| Constraints | 7条 | 9条 | 新增2条(禁止跳过步骤、红线执行强制) |
| Output Format | 6章节列表 | 6章节表格+下游字段 | 增强编排兼容性 |
| 免责声明 | 无 | 引用shared/disclaimer-template.md | 新增合规要求 |
| provenance/ | 不存在 | 4个子目录+README | 新增生产溯源体系 |
| test_cases.yaml | 不存在 | 12个用例 | 新增测试覆盖 |

### 合规检查

- [x] SKILL.md行数 < 500行 (当前约420行)
- [x] Frontmatter完整(包含name/description/target_role/business_domain/risk_level/version/status)
- [x] 编排字段正确(upstream_skills/downstream_skills)
- [x] 9条Constraints(含3条红线)
- [x] 6步Workflow(含步骤0先读后写)
- [x] 门控路径完整(模式B-步骤门控型)
- [x] 输出格式结构化(下游可解析)
- [x] 免责声明引用
- [x] 5条Gotchas
- [x] 2个Examples
- [x] Out of Scope章节
- [x] PROVENANCE.md ≥ 100行
- [x] provenance/ 4个子目录均有README.md
- [x] test_cases.yaml覆盖4种类型

## 输出

- 新版SKILL.md(约420行)
- PROVENANCE.md(约180行)
- provenance/ 目录(4个子目录)
- tests/test_cases.yaml(12个用例)
- audit/ 审计日志
