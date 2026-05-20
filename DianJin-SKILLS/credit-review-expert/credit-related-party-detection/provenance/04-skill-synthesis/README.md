# 步骤4:Skill合成与迭代

## 输入
- 5步Workflow设计(来自[步骤3](../03-structured-design/README.md))
- 核心业务规则保护清单(8条规则,来自步骤2)
- METHODOLOGY.md金融级标准

## 过程
1. 重构YAML Frontmatter:
   - name:credit-related-party-detection(保留)
   - description:增加触发词+反触发条件(功能+触发+反触发)
   - 新增字段:target_role/business_domain/risk_level/version(2.0.0)/status(draft)/data_sources/upstream_skills/downstream_skills
2. 增加Target Role章节:角色/场景/输出用途/决策层级/执行频率
3. 增加Data Sources章节:5项必需数据+4种降级策略
4. 重构Workflow:4步→5步(增加步骤0),增加门控路径标注
5. 强化Constraints:5条→7条(增加禁止越权/红线强制)
6. 增加Audit Trail:JSON格式审计日志模板
7. 增加Gotchas:4条踩坑记录(股权穿透/隐性关联/四流合一/循环担保)
8. 增加Examples:2个端到端示例(贷前排查/贷后预警)
9. 增加Output Format:结构化表格(7章节)+下游兼容性+免责声明
10. 增加Out of Scope:5项非功能范围界定

## 输出
- [新版SKILL.md](../SKILL.md)(420行)
- [PROVENANCE.md](../PROVENANCE.md)(180行)
- [tests/test_cases.yaml](../tests/test_cases.yaml)(15个测试用例)

## 核心业务逻辑保护清单
改造前后核对,确保以下8条核心规则一字未改:
- [x] R1: 穿透至最终自然人,至少3层穿透
- [x] R2: 六维关联全部排查,不得选择性忽略
- [x] R3: 四流合一验证(合同/发票/物流/资金)
- [x] R4: 关联交易占比计算(关联交易金额/同类交易总金额)
- [x] R5: 担保圈≥3家形成担保圈,≥5家大型担保圈
- [x] R6: 一票否决条件R1-R6优先检查,触发即标注
- [x] R7: 隐性关联六项信号检查
- [x] R8: 客户拒绝配合调查须显式记录

## 改造前后对比
| 维度 | 改造前 | 改造后 | 变更说明 |
|------|--------|--------|---------|
| 文件行数 | 270行 | 420行 | +150行(仍<500行要求) |
| Frontmatter | 2字段 | 10字段 | +8字段 |
| Workflow | 4步(无门控) | 5步(带门控路径) | +1步(步骤0),增加分支路径标注 |
| Constraints | 5条 | 7条 | +2条(禁止越权/红线强制) |
| 降级策略 | 无 | 4种场景 | 新增 |
| Audit Trail | 无 | JSON格式 | 新增 |
| Gotchas | 无 | 4条 | 新增 |
| Examples | 无 | 2个 | 新增 |
| Output Format | 无 | 结构化表格(7章节) | 新增+下游兼容性+免责声明 |
| provenance/ | 无 | 4个子目录 | 新增 |
| PROVENANCE.md | 无 | 180行 | 新增 |
| tests/ | 无 | 15个测试用例 | 新增 |

## 质量验证
- [x] SKILL.md行数420行(<500行要求)
- [x] Frontmatter完整(10字段)
- [x] Workflow门控路径完整(步骤0-4均有通过/不通过/预警分支)
- [x] Constraints 7条(≥5条要求)
- [x] Gotchas 4条(≥3条要求)
- [x] PROVENANCE.md 180行(≥100行要求)
- [x] provenance/子目录4个,每个含README.md
- [x] 核心业务逻辑保护(8条规则全部保留)
