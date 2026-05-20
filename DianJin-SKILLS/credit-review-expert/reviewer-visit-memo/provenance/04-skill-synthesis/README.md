# 步骤4:Skill合成与迭代

## 输入
- 8步Workflow设计(来自[步骤3](../03-regulatory-analysis/README.md))
- 核心业务规则保护清单(8条规则,来自步骤1)
- 专家经验规则(4条,来自步骤2)
- METHODOLOGY.md金融级标准

## 过程
1. 重构YAML Frontmatter:
   - name:reviewer-visit-memo(保留)
   - description:增加触发词+反触发条件(功能+触发+反触发)
   - 新增字段:target_role/business_domain/risk_level/version(2.0.0)/status(draft)/data_sources/upstream_skills/downstream_skills
2. 增加Target Role章节:角色/场景/输出用途/决策层级/执行频率
3. 增加Data Sources章节:5项必需数据+4种降级策略
4. 重构Workflow:原有工作规则→8步(增加步骤0先读后写+步骤7一票否决自检),增加分支路径标注
5. 强化Constraints:5条→7条(增加禁止跳过步骤/禁止越权建议)
6. 增加Audit Trail:JSON格式审计日志模板
7. 增加Gotchas:4条踩坑记录(主观推断/增量更新/CM偏差/一票否决遗漏)
8. 增加Examples:2个端到端示例(新客户首笔/风险预警复核)
9. 增加Interaction Pattern:模式D对话辅助型(开始/中间/结束阶段)
10. 增加Context Management:上下文累积与清理规则
11. 增加Output Format:结构化表格(10章节)+下游兼容性+免责声明
12. 增加Out of Scope:5项非功能范围界定
13. 将六维度详细表格(约200行)移至references/visit-memo-dimensions.md

## 输出
- 新版SKILL.md(451行,符合<500行要求)
- references/visit-memo-dimensions.md(98行,六维度详细框架)
- PROVENANCE.md(生产溯源文件)
- tests/test_cases.yaml(15个测试用例)

## 核心业务逻辑保护清单
改造过程中严格保留以下核心规则,未篡改:
1. 事实优先:每条记录须有观察来源,不得用主观推断代替事实
2. CM交叉验证:走访发现与CM报告的偏差须显式标注
3. 信息修正可追溯:保留原记录并标注修正时间,不得直接覆盖
4. 敏感信号必报:发现停工/欠薪/民间融资等信号须如实记录
5. 笔记归档确认:每次保存后须确认笔记已正确归档
6. 一票否决条件S1-S6优先检查,触发须红色标注
7. 量化优先:规模/金额/比例等须用具体数字,不得用模糊表述
8. 版本可追溯:增量更新保留历史记录,支持审计调阅

## 改造前后对比
| 维度 | 改造前 | 改造后 | 变更说明 |
|------|--------|--------|---------|
| 文件行数 | 274行 | 451行 | +177行(仍<500行要求) |
| Frontmatter | 2字段 | 10字段 | +8字段 |
| Workflow | 7条工作规则 | 8步(带分支路径) | 增加步骤0/步骤7 |
| Constraints | 5条 | 7条 | +2条 |
| 降级策略 | 无 | 4种场景 | 新增 |
| Audit Trail | 无 | JSON格式 | 新增 |
| Gotchas | 无 | 4条 | 新增 |
| Examples | 无 | 2个 | 新增 |
| Output Format | 代码块 | 结构化表格(10章节) | 新增+下游兼容性+免责声明 |
| Interaction Pattern | 无 | 模式D(3阶段) | 新增 |
| Context Management | 无 | 累积/清理规则 | 新增 |
| provenance/ | 无 | 4个子目录 | 新增 |
| PROVENANCE.md | 无 | 160行 | 新增 |
| tests/ | 无 | 15个测试用例 | 新增 |
| references/ | 无 | visit-memo-dimensions.md | 新增 |
