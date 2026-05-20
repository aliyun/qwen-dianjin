# 步骤4:Skill合成与迭代

## 输入
- 6步Workflow设计(来自[步骤3](../03-structured-design/README.md))
- 核心业务规则保护清单(8条规则,来自步骤2)
- METHODOLOGY.md金融级标准

## 过程
1. 重构YAML Frontmatter:
   - name:credit-collateral-risk-mgmt(保留)
   - description:增加触发词+反触发条件(功能+触发+反触发)
   - 新增字段:target_role/business_domain/risk_level/version(2.0.0)/status(draft)/data_sources/upstream_skills/downstream_skills
2. 增加Target Role章节:角色/场景/输出用途/决策层级/执行频率
3. 增加Data Sources章节:5项必需数据+4种降级策略
4. 重构Workflow:5步→6步(增加步骤0),增加门控路径标注
5. 强化Constraints:5条→7条(增加禁止越权/红线强制)
6. 增加Audit Trail:JSON格式审计日志模板
7. 增加Gotchas:4条踩坑记录(评估限制/多顺位抵押/保险续期/处置程序)
8. 增加Examples:2个端到端示例(贷前评估/贷后监控)
9. 增加Output Format:结构化表格(9章节)+下游兼容性说明+免责声明
10. 增加Out of Scope:5项非功能范围界定
11. 构建L3资源目录:references/(3个文件)/assets/(1个模板)/scripts/(1个验证脚本)/tests/(测试用例)/provenance/(4个子目录)
12. 生成PROVENANCE.md:180行,6个必需章节

## 输出
- 新版SKILL.md(351行,<500行要求)
- PROVENANCE.md(180行)
- provenance/目录(4个子目录,每个含README.md)
- references/目录(待填充3个文件)
- assets/目录(待填充报告模板)
- scripts/目录(待填充验证脚本)
- tests/目录(待填充测试用例)

## 核心业务逻辑保护清单
改造过程中严格保留以下8条核心业务规则,未做任何篡改:
1. ✅ R1:押品合法所有,无查封/冻结/扣押
2. ✅ R2:评估报告在有效期内
3. ✅ R3:登记手续完备有效
4. ✅ R4:抵押率不超过品类上限(房产70%/土地60%/设备50%/应收账款80%/知识产权30%/存货60%/金融资产90%)
5. ✅ R5:存续期监控按频率执行(价值跌幅≥15%预警,≥25%补足)
6. ✅ R6:一票否决条件C1-C6优先检查,触发即终止
7. ✅ R7:价值补足机制15工作日时限
8. ✅ R8:处置程序合法合规,拍卖底价≥评估价70%

## 改造前后对比
| 维度 | 改造前 | 改造后 | 变更说明 |
|------|--------|--------|---------|
| 文件行数 | 288行 | 351行 | +63行(增加Frontmatter/Target Role/Data Sources/Audit Trail等) |
| Frontmatter | 2字段(name/description) | 10字段 | +8字段(target_role/business_domain/risk_level/version/data_sources等) |
| Workflow | 5步(无门控) | 6步(带门控路径) | +1步(步骤0先读后写),增加分支路径标注 |
| Constraints | 5条 | 7条 | +2条(禁止越权/红线强制) |
| 降级策略 | 无 | 4种场景 | 新增(评估接口/市场数据/登记系统/保险信息不可用) |
| Audit Trail | 无 | JSON格式 | 新增 |
| Gotchas | 无 | 4条 | 新增 |
| Examples | 无 | 2个 | 新增 |
| Output Format | 无 | 结构化表格(9章节) | 新增 |
| Out of Scope | 无 | 5项 | 新增 |
| provenance/ | 无 | 4个子目录 | 新增(混合型管线模板) |
| PROVENANCE.md | 无 | 180行 | 新增 |
| tests/ | 无 | 待填充 | 新增目录 |
