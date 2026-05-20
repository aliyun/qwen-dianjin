# 步骤 4：Skill 合成与迭代

## 输入
- 步骤 1-3 的全部产出
- CLAUDE.md 项目指令
- skill-refactor-checklist.md 快速检查清单

## 过程
1. 按照 METHODOLOGY.md 的黄金骨架结构编写 SKILL.md
2. 将原 SKILL.md 中的长篇内容下沉至 references/ 和 assets/：
   - 检查频率矩阵 → references/check-frequency-policy.md
   - 资金用途核查表 → references/fund-usage-policy.md
   - 财务指标阈值表 → references/industry-benchmarks.md
   - 担保核查表 → references/collateral-policy.md
   - 五级分类标准 → references/five-classification-policy.md
   - 预警指标体系 → references/early-warning-indicators.md
   - 贷后报告模板 → assets/post-loan-report-template.md
   - 免责声明 → assets/disclaimer-template.md
3. 创建 PROVENANCE.md 生产溯源文件
4. 创建各 provenance 子目录的 README.md
5. 执行最终校验：
   - SKILL.md 行数 < 500（实际 405 行）✅
   - 目录名 kebab-case ✅
   - 无 README 文件 ✅
   - Frontmatter 完整 ✅
   - 交互模式 B 门控路径完整 ✅
   - 测试用例 4 种类型全覆盖 ✅
   - 核心逻辑保护清单 14 项全部保留 ✅

## 输出
- 新版 SKILL.md（405 行）
- PROVENANCE.md
- 完整 L3 资源目录结构
- 改造执行报告

## 日期
2026-05-06

## 负责人
AI 架构师
