# 步骤 4：Skill 合成与迭代

## 输入
- 步骤 1-3 的全部产出
- CLAUDE.md 项目指令
- skill-refactor-checklist.md 快速检查清单

## 过程
1. 按照 METHODOLOGY.md 的黄金骨架结构编写 SKILL.md
2. 将原 SKILL.md 中的长篇内容下沉至 references/ 和 assets/：
   - 信号类型库 → references/cot-signal-library.md
   - 行业基准数据 → references/industry-benchmarks.md
   - 政策文件清单 → references/policy-regulatory-framework.md
   - 压力测试参数 → references/financial-stress-test-params.md
   - 报告模板 → assets/cot-report-template.md
   - 免责声明 → assets/disclaimer-template.md
3. 创建 PROVENANCE.md 生产溯源文件
4. 创建各 provenance 子目录的 README.md
5. 执行最终校验：
   - SKILL.md 行数 < 500（实际 351 行）✅
   - 目录名 kebab-case ✅
   - 无 README 文件 ✅
   - Frontmatter 完整 ✅
   - 交互模式 A 线性流程 ✅
   - 测试用例 4 种类型全覆盖 ✅
   - 核心逻辑保护清单 9 项全部保留 ✅
   - 所有 references/ 文件标注 data_valid_until ✅

## 输出
- 新版 SKILL.md（351 行）
- PROVENANCE.md
- 完整 L3 资源目录结构
- 改造执行报告

## 日期
2026-05-06

## 负责人
AI 架构师
