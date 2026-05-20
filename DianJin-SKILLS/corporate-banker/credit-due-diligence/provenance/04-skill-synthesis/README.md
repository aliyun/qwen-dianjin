# 步骤4：Skill 合成与迭代

## 输入
- 步骤3输出的结构化设计方案
- 现有 SKILL.md（v1.0.0）的业务逻辑保护清单
- METHODOLOGY.md 快速检查清单

## 过程
1. 按照步骤3的设计方案，逐项修改 SKILL.md
2. 保留原有15条核心业务规则（财务指标阈值、关联风险判断标准等）
3. 重构 Frontmatter（version: 1.0.0 → 2.0.0，status: draft）
4. 重写 Workflow，增加步骤0和门控条件
5. 优化 Output Format，增强结构化程度
6. 更新 Constraints（增加2条）
7. 校验 SKILL.md 行数（确保 < 500行）
8. 创建 provenance/ 目录结构和 PROVENANCE.md
9. 更新测试用例，增加 rejection 类型

## 输出
- 新版 SKILL.md（v2.0.0，行数：~450行）
- PROVENANCE.md（生产溯源文件，~150行）
- provenance/ 目录（4个子目录，每个包含 README.md）
- 更新后的 tests/test_cases.yaml（12个用例，覆盖4种类型）

## 日期
2026-05-05

## 负责人
AI架构师团队

## 迭代记录
- **第1轮**：完成 SKILL.md 主体改造，校验通过
- **第2轮**：创建 provenance/ 目录和 PROVENANCE.md
- **第3轮**：更新测试用例，增加 rejection 类型
- **第4轮**：最终校验，所有检查项通过

## 质量验证
- ✅ SKILL.md < 500行（~450行）
- ✅ Frontmatter 完整（name、description三要素、target_role、risk_level、version、status、upstream_skills、downstream_skills）
- ✅ Workflow 使用指令式动词，无客套语
- ✅ 每个步骤标注门控条件（✅/❌/⚠️）
- ✅ 步骤0实现"先读后写"
- ✅ 核心步骤包含对抗模型偷懒指令
- ✅ Output Format 结构化（表格/JSON格式）
- ✅ Constraints 11条，包含金融合规红线
- ✅ Audit Trail 格式完整
- ✅ Gotchas 5条实战踩坑记录
- ✅ Examples 2个端到端案例
- ✅ Out of Scope 已界定
- ✅ references/ 文件标注 data_valid_until
- ✅ provenance/ 目录结构完整（4个子目录）
- ✅ PROVENANCE.md 内容完整（≥100行）
- ✅ tests/test_cases.yaml 覆盖4种类型（positive/edge_case/negative/rejection）

## 审核状态
待审核
