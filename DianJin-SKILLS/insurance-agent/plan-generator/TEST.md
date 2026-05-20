# insurance-underwriting-plan-generator 手动测试指南

> **功能概述**: "保险计划书生成器。根据客户画像、保障缺口分析和产品组合推荐，自动生成专业Word(.docx)格式保险计划书。支持从客户画像技能和产品推荐技能导入数据，也支持独立手动录入。自动填充客户信息、产品方案...
> **所属类别**: 承保

## 测试目标
验证 [insurance-underwriting-plan-generator](/Users/zhangjia/Documents/客户/保险/技能/保险技能/.qoder/skills/insurance-underwriting-plan-generator/SKILL.md) 的触发条件和核心功能是否正常运作。

## 测试数据

测试数据位于 `/Users/zhangjia/Documents/客户/保险/技能/保险技能/test_data/` 和 `/Users/zhangjia/Documents/客户/保险/技能/保险技能/test_images/`：
- `customer_renewal.xlsx` — 车险续保客户数据
- `bid_document.txt` — 招标文件
- `policy_rules.txt` — 承保政策文本
- `policy_guidelines.xlsx` — 政策指引Excel
- `medical_report.png` — 体检报告
- `underwriting_materials/` — 核保材料目录

## 测试步骤

### 用例1：正常场景
1. 输入："帮我生成一份保险计划书"
2. 提供客户画像：35岁男，需求重疾+医疗
3. 观察输出是否为Word格式计划书
4. 验证文件生成成功且格式合规

**预期结果**: Skill 正常触发，输出结果完整、格式正确、逻辑合理。

### 用例2：边界场景
1. 提供不完整或边界数据（如仅提供部分材料、预算极低/极高）
2. 观察 Skill 是否能 graceful 处理

**预期结果**: 不崩溃，给出明确提示（如"数据不足"、"建议补充XXX"）。

### 用例3：异常场景
1. 提供明显错误数据（如不存在的文件路径、无关图片、矛盾信息）
2. 观察 Skill 的错误处理

**预期结果**: 给出清晰的错误提示，不编造数据，不输出误导信息。

## 验收 checklist

- [ ] Skill 能被正确触发
- [ ] 输出包含预期字段/章节
- [ ] 对异常输入有合理处理
- [ ] 输出语言通俗易懂（如适用）
- [ ] 不泄露敏感信息
- [ ] 响应时间在合理范围内（<30s）

## 问题记录

| 问题描述 | 严重程度 | 复现步骤 | 截图/日志 |
|---------|---------|---------|----------|
|         |         |         |          |

---
*本测试指南由自动化脚本生成，如有疑问请查阅 SKILL.md*
