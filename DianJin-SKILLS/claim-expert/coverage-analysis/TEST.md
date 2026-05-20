# insurance-claim-coverage-analysis 手动测试指南

> **功能概述**: 识别理赔案件中可能触发的保险免责条款，评估拒赔风险。内部自动执行OCR提取。
> **所属类别**: 理赔

## 测试目标
验证 [insurance-claim-coverage-analysis](/Users/zhangjia/Documents/客户/保险/技能/保险技能/.qoder/skills/insurance-claim-coverage-analysis/SKILL.md) 的触发条件和核心功能是否正常运作。

## 测试数据

测试图片位于 `/Users/zhangjia/Documents/客户/保险/技能/保险技能/test_images/`：
- `invoice_sample.png` — 医疗发票
- `medical_record_sample.png` — 门诊病历
- `diagnosis_certificate_sample.png` — 诊断证明
- `fee_list_sample.png` — 费用清单

## 测试步骤

### 用例1：正常场景
1. 上传 `medical_record_sample.png`
2. 输入："请检查这个案子有没有触免责"
3. 提供保单条款（可用 `policy_rules.txt`）
4. 观察输出是否列出可能触发的免责项及风险等级

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
