# 步骤 3：测试验证体系构建

## 输入
- 步骤 1-2 的产出
- skill-refactor 操作指南的测试要求

## 过程
1. 设计测试用例覆盖 4 种类型（共 9 个用例）：
   - **positive（3 个）**：标准制造业财务异常、多重风险信号叠加、技术迭代前沿企业
   - **edge_case（2 个）**：仅一年财务报表、行业基准数据不可用
   - **negative（2 个）**：财务数据勾稽不平衡、金融企业不适用制造业指标
   - **rejection（2 个）**：无原始数据源（终止输出）、个人信贷风险评估（超范围）
2. 为每个用例定义完整的 expected 字段：
   - contains / must_not_contain / disclaimer_present / audit_log_generated
   - degraded_mode / validation_failed / framework_adapted / data_termination_triggered / rejected
3. 创建验证脚本 validate_financial_data.py：
   - 报表平衡性检查（资产=负债+权益，1% 容差）
   - 数据连续性检查（至少 2 期）
   - 关键科目非空检查（9 个必检科目）
4. 创建 tests/golden/ 目录（预留给标准输出参照文件）

## 输出
- tests/test_cases.yaml（9 个用例，4 种类型）
- scripts/validate_financial_data.py
- scripts/requirements.txt
- tests/golden/ 目录

## 日期
2026-05-06

## 负责人
AI 架构师
