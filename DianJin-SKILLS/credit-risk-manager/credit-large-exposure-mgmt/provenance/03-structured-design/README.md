# 步骤3: 结构化设计

## 输入
- 规则提取成果
- 输出格式JSON Schema定义

## 过程
- 设计验证脚本validate_exposure.py
- 定义13个测试用例(覆盖positive/edge_case/negative/rejection 4种类型)
- 设计输出格式JSON结构(exposure_meta/total_exposure/exposure_breakdown/warning_level等)
- 创建免责声明模板assets/disclaimer-template.md
- 设计审计日志JSON格式

## 输出
- 验证脚本scripts/validate_exposure.py(259行)
- 测试用例tests/test_cases.yaml(13个用例)
- 输出格式JSON Schema

## 验证逻辑
- 必填字段检查(7个顶层字段)
- exposure_meta合法性验证(客户类型/业务场景/资本净额)
- total_exposure合法性验证(净风险暴露/集中度比例/监管红线)
- exposure_breakdown数量验证(≥1个业务类型)
- warning_level合法性验证(关注/黄色/橙色/红色/突破红线)
- penetration_summary合法性验证(已穿透/未穿透/匿名客户比例)
- disclaimer验证(包含"仅供参考"、"不构成"关键词)
