# 步骤2:工具开发

## 输入
- 步骤1输出的方法论设计
- Python 编程环境
- 财务数据验证规则
- 输出报告模板

## 过程
1. 开发财务分析验证脚本 `scripts/validate_financial_report.py`,用于检查输入参数完整性和三表勾稽关系
2. 设计验证规则:
   - 企业名称、报告期格式校验
   - 三表勾稽关系检查(资产负债表期末与期初差 = 现金流量表期末现金)
   - 财务指标计算准确性检查
   - 红线核查完整性检查
3. 设计输出模板 `assets/financial-report-template.md`
4. 创建测试用例集 `tests/test_cases.yaml`,覆盖正例、反例、边界案例

## 输出
- 验证脚本 `scripts/validate_financial_report.py`(已创建)
- 输出模板 `assets/financial-report-template.md`(已创建)
- 测试用例集 `tests/test_cases.yaml`(12个用例,覆盖4种类型)
- Golden file `tests/golden/financial-report.md`(标准输出参照)

## 日期
2026-05-05

## 负责人
AI架构师团队 + 金融科技部

## 技术要点
1. 验证脚本必须检查三表勾稽关系,这是财务数据质量的核心指标
2. 测试用例覆盖4种类型:positive(5个)、edge_case(3个)、negative(3个)、rejection(1个)
3. 输出模板必须包含免责声明,引用 `shared/disclaimer-template.md`
