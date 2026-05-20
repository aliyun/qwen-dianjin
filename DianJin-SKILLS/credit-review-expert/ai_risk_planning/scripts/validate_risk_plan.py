#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信贷风控任务规划数据格式验证脚本
用于验证 ai-risk-planning Skill 步骤0的输入数据完整性和格式正确性
"""

import re
import sys
from datetime import datetime


class RiskPlanValidator:
    """风控任务规划数据验证器"""
    
    # 合法的产品名称枚举
    VALID_PRODUCTS = [
        "流动资金贷款",
        "固定资产贷款",
        "项目贷款",
        "银行承兑汇票",
        "商业承兑汇票贴现",
        "保函",
        "信用证",
        "贸易融资",
        "科技授信",
        "创业贷款",
        "项目开发贷款"
    ]
    
    def __init__(self, input_data):
        """
        初始化验证器
        
        Args:
            input_data: 字典,包含输入数据
                - applicant: 申请人姓名(必填)
                - company: 借款企业名称(必填)
                - industry: 所属行业(必填)
                - product: 产品类型(必填)
                - amount: 贷款金额(必填)
                - purpose: 贷款用途(必填)
                - term_months: 贷款期限(建议)
        """
        self.input_data = input_data
        self.errors = []
        self.warnings = []
    
    def validate_required_fields(self):
        """验证必填字段"""
        required_fields = ['applicant', 'company', 'industry', 'product', 'amount', 'purpose']
        
        for field in required_fields:
            if field not in self.input_data or not self.input_data[field]:
                self.errors.append(f"必填字段缺失: {field}")
    
    def validate_applicant_name(self):
        """验证申请人姓名"""
        applicant = self.input_data.get('applicant', '')
        if applicant and len(applicant.strip()) < 2:
            self.errors.append("申请人姓名长度不足(≥2字符)")
    
    def validate_company_name(self):
        """验证企业名称"""
        company = self.input_data.get('company', '')
        if company and len(company.strip()) < 2:
            self.errors.append("企业名称长度不足(≥2字符)")
    
    def validate_industry(self):
        """验证行业信息"""
        industry = self.input_data.get('industry', '')
        if industry:
            # 检查是否精确到二级行业(建议格式: C34 通用设备制造业)
            if len(industry) < 5 or ' ' not in industry:
                self.warnings.append("行业信息建议精确到二级行业(如C34 通用设备制造业)")
    
    def validate_product(self):
        """验证产品名称"""
        product = self.input_data.get('product', '')
        if product and product not in self.VALID_PRODUCTS:
            self.warnings.append(f"产品名称'{product}'不在标准产品清单中,建议使用标准产品名称")
    
    def validate_amount(self):
        """验证贷款金额"""
        amount = self.input_data.get('amount')
        if amount is not None:
            try:
                amount = float(amount)
                if amount <= 0:
                    self.errors.append("贷款金额必须为正数")
                elif amount > 1000000000:  # 10亿
                    self.warnings.append("贷款金额超过10亿元,建议加强审查")
            except (ValueError, TypeError):
                self.errors.append("贷款金额格式错误(应为数字)")
    
    def validate_purpose(self):
        """验证贷款用途"""
        purpose = self.input_data.get('purpose', '')
        if purpose and len(purpose.strip()) < 5:
            self.errors.append("贷款用途描述过短(≥5字符)")
        
        # 检查违规用途关键词
        违规关键词 = ['股市', '房地产投机', '虚拟货币', '期货投机']
        for keyword in 违规关键词:
            if keyword in purpose:
                self.errors.append(f"贷款用途包含违规关键词: {keyword}")
    
    def validate(self) -> bool:
        """执行完整验证"""
        self.validate_required_fields()
        self.validate_applicant_name()
        self.validate_company_name()
        self.validate_industry()
        self.validate_product()
        self.validate_amount()
        self.validate_purpose()
        
        return len(self.errors) == 0
    
    def get_report(self) -> str:
        """生成验证报告"""
        report = []
        report.append("=" * 60)
        report.append("信贷风控任务规划 - 数据验证报告")
        report.append("=" * 60)
        report.append(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"客户名称: {self.input_data.get('company', '未知')}")
        report.append(f"产品名称: {self.input_data.get('product', '未知')}")
        report.append(f"申请金额: {self.input_data.get('amount', '未知')}元")
        report.append("")
        
        if self.errors:
            report.append(f"❌ 错误数: {len(self.errors)}")
            for i, error in enumerate(self.errors, 1):
                report.append(f"  {i}. {error}")
            report.append("")
        
        if self.warnings:
            report.append(f"⚠️ 警告数: {len(self.warnings)}")
            for i, warning in enumerate(self.warnings, 1):
                report.append(f"  {i}. {warning}")
            report.append("")
        
        if not self.errors and not self.warnings:
            report.append("✅ 验证通过,无错误或警告")
        elif not self.errors:
            report.append("✅ 验证通过(有警告)")
        else:
            report.append("❌ 验证失败")
        
        report.append("=" * 60)
        return "\n".join(report)


def main():
    """主函数 - 测试验证脚本"""
    print("=" * 60)
    print("信贷风控任务规划 - 验证脚本测试")
    print("=" * 60)
    print()
    
    # 测试用例1: 标准新客户首笔授信
    print("测试用例1: 标准制造业企业流动资金贷款")
    test_data_1 = {
        "applicant": "张三",
        "company": "XX机械制造有限公司",
        "industry": "C34 通用设备制造业",
        "product": "流动资金贷款",
        "amount": 50000000,
        "purpose": "购买原材料",
        "term_months": 12
    }
    validator_1 = RiskPlanValidator(test_data_1)
    result_1 = validator_1.validate()
    print(validator_1.get_report())
    print()
    
    # 测试用例2: 申请人姓名为空
    print("测试用例2: 申请人姓名为空")
    test_data_2 = {
        "applicant": "",
        "company": "XX公司",
        "industry": "C34 通用设备制造业",
        "product": "流动资金贷款",
        "amount": 50000000,
        "purpose": "购买原材料",
        "term_months": 12
    }
    validator_2 = RiskPlanValidator(test_data_2)
    result_2 = validator_2.validate()
    print(validator_2.get_report())
    print()
    
    # 测试用例3: 非法产品名称
    print("测试用例3: 非法产品名称")
    test_data_3 = {
        "applicant": "张三",
        "company": "XX公司",
        "industry": "C34 通用设备制造业",
        "product": "未知产品",
        "amount": 50000000,
        "purpose": "购买原材料",
        "term_months": 12
    }
    validator_3 = RiskPlanValidator(test_data_3)
    result_3 = validator_3.validate()
    print(validator_3.get_report())
    print()
    
    # 测试用例4: 贷款金额为负数
    print("测试用例4: 贷款金额为负数")
    test_data_4 = {
        "applicant": "张三",
        "company": "XX公司",
        "industry": "C34 通用设备制造业",
        "product": "流动资金贷款",
        "amount": -50000000,
        "purpose": "购买原材料",
        "term_months": 12
    }
    validator_4 = RiskPlanValidator(test_data_4)
    result_4 = validator_4.validate()
    print(validator_4.get_report())
    print()
    
    # 总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"测试用例1(标准场景): {'✅ 通过' if result_1 else '❌ 失败'}")
    print(f"测试用例2(姓名为空): {'❌ 失败(预期)' if not result_2 else '✅ 通过(异常)'}")
    print(f"测试用例3(非法产品): {'✅ 通过(带警告)' if result_3 else '❌ 失败'}")
    print(f"测试用例4(金额负数): {'❌ 失败(预期)' if not result_4 else '✅ 通过(异常)'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
