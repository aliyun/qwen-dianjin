#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准入规则扫描数据格式验证脚本
用于验证 admission-rules-scan Skill 步骤0的输入数据完整性和格式正确性
"""

import re
import sys
from datetime import datetime


class AdmissionScanValidator:
    """准入规则扫描数据验证器"""
    
    # 合法的产品名称枚举
    VALID_PRODUCTS = [
        "流动资金贷款",
        "固定资产贷款",
        "项目贷款",
        "银行承兑汇票",
        "商业承兑汇票贴现",
        "保函",
        "信用证",
        "贸易融资"
    ]
    
    def __init__(self, input_data):
        """
        初始化验证器
        
        Args:
            input_data: 字典,包含输入数据
                - customer_name: 客户名称(必填)
                - product_name: 产品名称(必填)
                - amount: 申请金额(建议)
        """
        self.input_data = input_data
        self.errors = []
        self.warnings = []
    
    def validate(self) -> bool:
        """执行完整验证"""
        self._validate_required_fields()
        self._validate_customer_name()
        self._validate_product_name()
        self._validate_amount()
        return len(self.errors) == 0
    
    def _validate_required_fields(self):
        """验证必填字段"""
        if not self.input_data.get('customer_name'):
            self.errors.append("必填字段缺失:客户名称")
        
        if not self.input_data.get('product_name'):
            self.errors.append("必填字段缺失:产品名称")
    
    def _validate_customer_name(self):
        """验证客户名称"""
        customer_name = self.input_data.get('customer_name', '')
        if customer_name and len(customer_name.strip()) < 2:
            self.errors.append("客户名称长度无效:至少2个字符")
    
    def _validate_product_name(self):
        """验证产品名称"""
        product_name = self.input_data.get('product_name', '')
        if product_name and product_name not in self.VALID_PRODUCTS:
            self.warnings.append(f"产品名称'{product_name}'不在标准产品列表中,将使用自定义规则")
    
    def _validate_amount(self):
        """验证申请金额"""
        amount = self.input_data.get('amount')
        if amount is not None:
            if not isinstance(amount, (int, float)):
                self.errors.append("申请金额无效:必须为数字")
            elif amount <= 0:
                self.errors.append("申请金额无效:必须大于0")
            elif amount > 1000000000:  # 10亿
                self.warnings.append(f"申请金额过大({amount}元),建议确认是否为大额授信")
    
    def get_report(self) -> str:
        """生成验证报告"""
        report = []
        report.append("=" * 60)
        report.append("准入规则扫描数据验证报告")
        report.append("=" * 60)
        
        if self.errors:
            report.append(f"\n❌ 错误({len(self.errors)}个):")
            for error in self.errors:
                report.append(f"  - {error}")
        
        if self.warnings:
            report.append(f"\n⚠️ 警告({len(self.warnings)}个):")
            for warning in self.warnings:
                report.append(f"  - {warning}")
        
        if not self.errors and not self.warnings:
            report.append("\n✅ 验证通过,无错误或警告")
        
        report.append(f"\n验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """命令行测试入口"""
    print("准入规则扫描验证脚本 - 测试用例\n")
    
    # 测试用例1: 标准新客户首笔授信
    print("测试用例1: 标准新客户首笔授信")
    test_case_1 = {
        "customer_name": "XX机械制造有限公司",
        "product_name": "流动资金贷款",
        "amount": 50000000
    }
    validator_1 = AdmissionScanValidator(test_case_1)
    result_1 = validator_1.validate()
    print(validator_1.get_report())
    print(f"预期: 通过, 实际: {'通过' if result_1 else '失败'}\n")
    
    # 测试用例2: 客户名称为空
    print("测试用例2: 客户名称为空")
    test_case_2 = {
        "customer_name": "",
        "product_name": "流动资金贷款",
        "amount": 50000000
    }
    validator_2 = AdmissionScanValidator(test_case_2)
    result_2 = validator_2.validate()
    print(validator_2.get_report())
    print(f"预期: 失败, 实际: {'通过' if result_2 else '失败'}\n")
    
    # 测试用例3: 非法产品名称
    print("测试用例3: 非法产品名称")
    test_case_3 = {
        "customer_name": "XX制造企业",
        "product_name": "非法产品",
        "amount": 30000000
    }
    validator_3 = AdmissionScanValidator(test_case_3)
    result_3 = validator_3.validate()
    print(validator_3.get_report())
    print(f"预期: 通过(带警告), 实际: {'通过' if result_3 else '失败'}\n")
    
    # 测试用例4: 申请金额为负数
    print("测试用例4: 申请金额为负数")
    test_case_4 = {
        "customer_name": "XX制造企业",
        "product_name": "流动资金贷款",
        "amount": -50000000
    }
    validator_4 = AdmissionScanValidator(test_case_4)
    result_4 = validator_4.validate()
    print(validator_4.get_report())
    print(f"预期: 失败, 实际: {'通过' if result_4 else '失败'}\n")


if __name__ == "__main__":
    main()
