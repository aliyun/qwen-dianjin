#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
进件检查数据格式验证脚本
用于验证 credit-case-intake-check Skill 步骤0的输入数据完整性和格式正确性
"""

import re
import sys
from datetime import datetime


class IntakeCheckValidator:
    """进件检查数据验证器"""
    
    # 合法的产品名称枚举
    VALID_PRODUCTS = [
        "流动资金贷款",
        "固定资产贷款",
        "银行承兑汇票",
        "商业承兑汇票贴现",
        "保函",
        "信用证",
        "贸易融资",
        "项目融资"
    ]
    
    def __init__(self, input_data):
        """
        初始化验证器
        
        Args:
            input_data: 字典,包含输入数据
                - customer_name: 客户名称(必填)
                - case_id: 案件编号(必填)
                - product_name: 产品名称(必填)
                - amount: 申请金额(必填)
                - materials: 申请材料清单(必填)
        """
        self.input_data = input_data
        self.errors = []
        self.warnings = []
    
    def _validate_required_fields(self):
        """验证必填字段"""
        required_fields = ['customer_name', 'case_id', 'product_name', 'amount', 'materials']
        for field in required_fields:
            if field not in self.input_data or self.input_data[field] is None:
                self.errors.append(f"必填字段缺失: {field}")
            elif isinstance(self.input_data[field], str) and self.input_data[field].strip() == '':
                self.errors.append(f"必填字段为空: {field}")
    
    def _validate_customer_name(self):
        """验证客户名称"""
        customer_name = self.input_data.get('customer_name', '')
        if len(customer_name) < 2:
            self.errors.append("客户名称过短(至少2个字符)")
        if len(customer_name) > 100:
            self.errors.append("客户名称过长(最多100个字符)")
    
    def _validate_case_id(self):
        """验证案件编号格式"""
        case_id = self.input_data.get('case_id', '')
        # 案件编号格式: CASE-YYYY-XXX
        case_id_pattern = re.compile(r'^CASE-\d{4}-\d{3,}$')
        if case_id and not case_id_pattern.match(case_id):
            self.warnings.append(f"案件编号格式不规范: {case_id}(建议格式:CASE-YYYY-XXX)")
    
    def _validate_product_name(self):
        """验证产品名称合法性"""
        product_name = self.input_data.get('product_name', '')
        if product_name and product_name not in self.VALID_PRODUCTS:
            self.warnings.append(f"产品名称不在标准列表中: {product_name}(标准产品:{', '.join(self.VALID_PRODUCTS[:3])}等)")
    
    def _validate_amount(self):
        """验证申请金额合理性"""
        amount = self.input_data.get('amount')
        if amount is not None:
            if not isinstance(amount, (int, float)):
                self.errors.append("申请金额必须为数字")
            elif amount <= 0:
                self.errors.append("申请金额必须大于0")
            elif amount > 10000000000:  # 100亿元
                self.warnings.append("申请金额超过100亿元,请确认是否正确")
    
    def _validate_materials(self):
        """验证申请材料清单"""
        materials = self.input_data.get('materials', [])
        if not isinstance(materials, list):
            self.errors.append("申请材料清单必须为列表")
        elif len(materials) == 0:
            self.warnings.append("申请材料清单为空,无法进行完整性检查")
    
    def validate(self) -> bool:
        """
        执行完整验证
        
        Returns:
            bool: 验证是否通过
        """
        self._validate_required_fields()
        self._validate_customer_name()
        self._validate_case_id()
        self._validate_product_name()
        self._validate_amount()
        self._validate_materials()
        
        return len(self.errors) == 0
    
    def get_report(self) -> str:
        """
        生成验证报告
        
        Returns:
            str: 格式化的验证报告
        """
        if len(self.errors) == 0 and len(self.warnings) == 0:
            return "✅ 验证通过,无错误或警告"
        
        report = []
        if self.errors:
            report.append("❌ 错误:")
            for error in self.errors:
                report.append(f"  - {error}")
        
        if self.warnings:
            report.append("⚠️ 警告:")
            for warning in self.warnings:
                report.append(f"  - {warning}")
        
        return "\n".join(report)


def main():
    """测试入口"""
    print("=" * 60)
    print("进件检查数据验证脚本 - 测试用例")
    print("=" * 60)
    
    # 测试用例1: 标准新客户首笔授信
    print("\n测试用例1: 标准新客户首笔授信")
    test_case_1 = {
        "customer_name": "XX机械制造有限公司",
        "case_id": "CASE-2026-001",
        "product_name": "流动资金贷款",
        "amount": 50000000,
        "materials": ["授信申请书", "营业执照", "法定代表人身份证", "财务报表", "征信授权书", "公司章程"]
    }
    validator_1 = IntakeCheckValidator(test_case_1)
    result_1 = validator_1.validate()
    print(f"验证结果: {'✅ 通过' if result_1 else '❌ 失败'}")
    print(validator_1.get_report())
    
    # 测试用例2: 客户名称为空
    print("\n测试用例2: 客户名称为空")
    test_case_2 = {
        "customer_name": "",
        "case_id": "CASE-2026-002",
        "product_name": "流动资金贷款",
        "amount": 50000000,
        "materials": ["授信申请书"]
    }
    validator_2 = IntakeCheckValidator(test_case_2)
    result_2 = validator_2.validate()
    print(f"验证结果: {'✅ 通过' if result_2 else '❌ 失败'}(预期失败)")
    print(validator_2.get_report())
    
    # 测试用例3: 非法产品名称
    print("\n测试用例3: 非法产品名称")
    test_case_3 = {
        "customer_name": "XX制造有限公司",
        "case_id": "CASE-2026-003",
        "product_name": "非法产品",
        "amount": 50000000,
        "materials": ["授信申请书"]
    }
    validator_3 = IntakeCheckValidator(test_case_3)
    result_3 = validator_3.validate()
    print(f"验证结果: {'✅ 通过' if result_3 else '❌ 失败'}(预期通过,带警告)")
    print(validator_3.get_report())
    
    # 测试用例4: 申请金额为负数
    print("\n测试用例4: 申请金额为负数")
    test_case_4 = {
        "customer_name": "XX制造有限公司",
        "case_id": "CASE-2026-004",
        "product_name": "流动资金贷款",
        "amount": -1000000,
        "materials": ["授信申请书"]
    }
    validator_4 = IntakeCheckValidator(test_case_4)
    result_4 = validator_4.validate()
    print(f"验证结果: {'✅ 通过' if result_4 else '❌ 失败'}(预期失败)")
    print(validator_4.get_report())
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
