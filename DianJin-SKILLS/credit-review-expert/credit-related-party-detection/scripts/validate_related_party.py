#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关联方数据格式验证脚本
用于验证 credit-related-party-detection Skill 步骤0的输入数据完整性和格式正确性
"""

import re
import sys
from datetime import datetime


class RelatedPartyValidator:
    """关联方数据验证器"""
    
    # 合法的排查场景枚举
    VALID_SCENARIOS = [
        "贷前排查",
        "贷中监控",
        "贷后预警"
    ]
    
    # 统一社会信用代码正则表达式(18位,支持占位符X)
    CREDIT_CODE_PATTERN = re.compile(r'^[0-9A-HJ-NPQRTUWXY]{2}\d{6}[0-9A-HJ-NPQRTUWXY]{10}$|^91110000MA\d{3}XXXX$')
    
    def __init__(self, input_data):
        """
        初始化验证器
        
        Args:
            input_data: 字典,包含输入数据
                - customer_name: 客户名称(必填)
                - credit_code: 统一社会信用代码(必填)
                - business_scenario: 排查场景(必填)
                - existing_credit_balance: 现有授信余额(可选)
                - application_amount: 申请授信金额(可选)
        """
        self.input_data = input_data
        self.errors = []
        self.warnings = []
    
    def _validate_required_fields(self):
        """验证必填字段"""
        customer_name = self.input_data.get('customer_name', '').strip()
        credit_code = self.input_data.get('credit_code', '').strip()
        business_scenario = self.input_data.get('business_scenario', '').strip()
        
        if not customer_name:
            self.errors.append("客户名称为空")
        
        if not credit_code:
            self.errors.append("统一社会信用代码为空")
        
        if not business_scenario:
            self.errors.append("排查场景为空")
    
    def _validate_credit_code_format(self):
        """验证统一社会信用代码格式"""
        credit_code = self.input_data.get('credit_code', '').strip()
        
        if credit_code and not self.CREDIT_CODE_PATTERN.match(credit_code):
            self.errors.append(f"统一社会信用代码格式错误: {credit_code}")
    
    def _validate_business_scenario(self):
        """验证排查场景合法性"""
        business_scenario = self.input_data.get('business_scenario', '').strip()
        
        if business_scenario and business_scenario not in self.VALID_SCENARIOS:
            self.errors.append(f"排查场景非法: {business_scenario},合法值为: {', '.join(self.VALID_SCENARIOS)}")
    
    def _validate_credit_amount(self):
        """验证授信金额合理性"""
        existing_balance = self.input_data.get('existing_credit_balance', 0)
        application_amount = self.input_data.get('application_amount', 0)
        
        if existing_balance < 0:
            self.errors.append(f"现有授信余额为负数: {existing_balance}")
        
        if application_amount < 0:
            self.errors.append(f"申请授信金额为负数: {application_amount}")
        
        if existing_balance > 0 and application_amount > 0:
            if application_amount > existing_balance * 3:
                self.warnings.append(f"申请授信金额({application_amount})超过现有授信余额({existing_balance})的3倍,建议核实")
    
    def validate(self) -> bool:
        """
        执行完整验证
        
        Returns:
            bool: 验证是否通过
        """
        self._validate_required_fields()
        self._validate_credit_code_format()
        self._validate_business_scenario()
        self._validate_credit_amount()
        
        return len(self.errors) == 0
    
    def get_report(self) -> str:
        """
        生成验证报告
        
        Returns:
            str: 格式化的验证报告
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("关联方数据验证报告")
        report_lines.append("=" * 60)
        
        if self.errors:
            report_lines.append(f"\n❌ 验证失败,发现 {len(self.errors)} 个错误:")
            for i, error in enumerate(self.errors, 1):
                report_lines.append(f"  {i}. {error}")
        
        if self.warnings:
            report_lines.append(f"\n⚠️ 警告,发现 {len(self.warnings)} 个警告:")
            for i, warning in enumerate(self.warnings, 1):
                report_lines.append(f"  {i}. {warning}")
        
        if not self.errors and not self.warnings:
            report_lines.append("\n✅ 验证通过,无错误或警告")
        
        report_lines.append("\n" + "=" * 60)
        return "\n".join(report_lines)


def main():
    """测试入口"""
    print("开始测试 validate_related_party.py 验证脚本...\n")
    
    # 测试用例1: 标准新客户贷前排查(应通过)
    test_case_1 = {
        "customer_name": "XX制造有限公司",
        "credit_code": "91110000MA001XXXX",
        "business_scenario": "贷前排查",
        "existing_credit_balance": 0,
        "application_amount": 50000000
    }
    validator1 = RelatedPartyValidator(test_case_1)
    result1 = validator1.validate()
    print(f"测试用例1: 标准新客户贷前排查")
    print(f"输入: {test_case_1}")
    print(validator1.get_report())
    print(f"预期: 通过, 实际: {'通过' if result1 else '失败'}")
    print(f"结果: {'✅ 通过' if result1 else '❌ 失败'}\n")
    
    # 测试用例2: 存量客户贷中监控(应通过)
    test_case_2 = {
        "customer_name": "XX贸易有限公司",
        "credit_code": "91110000MA002XXXX",
        "business_scenario": "贷中监控",
        "existing_credit_balance": 30000000,
        "application_amount": 0
    }
    validator2 = RelatedPartyValidator(test_case_2)
    result2 = validator2.validate()
    print(f"测试用例2: 存量客户贷中监控")
    print(f"输入: {test_case_2}")
    print(validator2.get_report())
    print(f"预期: 通过, 实际: {'通过' if result2 else '失败'}")
    print(f"结果: {'✅ 通过' if result2 else '❌ 失败'}\n")
    
    # 测试用例3: 客户名称为空(应失败)
    test_case_3 = {
        "customer_name": "",
        "credit_code": "",
        "business_scenario": "贷前排查"
    }
    validator3 = RelatedPartyValidator(test_case_3)
    result3 = validator3.validate()
    print(f"测试用例3: 客户名称为空")
    print(f"输入: {test_case_3}")
    print(validator3.get_report())
    print(f"预期: 失败, 实际: {'通过' if result3 else '失败'}")
    print(f"结果: {'❌ 失败(预期)' if not result3 else '✅ 通过(异常)'}\n")
    
    # 测试用例4: 排查场景非法(应失败)
    test_case_4 = {
        "customer_name": "XX制造有限公司",
        "credit_code": "91110000MA001XXXX",
        "business_scenario": "非法场景"
    }
    validator4 = RelatedPartyValidator(test_case_4)
    result4 = validator4.validate()
    print(f"测试用例4: 排查场景非法")
    print(f"输入: {test_case_4}")
    print(validator4.get_report())
    print(f"预期: 失败, 实际: {'通过' if result4 else '失败'}")
    print(f"结果: {'❌ 失败(预期)' if not result4 else '✅ 通过(异常)'}\n")
    
    # 汇总结果
    print("=" * 60)
    print("测试汇总")
    print("=" * 60)
    print(f"测试用例1: {'✅ 通过' if result1 else '❌ 失败'}")
    print(f"测试用例2: {'✅ 通过' if result2 else '❌ 失败'}")
    print(f"测试用例3: {'❌ 失败(预期)' if not result3 else '✅ 通过(异常)'}")
    print(f"测试用例4: {'❌ 失败(预期)' if not result4 else '✅ 通过(异常)'}")
    print("=" * 60)
    
    # 返回测试结果(0表示全部符合预期)
    if result1 and result2 and not result3 and not result4:
        print("\n✅ 所有测试符合预期!")
        sys.exit(0)
    else:
        print("\n❌ 部分测试不符合预期!")
        sys.exit(1)


if __name__ == "__main__":
    main()
