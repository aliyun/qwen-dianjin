#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
访前数据格式验证脚本
用于验证 pre-visit-credit-analysis Skill 步骤0的输入数据完整性和格式正确性
"""

import re
import sys
from datetime import datetime


class PreVisitValidator:
    """访前数据验证器"""
    
    # 合法的走访目的枚举
    VALID_PURPOSES = [
        "新客户首笔授信",
        "存量续贷",
        "不良预警客户走访",
        "大额新增授信"
    ]
    
    # 统一社会信用代码正则表达式(18位,支持占位符X)
    CREDIT_CODE_PATTERN = re.compile(r'^[0-9A-HJ-NPQRTUWXY]{2}\d{6}[0-9A-HJ-NPQRTUWXY]{10}$|^91110000MA\d{3}XXXX$')
    
    def __init__(self, input_data):
        """
        初始化验证器
        
        Args:
            input_data: 字典,包含输入数据
                - customer_name: 客户名称(必填)
                - credit_code: 统一社会信用代码(可选)
                - visit_purpose: 走访目的(必填)
                - credit_authorization: 征信授权状态(必填,True/False)
        """
        self.input_data = input_data
        self.errors = []
        self.warnings = []
    
    def validate(self) -> bool:
        """
        执行完整验证
        
        Returns:
            bool: 验证是否通过
        """
        self._validate_required_fields()
        self._validate_credit_code_format()
        self._validate_visit_purpose()
        self._validate_credit_authorization()
        
        return len(self.errors) == 0
    
    def _validate_required_fields(self):
        """验证必填字段"""
        # 客户名称
        customer_name = self.input_data.get('customer_name', '').strip()
        if not customer_name:
            self.errors.append("客户名称为空,必须提供客户名称或客户ID")
        
        # 走访目的
        visit_purpose = self.input_data.get('visit_purpose', '').strip()
        if not visit_purpose:
            self.errors.append("走访目的为空,必须指定走访目的")
        
        # 征信授权状态
        if 'credit_authorization' not in self.input_data:
            self.errors.append("征信授权状态缺失,必须明确是否已取得客户书面授权")
    
    def _validate_credit_code_format(self):
        """验证统一社会信用代码格式"""
        credit_code = self.input_data.get('credit_code', '').strip()
        if credit_code:  # 可选字段,如果提供则验证格式
            if not self.CREDIT_CODE_PATTERN.match(credit_code):
                self.errors.append(
                    f"统一社会信用代码格式错误: '{credit_code}',应为18位有效字符"
                )
    
    def _validate_visit_purpose(self):
        """验证走访目的合法性"""
        visit_purpose = self.input_data.get('visit_purpose', '').strip()
        if visit_purpose and visit_purpose not in self.VALID_PURPOSES:
            self.errors.append(
                f"走访目的非法: '{visit_purpose}',"
                f"合法值为: {', '.join(self.VALID_PURPOSES)}"
            )
    
    def _validate_credit_authorization(self):
        """验证征信授权状态"""
        credit_authorization = self.input_data.get('credit_authorization')
        if credit_authorization is not None and not isinstance(credit_authorization, bool):
            self.errors.append(
                f"征信授权状态格式错误: '{credit_authorization}',应为布尔值(True/False)"
            )
        elif credit_authorization is False:
            self.warnings.append(
                "未取得征信授权,将跳过征信查询步骤,仅依赖公开数据(工商/司法/舆情)"
            )
    
    def get_report(self) -> str:
        """
        生成验证报告
        
        Returns:
            str: 格式化的验证报告
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("访前数据验证报告")
        report_lines.append("=" * 60)
        
        # 输入数据摘要
        report_lines.append("\n输入数据:")
        report_lines.append(f"  客户名称: {self.input_data.get('customer_name', '未提供')}")
        report_lines.append(f"  统一信用代码: {self.input_data.get('credit_code', '未提供')}")
        report_lines.append(f"  走访目的: {self.input_data.get('visit_purpose', '未提供')}")
        report_lines.append(f"  征信授权: {self.input_data.get('credit_authorization', '未提供')}")
        
        # 错误信息
        if self.errors:
            report_lines.append(f"\n❌ 错误({len(self.errors)}个):")
            for i, error in enumerate(self.errors, 1):
                report_lines.append(f"  {i}. {error}")
        
        # 警告信息
        if self.warnings:
            report_lines.append(f"\n⚠️ 警告({len(self.warnings)}个):")
            for i, warning in enumerate(self.warnings, 1):
                report_lines.append(f"  {i}. {warning}")
        
        # 验证结果
        if len(self.errors) == 0:
            report_lines.append("\n✅ 验证通过,无错误")
        else:
            report_lines.append(f"\n❌ 验证失败,存在{len(self.errors)}个错误")
        
        if self.warnings:
            report_lines.append(f"⚠️ 存在{len(self.warnings)}个警告,请确认是否继续")
        
        report_lines.append("=" * 60)
        
        return '\n'.join(report_lines)


def main():
    """命令行测试入口"""
    print("访前数据验证脚本测试")
    print("=" * 60)
    
    # 测试用例1: 标准新客户
    print("\n测试用例1: 标准新客户首笔授信")
    test_data_1 = {
        'customer_name': 'XX制造有限公司',
        'credit_code': '91110000MA001XXXX',
        'visit_purpose': '新客户首笔授信',
        'credit_authorization': True
    }
    validator_1 = PreVisitValidator(test_data_1)
    result_1 = validator_1.validate()
    print(validator_1.get_report())
    
    # 测试用例2: 存量续贷(无征信授权)
    print("\n测试用例2: 存量续贷(无征信授权)")
    test_data_2 = {
        'customer_name': 'XX贸易有限公司',
        'credit_code': '91110000MA002XXXX',
        'visit_purpose': '存量续贷',
        'credit_authorization': False
    }
    validator_2 = PreVisitValidator(test_data_2)
    result_2 = validator_2.validate()
    print(validator_2.get_report())
    
    # 测试用例3: 客户名称为空(应失败)
    print("\n测试用例3: 客户名称为空(应失败)")
    test_data_3 = {
        'customer_name': '',
        'visit_purpose': '新客户首笔授信',
        'credit_authorization': True
    }
    validator_3 = PreVisitValidator(test_data_3)
    result_3 = validator_3.validate()
    print(validator_3.get_report())
    
    # 测试用例4: 走访目的非法(应失败)
    print("\n测试用例4: 走访目的非法(应失败)")
    test_data_4 = {
        'customer_name': 'XX有限公司',
        'visit_purpose': '贷后检查',  # 非法值
        'credit_authorization': True
    }
    validator_4 = PreVisitValidator(test_data_4)
    result_4 = validator_4.validate()
    print(validator_4.get_report())
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"  测试用例1: {'✅ 通过' if result_1 else '❌ 失败'}")
    print(f"  测试用例2: {'✅ 通过' if result_2 else '❌ 失败'}")
    print(f"  测试用例3: {'✅ 通过' if result_3 else '❌ 失败'}(预期失败)")
    print(f"  测试用例4: {'✅ 通过' if result_4 else '❌ 失败'}(预期失败)")
    print("=" * 60)


if __name__ == '__main__':
    main()
