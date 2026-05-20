#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
押品数据格式验证脚本
用于验证 credit-collateral-risk-mgmt Skill 步骤0的输入数据完整性和格式正确性
"""

import re
import sys
from datetime import datetime


class CollateralValidator:
    """押品数据验证器"""
    
    # 合法的业务场景枚举
    VALID_SCENARIOS = [
        "贷前评估",
        "审批审查",
        "贷后监控",
        "风险处置"
    ]
    
    # 合法的押品类别枚举
    VALID_COLLATERAL_TYPES = [
        "房产",
        "土地",
        "设备",
        "交通运输工具",
        "存货",
        "贵金属",
        "应收账款",
        "知识产权",
        "特许经营权",
        "收费权",
        "金融资产",
        "本行定期存单",
        "国债",
        "银行承兑汇票",
        "上市公司股票",
        "基金"
    ]
    
    # 抵押率上限字典
    MAX_LOAN_TO_VALUE = {
        "房产": 70,
        "土地": 60,
        "设备": 50,
        "交通运输工具": 60,
        "存货": 60,
        "贵金属": 80,
        "应收账款": 80,
        "知识产权": 30,
        "特许经营权": 40,
        "收费权": 50,
        "金融资产": 90,
        "本行定期存单": 90,
        "国债": 90,
        "银行承兑汇票": 85,
        "上市公司股票": 50,
        "基金": 60
    }
    
    # 日期格式正则表达式(YYYY-MM-DD)
    DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    
    def __init__(self, input_data):
        """
        初始化验证器
        
        Args:
            input_data: 字典,包含输入数据
                - customer_name: 客户名称(必填)
                - business_scenario: 业务场景(必填)
                - collateral_list: 押品清单(必填,列表)
                    - name: 押品名称(必填)
                    - type: 押品类别(必填)
                    - owner: 权属人(必填)
                    - appraised_value: 评估价值(必填,数字)
                    - appraisal_date: 评估基准日(必填,YYYY-MM-DD)
                - application_amount: 申请金额(可选,数字)
        """
        self.input_data = input_data
        self.errors = []
        self.warnings = []
    
    def validate(self) -> bool:
        """执行完整验证"""
        self._validate_required_fields()
        self._validate_business_scenario()
        self._validate_collateral_list()
        
        if 'application_amount' in self.input_data:
            self._validate_application_amount()
        
        return len(self.errors) == 0
    
    def _validate_required_fields(self):
        """验证必填字段"""
        required_fields = ['customer_name', 'business_scenario', 'collateral_list']
        
        for field in required_fields:
            if field not in self.input_data:
                self.errors.append(f"缺少必填字段: {field}")
            elif field == 'collateral_list':
                if not isinstance(self.input_data[field], list) or len(self.input_data[field]) == 0:
                    self.errors.append("押品清单必须为非空列表")
            elif not self.input_data[field]:
                self.errors.append(f"字段 {field} 不能为空")
    
    def _validate_business_scenario(self):
        """验证业务场景合法性"""
        scenario = self.input_data.get('business_scenario', '')
        
        if scenario and scenario not in self.VALID_SCENARIOS:
            self.errors.append(
                f"非法业务场景: {scenario}。合法值: {', '.join(self.VALID_SCENARIOS)}"
            )
    
    def _validate_collateral_list(self):
        """验证押品清单"""
        collateral_list = self.input_data.get('collateral_list', [])
        
        for i, collateral in enumerate(collateral_list):
            self._validate_collateral_item(collateral, i + 1)
    
    def _validate_collateral_item(self, collateral, index):
        """验证单个押品项"""
        # 验证必填字段
        required_fields = ['name', 'type', 'owner', 'appraised_value', 'appraisal_date']
        
        for field in required_fields:
            if field not in collateral:
                self.errors.append(f"押品{index}缺少必填字段: {field}")
            elif field == 'name' and not collateral[field]:
                self.errors.append(f"押品{index}名称为空")
            elif field == 'type' and collateral[field] not in self.VALID_COLLATERAL_TYPES:
                self.errors.append(
                    f"押品{index}非法类别: {collateral[field]}。"
                    f"合法值: {', '.join(self.VALID_COLLATERAL_TYPES)}"
                )
            elif field == 'appraised_value':
                if not isinstance(collateral[field], (int, float)) or collateral[field] <= 0:
                    self.errors.append(f"押品{index}评估价值必须为正数")
            elif field == 'appraisal_date':
                if not self.DATE_PATTERN.match(str(collateral[field])):
                    self.errors.append(f"押品{index}评估基准日格式错误,应为YYYY-MM-DD")
                else:
                    self._validate_appraisal_date(collateral[field], index)
    
    def _validate_appraisal_date(self, appraisal_date, index):
        """验证评估基准日有效性(≤6个月)"""
        try:
            appraisal_dt = datetime.strptime(appraisal_date, '%Y-%m-%d')
            today = datetime.now()
            months_diff = (today.year - appraisal_dt.year) * 12 + (today.month - appraisal_dt.month)
            
            if months_diff > 6:
                self.errors.append(f"押品{index}评估基准日超期失效(距申请日{months_diff}个月>6个月)")
            elif months_diff >= 5:
                self.warnings.append(f"押品{index}评估报告即将过期(距申请日{months_diff}个月),建议尽快重新评估")
        except ValueError:
            self.errors.append(f"押品{index}评估基准日解析失败")
    
    def _validate_application_amount(self):
        """验证申请金额合理性"""
        amount = self.input_data.get('application_amount', 0)
        
        if not isinstance(amount, (int, float)) or amount <= 0:
            self.errors.append("申请金额必须为正数")
        
        # 计算押品总价值
        collateral_list = self.input_data.get('collateral_list', [])
        total_appraised_value = sum(
            c.get('appraised_value', 0) for c in collateral_list
            if isinstance(c.get('appraised_value'), (int, float))
        )
        
        # 计算最大可担保金额
        max_guarantee_amount = 0
        for collateral in collateral_list:
            collateral_type = collateral.get('type', '')
            appraised_value = collateral.get('appraised_value', 0)
            max_ltv = self.MAX_LOAN_TO_VALUE.get(collateral_type, 50)
            max_guarantee_amount += appraised_value * max_ltv / 100
        
        # 检查是否超过最大可担保金额
        if amount > max_guarantee_amount and max_guarantee_amount > 0:
            self.warnings.append(
                f"申请金额{amount}万元超过最大可担保金额{max_guarantee_amount:.2f}万元,担保不足"
            )
    
    def get_report(self) -> str:
        """生成验证报告"""
        report = []
        report.append("=" * 60)
        report.append("押品数据验证报告")
        report.append("=" * 60)
        
        if self.errors:
            report.append(f"\n❌ 错误({len(self.errors)}项):")
            for error in self.errors:
                report.append(f"  - {error}")
        
        if self.warnings:
            report.append(f"\n⚠️ 警告({len(self.warnings)}项):")
            for warning in self.warnings:
                report.append(f"  - {warning}")
        
        if not self.errors and not self.warnings:
            report.append("\n✅ 验证通过,无错误或警告")
        elif not self.errors:
            report.append(f"\n✅ 验证通过,但有{len(self.warnings)}项警告")
        else:
            report.append(f"\n❌ 验证失败,存在{len(self.errors)}项错误")
        
        report.append("=" * 60)
        return '\n'.join(report)


def main():
    """测试入口"""
    # 测试用例1: 标准贷前押品评估
    test_case_1 = {
        "customer_name": "XX制造有限公司",
        "business_scenario": "贷前评估",
        "collateral_list": [
            {
                "name": "商业用房",
                "type": "房产",
                "owner": "XX制造有限公司",
                "appraised_value": 15000000,
                "appraisal_date": "2026-03-15"
            },
            {
                "name": "通用生产设备",
                "type": "设备",
                "owner": "XX制造有限公司",
                "appraised_value": 3000000,
                "appraisal_date": "2026-04-01"
            }
        ],
        "application_amount": 10000000
    }
    
    # 测试用例2: 押品名称为空
    test_case_2 = {
        "customer_name": "XX企业",
        "business_scenario": "贷前评估",
        "collateral_list": [
            {
                "name": "",
                "type": "房产",
                "owner": "XX企业",
                "appraised_value": 10000000,
                "appraisal_date": "2026-04-01"
            }
        ]
    }
    
    # 测试用例3: 非法业务场景
    test_case_3 = {
        "customer_name": "XX企业",
        "business_scenario": "非法场景",
        "collateral_list": [
            {
                "name": "商业用房",
                "type": "房产",
                "owner": "XX企业",
                "appraised_value": 10000000,
                "appraisal_date": "2026-04-01"
            }
        ]
    }
    
    # 测试用例4: 评估报告超期
    test_case_4 = {
        "customer_name": "XX企业",
        "business_scenario": "贷前评估",
        "collateral_list": [
            {
                "name": "商业用房",
                "type": "房产",
                "owner": "XX企业",
                "appraised_value": 10000000,
                "appraisal_date": "2025-01-01"
            }
        ]
    }
    
    print("测试用例1: 标准贷前押品评估")
    validator1 = CollateralValidator(test_case_1)
    result1 = validator1.validate()
    print(validator1.get_report())
    print()
    
    print("测试用例2: 押品名称为空")
    validator2 = CollateralValidator(test_case_2)
    result2 = validator2.validate()
    print(validator2.get_report())
    print()
    
    print("测试用例3: 非法业务场景")
    validator3 = CollateralValidator(test_case_3)
    result3 = validator3.validate()
    print(validator3.get_report())
    print()
    
    print("测试用例4: 评估报告超期")
    validator4 = CollateralValidator(test_case_4)
    result4 = validator4.validate()
    print(validator4.get_report())


if __name__ == '__main__':
    main()
