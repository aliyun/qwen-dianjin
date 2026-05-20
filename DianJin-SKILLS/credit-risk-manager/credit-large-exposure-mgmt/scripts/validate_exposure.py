#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大额风险暴露集中度监控输出验证脚本

验证JSON输出格式的合法性和业务规则合规性
"""

import json
import sys


class ExposureValidator:
    """大额风险暴露数据验证器"""
    
    def __init__(self, input_data):
        self.input_data = input_data
        self.errors = []
        self.warnings = []
    
    def validate_required_fields(self):
        """验证必填字段"""
        required_fields = [
            'exposure_meta', 'total_exposure', 'exposure_breakdown',
            'warning_level', 'penetration_summary', 'disclaimer'
        ]
        
        for field in required_fields:
            if field not in self.input_data or self.input_data[field] is None:
                self.errors.append(f"必填字段缺失: {field}")
    
    def validate_exposure_meta(self):
        """验证分析元数据"""
        meta = self.input_data.get('exposure_meta', {})
        
        if 'customer_name' not in meta or not meta['customer_name']:
            self.errors.append("客户名称缺失")
        
        if 'customer_type' not in meta:
            self.errors.append("客户类型缺失")
        elif meta['customer_type'] not in ['单一客户', '集团客户', '关联方', '匿名客户']:
            self.errors.append(f"客户类型非法: {meta['customer_type']}")
        
        if 'business_scenario' not in meta:
            self.errors.append("业务场景缺失")
        elif meta['business_scenario'] not in ['新增预检', '日常监测', '集团管理', '压降跟踪', '监管报送']:
            self.errors.append(f"业务场景非法: {meta['business_scenario']}")
        
        if 'tier1_capital_net' not in meta or meta['tier1_capital_net'] <= 0:
            self.errors.append("一级资本净额必须为正数")
        
        if 'total_capital_net' not in meta or meta['total_capital_net'] <= 0:
            self.errors.append("资本净额必须为正数")
    
    def validate_total_exposure(self):
        """验证风险暴露汇总"""
        exposure = self.input_data.get('total_exposure', {})
        
        if 'net_exposure' not in exposure or exposure['net_exposure'] < 0:
            self.errors.append("净风险暴露必须为非负数")
        
        if 'concentration_ratio' not in exposure:
            self.errors.append("集中度比例缺失")
        elif exposure['concentration_ratio'] < 0:
            self.errors.append("集中度比例必须为非负数")
        
        if 'regulatory_limit' not in exposure or exposure['regulatory_limit'] <= 0:
            self.errors.append("监管红线必须为正数")
        
        if 'internal_limit' not in exposure or exposure['internal_limit'] <= 0:
            self.errors.append("内部管控线必须为正数")
    
    def validate_exposure_breakdown(self):
        """验证风险暴露明细"""
        breakdown = self.input_data.get('exposure_breakdown', [])
        
        if len(breakdown) == 0:
            self.errors.append("风险暴露明细不能为空")
        
        valid_business_types = ['贷款', '票据', '债券投资', '担保承诺', '同业投资', '衍生品']
        
        for item in breakdown:
            if 'business_type' not in item:
                self.errors.append("风险暴露明细缺少业务类型")
            elif item['business_type'] not in valid_business_types:
                self.warnings.append(f"非标准业务类型: {item['business_type']}")
            
            if 'net_amount' not in item or item['net_amount'] < 0:
                self.errors.append("风险暴露明细净金额必须为非负数")
    
    def validate_warning_level(self):
        """验证预警等级"""
        warning_level = self.input_data.get('warning_level', '')
        
        valid_levels = ['关注', '黄色', '橙色', '红色', '突破红线']
        
        if warning_level not in valid_levels:
            self.errors.append(f"预警等级非法: {warning_level}, 应为: {', '.join(valid_levels)}")
    
    def validate_penetration_summary(self):
        """验证穿透计量汇总"""
        penetration = self.input_data.get('penetration_summary', {})
        
        if 'total_penetrated' not in penetration or penetration['total_penetrated'] < 0:
            self.errors.append("已穿透数量必须为非负数")
        
        if 'total_unpenetrated' not in penetration or penetration['total_unpenetrated'] < 0:
            self.errors.append("未穿透数量必须为非负数")
        
        if 'anonymous_customer_ratio' in penetration:
            ratio = penetration['anonymous_customer_ratio']
            if ratio < 0 or ratio > 100:
                self.errors.append("匿名客户比例必须在0-100之间")
            elif ratio > 12:
                self.warnings.append(f"匿名客户比例过高: {ratio}%, 建议关注")
    
    def validate_disclaimer(self):
        """验证免责声明"""
        disclaimer = self.input_data.get('disclaimer', '')
        
        if not disclaimer:
            self.errors.append("免责声明缺失")
        elif '仅供参考' not in disclaimer or '不构成' not in disclaimer:
            self.errors.append("免责声明格式不正确,须包含'仅供参考'和'不构成'关键词")
    
    def validate(self):
        """执行完整验证"""
        self.validate_required_fields()
        self.validate_exposure_meta()
        self.validate_total_exposure()
        self.validate_exposure_breakdown()
        self.validate_warning_level()
        self.validate_penetration_summary()
        self.validate_disclaimer()
        
        return len(self.errors) == 0
    
    def get_report(self):
        """生成验证报告"""
        is_valid = self.validate()
        
        report = []
        report.append("=" * 60)
        report.append("大额风险暴露输出验证报告")
        report.append("=" * 60)
        report.append(f"验证结果: {'✅ 通过' if is_valid else '❌ 失败'}")
        report.append(f"错误数量: {len(self.errors)}")
        report.append(f"警告数量: {len(self.warnings)}")
        report.append("")
        
        if self.errors:
            report.append("错误清单:")
            for i, error in enumerate(self.errors, 1):
                report.append(f"  {i}. {error}")
            report.append("")
        
        if self.warnings:
            report.append("警告清单:")
            for i, warning in enumerate(self.warnings, 1):
                report.append(f"  {i}. {warning}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """测试验证脚本"""
    print("测试大额风险暴露验证脚本...")
    
    # 测试用例1:标准场景
    standard_data = {
        "exposure_meta": {
            "customer_name": "XX集团",
            "customer_type": "集团客户",
            "business_scenario": "日常监测",
            "data_as_of": "2026-05-05",
            "tier1_capital_net": 800000,
            "total_capital_net": 1000000
        },
        "total_exposure": {
            "gross_exposure": 120000,
            "risk_mitigation_amount": 10000,
            "net_exposure": 110000,
            "concentration_ratio": 13.75,
            "limit_type": "单一集团客户风险暴露",
            "regulatory_limit": 20.0,
            "internal_limit": 16.0
        },
        "exposure_breakdown": [
            {
                "business_type": "贷款",
                "gross_amount": 50000,
                "risk_mitigation": 5000,
                "net_amount": 45000,
                "penetration_required": False,
                "penetration_completed": True
            },
            {
                "business_type": "同业投资",
                "gross_amount": 70000,
                "risk_mitigation": 5000,
                "net_amount": 65000,
                "penetration_required": True,
                "penetration_completed": True
            }
        ],
        "warning_level": "黄色",
        "warning_details": {
            "triggered_at": "85.94%",
            "response_required_by": "3个工作日内",
            "response_measures": "限制新增非低风险业务,发送风险提示函"
        },
        "penetration_summary": {
            "total_penetrated": 8,
            "total_unpenetrated": 2,
            "anonymous_customer_exposure": 50000,
            "anonymous_customer_ratio": 6.25
        },
        "disclaimer": "本监控报告由AI辅助生成,基于公开数据和系统数据进行分析,仅供参考,不构成任何授信审批意见或风险决策依据。"
    }
    
    validator1 = ExposureValidator(standard_data)
    result1 = validator1.validate()
    print(f"\n测试用例1(标准场景): {'✅ 通过' if result1 else '❌ 失败'}")
    print(validator1.get_report())
    
    # 测试用例2:缺少免责声明
    no_disclaimer_data = standard_data.copy()
    no_disclaimer_data['disclaimer'] = ""
    
    validator2 = ExposureValidator(no_disclaimer_data)
    result2 = validator2.validate()
    print(f"\n测试用例2(缺少免责声明): {'✅ 通过(预期失败)' if not result2 else '❌ 失败(预期通过)'}")
    print(validator2.get_report())
    
    # 测试用例3:预警等级非法
    invalid_warning_data = standard_data.copy()
    invalid_warning_data['warning_level'] = "严重"
    
    validator3 = ExposureValidator(invalid_warning_data)
    result3 = validator3.validate()
    print(f"\n测试用例3(预警等级非法): {'✅ 通过(预期失败)' if not result3 else '❌ 失败(预期通过)'}")
    print(validator3.get_report())
    
    # 测试用例4:匿名客户比例过高
    high_anonymous_data = standard_data.copy()
    high_anonymous_data['penetration_summary']['anonymous_customer_ratio'] = 18.0
    
    validator4 = ExposureValidator(high_anonymous_data)
    result4 = validator4.validate()
    print(f"\n测试用例4(匿名客户比例过高): {'✅ 通过(带警告)' if result4 else '❌ 失败'}")
    print(validator4.get_report())


if __name__ == "__main__":
    main()
