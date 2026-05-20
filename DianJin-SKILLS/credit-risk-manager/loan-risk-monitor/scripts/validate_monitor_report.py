#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
贷后风险监测报告验证脚本

验证Markdown报告格式的合法性和业务规则合规性
"""

import sys


class MonitorReportValidator:
    """贷后监测报告数据验证器"""
    
    def __init__(self, input_data):
        self.input_data = input_data
        self.errors = []
        self.warnings = []
    
    def validate_required_sections(self):
        """验证必填章节"""
        required_sections = [
            '报告基本信息',
            '整体风险概览',
            '风险汇总表',
            '各企业详细监测结果',
            '风险预警与处置建议',
            '数据来源与免责说明'
        ]
        
        for section in required_sections:
            if section not in self.input_data:
                self.errors.append(f"必填章节缺失: {section}")
    
    def validate_risk_summary(self):
        """验证风险概览"""
        if '整体风险概览' in self.input_data:
            summary = self.input_data['整体风险概览']
            if '红色预警' not in summary:
                self.errors.append("风险概览缺少红色预警企业数量")
            if '橙色预警' not in summary:
                self.errors.append("风险概览缺少橙色预警企业数量")
            if '黄色预警' not in summary:
                self.errors.append("风险概览缺少黄色预警企业数量")
            if '绿色正常' not in summary:
                self.errors.append("风险概览缺少绿色正常企业数量")
    
    def validate_disclaimer(self):
        """验证免责声明"""
        if '数据来源与免责说明' not in self.input_data:
            self.errors.append("免责声明章节缺失")
        elif '不构成' not in self.input_data['数据来源与免责说明']:
            self.errors.append("免责声明缺少'不构成'关键词")
        elif '仅供参考' not in self.input_data['数据来源与免责说明']:
            self.errors.append("免责声明缺少'仅供参考'关键词")
    
    def validate(self):
        """执行完整验证"""
        self.errors = []
        self.warnings = []
        
        self.validate_required_sections()
        self.validate_risk_summary()
        self.validate_disclaimer()
        
        return len(self.errors) == 0
    
    def get_report(self):
        """生成验证报告"""
        if self.validate():
            return "✅ 验证通过:贷后监测报告格式合规"
        else:
            report = "❌ 验证失败:\n"
            for error in self.errors:
                report += f"  - {error}\n"
            if self.warnings:
                report += "\n⚠️ 警告:\n"
                for warning in self.warnings:
                    report += f"  - {warning}\n"
            return report


def main():
    """测试验证脚本"""
    # 测试用例1:标准报告
    test_report_1 = {
        '报告基本信息': '报告生成日期:2026-05-05,监测企业数量:5家',
        '整体风险概览': '红色预警:1,橙色预警:2,黄色预警:1,绿色正常:1',
        '风险汇总表': '序号|企业名称|风险等级',
        '各企业详细监测结果': 'XX制造有限公司(红色预警)',
        '风险预警与处置建议': '红色预警企业:XX制造有限公司',
        '数据来源与免责说明': '本监测报告仅供参考,不构成任何投资建议或授信审批意见'
    }
    
    # 测试用例2:缺少章节
    test_report_2 = {
        '报告基本信息': '报告生成日期:2026-05-05',
        '整体风险概览': '红色预警:1',
        '风险汇总表': '序号|企业名称|风险等级'
    }
    
    # 测试用例3:缺少免责声明
    test_report_3 = {
        '报告基本信息': '报告生成日期:2026-05-05',
        '整体风险概览': '红色预警:1,橙色预警:2,黄色预警:1,绿色正常:1',
        '风险汇总表': '序号|企业名称|风险等级',
        '各企业详细监测结果': 'XX制造有限公司(红色预警)',
        '风险预警与处置建议': '红色预警企业:XX制造有限公司',
        '数据来源与免责说明': '数据来源:中国执行信息公开网'
    }
    
    print("=" * 50)
    print("贷后风险监测报告验证脚本测试")
    print("=" * 50)
    
    validator1 = MonitorReportValidator(test_report_1)
    result1 = validator1.get_report()
    print(f"\n测试用例1(标准报告): {result1}")
    
    validator2 = MonitorReportValidator(test_report_2)
    result2 = validator2.get_report()
    print(f"\n测试用例2(缺少章节): {result2}")
    
    validator3 = MonitorReportValidator(test_report_3)
    result3 = validator3.get_report()
    print(f"\n测试用例3(缺少免责声明): {result3}")
    
    print("\n" + "=" * 50)
    print("测试完成!")
    print("=" * 50)


if __name__ == '__main__':
    main()
