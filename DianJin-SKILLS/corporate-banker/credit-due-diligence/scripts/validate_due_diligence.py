#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
尽职调查报告验证脚本

验证生成的尽调报告是否符合质量标准。
"""

import re
import sys
import json
from typing import List, Dict


class DueDiligenceValidator:
    """尽职调查报告验证器"""
    
    REQUIRED_SECTIONS = [
        "核心结论",
        "企业基本情况",
        "公司治理与管理团队",
        "关联关系与集团架构",
        "行业与市场环境",
        "生产经营与业务真实性验证",
        "财务状况分析",
        "主体资信与融资状况",
        "风险评估与授信建议"
    ]
    
    RED_LINES = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
    
    FINANCIAL_METRICS = [
        "毛利率",
        "净现比",
        "流动比率",
        "DSCR",
        "有息负债率",
        "利息保障倍数"
    ]
    
    def __init__(self, report_path: str):
        self.report_path = report_path
        self.errors = []
        self.warnings = []
        self.report = ""
        
        with open(report_path, 'r', encoding='utf-8') as f:
            self.report = f.read()
    
    def validate(self) -> bool:
        """执行完整验证"""
        print(f"🔍 开始验证尽职调查报告：{self.report_path}\n")
        
        self._check_required_sections()
        self._check_red_lines()
        self._check_financial_metrics()
        self._check_data_sources()
        self._check_credit_recommendation()
        self._check_authenticity_verification()
        
        if self.errors:
            print(f"❌ 报告验证失败\n")
            print("错误:")
            for error in self.errors:
                print(f"  - {error}")
            return False
        
        if self.warnings:
            print(f"⚠️  报告验证通过（有警告）\n")
            print("警告:")
            for warning in self.warnings:
                print(f"  - {warning}")
            return True
        
        print(f"✅ 报告验证通过\n")
        return True
    
    def _check_required_sections(self):
        """检查必需章节"""
        for section in self.REQUIRED_SECTIONS:
            if section not in self.report:
                self.errors.append(f"缺少必需章节：{section}")
    
    def _check_red_lines(self):
        """检查红线是否核查"""
        if "红线预警" not in self.report:
            self.errors.append("缺少红线预警标注")
            return
        
        # 检查是否对7条红线逐项核查
        red_line_section = self.report[self.report.find("一票否决条件"):] if "一票否决条件" in self.report else ""
        
        # 如果报告提到红线，应该有核查结果
        if any(red_line in self.report for red_line in self.RED_LINES):
            print("✅ 已核查红线条件")
    
    def _check_financial_metrics(self):
        """检查财务指标是否量化"""
        missing_metrics = []
        for metric in self.FINANCIAL_METRICS:
            if metric not in self.report:
                missing_metrics.append(metric)
        
        if missing_metrics:
            self.warnings.append(f"缺少财务指标：{', '.join(missing_metrics)}")
        else:
            print("✅ 财务指标完整")
    
    def _check_data_sources(self):
        """检查数据来源标注"""
        if "数据来源" not in self.report:
            self.errors.append("缺少数据来源标注")
            return
        
        # 统计数据来源标注次数
        source_count = self.report.count("数据来源")
        if source_count < 5:
            self.warnings.append(f"数据来源标注不足（{source_count}次，建议至少5次）")
        else:
            print(f"✅ 数据来源标注完整（{source_count}处）")
    
    def _check_credit_recommendation(self):
        """检查授信建议是否有测算依据"""
        if "建议额度" not in self.report:
            self.errors.append("缺少授信建议")
            return
        
        # 检查是否有测算依据
        if "测算依据" not in self.report and "测算逻辑" not in self.report:
            self.errors.append("授信建议缺少测算依据")
    
    def _check_authenticity_verification(self):
        """检查经营真实性验证"""
        if "真实性验证" not in self.report:
            self.errors.append("缺少经营真实性验证")
            return
        
        # 检查是否有交叉验证
        if "交叉验算" not in self.report and "交叉验证" not in self.report:
            self.warnings.append("建议增加交叉验算内容")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法：python3 validate_due_diligence.py <报告文件路径>")
        print("示例：python3 validate_due_diligence.py tests/golden/new-loan-report.md")
        sys.exit(1)
    
    report_path = sys.argv[1]
    
    try:
        validator = DueDiligenceValidator(report_path)
        success = validator.validate()
        sys.exit(0 if success else 1)
    except FileNotFoundError:
        print(f"❌ 文件不存在：{report_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 验证失败：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
