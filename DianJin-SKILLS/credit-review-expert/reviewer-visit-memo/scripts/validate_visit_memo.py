#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
走访纪要数据格式验证脚本

用于验证 reviewer-visit-memo Skill 输入数据的完整性和格式正确性。
验证项包括:
1. 必填字段检查(客户名称、走访时间、走访人)
2. 走访时间格式验证(YYYY-MM-DD HH:MM)
3. 走访目的合法性检查
4. 口述观察内容非空检查
"""

import sys
import re
from datetime import datetime
from typing import Dict, Any, List


class VisitMemoValidator:
    """走访纪要数据验证器"""
    
    # 合法的走访目的枚举
    VALID_PURPOSES = [
        "新客户首笔授信",
        "存量续贷",
        "大额新增授信",
        "风险预警复核",
        "补充调查"
    ]
    
    # 走访时间正则表达式(YYYY-MM-DD HH:MM)
    TIME_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}$')
    
    def __init__(self, input_data: Dict[str, Any]):
        """
        初始化验证器
        
        Args:
            input_data: 输入数据字典
        """
        self.input_data = input_data
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self) -> bool:
        """
        执行完整验证
        
        Returns:
            bool: 验证是否通过
        """
        self._validate_required_fields()
        self._validate_visit_time_format()
        self._validate_visit_purpose()
        self._validate_oral_observation()
        
        return len(self.errors) == 0
    
    def _validate_required_fields(self):
        """验证必填字段"""
        required_fields = ['customer_name', 'visit_time', 'reviewer_name']
        
        for field in required_fields:
            value = self.input_data.get(field, '').strip()
            if not value:
                self.errors.append(f"必填字段缺失: {field}")
    
    def _validate_visit_time_format(self):
        """验证走访时间格式"""
        visit_time = self.input_data.get('visit_time', '').strip()
        
        if not visit_time:
            return  # 已在必填字段检查中处理
        
        if not self.TIME_PATTERN.match(visit_time):
            self.errors.append(f"走访时间格式错误: {visit_time}, 应为 YYYY-MM-DD HH:MM")
            return
        
        # 验证时间是否合法
        try:
            datetime.strptime(visit_time, '%Y-%m-%d %H:%M')
        except ValueError:
            self.errors.append(f"走访时间非法: {visit_time}")
    
    def _validate_visit_purpose(self):
        """验证走访目的"""
        visit_purpose = self.input_data.get('visit_purpose', '').strip()
        
        if not visit_purpose:
            self.warnings.append("走访目的未填写(建议填写)")
            return
        
        if visit_purpose not in self.VALID_PURPOSES:
            self.warnings.append(
                f"走访目的 '{visit_purpose}' 不在标准列表中,建议使用: {', '.join(self.VALID_PURPOSES)}"
            )
    
    def _validate_oral_observation(self):
        """验证口述观察内容"""
        oral_observation = self.input_data.get('oral_observation', '').strip()
        
        if not oral_observation:
            self.errors.append("口述观察内容不能为空")
        elif len(oral_observation) < 10:
            self.warnings.append("口述观察内容过短,建议提供更详细的观察记录")
    
    def get_errors(self) -> List[str]:
        """获取错误列表"""
        return self.errors
    
    def get_warnings(self) -> List[str]:
        """获取警告列表"""
        return self.warnings
    
    def get_report(self) -> str:
        """
        获取验证报告
        
        Returns:
            str: 格式化的验证报告
        """
        report = []
        report.append("=" * 60)
        report.append("走访纪要数据验证报告")
        report.append("=" * 60)
        
        if self.errors:
            report.append(f"\n❌ 错误 ({len(self.errors)}个):")
            for error in self.errors:
                report.append(f"  - {error}")
        
        if self.warnings:
            report.append(f"\n⚠️ 警告 ({len(self.warnings)}个):")
            for warning in self.warnings:
                report.append(f"  - {warning}")
        
        if not self.errors and not self.warnings:
            report.append("\n✅ 验证通过,无错误或警告")
        elif not self.errors:
            report.append("\n✅ 验证通过(存在警告)")
        else:
            report.append("\n❌ 验证失败")
        
        report.append("=" * 60)
        return '\n'.join(report)


def main():
    """主函数 - 用于命令行测试"""
    # 示例测试数据
    test_data = {
        "customer_name": "XX制造有限公司",
        "visit_time": "2026-05-05 10:00",
        "reviewer_name": "张审批",
        "visit_purpose": "新客户首笔授信",
        "oral_observation": "今天去了XX制造有限公司,地址是北京市朝阳区XX路123号,跟营业执照一致。厂房大概5000平米,目测开工率约60%,设备运转正常。"
    }
    
    validator = VisitMemoValidator(test_data)
    is_valid = validator.validate()
    
    print(validator.get_report())
    
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
