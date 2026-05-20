#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
访前规划报告验证脚本

验证生成的访前规划报告是否符合质量标准：
1. 包含所有必需章节
2. 目标数量符合要求（1-3个主目标）
3. 风险信号正确置顶
4. 行业分类精确到中类或小类
5. 数据标注完整
"""

import sys
import re
import json
from typing import List, Tuple


class VisitPlanValidator:
    """访前规划报告验证器"""
    
    REQUIRED_SECTIONS = [
        "客户基本面回顾",
        "近期动态分析",
        "本次拜访目标",
        "材料准备清单",
        "沟通策略"
    ]
    
    VALID_VISIT_TYPES = [
        "首次拜访",
        "定期维护",
        "续贷跟进",
        "风险排查",
        "营销推进"
    ]
    
    def __init__(self, report_content: str):
        self.report = report_content
        self.errors = []
        self.warnings = []
    
    def validate_all(self) -> Tuple[bool, List[str], List[str]]:
        """执行所有验证"""
        self._check_required_sections()
        self._check_visit_type()
        self._check_objectives()
        self._check_risk_placement()
        self._check_industry_classification()
        self._check_data_sources()
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _check_required_sections(self):
        """检查必需章节是否存在"""
        for section in self.REQUIRED_SECTIONS:
            if section not in self.report:
                self.errors.append(f"缺少必需章节: {section}")
    
    def _check_visit_type(self):
        """检查拜访类型是否合法"""
        if "**本次拜访类型**" not in self.report:
            self.errors.append("缺少拜访类型标注")
            return
        
        # 提取拜访类型（支持方括号和无方括号两种格式）
        match = re.search(r"\*\*本次拜访类型\*\*[：:]\s*([^\n]+)", self.report)
        if not match:
            self.errors.append("拜访类型格式错误")
            return
        
        # 清理提取的文本，移除方括号、括号等
        visit_type = match.group(1).strip()
        visit_type = visit_type.replace('[', '').replace(']', '').replace('（', '').replace('）', '')
        visit_type = visit_type.strip()
        
        if visit_type not in self.VALID_VISIT_TYPES:
            self.errors.append(f"拜访类型不合法: {visit_type}，应为 {self.VALID_VISIT_TYPES} 之一")
    
    def _check_objectives(self):
        """检查目标数量和质量"""
        if "**主目标**" not in self.report:
            self.errors.append("缺少主目标")
            return
        
        # 提取主目标部分（到次目标或信息收集任务为止）
        objectives_section = self.report.split("**主目标**")[1] if "**主目标**" in self.report else ""
        if not objectives_section:
            self.errors.append("主目标部分为空")
            return
        
        # 截取到次目标之前
        if "**次目标**" in objectives_section:
            objectives_section = objectives_section.split("**次目标**")[0]
        
        # 计算目标数量（查找数字编号，支持多种格式）
        objectives = re.findall(r'\d+\.\s+[^\[]', objectives_section)
        num_objectives = len(objectives)
        
        if num_objectives == 0:
            self.errors.append("未找到具体的主目标")
        elif num_objectives > 3:
            self.errors.append(f"主目标数量过多（{num_objectives}个），应不超过3个")
        elif num_objectives == 1:
            self.warnings.append("仅设置1个主目标，建议根据实际情况增设次目标")
    
    def _check_risk_placement(self):
        """检查风险信号是否正确置顶"""
        if "⚠️ 风险提示" in self.report:
            # 检查风险提示是否在报告前30%
            risk_pos = self.report.find("⚠️ 风险提示")
            if risk_pos > len(self.report) * 0.3:
                self.errors.append("风险提示位置过于靠后，应放在报告开头")
    
    def _check_industry_classification(self):
        """检查行业分类是否精确"""
        if "行业分类" not in self.report:
            self.errors.append("缺少行业分类信息")
            return
        
        # 提取行业分类
        industry_match = re.search(r"\|\s*行业分类\s*\|\s*(.+?)\s*\|", self.report)
        if not industry_match:
            self.errors.append("行业分类格式错误")
            return
        
        industry = industry_match.group(1)
        
        # 检查是否过于宽泛
        broad_categories = ["制造业", "批发和零售业", "建筑业", "服务业"]
        if industry in broad_categories:
            self.errors.append(f"行业分类过于宽泛：'{industry}'，应精确到中类或小类（如'C3360 金属表面处理及热处理加工'）")
    
    def _check_data_sources(self):
        """检查数据标注是否完整"""
        if "数据来源" not in self.report:
            self.warnings.append("缺少数据来源声明")


def validate_report_file(file_path: str):
    """验证报告文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        validator = VisitPlanValidator(content)
        is_valid, errors, warnings = validator.validate_all()
        
        if is_valid:
            print("✅ 报告验证通过")
        else:
            print("❌ 报告验证失败")
        
        if errors:
            print("\n错误:")
            for error in errors:
                print(f"  - {error}")
        
        if warnings:
            print("\n警告:")
            for warning in warnings:
                print(f"  - {warning}")
        
        return 0 if is_valid else 1
        
    except FileNotFoundError:
        print(f"❌ 文件不存在: {file_path}")
        return 1
    except Exception as e:
        print(f"❌ 验证失败: {str(e)}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python validate_visit_plan.py <报告文件路径>")
        sys.exit(1)
    
    exit_code = validate_report_file(sys.argv[1])
    sys.exit(exit_code)
