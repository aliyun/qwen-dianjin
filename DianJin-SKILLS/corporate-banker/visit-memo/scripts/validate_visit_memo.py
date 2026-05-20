#!/usr/bin/env python3
"""拜访纪要验证脚本"""

import sys
import re
from pathlib import Path


class VisitMemoValidator:
    """拜访纪要验证器"""
    
    REQUIRED_DIMENSIONS = [
        "企业概况",
        "财务线索",
        "上下游关系",
        "担保物与资产",
        "人物印象"
    ]
    
    RISK_SIGNALS = [
        "经营类风险",
        "财务类风险",
        "关联方风险",
        "法律与合规风险"
    ]
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.content = self.file_path.read_text(encoding='utf-8')
        self.errors = []
        self.warnings = []
        
    def validate_structure(self):
        """验证基本结构"""
        for dimension in self.REQUIRED_DIMENSIONS:
            if dimension not in self.content:
                self.errors.append(f"缺少必需维度：{dimension}")
        
        # 检查是否有风险信号部分
        if "⚠️ 风险信号" not in self.content:
            self.warnings.append("未找到风险信号标注（⚠️）")
        
        # 检查是否有建议追问
        if "❓ 建议追问" not in self.content:
            self.warnings.append("未找到建议追问部分")
    
    def validate_data_tracing(self):
        """验证数据溯源"""
        # 检查是否有来源标注
        source_patterns = ["（口述）", "（现场观察）", "（企业方口述）", "（CM判断）"]
        has_source = any(pattern in self.content for pattern in source_patterns)
        
        if not has_source:
            self.warnings.append("未找到数据来源标注（口述/现场观察/CM判断）")
        
        # 检查金额是否有单位
        money_pattern = r'\d+[^万亿元亿元]'
        money_matches = re.findall(money_pattern, self.content)
        if money_matches:
            self.warnings.append(f"发现可能缺少单位的金额：{money_matches[:3]}")
    
    def validate_risk_signals(self):
        """验证风险信号识别"""
        # 检查是否识别了常见风险
        risk_keywords = ["停工", "减产", "积压", "民间融资", "诉讼", "仲裁"]
        found_risks = []
        
        for keyword in risk_keywords:
            if keyword in self.content:
                found_risks.append(keyword)
        
        if found_risks and "⚠️" not in self.content:
            self.errors.append(f"发现风险关键词{found_risks}，但未标注风险信号")
    
    def validate_quality(self):
        """验证质量要求"""
        # 检查追问数量
        question_count = self.content.count("？") + self.content.count("?")
        if question_count > 4:  # 最多2个追问，每个追问可能包含多个问号
            self.warnings.append(f"追问数量可能过多（发现{question_count}个问号）")
        
        # 检查是否有"（已更正）"标注（如果有修正场景）
        if "更正" in self.content and "（已更正）" not in self.content:
            self.warnings.append("发现更正信息，但未使用标准格式'（已更正）'标注")
    
    def validate(self) -> dict:
        """执行完整验证"""
        self.validate_structure()
        self.validate_data_tracing()
        self.validate_risk_signals()
        self.validate_quality()
        
        return {
            "file": str(self.file_path),
            "passed": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "score": max(0, 100 - len(self.errors) * 20 - len(self.warnings) * 10)
        }


def main():
    if len(sys.argv) < 2:
        print("用法：python validate_visit_memo.py <拜访纪要文件路径>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    validator = VisitMemoValidator(file_path)
    result = validator.validate()
    
    print(f"\n{'='*60}")
    print(f"拜访纪要验证报告")
    print(f"{'='*60}")
    print(f"文件：{result['file']}")
    print(f"得分：{result['score']}/100")
    print(f"状态：{'✅ 通过' if result['passed'] else '❌ 未通过'}")
    
    if result['errors']:
        print(f"\n❌ 错误 ({len(result['errors'])}个)：")
        for error in result['errors']:
            print(f"  - {error}")
    
    if result['warnings']:
        print(f"\n⚠️ 警告 ({len(result['warnings'])}个)：")
        for warning in result['warnings']:
            print(f"  - {warning}")
    
    print(f"\n{'='*60}\n")
    
    sys.exit(0 if result['passed'] else 1)


if __name__ == "__main__":
    main()
