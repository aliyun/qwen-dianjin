#!/usr/bin/env python3
"""财务报表分析验证脚本 - 金融级标准"""

import sys
from pathlib import Path


class FinancialReportValidator:
    """财务报表分析验证器 - 符合 METHODOLOGY.md 金融级标准"""
    
    REQUIRED_SECTIONS = [
        "盈利能力分析",
        "资产质量分析",
        "负债与偿债能力",
        "现金流分析",
        "杜邦分析",
        "财务预警扫描",
        "综合评价与建议"
    ]
    
    FINANCIAL_INDICATORS = [
        "毛利率",
        "净利率",
        "ROE",
        "收现比",
        "净现比",
        "流动比率",
        "速动比率",
        "DSCR",
        "有息负债率",
        "利息保障倍数",
        "自由现金流"
    ]
    
    RED_LINES = [
        "R1：审计意见为否定意见或无法表示意见",
        "R2：经营现金流连续3年为负，且有息负债持续增加",
        "R3：其他应收款 / 净资产 > 30%，且主要对手方为关联企业",
        "R4：货币资金余额 > 总负债20% 但同时借有大量短期有息负债（存贷双高）",
        "R5：应收账款增速连续2年超营业收入增速超过50个百分点",
        "R6：重大在建工程长期（>3年）未转固，且无合理建设周期说明",
        "R7：商誉占净资产比例 > 50%，且减值测试关键假设明显乐观",
        "R8：最近1年内实际控制人或主要股东股权高比例质押（> 80%）且无补救措施"
    ]
    
    FORBIDDEN_PHRASES = [
        "投资建议",
        "保证收益",
        "年化收益率"
    ]
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.content = self.file_path.read_text(encoding='utf-8')
        self.errors = []
        self.warnings = []
        
    def validate(self) -> dict:
        """执行验证"""
        self._check_required_sections()
        self._check_financial_indicators()
        self._check_red_lines()
        self._check_forbidden_phrases()
        self._check_disclaimer()
        self._check_data_sources()
        
        return {
            "file": str(self.file_path),
            "passed": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings
        }
    
    def _check_required_sections(self):
        """检查必需章节"""
        for section in self.REQUIRED_SECTIONS:
            if section not in self.content:
                self.errors.append(f"缺少必需章节：{section}")
    
    def _check_financial_indicators(self):
        """检查核心财务指标"""
        missing_indicators = []
        for indicator in self.FINANCIAL_INDICATORS:
            if indicator not in self.content:
                missing_indicators.append(indicator)
        
        if missing_indicators:
            self.errors.append(f"缺少核心财务指标：{', '.join(missing_indicators)}")
    
    def _check_red_lines(self):
        """检查红线核查情况"""
        if "红线" not in self.content and "红色警示" not in self.content:
            self.warnings.append("未包含红线核查信息")
    
    def _check_forbidden_phrases(self):
        """检查禁止用语（排除免责声明中的合规表述）"""
        # 先提取免责声明部分
        disclaimer_section = ""
        if "免责声明" in self.content:
            disclaimer_start = self.content.find("免责声明")
            disclaimer_section = self.content[disclaimer_start:disclaimer_start+500]
        elif "重要声明" in self.content:
            disclaimer_start = self.content.find("重要声明")
            disclaimer_section = self.content[disclaimer_start:disclaimer_start+500]
        
        # 检查禁止用语（排除免责声明部分）
        content_without_disclaimer = self.content.replace(disclaimer_section, "")
        
        for phrase in self.FORBIDDEN_PHRASES:
            if phrase in content_without_disclaimer:
                self.errors.append(f"包含禁止用语：{phrase}")
    
    def _check_disclaimer(self):
        """检查免责声明"""
        if "免责声明" not in self.content and "重要声明" not in self.content:
            self.errors.append("缺少免责声明")
    
    def _check_data_sources(self):
        """检查数据来源标注"""
        if "数据来源" not in self.content:
            self.warnings.append("未标注数据来源")


def main():
    if len(sys.argv) < 2:
        print("用法：python validate_financial_report.py <文件路径>")
        sys.exit(1)
    
    validator = FinancialReportValidator(sys.argv[1])
    result = validator.validate()
    
    print(f"\n验证报告：{result['file']}")
    print(f"状态：{'✅ 通过' if result['passed'] else '❌ 未通过'}")
    
    if result['errors']:
        print(f"\n错误：")
        for error in result['errors']:
            print(f"  - {error}")
    
    sys.exit(0 if result['passed'] else 1)


if __name__ == "__main__":
    main()
