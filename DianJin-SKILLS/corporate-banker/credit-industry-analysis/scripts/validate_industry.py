#!/usr/bin/env python3
"""行业分析验证脚本 - 金融级标准"""

import sys
from pathlib import Path


class IndustryAnalysisValidator:
    """行业分析验证器 - 符合 METHODOLOGY.md 金融级标准"""
    
    REQUIRED_SECTIONS = [
        "行业概况",
        "产业链分析",
        "竞争格局",
        "PEST",
        "供需分析",
        "发展趋势",
        "波特五力",
        "市场规模",
        "风险矩阵",
        "情景分析"
    ]
    
    FINANCIAL_INDICATORS = [
        "CR3",
        "CR5",
        "HHI",
        "产能利用率",
        "毛利率",
        "市场规模"
    ]
    
    RED_LINES = [
        "I1：淘汰类行业",
        "I2：严重产能过剩",
        "I3：行业集体亏损",
        "I4：技术颠覆风险",
        "I5：监管政策收紧",
        "I6：补贴退坡风险"
    ]
    
    FORBIDDEN_PHRASES = [
        "投资建议",
        "保证收益",
        "年化收益率预计",
        "保本"
    ]
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.content = self.file_path.read_text(encoding='utf-8')
        self.errors = []
        self.warnings = []
        
    def validate(self) -> dict:
        """执行验证"""
        # 1. 检查必需章节
        self._check_required_sections()
        
        # 2. 检查财务指标
        self._check_financial_indicators()
        
        # 3. 检查数据来源标注
        self._check_data_sources()
        
        # 4. 检查合规红线
        self._check_red_lines()
        
        # 5. 检查禁止用语
        self._check_forbidden_phrases()
        
        # 6. 检查免责声明
        self._check_disclaimer()
        
        # 7. 检查情景分析差异性
        self._check_scenario_analysis()
        
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
        """检查财务指标完整性"""
        for indicator in self.FINANCIAL_INDICATORS:
            if indicator not in self.content:
                self.errors.append(f"缺少必需财务指标：{indicator}")
    
    def _check_data_sources(self):
        """检查数据来源标注"""
        if "来源" not in self.content and "Source" not in self.content:
            self.errors.append("缺少数据来源标注")
        
        # 检查是否有数据日期标注
        if "数据日期" not in self.content and "截至" not in self.content:
            self.warnings.append("建议添加数据日期标注")
    
    def _check_red_lines(self):
        """检查合规红线"""
        # 检查是否进行了红线核查
        if "红线" not in self.content and "Red Line" not in self.content:
            self.errors.append("缺少合规红线核查")
        
        # 检查红线编号格式
        red_line_found = False
        for red_line in self.RED_LINES:
            if red_line.split("：")[0] in self.content:
                red_line_found = True
                break
        
        if not red_line_found:
            self.warnings.append("未发现明确的红线编号（I1-I6）")
    
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
        disclaimer_keywords = ["免责声明", "重要声明", "不构成"]
        has_disclaimer = any(keyword in self.content for keyword in disclaimer_keywords)
        
        if not has_disclaimer:
            self.errors.append("缺少免责声明")
    
    def _check_scenario_analysis(self):
        """检查情景分析差异性"""
        if "情景分析" not in self.content:
            self.warnings.append("建议添加情景分析（乐观/中性/悲观）")
            return
        
        # 检查是否有三种情景
        scenarios = ["乐观", "中性", "悲观"]
        scenarios_found = [s for s in scenarios if s in self.content]
        
        if len(scenarios_found) < 3:
            self.warnings.append(f"情景分析不完整，仅包含：{', '.join(scenarios_found)}")


def main():
    if len(sys.argv) < 2:
        print("用法：python validate_industry.py <文件路径>")
        sys.exit(1)
    
    validator = IndustryAnalysisValidator(sys.argv[1])
    result = validator.validate()
    
    print(f"\n{'='*60}")
    print(f"行业分析报告验证：{result['file']}")
    print(f"{'='*60}")
    print(f"状态：{'✅ 通过' if result['passed'] else '❌ 未通过'}")
    
    if result['errors']:
        print(f"\n❌ 错误 ({len(result['errors'])}个)：")
        for error in result['errors']:
            print(f"  - {error}")
    
    if result['warnings']:
        print(f"\n⚠️ 警告 ({len(result['warnings'])}个)：")
        for warning in result['warnings']:
            print(f"  - {warning}")
    
    if result['passed']:
        print(f"\n✅ 报告验证通过")
        print(f"   - 已核查合规红线")
        print(f"   - 财务指标完整")
        print(f"   - 数据来源标注完整")
    
    print(f"{'='*60}\n")
    
    sys.exit(0 if result['passed'] else 1)


if __name__ == "__main__":
    main()
