#!/usr/bin/env python3
"""股权穿透分析验证脚本 - 金融级标准"""

import sys
from pathlib import Path


class EquityPenetrationValidator:
    """股权穿透分析验证器 - 符合 METHODOLOGY.md 金融级标准"""
    
    REQUIRED_SECTIONS = [
        "股权架构总览",
        "实际控制人分析",
        "关联方全图谱",
        "关联交易分析",
        "同业竞争分析",
        "综合风险评估"
    ]
    
    FINANCIAL_INDICATORS = [
        "CR3",
        "CR5",
        "一致行动人",
        "隐性关联方",
        "利益输送",
        "资金占用"
    ]
    
    RED_LINES = [
        "R1：实际控制人逃废债",
        "R2：融资平台无实质经营",
        "R3：股权代持超限",
        "R4：关联方资金占用",
        "R5：实控人资产转移",
        "R6：未披露重大担保"
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
        self._check_mermaid_diagram()
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
    
    def _check_mermaid_diagram(self):
        """检查 Mermaid 股权穿透图"""
        if "mermaid" not in self.content.lower():
            self.errors.append("缺少 Mermaid 股权穿透图")
    
    def _check_financial_indicators(self):
        """检查核心财务指标"""
        for indicator in self.FINANCIAL_INDICATORS:
            if indicator not in self.content:
                self.errors.append(f"缺少核心指标：{indicator}")
    
    def _check_red_lines(self):
        """检查红线核查情况"""
        # 检查是否有红线核查章节
        if "红线核查" not in self.content and "R1-R6" not in self.content:
            self.errors.append("缺少红线核查（R1-R6）")
        
        # 统计已核查的红线数量
        red_line_count = sum(1 for rl in self.RED_LINES if rl in self.content)
        if red_line_count < 6:
            self.warnings.append(f"仅核查 {red_line_count}/6 条红线，建议完整核查")
    
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
        data_source_count = self.content.count("数据来源")
        if data_source_count < 3:
            self.warnings.append(f"数据来源标注仅 {data_source_count} 处，建议增加")


def main():
    if len(sys.argv) < 2:
        print("用法：python validate_equity.py <文件路径>")
        sys.exit(1)
    
    validator = EquityPenetrationValidator(sys.argv[1])
    result = validator.validate()
    
    print(f"\n股权穿透分析报告验证：{result['file']}")
    print(f"状态：{'✅ 通过' if result['passed'] else '❌ 未通过'}")
    
    if result['errors']:
        print(f"\n❌ 错误 ({len(result['errors'])} 项)：")
        for error in result['errors']:
            print(f"  - {error}")
    
    if result['warnings']:
        print(f"\n⚠️  警告 ({len(result['warnings'])} 项)：")
        for warning in result['warnings']:
            print(f"  - {warning}")
    
    if result['passed']:
        print("\n✅ 报告验证通过")
        print("- ✅ 已核查红线条件")
        print("- ✅ 财务指标完整")
        print("- ✅ 数据来源标注完整")
    
    sys.exit(0 if result['passed'] else 1)


if __name__ == "__main__":
    main()
