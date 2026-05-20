#!/usr/bin/env python3
"""授信申请验证脚本"""

import sys
from pathlib import Path


class CreditApplicationValidator:
    """授信申请验证器"""
    
    REQUIRED_SECTIONS = [
        "材料完备性评估",
        "案件冲突检查",
        "确认提交"
    ]
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.content = self.file_path.read_text(encoding='utf-8')
        self.errors = []
        self.warnings = []
        
    def validate(self) -> dict:
        """执行验证"""
        for section in self.REQUIRED_SECTIONS:
            if section not in self.content:
                self.errors.append(f"缺少必需部分：{section}")
        
        if "✅" not in self.content or "⚠️" not in self.content or "❌" not in self.content:
            self.warnings.append("未找到三级评估符号（✅/⚠️/❌）")
        
        return {
            "file": str(self.file_path),
            "passed": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings
        }


def main():
    if len(sys.argv) < 2:
        print("用法：python validate_credit_application.py <文件路径>")
        sys.exit(1)
    
    validator = CreditApplicationValidator(sys.argv[1])
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
