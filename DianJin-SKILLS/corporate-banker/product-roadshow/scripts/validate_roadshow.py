#!/usr/bin/env python3
"""产品路演方案验证脚本"""

import sys
from pathlib import Path


class ProductRoadshowValidator:
    """产品路演方案验证器"""
    
    REQUIRED_SECTIONS = [
        "客户需求诊断",
        "推荐产品组合",
        "竞品对比分析",
        "收益测算",
        "营销话术",
        "后续跟进计划"
    ]
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.content = self.file_path.read_text(encoding='utf-8')
        self.errors = []
        
    def validate(self) -> dict:
        """执行验证"""
        for section in self.REQUIRED_SECTIONS:
            if section not in self.content:
                self.errors.append(f"缺少必需章节：{section}")
        
        # 检查是否有风险提示
        if "理财" in self.content and "风险提示" not in self.content:
            self.errors.append("理财类产品介绍缺少风险提示")
        
        return {
            "file": str(self.file_path),
            "passed": len(self.errors) == 0,
            "errors": self.errors
        }


def main():
    if len(sys.argv) < 2:
        print("用法：python validate_roadshow.py <文件路径>")
        sys.exit(1)
    
    validator = ProductRoadshowValidator(sys.argv[1])
    result = validator.validate()
    
    print(f"\n验证报告：{result['file']}")
    print(f"状态：{'✅ 通过' if result['passed'] else '❌ 未通过'}")
    
    if result['errors']:
        for error in result['errors']:
            print(f"  - {error}")
    
    sys.exit(0 if result['passed'] else 1)


if __name__ == "__main__":
    main()
