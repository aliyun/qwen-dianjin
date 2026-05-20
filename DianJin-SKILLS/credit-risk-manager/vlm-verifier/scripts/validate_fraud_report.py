"""
企业信贷反欺诈评估报告验证脚本
验证报告结构完整性、必填字段、免责声明等
"""

import yaml
import sys
from datetime import datetime


class FraudVerificationReportValidator:
    """反欺诈评估报告数据验证器"""

    def __init__(self, report_data: dict):
        self.report_data = report_data
        self.errors = []
        self.warnings = []

    def validate_required_sections(self):
        """验证必填章节"""
        required_sections = [
            '报告基本信息',
            '材料分类汇总',
            '反欺诈检测结果',
            '异常信号汇总',
            '建议核实清单',
            '覆盖度说明',
            '数据来源与免责说明'
        ]

        for section in required_sections:
            if section not in self.report_data:
                self.errors.append(f"必填章节缺失: {section}")

    def validate_basic_info(self):
        """验证报告基本信息"""
        if '报告基本信息' not in self.report_data:
            return

        basic_info = self.report_data['报告基本信息']
        required_fields = ['企业名称', '评估时间', '适用行业', '输入材料清单']

        for field in required_fields:
            if field not in basic_info:
                self.errors.append(f"报告基本信息缺少字段: {field}")

        # 验证评估时间格式
        if '评估时间' in basic_info:
            try:
                datetime.strptime(basic_info['评估时间'], '%Y-%m-%d')
            except ValueError:
                self.errors.append("评估时间格式错误，应为YYYY-MM-DD")

    def validate_detection_results(self):
        """验证检测结果结构"""
        if '反欺诈检测结果' not in self.report_data:
            return

        results = self.report_data['反欺诈检测结果']
        if not isinstance(results, list) or len(results) == 0:
            self.errors.append("反欺诈检测结果为空，应至少包含1个检测点")
            return

        # 验证每个检测点的必填字段
        for idx, detection in enumerate(results):
            required_fields = ['验证目标', '数据来源', '比对结果', '推理过程', '检测结论']
            for field in required_fields:
                if field not in detection:
                    self.errors.append(f"检测点{idx+1}缺少字段: {field}")

            # 验证检测结论格式
            if '检测结论' in detection:
                conclusion = detection['检测结论']
                valid_conclusions = ['未发现异常', '存在异常信号', '待核实']
                if not any(vc in conclusion for vc in valid_conclusions):
                    self.errors.append(f"检测点{idx+1}结论格式错误，应包含：未发现异常/存在异常信号/待核实")

    def validate_coverage(self):
        """验证覆盖度说明"""
        if '覆盖度说明' not in self.report_data:
            return

        coverage = self.report_data['覆盖度说明']
        required_categories = ['财务造假类', '经营虚假类', '身份与资质类', '抵押物类']

        for category in required_categories:
            if category not in coverage:
                self.warnings.append(f"未覆盖欺诈类别: {category}")

    def validate_disclaimer(self):
        """验证免责声明"""
        if '数据来源与免责说明' not in self.report_data:
            self.errors.append("免责声明章节缺失")
            return

        disclaimer = self.report_data['数据来源与免责说明']
        if '不构成' not in disclaimer:
            self.errors.append("免责声明缺少'不构成'关键词")
        if '仅供参考' not in disclaimer:
            self.errors.append("免责声明缺少'仅供参考'关键词")

    def validate(self) -> bool:
        """执行完整验证"""
        self.validate_required_sections()
        self.validate_basic_info()
        self.validate_detection_results()
        self.validate_coverage()
        self.validate_disclaimer()
        return len(self.errors) == 0

    def get_report(self) -> str:
        """生成验证报告"""
        status = "✅ 通过" if self.validate() else "❌ 失败"
        report = f"验证结果: {status}\n"
        if self.errors:
            report += f"\n错误 ({len(self.errors)}):\n"
            for error in self.errors:
                report += f"  - {error}\n"
        if self.warnings:
            report += f"\n警告 ({len(self.warnings)}):\n"
            for warning in self.warnings:
                report += f"  - {warning}\n"
        return report


def main():
    """测试入口"""
    # 测试用例1：完整报告（应通过）
    valid_report = {
        '报告基本信息': {
            '企业名称': '测试企业',
            '评估时间': '2026-05-05',
            '适用行业': 'H62 餐饮业',
            '输入材料清单': ['门头照.jpg', '流水.xlsx']
        },
        '材料分类汇总': [],
        '反欺诈检测结果': [
            {
                '验证目标': '验证装修投入真实性',
                '数据来源': {'材料A': '现场照片', '基准源B': '银行流水'},
                '比对结果': '视觉估值10万 vs 实际支出8万，差异20%',
                '推理过程': '差异在合理范围内',
                '检测结论': '未发现异常 — 差异率20% < 30%阈值'
            }
        ],
        '异常信号汇总': {},
        '建议核实清单': [],
        '覆盖度说明': {
            '财务造假类': '✅',
            '经营虚假类': '✅',
            '身份与资质类': '✅',
            '抵押物类': '⚠️'
        },
        '数据来源与免责说明': '本报告不构成信贷审批意见，检测结论仅供参考'
    }

    validator1 = FraudVerificationReportValidator(valid_report)
    print("测试用例1 - 完整报告:")
    print(validator1.get_report())
    print()

    # 测试用例2：缺少免责声明（应失败）
    invalid_report = valid_report.copy()
    del invalid_report['数据来源与免责说明']

    validator2 = FraudVerificationReportValidator(invalid_report)
    print("测试用例2 - 缺少免责声明:")
    print(validator2.get_report())
    print()

    # 测试用例3：检测点缺少字段（应失败）
    incomplete_report = valid_report.copy()
    incomplete_report['反欺诈检测结果'] = [
        {
            '验证目标': '验证经营规模',
            # 缺少其他必填字段
        }
    ]

    validator3 = FraudVerificationReportValidator(incomplete_report)
    print("测试用例3 - 检测点字段不完整:")
    print(validator3.get_report())


if __name__ == '__main__':
    main()
