"""
信贷风险结构化提取报告验证脚本
验证报告结构完整性、必填字段、风险点数量一致性、免责声明等
"""

import yaml
import sys
from datetime import datetime


class RiskExtractionReportValidator:
    """信贷风险提取报告数据验证器"""

    def __init__(self, report_data: dict):
        self.report_data = report_data
        self.errors = []
        self.warnings = []

    def validate_required_sections(self):
        """验证必填章节"""
        required_sections = [
            '报告基本信息',
            '【企业标签画像】',
            '【风险结构化分析】',
            '完整性声明',
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
        required_fields = ['企业名称', '提取时间', '适用行业', '输入材料清单']

        for field in required_fields:
            if field not in basic_info:
                self.errors.append(f"报告基本信息缺少字段: {field}")

    def validate_tag_profile(self):
        """验证企业标签画像"""
        if '【企业标签画像】' not in self.report_data:
            return

        tag_profile = self.report_data['【企业标签画像】']
        
        # 验证行业分类四级到底
        if '行业分类' not in tag_profile:
            self.errors.append("标签画像缺少行业分类")
        else:
            industry = tag_profile['行业分类']
            if '>' not in industry:
                self.errors.append("行业分类未按四级格式展示（应包含门类>大类>中类>小类）")
            else:
                levels = industry.split('>')
                if len(levels) < 4:
                    self.errors.append(f"行业分类仅包含{len(levels)}级，必须包含4级（门类>大类>中类>小类）")

        # 验证必填标签
        required_tags = ['基础属性', '经营特征', '风险特征', '外部环境']
        for tag in required_tags:
            if tag not in tag_profile:
                self.warnings.append(f"标签画像缺少{tag}（可选）")

    def validate_risk_points(self):
        """验证风险点结构化分析"""
        if '【风险结构化分析】' not in self.report_data:
            return

        risk_analysis = self.report_data['【风险结构化分析】']
        
        # 验证风险点总数声明
        if '风险点总数' not in risk_analysis:
            self.errors.append("风险结构化分析缺少风险点总数声明")

        # 验证每个风险点的必填字段
        risk_fields = [
            '风险定义',
            '企业风险原文描述',
            '风险分析方向',
            '数据来源说明',
            '适用场景',
            '建议关注事项'
        ]

        # 查找所有风险点（以###开头的章节）
        risk_points = [k for k in risk_analysis.keys() if k != '风险点总数']
        
        if len(risk_points) == 0:
            self.errors.append("未找到任何风险点分析")
        else:
            for risk_point in risk_points:
                risk_data = risk_analysis[risk_point]
                for field in risk_fields:
                    if field not in risk_data:
                        self.errors.append(f"风险点'{risk_point}'缺少字段: {field}")

    def validate_disclaimer(self):
        """验证免责声明"""
        if '数据来源与免责说明' not in self.report_data:
            self.errors.append("免责声明章节缺失")
        elif '不构成' not in self.report_data['数据来源与免责说明']:
            self.errors.append("免责声明缺少'不构成'关键词")
        elif '仅供参考' not in self.report_data['数据来源与免责说明']:
            self.errors.append("免责声明缺少'仅供参考'关键词")

    def validate_no_risk_grading(self):
        """验证禁止风险分级表述"""
        report_text = str(self.report_data)
        grading_words = ['高风险', '中风险', '低风险']
        
        for word in grading_words:
            if word in report_text:
                self.errors.append(f"检测到禁止的风险分级表述: {word}")

    def validate_no_vague_words(self):
        """验证禁止模糊词汇"""
        report_text = str(self.report_data)
        vague_words = ['相关', '某些', '适当', '尽量', '原则上']
        
        for word in vague_words:
            if word in report_text:
                self.warnings.append(f"检测到模糊词汇: {word}（建议替换为具体表述）")

    def validate(self):
        """执行完整验证"""
        self.validate_required_sections()
        self.validate_basic_info()
        self.validate_tag_profile()
        self.validate_risk_points()
        self.validate_disclaimer()
        self.validate_no_risk_grading()
        self.validate_no_vague_words()
        
        return len(self.errors) == 0

    def get_report(self):
        """生成格式化验证报告"""
        is_valid = len(self.errors) == 0
        status = "✅ 通过" if is_valid else "❌ 失败"
        report = f"信贷风险提取报告验证结果: {status}\n"
        report += f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"错误数: {len(self.errors)}\n"
        report += f"警告数: {len(self.warnings)}\n\n"
        
        if self.errors:
            report += "错误列表:\n"
            for i, error in enumerate(self.errors, 1):
                report += f"  {i}. {error}\n"
            report += "\n"
        
        if self.warnings:
            report += "警告列表:\n"
            for i, warning in enumerate(self.warnings, 1):
                report += f"  {i}. {warning}\n"
        
        return report


def main():
    """命令行测试入口"""
    print("=" * 60)
    print("信贷风险结构化提取报告验证脚本 - 测试用例")
    print("=" * 60)

    # 测试用例1：完整合规报告
    print("\n测试用例1: 完整合规报告（应通过）")
    valid_report = {
        '报告基本信息': {
            '企业名称': 'XX制造有限公司',
            '提取时间': '2026-05-05',
            '适用行业': 'C制造业>C36汽车制造业>C367汽车零部件及配件制造>C3670汽车零部件及配件制造',
            '输入材料清单': ['信贷报告.pdf', '基本情况描述.docx']
        },
        '【企业标签画像】': {
            '行业分类': 'C制造业>C36汽车制造业>C367汽车零部件及配件制造>C3670汽车零部件及配件制造',
            '基础属性': '中型、民营企业、非上市公司、单一企业、成熟期',
            '经营特征': '国内采购主导、离散制造、传统制造、直销主导',
            '风险特征': '需关注安全生产、现金流充裕、无司法风险',
            '外部环境': '政策中性、存量竞争'
        },
        '【风险结构化分析】': {
            '风险点总数': 2,
            '供应链安全风险': {
                '风险定义': '核心供应商集中度过高可能导致原材料供应中断',
                '企业风险原文描述': '公司前三大供应商采购占比超过60%，存在供应链集中的问题',
                '风险分析方向': ['用采购数据分析供应商集中度', '通过访谈验证替代供应商可用性'],
                '数据来源说明': '已知数据源：采购台账；可推断数据源：供应商工商登记信息',
                '适用场景': '工商经营范围包含原材料生产（数据来源：工商经营范围）',
                '建议关注事项': '建议查询前五大供应商合作年限及合同有效期'
            },
            '财务流动性风险': {
                '风险定义': '应收账款账期延长可能影响企业现金流',
                '企业风险原文描述': '公司应收账款周转天数从90天延长至150天',
                '风险分析方向': ['用财务报表分析应收账款账龄', '通过客户访谈验证回款计划'],
                '数据来源说明': '已知数据源：财务报表；可推断数据源：客户征信报告',
                '适用场景': '征信报告显示应收账款余额（数据来源：人民银行征信系统）',
                '建议关注事项': '建议对比同业上市公司年报应收账款周转率'
            }
        },
        '完整性声明': '本次分析已覆盖原文全部2个风险点，无遗漏。',
        '数据来源与免责说明': '本风险结构化提取报告仅基于用户提供的原始文本进行客观提取与结构化整理，不构成任何信贷审批意见、投资建议或风险评级。报告中的风险定义、分析方向及建议关注事项仅供参考，实际信贷决策需结合全面尽职调查、财务数据分析及专业判断。'
    }
    
    validator1 = RiskExtractionReportValidator(valid_report)
    result1 = validator1.validate()
    print(f"结果: {'✅ 通过' if result1 else '❌ 失败'}")
    print(validator1.get_report())

    # 测试用例2：缺少免责声明（应失败）
    print("\n测试用例2: 缺少免责声明（应失败）")
    invalid_report = {
        '报告基本信息': {
            '企业名称': 'XX制造有限公司',
            '提取时间': '2026-05-05',
            '适用行业': 'C制造业>C36汽车制造业',
            '输入材料清单': ['信贷报告.pdf']
        },
        '【企业标签画像】': {
            '行业分类': 'C制造业>C36汽车制造业',
            '基础属性': '中型、民营企业'
        },
        '【风险结构化分析】': {
            '风险点总数': 1,
            '供应链问题': {
                '风险定义': '供应链集中',
                '企业风险原文描述': '供应商集中',
                '风险分析方向': ['分析供应商'],
                '数据来源说明': '已知：采购数据',
                '适用场景': '工商数据',
                '建议关注事项': '关注供应商'
            }
        },
        '完整性声明': '已覆盖全部1个风险点'
    }
    
    validator2 = RiskExtractionReportValidator(invalid_report)
    result2 = validator2.validate()
    print(f"结果: {'✅ 通过' if result2 else '❌ 失败'}")
    print(validator2.get_report())

    # 测试用例3：风险点字段不完整（应失败）
    print("\n测试用例3: 风险点字段不完整（应失败）")
    incomplete_report = {
        '报告基本信息': {
            '企业名称': 'XX制造有限公司',
            '提取时间': '2026-05-05',
            '适用行业': 'C制造业>C36汽车制造业>C367汽车零部件及配件制造>C3670汽车零部件及配件制造',
            '输入材料清单': ['信贷报告.pdf']
        },
        '【企业标签画像】': {
            '行业分类': 'C制造业>C36汽车制造业>C367汽车零部件及配件制造>C3670汽车零部件及配件制造',
            '基础属性': '中型',
            '经营特征': '不适用',
            '风险特征': '无司法问题',
            '外部环境': '政策中性'
        },
        '【风险结构化分析】': {
            '风险点总数': 1,
            '财务问题': {
                '风险定义': '财务指标异常',
                '企业风险原文描述': '财务指标异常'
                # 缺少风险分析方向、数据来源说明、适用场景、建议关注事项
            }
        },
        '完整性声明': '已覆盖全部1个风险点',
        '数据来源与免责说明': '本报告仅供参考，不构成投资建议'
    }
    
    validator3 = RiskExtractionReportValidator(incomplete_report)
    result3 = validator3.validate()
    print(f"结果: {'✅ 通过' if result3 else '❌ 失败'}")
    print(validator3.get_report())


if __name__ == '__main__':
    main()
