#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信贷政策环境分析输出验证脚本

验证JSON输出格式的合法性和业务规则合规性
"""

import json
import sys


class PolicyAnalysisValidator:
    """信贷政策环境分析数据验证器"""
    
    def __init__(self, input_data):
        self.input_data = input_data
        self.errors = []
        self.warnings = []
    
    def validate_required_fields(self):
        """验证必填字段"""
        required_fields = [
            'analysis_meta', 'policy_context', 'summary', 'focus_indicators',
            'key_findings', 'transmission_path', 'impact_assessment',
            'differentiated_impact', 'trend_prediction', 'risk_alerts',
            'credit_strategy_suggestions', 'raw_report', 'disclaimer'
        ]
        
        for field in required_fields:
            if field not in self.input_data or self.input_data[field] is None:
                self.errors.append(f"必填字段缺失: {field}")
    
    def validate_analysis_meta(self):
        """验证分析元数据"""
        meta = self.input_data.get('analysis_meta', {})
        
        if not meta.get('dimension'):
            self.errors.append("analysis_meta.dimension不能为空")
        
        if meta.get('overall_signal') not in ['positive', 'negative', 'neutral', None]:
            self.errors.append("analysis_meta.overall_signal必须为positive/negative/neutral")
        
        if not meta.get('data_as_of'):
            self.warnings.append("analysis_meta.data_as_of未标注数据截止时间")
    
    def validate_focus_indicators(self):
        """验证核心关注指标"""
        indicators = self.input_data.get('focus_indicators', [])
        
        if len(indicators) < 4:
            self.errors.append(f"focus_indicators数量不足(至少4个,当前{len(indicators)}个)")
        elif len(indicators) > 8:
            self.warnings.append(f"focus_indicators数量过多(建议4-8个,当前{len(indicators)}个)")
        
        for i, indicator in enumerate(indicators):
            if not indicator.get('name'):
                self.errors.append(f"focus_indicators[{i}].name不能为空")
            if not indicator.get('value'):
                self.errors.append(f"focus_indicators[{i}].value不能为空(必须有具体数值)")
            if not indicator.get('data_source'):
                self.warnings.append(f"focus_indicators[{i}].data_source未标注数据来源")
            if not indicator.get('publish_date'):
                self.warnings.append(f"focus_indicators[{i}].publish_date未标注数据发布日期")
            
            confidence = indicator.get('confidence')
            if confidence is not None:
                if not isinstance(confidence, int) or confidence < 0 or confidence > 100:
                    self.errors.append(f"focus_indicators[{i}].confidence必须为0-100的整数")
    
    def validate_key_findings(self):
        """验证关键发现"""
        findings = self.input_data.get('key_findings', [])
        
        if len(findings) < 4:
            self.errors.append(f"key_findings数量不足(至少4条,当前{len(findings)}条)")
        elif len(findings) > 6:
            self.warnings.append(f"key_findings数量过多(建议4-6条,当前{len(findings)}条)")
        
        for i, finding in enumerate(findings):
            if not finding.get('affected_segments') or len(finding.get('affected_segments', [])) == 0:
                self.errors.append(f"key_findings[{i}].affected_segments不能为空")
    
    def validate_transmission_path(self):
        """验证传导路径"""
        path = self.input_data.get('transmission_path', {})
        chain = path.get('chain', [])
        
        if len(chain) < 3:
            self.errors.append(f"transmission_path.chain节点数量不足(至少3个,当前{len(chain)}个)")
        elif len(chain) > 5:
            self.warnings.append(f"transmission_path.chain节点数量过多(建议3-5个,当前{len(chain)}个)")
    
    def validate_risk_alerts(self):
        """验证风险预警"""
        alerts = self.input_data.get('risk_alerts', [])
        
        if len(alerts) < 2:
            self.errors.append(f"risk_alerts数量不足(至少2条,当前{len(alerts)}条)")
        elif len(alerts) > 4:
            self.warnings.append(f"risk_alerts数量过多(建议2-4条,当前{len(alerts)}条)")
        
        for i, alert in enumerate(alerts):
            if alert.get('level') not in ['high', 'medium', 'low']:
                self.errors.append(f"risk_alerts[{i}].level必须为high/medium/low")
            if alert.get('level') == 'high' and not alert.get('suggestion'):
                self.errors.append(f"risk_alerts[{i}]为high级别,必须包含应对建议")
    
    def validate_disclaimer(self):
        """验证免责声明"""
        disclaimer = self.input_data.get('disclaimer', '')
        
        if not disclaimer or len(disclaimer.strip()) < 50:
            self.errors.append("disclaimer不能为空或过短(至少50字符)")
        
        # 检查是否包含关键声明
        if '不构成' not in disclaimer and '仅供参考' not in disclaimer:
            self.warnings.append("disclaimer可能缺少'不构成投资建议'或'仅供参考'等关键声明")
    
    def validate(self):
        """执行完整验证"""
        self.validate_required_fields()
        self.validate_analysis_meta()
        self.validate_focus_indicators()
        self.validate_key_findings()
        self.validate_transmission_path()
        self.validate_risk_alerts()
        self.validate_disclaimer()
        
        return len(self.errors) == 0
    
    def get_report(self):
        """生成验证报告"""
        report = []
        report.append("=" * 60)
        report.append("信贷政策环境分析输出验证报告")
        report.append("=" * 60)
        report.append("")
        
        if self.errors:
            report.append(f"❌ 验证失败,发现{len(self.errors)}个错误:")
            for error in self.errors:
                report.append(f"  - {error}")
        else:
            report.append("✅ 验证通过,无错误")
        
        if self.warnings:
            report.append(f"\n⚠️ 发现{len(self.warnings)}个警告:")
            for warning in self.warnings:
                report.append(f"  - {warning}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """测试入口"""
    print("信贷政策环境分析验证脚本")
    print("=" * 60)
    
    # 测试用例1:标准场景
    test_data_1 = {
        "analysis_meta": {
            "dimension": "货币政策",
            "target": "制造业",
            "data_as_of": "2024年Q3",
            "overall_signal": "positive"
        },
        "policy_context": {
            "current_stance": "稳健偏宽松",
            "key_events": []
        },
        "summary": "货币政策保持稳健偏宽松基调",
        "focus_indicators": [
            {
                "name": "LPR 1年期",
                "type": "利率工具",
                "confidence": 95,
                "trend": "down",
                "signal": "positive",
                "value": "3.45%,较上期-10BP",
                "data_source": "中国人民银行",
                "publish_date": "2024-09-20",
                "remark": "连续下调"
            },
            {
                "name": "M2增速",
                "type": "货币供应量",
                "confidence": 90,
                "trend": "stable",
                "signal": "neutral",
                "value": "10.3%",
                "data_source": "中国人民银行",
                "publish_date": "2024-09-15",
                "remark": "保持平稳"
            },
            {
                "name": "社融规模",
                "type": "融资指标",
                "confidence": 85,
                "trend": "up",
                "signal": "positive",
                "value": "3.8万亿元",
                "data_source": "中国人民银行",
                "publish_date": "2024-09-15",
                "remark": "同比多增"
            },
            {
                "name": "存款准备金率",
                "type": "数量工具",
                "confidence": 90,
                "trend": "down",
                "signal": "positive",
                "value": "7.0%,较上期-25BP",
                "data_source": "中国人民银行",
                "publish_date": "2024-09-01",
                "remark": "全面降准"
            }
        ],
        "key_findings": [
            {
                "title": "融资成本持续下降",
                "detail": "LPR连续下调带动企业融资成本降低",
                "impact": "positive",
                "affected_segments": ["制造业", "中小微企业"]
            },
            {
                "title": "流动性保持充裕",
                "detail": "M2增速稳定,社融规模同比多增",
                "impact": "positive",
                "affected_segments": ["全行业"]
            },
            {
                "title": "银行息差收窄压力",
                "detail": "LPR下行速度快于存款利率,净息差持续收窄",
                "impact": "negative",
                "affected_segments": ["商业银行"]
            },
            {
                "title": "信贷结构优化",
                "detail": "制造业中长期贷款增速高于各项贷款平均增速",
                "impact": "positive",
                "affected_segments": ["制造业"]
            }
        ],
        "transmission_path": {
            "description": "LPR下调传导至实体经济融资成本降低",
            "chain": ["LPR下调", "企业融资成本降低", "制造业信贷需求上升", "制造业信贷规模扩大"]
        },
        "impact_assessment": {
            "overall": "货币政策宽松对信贷业务整体有利",
            "aspects": [
                {
                    "name": "信贷定价",
                    "score": 2,
                    "description": "LPR下行导致贷款定价承压"
                },
                {
                    "name": "信贷需求",
                    "score": 4,
                    "description": "融资成本降低刺激信贷需求"
                },
                {
                    "name": "资产质量",
                    "score": 4,
                    "description": "企业还款压力减轻,资产质量改善"
                },
                {
                    "name": "流动性管理",
                    "score": 5,
                    "description": "降准释放流动性,银行资金充裕"
                }
            ]
        },
        "differentiated_impact": [
            {
                "segment": "大型国有企业",
                "impact": "positive",
                "description": "融资成本下降,信贷可获得性提升"
            },
            {
                "segment": "中小微企业",
                "impact": "positive",
                "description": "普惠金融政策支持下,融资环境改善"
            },
            {
                "segment": "商业银行",
                "impact": "negative",
                "description": "净息差收窄,盈利能力承压"
            }
        ],
        "trend_prediction": {
            "short_term": "未来1-3个月LPR可能继续小幅下调",
            "medium_term": "未来6-12个月货币政策保持稳健偏宽松",
            "key_variables": ["美联储降息节奏", "国内经济复苏情况"],
            "outlook": "positive"
        },
        "risk_alerts": [
            {
                "level": "high",
                "title": "银行息差收窄风险",
                "description": "LPR下行速度快于存款利率,净息差持续收窄至历史低位",
                "probability": "high",
                "suggestion": "优化资产负债结构,加大低成本负债吸收"
            },
            {
                "level": "medium",
                "title": "资金空转风险",
                "description": "流动性充裕可能导致资金在金融体系内空转",
                "probability": "medium",
                "suggestion": "加强信贷资金用途监控,防止资金脱实向虚"
            }
        ],
        "credit_strategy_suggestions": [
            {
                "strategy": "加大制造业中长期信贷投放",
                "rationale": "融资成本降低,制造业信贷需求上升",
                "applicable_scope": "制造业",
                "precondition": "客户经营正常,符合授信条件"
            },
            {
                "strategy": "优化负债结构降低资金成本",
                "rationale": "净息差收窄压力加大",
                "applicable_scope": "全行业",
                "precondition": "无"
            }
        ],
        "raw_report": "# 货币政策分析报告\n\n## 摘要\n...\n\n## 政策背景\n...\n\n(完整报告内容)",
        "disclaimer": "本分析报告由AI辅助生成,基于公开数据和政策信息进行分析,仅供参考,不构成任何投资建议或授信决策依据。数据时效性:报告中使用的数据均来自公开渠道,可能存在滞后或更新不及时的情况。"
    }
    
    validator1 = PolicyAnalysisValidator(test_data_1)
    result1 = validator1.validate()
    print(f"\n测试用例1(标准场景): {'✅ 通过' if result1 else '❌ 失败'}")
    print(validator1.get_report())
    
    # 测试用例2:指标数量不足
    test_data_2 = test_data_1.copy()
    test_data_2['focus_indicators'] = test_data_1['focus_indicators'][:2]  # 仅2个指标
    
    validator2 = PolicyAnalysisValidator(test_data_2)
    result2 = validator2.validate()
    print(f"\n测试用例2(指标数量不足): {'✅ 通过' if not result2 else '❌ 失败(预期应失败)'}")
    print(validator2.get_report())
    
    # 测试用例3:传导路径节点不足
    test_data_3 = test_data_1.copy()
    test_data_3['transmission_path']['chain'] = ["LPR下调", "信贷扩大"]  # 仅2个节点
    
    validator3 = PolicyAnalysisValidator(test_data_3)
    result3 = validator3.validate()
    print(f"\n测试用例3(传导路径节点不足): {'✅ 通过' if not result3 else '❌ 失败(预期应失败)'}")
    print(validator3.get_report())
    
    # 测试用例4:缺少免责声明
    test_data_4 = test_data_1.copy()
    test_data_4['disclaimer'] = ""
    
    validator4 = PolicyAnalysisValidator(test_data_4)
    result4 = validator4.validate()
    print(f"\n测试用例4(缺少免责声明): {'✅ 通过' if not result4 else '❌ 失败(预期应失败)'}")
    print(validator4.get_report())
    
    print("\n" + "=" * 60)
    print("验证脚本测试完成")


if __name__ == "__main__":
    main()
