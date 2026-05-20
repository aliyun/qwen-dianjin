#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行业信贷风险审查规则输出验证脚本

验证JSON输出格式的合法性和业务规则合规性
"""

import json
import sys


class RuleOutputValidator:
    """行业信贷规则输出数据验证器"""
    
    def __init__(self, input_data):
        self.input_data = input_data
        self.errors = []
        self.warnings = []
    
    def validate_required_fields(self):
        """验证必填字段"""
        required_fields = [
            'industry', 'industry_analysis', 'industry_chain', 'risk_profile',
            'basic_check_rules', 'deep_analysis_rules', 'summary', 'disclaimer'
        ]
        
        for field in required_fields:
            if field not in self.input_data or self.input_data[field] is None:
                self.errors.append(f"必填字段缺失: {field}")
    
    def validate_industry_analysis(self):
        """验证行业分析字段"""
        analysis = self.input_data.get('industry_analysis', {})
        
        # 验证overview长度
        overview = analysis.get('overview', '')
        if overview and (len(overview) < 150 or len(overview) > 250):
            self.warnings.append(f"industry_analysis.overview长度不符合150-250字要求(当前{len(overview)}字)")
        
        # 验证lifecycle_stage
        valid_stages = ['初创期', '成长期', '成熟期', '衰退期']
        lifecycle = analysis.get('lifecycle_stage', '')
        if lifecycle not in valid_stages:
            self.errors.append(f"lifecycle_stage必须为{valid_stages}之一(当前:{lifecycle})")
        
        # 验证数组字段非空
        for field in ['risk_characteristics', 'regulatory_requirements', 'key_risk_factors']:
            if not analysis.get(field) or len(analysis[field]) == 0:
                self.errors.append(f"industry_analysis.{field}不能为空数组")
    
    def validate_basic_rules(self):
        """验证基础核查规则"""
        basic_rules = self.input_data.get('basic_check_rules', [])
        
        # 验证规则数量
        if len(basic_rules) < 5:
            self.errors.append(f"basic_check_rules数量不足(当前{len(basic_rules)}条,要求≥5条)")
        
        for i, rule in enumerate(basic_rules):
            # 验证必填字段
            for field in ['name', 'description', 'rule_type', 'category', 'risk_target', 'check_method', 'check_rules', 'confidence']:
                if field not in rule or not rule[field]:
                    self.errors.append(f"basic_check_rules[{i}].{field}缺失")
            
            # 验证rule_type
            valid_types = ['数据校验', '资产核验', '文件校验', '资质核验']
            if rule.get('rule_type') not in valid_types:
                self.errors.append(f"basic_check_rules[{i}].rule_type必须为{valid_types}之一")
            
            # 验证check_rules数量和质量
            check_rules = rule.get('check_rules', [])
            if len(check_rules) < 2:
                self.errors.append(f"basic_check_rules[{i}].check_rules至少需要2条(当前{len(check_rules)}条)")
            
            # 验证confidence
            confidence = rule.get('confidence', 0)
            if not isinstance(confidence, (int, float)) or confidence < 60 or confidence > 100:
                self.errors.append(f"basic_check_rules[{i}].confidence必须在60-100之间(当前:{confidence})")
    
    def validate_deep_rules(self):
        """验证深度推理规则"""
        deep_rules = self.input_data.get('deep_analysis_rules', [])
        
        # 验证规则数量
        if len(deep_rules) < 3:
            self.errors.append(f"deep_analysis_rules数量不足(当前{len(deep_rules)}条,要求≥3条)")
        
        for i, rule in enumerate(deep_rules):
            # 验证必填字段
            for field in ['name', 'description', 'rule_type', 'category', 'risk_target', 'check_method', 'check_rules', 'confidence']:
                if field not in rule or not rule[field]:
                    self.errors.append(f"deep_analysis_rules[{i}].{field}缺失")
            
            # 验证rule_type
            valid_types = ['交叉核验', '趋势分析', '风险模型', '压力测试']
            if rule.get('rule_type') not in valid_types:
                self.errors.append(f"deep_analysis_rules[{i}].rule_type必须为{valid_types}之一")
            
            # 验证confidence
            confidence = rule.get('confidence', 0)
            if not isinstance(confidence, (int, float)) or confidence < 60 or confidence > 100:
                self.errors.append(f"deep_analysis_rules[{i}].confidence必须在60-100之间(当前:{confidence})")
    
    def validate_summary(self):
        """验证总结字段"""
        summary = self.input_data.get('summary', '')
        if summary and (len(summary) < 80 or len(summary) > 120):
            self.warnings.append(f"summary长度不符合80-120字要求(当前{len(summary)}字)")
    
    def validate_disclaimer(self):
        """验证免责声明"""
        disclaimer = self.input_data.get('disclaimer', '')
        if not disclaimer or len(disclaimer.strip()) < 20:
            self.errors.append("免责声明缺失或过短")
        
        # 检查关键声明词
        required_keywords = ['仅供参考', '不构成']
        for keyword in required_keywords:
            if keyword not in disclaimer:
                self.warnings.append(f"免责声明建议包含关键词: {keyword}")
    
    def validate(self):
        """执行完整验证"""
        self.validate_required_fields()
        self.validate_industry_analysis()
        self.validate_basic_rules()
        self.validate_deep_rules()
        self.validate_summary()
        self.validate_disclaimer()
        
        return len(self.errors) == 0
    
    def get_report(self):
        """生成验证报告"""
        if self.validate():
            status = "✅ 验证通过"
        else:
            status = f"❌ 验证失败({len(self.errors)}个错误)"
        
        report = [
            f"验证结果: {status}",
            f"错误数: {len(self.errors)}",
            f"警告数: {len(self.warnings)}",
            ""
        ]
        
        if self.errors:
            report.append("错误列表:")
            for error in self.errors:
                report.append(f"  - {error}")
            report.append("")
        
        if self.warnings:
            report.append("警告列表:")
            for warning in self.warnings:
                report.append(f"  - {warning}")
            report.append("")
        
        return "\n".join(report)


def main():
    """测试入口"""
    # 测试用例1: 标准场景
    test_data_1 = {
        "industry": "食用菌种植",
        "industry_analysis": {
            "overview": "食用菌种植行业近年来快速发展，市场规模持续扩大。该行业具有投资小、见效快、劳动力密集等特点，主要采用工厂化栽培模式。产业链上游为菌种研发和原材料供应，下游为批发商和零售商。行业整体呈现集中度低、竞争激烈的发展趋势。",
            "lifecycle_stage": "成长期",
            "risk_characteristics": ["季节性资金需求波动大", "技术门槛较低导致竞争加剧"],
            "regulatory_requirements": ["食品生产许可证(市场监管局)", "环保排污许可(生态环境局)"],
            "key_risk_factors": ["虚报种植面积风险", "季节性现金流压力"]
        },
        "industry_chain": {
            "upstream": "菌种供应商集中度低，原材料(木屑、棉籽壳)供应充足",
            "downstream": "下游客户主要为批发商，账期30-60天，集中度中等",
            "key_pain_points": ["菌种退化风险", "市场价格波动大"],
            "cash_cycle_days": "60-90天"
        },
        "risk_profile": {
            "main_fraud_patterns": ["虚报种植面积", "虚增产量"],
            "seasonal_risk": "旺季备货资金需求峰值超月均营收2倍",
            "collateral_quality": "主要抵押物为厂房和设备，变现能力中等",
            "benchmark_metrics": {
                "gross_margin": "20%-30%",
                "ar_days": "45-60天",
                "inventory_days": "30-45天"
            }
        },
        "basic_check_rules": [
            {
                "name": "种植面积与产量合理性验证",
                "description": "防止虚报种植规模，规避产量造假",
                "rule_type": "数据校验",
                "category": "行业项",
                "risk_target": "虚报种植面积",
                "check_method": "实地核查+卫星图像比对+政府农业档案三方验证",
                "check_rules": [
                    "申报产量与行业单产均值×面积之差不超过±20%",
                    "卫星图像核查面积与申报面积偏差不超过±15%",
                    "存在伪造政府档案迹象，直接否决"
                ],
                "confidence": 85
            },
            {
                "name": "食品生产许可证核查",
                "description": "验证企业是否具备合法生产资质",
                "rule_type": "资质核验",
                "category": "行业项",
                "risk_target": "无证生产风险",
                "check_method": "市场监管局官网查询+实地核查许可证原件",
                "check_rules": [
                    "许可证必须在有效期内",
                    "许可范围必须包含食用菌种植",
                    "许可证缺失或过期，直接否决"
                ],
                "confidence": 95
            },
            {
                "name": "环保排污许可核查",
                "description": "验证企业环保合规性",
                "rule_type": "资质核验",
                "category": "行业项",
                "risk_target": "环保违规风险",
                "check_method": "生态环境局官网查询+年度自行监测报告核查",
                "check_rules": [
                    "排污许可证必须在有效期内",
                    "年度自行监测报告必须齐全",
                    "存在重大环保违规记录，直接否决"
                ],
                "confidence": 90
            },
            {
                "name": "能耗与产量交叉验证",
                "description": "通过用电量验证产量真实性",
                "rule_type": "数据校验",
                "category": "行业项",
                "risk_target": "产量造假",
                "check_method": "电力公司用电数据+企业产量申报数据交叉验证",
                "check_rules": [
                    "单位产量用电量与行业均值偏差不超过±25%",
                    "用电趋势与产量趋势必须一致",
                    "用电量与产量严重不匹配(偏差>50%)，直接否决"
                ],
                "confidence": 80
            },
            {
                "name": "客户关联关系核查",
                "description": "识别关联方交易虚增收入风险",
                "rule_type": "数据校验",
                "category": "行业项",
                "risk_target": "关联方交易虚增收入",
                "check_method": "工商登记信息核查+实际控制人比对+资金流水追踪",
                "check_rules": [
                    "前5大客户中关联方占比不超过50%",
                    "关联方交易价格与市场均价偏差不超过±20%",
                    "发现故意隐瞒关联关系，直接否决"
                ],
                "confidence": 85
            }
        ],
        "deep_analysis_rules": [
            {
                "name": "产量-价格-收入三角交叉验证",
                "description": "通过多维数据交叉验证收入合理性",
                "rule_type": "交叉核验",
                "category": "行业项",
                "risk_target": "收入虚增",
                "check_method": "产量数据×市场均价→计算理论收入→与申报收入对比→验证偏差",
                "check_rules": [
                    "引入产量数据、市场价格数据、申报收入数据",
                    "收入验算偏离度不超过±15%",
                    "偏离度>30%且无法合理解释，触发深度调查"
                ],
                "confidence": 82
            },
            {
                "name": "季节性资金压力测试",
                "description": "评估旺季备货资金缺口风险",
                "rule_type": "压力测试",
                "category": "行业项",
                "risk_target": "现金流断裂风险",
                "check_method": "基于历史资金流水数据，测算旺季资金需求峰值，评估还款能力",
                "check_rules": [
                    "引入历史资金流水、季节性销售数据、原材料采购计划",
                    "资金缺口峰值不超过企业净资产的50%",
                    "资金缺口>净资产80%且无明确融资渠道，触发预警"
                ],
                "confidence": 78
            },
            {
                "name": "毛利率趋势合理性验证",
                "description": "验证毛利率变化是否符合行业趋势",
                "rule_type": "趋势分析",
                "category": "行业项",
                "risk_target": "成本压缩或收入虚增",
                "check_method": "连续3年毛利率数据对比行业基准，分析趋势合理性",
                "check_rules": [
                    "引入企业财务报表数据、行业毛利率基准",
                    "毛利率波动幅度不超过行业均值±10个百分点",
                    "毛利率连续3年异常高于行业均值20个百分点以上，触发深度调查"
                ],
                "confidence": 80
            }
        ],
        "summary": "本次共生成8条规则(5条基础核查+3条深度推理)，覆盖资质核查、产量验证、收入交叉核验、季节性资金压力测试等核心风险维度。重点关注虚报种植面积和关联方交易虚增收入两类欺诈风险，所有规则均含量化阈值和一票否决条件。",
        "disclaimer": "本规则集由AI辅助生成,基于公开行业数据和政策信息分析,仅供参考,不构成任何授信决策依据。使用前请核实最新数据和政策。"
    }
    
    validator1 = RuleOutputValidator(test_data_1)
    print("测试用例1(标准场景):")
    print(validator1.get_report())
    print()
    
    # 测试用例2: 规则数量不足
    test_data_2 = test_data_1.copy()
    test_data_2['basic_check_rules'] = test_data_2['basic_check_rules'][:3]  # 仅3条
    test_data_2['deep_analysis_rules'] = test_data_2['deep_analysis_rules'][:2]  # 仅2条
    
    validator2 = RuleOutputValidator(test_data_2)
    print("测试用例2(规则数量不足):")
    print(validator2.get_report())
    print()
    
    # 测试用例3: 缺少免责声明
    test_data_3 = test_data_1.copy()
    test_data_3['disclaimer'] = ""
    
    validator3 = RuleOutputValidator(test_data_3)
    print("测试用例3(缺少免责声明):")
    print(validator3.get_report())
    print()
    
    # 测试用例4: lifecycle_stage非法
    test_data_4 = test_data_1.copy()
    test_data_4['industry_analysis']['lifecycle_stage'] = "未知阶段"
    
    validator4 = RuleOutputValidator(test_data_4)
    print("测试用例4(lifecycle_stage非法):")
    print(validator4.get_report())


if __name__ == "__main__":
    main()
