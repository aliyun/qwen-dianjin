# -*- coding: utf-8 -*-
"""Knowledge base for claim document types and requirements."""

# Document types used for classification
DOCUMENT_TYPES = [
    "门诊病历",
    "住院病历", 
    "诊断证明",
    "费用清单",
    "发票",
    "身份证",
    "银行卡",
    "出院小结",
    "手术记录",
    "检查报告",
    "处方",
    "检验报告",
    "住院费用汇总",
    "门诊收费票据",
    "病理报告",
    "影像报告",
    "其他",
]

# Medical record types (for medical course skill)
MEDICAL_RECORD_TYPES = {
    "门诊病历", "住院病历", "出院小结", "手术记录", 
    "检查报告", "检验报告", "病理报告", "影像报告",
}

# Drug related types (for drug indication skill)
DRUG_RELATED_TYPES = {
    "处方", "门诊病历", "住院病历", "诊断证明",
}

# Liability related types (for liability exclusion skill)
LIABILITY_RELATED_TYPES = {
    "门诊病历", "住院病历", "诊断证明", "检查报告",
}

# Calculation related types (for claim calculation skill)
CALCULATION_RELATED_TYPES = {
    "发票", "费用清单", "住院费用汇总", "门诊收费票据",
}

# Required documents by claim type
REQUIRED_DOCUMENTS = {
    "门诊理赔": [
        "门诊病历",
        "发票",
        "费用清单",
        "身份证",
        "银行卡",
    ],
    "住院理赔": [
        "住院病历",
        "出院小结",
        "发票",
        "费用清单",
        "身份证",
        "银行卡",
    ],
    "手术理赔": [
        "住院病历",
        "手术记录",
        "出院小结",
        "发票",
        "费用清单",
        "身份证",
        "银行卡",
    ],
    "重疾理赔": [
        "病理报告",
        "诊断证明",
        "住院病历",
        "检查报告",
        "身份证",
        "银行卡",
    ],
    "意外理赔": [
        "门诊病历",
        "诊断证明",
        "发票",
        "身份证",
        "银行卡",
    ],
}
