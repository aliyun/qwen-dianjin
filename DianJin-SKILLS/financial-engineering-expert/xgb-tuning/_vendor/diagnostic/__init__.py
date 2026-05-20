# -*- coding: utf-8 -*-
"""shared/diagnostic — 结构化诊断子包。

提供多维诊断能力，输出统一的 DiagnosticReport：
  - fit_check    : 拟合度诊断（扩展自 DiagnosisResult，包含 Gap 判定）
  - ks_structure : KS 结构诊断（切点位置 / 漂移 / 陡峭度 / KS-AUC 一致性）
  - report       : DiagnosticReport 聚合器
"""

from diagnostic.fit_check import FitDiagnosis, run_fit_check
from diagnostic.ks_structure import KSStructureDiagnosis, run_ks_structure_check
from diagnostic.report import DiagnosticReport, run_diagnostics

__all__ = [
    "FitDiagnosis",
    "run_fit_check",
    "KSStructureDiagnosis",
    "run_ks_structure_check",
    "DiagnosticReport",
    "run_diagnostics",
]
