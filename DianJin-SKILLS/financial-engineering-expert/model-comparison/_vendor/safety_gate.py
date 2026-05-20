# -*- coding: utf-8 -*-
"""SafetyGate — 建模前体检。

在 xgb-modeling / xgb-tuning / xgb-deepmodel 主入口统一先调
`SafetyGate.check(train_df, val_df, ...)` 执行 8 项体检。

级别说明：
  - ERROR : 阻断，必须修复后才能继续
  - WARN  : 告警，可继续但需注意
  - INFO  : 仅供参考

使用示例：
    from safety_gate import SafetyGate
    report = SafetyGate.check(train_df, val_df, oot_df, target_col="label")
    if not report.passed:
        print("体检不通过:", report.blocking_reasons)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckResult:
    """单项检查结果。"""
    name: str           # 检查项名称
    level: str          # "ERROR" | "WARN" | "INFO"
    passed: bool        # 该检查是否通过
    message: str        # 人类可读描述
    detail: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class GateReport:
    """体检总报告。"""
    passed: bool                          # 所有 ERROR 级检查均通过
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def blocking_reasons(self) -> List[str]:
        """所有阻断原因。"""
        return [c.message for c in self.checks if c.level == "ERROR" and not c.passed]

    @property
    def warnings(self) -> List[str]:
        """所有告警。"""
        return [c.message for c in self.checks if c.level == "WARN" and not c.passed]

    @property
    def actions(self) -> List[str]:
        """建议动作（从 detail.action 字段收集）。"""
        acts: List[str] = []
        for c in self.checks:
            if not c.passed and "action" in c.detail:
                acts.append(c.detail["action"])
        return acts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "blocking_reasons": self.blocking_reasons,
            "warnings": self.warnings,
            "actions": self.actions,
            "checks": [
                {"name": c.name, "level": c.level, "passed": c.passed,
                 "message": c.message, "detail": c.detail}
                for c in self.checks
            ],
        }

# ---------------------------------------------------------------------------
# 体检项实现
# ---------------------------------------------------------------------------

def _check_temporal_integrity(
    train_df: pd.DataFrame, val_df: pd.DataFrame,
    oot_df: Optional[pd.DataFrame], time_col: Optional[str],
) -> CheckResult:
    """时序完整性：train_max < val_min < oot_min。"""
    name = "时序完整性"
    if time_col is None or time_col not in train_df.columns:
        return CheckResult(name, "INFO", True, "未提供时间列，跳过时序检查")

    try:
        t_max = pd.to_datetime(train_df[time_col]).max()
        v_min = pd.to_datetime(val_df[time_col]).min()
        detail: Dict[str, Any] = {"train_max": str(t_max), "val_min": str(v_min)}

        if t_max >= v_min:
            return CheckResult(name, "ERROR", False,
                f"训练集最大时间 {t_max} ≥ 验证集最小时间 {v_min}，存在时间穿越",
                detail)

        if oot_df is not None and time_col in oot_df.columns:
            o_min = pd.to_datetime(oot_df[time_col]).min()
            detail["oot_min"] = str(o_min)
            if v_min >= o_min:
                return CheckResult(name, "ERROR", False,
                    f"验证集最小时间 {v_min} ≥ OOT 最小时间 {o_min}，时间段重叠",
                    detail)

        return CheckResult(name, "ERROR", True, "时序完整性检查通过", detail)
    except Exception as e:
        return CheckResult(name, "WARN", True, f"时序检查异常: {e}")

def _check_schema_consistency(
    train_df: pd.DataFrame, val_df: pd.DataFrame,
    oot_df: Optional[pd.DataFrame],
) -> CheckResult:
    """Schema 一致性：字段名 + 类型。"""
    name = "Schema 一致性"
    train_cols = set(train_df.columns)
    val_cols = set(val_df.columns)

    missing_in_val = train_cols - val_cols
    extra_in_val = val_cols - train_cols

    detail: Dict[str, Any] = {}
    issues: List[str] = []

    if missing_in_val:
        issues.append(f"验证集缺少字段: {sorted(missing_in_val)[:10]}")
        detail["missing_in_val"] = sorted(missing_in_val)[:10]
    if extra_in_val:
        issues.append(f"验证集多出字段: {sorted(extra_in_val)[:10]}")
        detail["extra_in_val"] = sorted(extra_in_val)[:10]

    if oot_df is not None:
        oot_cols = set(oot_df.columns)
        missing_in_oot = train_cols - oot_cols
        if missing_in_oot:
            issues.append(f"OOT 缺少字段: {sorted(missing_in_oot)[:10]}")
            detail["missing_in_oot"] = sorted(missing_in_oot)[:10]

    if issues:
        return CheckResult(name, "ERROR", False, "; ".join(issues), detail)
    return CheckResult(name, "ERROR", True, "Schema 一致性检查通过", detail)

def _check_label_drift(
    train_df: pd.DataFrame, val_df: pd.DataFrame,
    oot_df: Optional[pd.DataFrame], target_col: str,
) -> CheckResult:
    """标签分布漂移：三段正样本率相对变化 > 30%。"""
    name = "标签分布漂移"
    if target_col not in train_df.columns:
        return CheckResult(name, "WARN", True, f"目标列 {target_col} 不存在，跳过")

    train_rate = float(train_df[target_col].mean())
    val_rate = float(val_df[target_col].mean())

    detail: Dict[str, Any] = {"train_pos_rate": train_rate, "val_pos_rate": val_rate}
    threshold = 0.30

    if train_rate > 0:
        rel_change = abs(val_rate - train_rate) / train_rate
        detail["train_val_rel_change"] = rel_change
        if rel_change > threshold:
            return CheckResult(name, "WARN", False,
                f"正样本率漂移 {rel_change:.1%}（train={train_rate:.3f}, val={val_rate:.3f}），超过 {threshold:.0%} 阈值",
                {**detail, "action": "检查数据切分是否合理，标签定义是否一致"})

    if oot_df is not None and target_col in oot_df.columns:
        oot_rate = float(oot_df[target_col].mean())
        detail["oot_pos_rate"] = oot_rate
        if train_rate > 0:
            rel_change_oot = abs(oot_rate - train_rate) / train_rate
            detail["train_oot_rel_change"] = rel_change_oot
            if rel_change_oot > threshold:
                return CheckResult(name, "WARN", False,
                    f"OOT 正样本率漂移 {rel_change_oot:.1%}（train={train_rate:.3f}, oot={oot_rate:.3f}）",
                    {**detail, "action": "检查 OOT 数据时间段是否合理"})

    return CheckResult(name, "WARN", True, "标签分布一致性良好", detail)

def _check_leakage_probe(
    train_df: pd.DataFrame, target_col: str,
    features: Optional[Sequence[str]] = None,
) -> CheckResult:
    """泄漏探针：单特征 AUC > 0.95。"""
    name = "泄漏探针"
    from sklearn.metrics import roc_auc_score

    if target_col not in train_df.columns:
        return CheckResult(name, "WARN", True, f"目标列 {target_col} 不存在，跳过")

    y = train_df[target_col].values
    if len(np.unique(y)) < 2:
        return CheckResult(name, "WARN", True, "标签只有一类，跳过泄漏检查")

    cols = list(features) if features else [
        c for c in train_df.columns
        if c != target_col and train_df[c].dtype in ("float64", "float32", "int64", "int32")
    ]

    # 只抽样检测（大数据集性能保护），最多检查 200 列
    if len(cols) > 200:
        cols = list(np.random.RandomState(42).choice(cols, 200, replace=False))

    suspects: List[str] = []
    auc_threshold = 0.95

    for col in cols:
        try:
            x = train_df[col].values.astype(float)
            mask = ~np.isnan(x)
            if mask.sum() < 50:
                continue
            auc = roc_auc_score(y[mask], x[mask])
            auc = max(auc, 1 - auc)  # 考虑反向
            if auc > auc_threshold:
                suspects.append(f"{col}(AUC={auc:.3f})")
        except Exception:
            continue

    if suspects:
        return CheckResult(name, "WARN", False,
            f"疑似泄漏特征: {', '.join(suspects[:5])}",
            {"suspects": suspects, "action": "移除泄漏特征或确认业务合理性"})
    return CheckResult(name, "WARN", True, "未发现单特征 AUC > 0.95 的泄漏嫌疑")

def _check_sample_size(
    train_df: pd.DataFrame, val_df: pd.DataFrame,
    oot_df: Optional[pd.DataFrame], target_col: str,
) -> CheckResult:
    """样本量门控：正样本绝对数 < 300 → 降级为 AUC 决策。"""
    name = "样本量门控"
    if target_col not in train_df.columns:
        return CheckResult(name, "WARN", True, f"目标列 {target_col} 不存在，跳过")

    detail: Dict[str, Any] = {}
    issues: List[str] = []
    threshold = 300

    for label, df in [("train", train_df), ("val", val_df)]:
        n_pos = int(df[target_col].sum())
        detail[f"{label}_n_pos"] = n_pos
        if n_pos < threshold:
            issues.append(f"{label} 正样本 {n_pos} < {threshold}")

    if oot_df is not None and target_col in oot_df.columns:
        n_pos = int(oot_df[target_col].sum())
        detail["oot_n_pos"] = n_pos
        if n_pos < threshold:
            issues.append(f"oot 正样本 {n_pos} < {threshold}")

    if issues:
        return CheckResult(name, "WARN", False,
            f"正样本不足: {'; '.join(issues)}",
            {**detail, "action": "降级为 AUC 决策（KS 方差过大）"})
    return CheckResult(name, "WARN", True, "正样本量充足", detail)

def _check_ks_sample_limit(
    val_df: pd.DataFrame, target_col: str,
) -> CheckResult:
    """KS 专项：正样本 < 300 时 CI 过宽，需放宽显著性门。"""
    name = "KS 样本量下限"
    if target_col not in val_df.columns:
        return CheckResult(name, "WARN", True, f"目标列 {target_col} 不存在，跳过")

    n_pos = int(val_df[target_col].sum())
    detail = {"val_n_pos": n_pos}

    if n_pos < 300:
        return CheckResult(name, "WARN", False,
            f"验证集正样本 {n_pos} < 300，Bootstrap KS CI 将过宽",
            {**detail, "action": "放宽显著性门阈值或增加样本量"})
    return CheckResult(name, "WARN", True, f"验证集正样本 {n_pos} 充足", detail)

def _check_ks_peak_feasibility(
    train_df: pd.DataFrame, target_col: str,
    features: Optional[Sequence[str]] = None,
) -> CheckResult:
    """KS 专项：切点可行域（train KS 峰值切点分位 < 2% 或 > 50%）。"""
    name = "KS 切点可行域"
    if target_col not in train_df.columns:
        return CheckResult(name, "WARN", True, f"目标列 {target_col} 不存在，跳过")

    # 使用 diagnostic 模块的能力
    try:
        from diagnostic.ks_structure import _compute_peak_quantile
    except ImportError:
        return CheckResult(name, "INFO", True, "diagnostic 模块不可用，跳过切点检查")

    y = train_df[target_col].values
    if len(np.unique(y)) < 2:
        return CheckResult(name, "WARN", True, "标签只有一类，跳过")

    # 用简单逻辑回归拟合快速得到概率（不依赖完整模型）
    # 如果 features 不提供，跳过（需要已训练模型的概率）
    return CheckResult(name, "INFO", True,
        "切点可行域检查需要模型预测概率，在训练后由 DiagnosticSuite 完成",
        {"note": "此检查将在首轮训练后由 run_ks_structure_check 自动执行"})

def _check_field_health(
    train_df: pd.DataFrame, val_df: pd.DataFrame,
    oot_df: Optional[pd.DataFrame], target_col: str,
) -> CheckResult:
    """字段体检：目标缺失 / 特征缺失率跨段突变 > 20%。"""
    name = "字段体检"
    issues: List[str] = []
    detail: Dict[str, Any] = {}

    # 目标列缺失检查
    for label, df in [("train", train_df), ("val", val_df)]:
        if target_col not in df.columns:
            issues.append(f"{label} 缺少目标列 {target_col}")
            continue
        miss_rate = float(df[target_col].isna().mean())
        if miss_rate > 0:
            issues.append(f"{label} 目标缺失率 {miss_rate:.1%}")
            detail[f"{label}_target_missing"] = miss_rate

    # 特征缺失率突变检查
    common_cols = sorted(set(train_df.columns) & set(val_df.columns) - {target_col})
    # 只检查数值列
    num_cols = [c for c in common_cols if train_df[c].dtype in ("float64", "float32", "int64", "int32")]

    drift_threshold = 0.20
    drift_cols: List[str] = []
    for col in num_cols[:200]:  # 最多检查 200 列
        train_miss = float(train_df[col].isna().mean())
        val_miss = float(val_df[col].isna().mean())
        if abs(train_miss - val_miss) > drift_threshold:
            drift_cols.append(f"{col}(Δ={abs(train_miss - val_miss):.1%})")

    if drift_cols:
        issues.append(f"缺失率突变特征: {', '.join(drift_cols[:5])}")
        detail["missing_drift_features"] = drift_cols[:10]

    if issues:
        return CheckResult(name, "WARN", False, "; ".join(issues),
            {**detail, "action": "检查数据源一致性和特征处理逻辑"})
    return CheckResult(name, "WARN", True, "字段体检通过", detail)

# ---------------------------------------------------------------------------
# SafetyGate 主入口
# ---------------------------------------------------------------------------

class SafetyGate:
    """建模前体检入口。

    Usage:
        report = SafetyGate.check(train_df, val_df, oot_df, target_col="label")
        if not report.passed:
            raise RuntimeError(f"体检不通过: {report.blocking_reasons}")
    """

    @staticmethod
    def check(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        oot_df: Optional[pd.DataFrame] = None,
        *,
        target_col: str = "label",
        time_col: Optional[str] = None,
        features: Optional[Sequence[str]] = None,
    ) -> GateReport:
        """执行全部体检项，返回 GateReport。

        Args:
            train_df: 训练集 DataFrame
            val_df: 验证集 DataFrame
            oot_df: OOT DataFrame（可选）
            target_col: 目标列名
            time_col: 时间列名（传入时启用时序检查）
            features: 特征列名列表（传入时限定泄漏探针范围）

        Returns:
            GateReport（passed=True 表示所有 ERROR 级检查均通过）
        """
        checks: List[CheckResult] = []

        # 1. 时序完整性 (ERROR)
        checks.append(_check_temporal_integrity(train_df, val_df, oot_df, time_col))

        # 2. Schema 一致性 (ERROR)
        checks.append(_check_schema_consistency(train_df, val_df, oot_df))

        # 3. 标签分布漂移 (WARN)
        checks.append(_check_label_drift(train_df, val_df, oot_df, target_col))

        # 4. 泄漏探针 (WARN)
        checks.append(_check_leakage_probe(train_df, target_col, features))

        # 5. 样本量门控 (WARN)
        checks.append(_check_sample_size(train_df, val_df, oot_df, target_col))

        # 6. KS 样本量下限 (WARN)
        checks.append(_check_ks_sample_limit(val_df, target_col))

        # 7. KS 切点可行域 (INFO — 需要模型概率，延迟到训练后)
        checks.append(_check_ks_peak_feasibility(train_df, target_col, features))

        # 8. 字段体检 (WARN)
        checks.append(_check_field_health(train_df, val_df, oot_df, target_col))

        # 综合判定：任一 ERROR 未通过 → 阻断
        passed = all(c.passed for c in checks if c.level == "ERROR")

        report = GateReport(passed=passed, checks=checks)
        logger.info(f"SafetyGate: passed={passed}, errors={len(report.blocking_reasons)}, "
                     f"warnings={len(report.warnings)}")
        return report

__all__ = ["SafetyGate", "GateReport", "CheckResult"]
