# -*- coding: utf-8 -*-
"""经典三段式数据切分（train / val / oot）。

职责：按时间 / 显式 filter / 随机三种策略把 DataFrame 切分为 train/val/oot，
并附带元信息与报告渲染辅助。不涉及数据读取（见 loading.py）。
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from .loading import read_dataset, infer_features

logger = logging.getLogger(__name__)

# ============================================================
# 数据结构
# ============================================================

class SplitResult(NamedTuple):
    """切分结果。

    用法：
        result = load_and_split_data(...)
        train, val, oot = result.train, result.val, result.oot

        # 兼容旧解包：train, val, oot, *_ = result
    """
    train: pd.DataFrame
    val: pd.DataFrame
    oot: pd.DataFrame
    features: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None

# ============================================================
# 默认值常量（供调用方与文档引用）
# ============================================================

DEFAULT_OOT_RATIO = 0.20
DEFAULT_VAL_RATIO = 0.25
DEFAULT_RANDOM_SEED = 42

# ============================================================
# 主切分入口
# ============================================================

def load_and_split_data(
    data_path: str,
    target: str,
    time_col: Optional[str] = "busi_dt",
    train_filter: Optional[str] = None,
    test_filter: Optional[str] = None,        # 兼容旧三件套，等价 val_filter
    oot_filter: Optional[str] = None,
    *,
    oot_ratio: float = DEFAULT_OOT_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED,
    label_filter: Optional[Iterable] = (0, 1),
    return_features: bool = False,
    exclude_cols: Optional[List[str]] = None,
    return_meta: bool = True,
) -> SplitResult:
    """加载数据并按经典三段式切分为 train / val / oot。

    切分优先级：
    1. 显式 filter（train_filter 或 oot_filter 任一给出）
    2. time_col 存在
    3. 随机切分

    Args:
        data_path: parquet 或 csv 路径
        target: 标签列名
        time_col: 时间列；为 None 或不存在时跳过策略 2
        train_filter / oot_filter: pandas query 条件
        test_filter: 兼容旧三件套，给出则视为 val 的显式约束（合并入 train_full）
        oot_ratio: train_full vs oot 比例，∈ (0, 1)
        val_ratio: val 在 train_full 内的比例，∈ (0, 0.5)，禁止为 0
        random_seed: 随机切分种子
        label_filter: 仅保留 target 在该集合内的样本；None 跳过
        return_features: 是否在 SplitResult.features 填充推断特征
        exclude_cols: 推断特征时额外排除的列
        return_meta: 是否填充 SplitResult.meta（默认 True）

    Returns:
        SplitResult(train, val, oot, features, meta)
    """
    # ---------- 参数校验 ----------
    _validate_ratios(oot_ratio, val_ratio)

    # ---------- 数据加载 ----------
    df = read_dataset(data_path, label_filter=label_filter, target=target)
    full_df = df  # 用于报告统计

    warnings: List[str] = []

    # ---------- 切分策略选择 ----------
    if train_filter is not None or oot_filter is not None:
        strategy = "explicit"
        train_full, oot, oot_boundary, t_filter_used, o_filter_used = _split_explicit(
            df, train_filter, oot_filter, test_filter, warnings
        )
    elif time_col and time_col in df.columns:
        strategy = "time"
        train_full, oot, oot_boundary = _split_by_time(df, time_col, oot_ratio)
        t_filter_used, o_filter_used = None, None
    else:
        strategy = "random"
        if time_col and time_col not in df.columns:
            warnings.append(
                f"time_col='{time_col}' 在数据中不存在，已 fallback 到随机切分"
            )
        train_full, oot, oot_boundary = _split_random(df, oot_ratio, random_seed)
        t_filter_used, o_filter_used = None, None

    # ---------- train_full → train / val ----------
    train, val = _split_train_val(
        train_full,
        time_col=time_col,
        val_ratio=val_ratio,
        random_seed=random_seed,
        follow_strategy=strategy,
    )

    # ---------- 切分后校验 ----------
    _validate_splits(train, val, oot, target, warnings)

    # ---------- 构造 meta ----------
    meta: Optional[Dict[str, Any]] = None
    if return_meta:
        meta = _build_meta(
            strategy=strategy,
            train=train, val=val, oot=oot,
            target=target,
            time_col=time_col if (strategy == "time" or strategy == "explicit") else None,
            oot_boundary=oot_boundary,
            train_filter_used=t_filter_used,
            oot_filter_used=o_filter_used,
            random_seed=random_seed,
            warnings=warnings,
        )

    # ---------- 推断特征 ----------
    features: Optional[List[str]] = None
    if return_features:
        features = infer_features(full_df, target, exclude_cols=exclude_cols)

    # ---------- 摘要日志 ----------
    logger.info(_summary_line(strategy, train, val, oot, target, oot_boundary))
    for w in warnings:
        logger.warning(f"[split] {w}")

    return SplitResult(train=train, val=val, oot=oot, features=features, meta=meta)

# ============================================================
# 内部实现
# ============================================================

def _validate_ratios(oot_ratio: float, val_ratio: float) -> None:
    if not (0.0 < oot_ratio < 1.0):
        raise ValueError(f"oot_ratio 必须 ∈ (0, 1)，当前 {oot_ratio}")
    if val_ratio <= 0.0:
        raise ValueError(
            f"val_ratio 必须 > 0（经典三段式禁止关闭 val），当前 {val_ratio}。"
            "如需 CV 模式或两段式，请另起重构 PR。"
        )
    if val_ratio >= 0.5:
        raise ValueError(f"val_ratio 必须 < 0.5（防止 train 过小），当前 {val_ratio}")
    train_ratio = (1 - oot_ratio) * (1 - val_ratio)
    if train_ratio < 0.1:
        raise ValueError(
            f"切分后 train 占比仅 {train_ratio:.2%}（< 10%），"
            f"请减小 oot_ratio({oot_ratio}) 或 val_ratio({val_ratio})"
        )

def _adapt_query_for_dtypes(df: pd.DataFrame, expr: str) -> str:
    """根据 DataFrame 列实际类型自动修正 query 表达式。

    当前支持的自动修正：
    1. datetime 列上误用 .str.startswith('YYYYMM') → 日期范围比较
    2. datetime 列上误用 .str.startswith('YYYYMMDD') → 日期范围比较
    3. 整数列上误用 .str.startswith('YYYYMM') → 数值范围比较
    4. 整数列上误用 .str.startswith('YYYYMMDD') → 数值范围比较
    5. col.astype(str).str.startswith(...) → 同上述修正
    6. ~col.str.startswith(...) 取反形式同样支持
    """
    if not expr:
        return expr

    adapted = expr

    # ---- 第一步：移除 .astype(str) 冗余（df.query 中 str 未定义） ----
    # 匹配形式: col.astype(str).str.startswith(...) 或 ~col.astype(str).str.startswith(...)
    astype_pattern = r'(~?)\s*(\w+)\.astype\(str\)\.str\.startswith'
    def _remove_astype(m: re.Match) -> str:
        negate = m.group(1)
        col = m.group(2)
        return f"{negate}{col}.str.startswith"
    adapted = re.sub(astype_pattern, _remove_astype, adapted)

    # ---- 第二步：修正 .str.startswith 用于 datetime / 整数列 ----
    # 匹配形式: col.str.startswith('...') 或 ~col.str.startswith('...')
    pattern = r'(~?)\s*(\w+)\.str\.startswith\([\'"](\d{6,8})[\'"]\)'

    for match in re.finditer(pattern, adapted):
        negate = match.group(1)  # ~ 或空
        col = match.group(2)
        prefix = match.group(3)

        if col not in df.columns:
            continue

        is_datetime = pd.api.types.is_datetime64_any_dtype(df[col])
        is_numeric = pd.api.types.is_numeric_dtype(df[col])

        if not is_datetime and not is_numeric:
            continue

        # 构建范围
        if len(prefix) == 6:  # YYYYMM
            year = int(prefix[:4])
            month = int(prefix[4:6])
            if month == 12:
                next_year, next_month = year + 1, 1
            else:
                next_year, next_month = year, month + 1
            if is_datetime:
                start = f"{prefix}01"
                end = f"{next_year:04d}{next_month:02d}01"
            else:  # numeric (e.g. int 20260101)
                start = int(f"{prefix}01")
                end = int(f"{next_year:04d}{next_month:02d}01")
        elif len(prefix) == 8:  # YYYYMMDD
            if is_datetime:
                start = prefix
                try:
                    dt = datetime.strptime(prefix, "%Y%m%d")
                    end = (dt + timedelta(days=1)).strftime("%Y%m%d")
                except ValueError:
                    continue
            else:  # numeric
                start = int(prefix)
                end = int(prefix) + 1
        else:
            continue

        if is_datetime:
            # datetime 列用字符串比较
            if negate:
                replacement = f"(({col} < '{start}') or ({col} >= '{end}'))"
            else:
                replacement = f"(({col} >= '{start}') and ({col} < '{end}'))"
        else:
            # 整数列用数值比较
            if negate:
                replacement = f"(({col} < {start}) or ({col} >= {end}))"
            else:
                replacement = f"(({col} >= {start}) and ({col} < {end}))"

        adapted = adapted.replace(match.group(0), replacement)
        logger.warning(
            f"[split] 自动修正 filter: {match.group(0)!r} → {replacement!r} "
            f"(列 '{col}' 为 {df[col].dtype} 类型)"
        )

    return adapted

def _split_explicit(
    df: pd.DataFrame,
    train_filter: Optional[str],
    oot_filter: Optional[str],
    test_filter: Optional[str],
    warnings: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, str, Optional[str], Optional[str]]:
    """显式 filter 模式。返回 (train_full, oot, oot_boundary, train_filter_used, oot_filter_used)"""

    def _safe_query(expr: str, name: str) -> pd.DataFrame:
        # 先尝试自动修正类型不匹配
        adapted_expr = _adapt_query_for_dtypes(df, expr)
        try:
            return df.query(adapted_expr)
        except Exception as e:
            err_msg = str(e)
            hint = ""
            if ".str accessor" in err_msg and "string values" in err_msg:
                hint = (
                    "（提示：该列可能是 datetime 类型而非字符串，"
                    "请改用日期范围比较，例如 'col >= \"20260101\" and col < \"20260201\"'）"
                )
            elif ".dt accessor" in err_msg:
                hint = "（提示：该列可能不是 datetime 类型，请检查数据类型）"
            raise ValueError(f"{name} 表达式无法解析: {expr!r}，错误: {e}{hint}")

    # 处理 test_filter（旧三件套兼容）
    train_filter_eff = train_filter
    if test_filter:
        if train_filter:
            train_filter_eff = f"({train_filter}) or ({test_filter})"
        else:
            train_filter_eff = test_filter
        warnings.append(
            "test_filter 已 deprecated，合并进 train_full 后仍按 val_ratio 在 train_full 内切 val"
        )

    if train_filter_eff and oot_filter:
        train_full = _safe_query(train_filter_eff, "train_filter").reset_index(drop=False)
        oot = _safe_query(oot_filter, "oot_filter").reset_index(drop=False)
        overlap = set(train_full["index"]).intersection(set(oot["index"]))
        if overlap:
            warnings.append(f"train_filter 与 oot_filter 存在 {len(overlap)} 条重叠样本")
        coverage = (len(train_full) + len(oot) - len(overlap)) / max(len(df), 1)
        if coverage < 0.95:
            warnings.append(f"train+oot 覆盖率仅 {coverage:.2%}，部分样本被丢弃")
        train_full = train_full.drop(columns=["index"]).reset_index(drop=True)
        oot = oot.drop(columns=["index"]).reset_index(drop=True)
        oot_boundary = f"oot_filter={oot_filter}"
    elif train_filter_eff and not oot_filter:
        train_full = _safe_query(train_filter_eff, "train_filter").reset_index(drop=True)
        oot = _safe_query(f"not ({train_filter_eff})", "oot_filter(auto)").reset_index(drop=True)
        oot_boundary = f"oot=NOT({train_filter_eff})"
    else:  # oot_filter only
        oot = _safe_query(oot_filter, "oot_filter").reset_index(drop=True)
        train_full = _safe_query(f"not ({oot_filter})", "train_filter(auto)").reset_index(drop=True)
        oot_boundary = f"oot_filter={oot_filter}"

    return train_full, oot, oot_boundary, train_filter_eff, oot_filter

def _split_by_time(
    df: pd.DataFrame,
    time_col: str,
    oot_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """按 time_col 唯一值切分；同一业务日不会被切散。"""
    unique_times = sorted(df[time_col].unique())
    n = len(unique_times)
    cut = int(n * (1 - oot_ratio))
    cut = max(1, min(cut, n - 1))
    train_times = set(unique_times[:cut])
    oot_times = set(unique_times[cut:])

    train_full = df[df[time_col].isin(train_times)].reset_index(drop=True)
    oot = df[df[time_col].isin(oot_times)].reset_index(drop=True)
    oot_boundary = f"{time_col} >= {unique_times[cut]}"
    return train_full, oot, oot_boundary

def _split_random(
    df: pd.DataFrame,
    oot_ratio: float,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """随机切分。"""
    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(len(df))
    cut = int(len(df) * (1 - oot_ratio))
    train_idx = perm[:cut]
    oot_idx = perm[cut:]
    train_full = df.iloc[train_idx].reset_index(drop=True)
    oot = df.iloc[oot_idx].reset_index(drop=True)
    oot_boundary = f"random seed={random_seed} (前 {1-oot_ratio:.0%} 行)"
    return train_full, oot, oot_boundary

def _split_train_val(
    train_full: pd.DataFrame,
    time_col: Optional[str],
    val_ratio: float,
    random_seed: int,
    follow_strategy: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """train_full → train/val。跟随 OOT 策略：
    - explicit / time：若 time_col 存在 → val 取末端 val_ratio
    - random / 无 time_col：val 在 train_full 内随机抽 val_ratio
    """
    use_time = (
        follow_strategy in ("explicit", "time")
        and time_col
        and time_col in train_full.columns
    )

    if use_time:
        unique_times = sorted(train_full[time_col].unique())
        n = len(unique_times)
        cut = int(n * (1 - val_ratio))
        cut = max(1, min(cut, n - 1))
        train_times = set(unique_times[:cut])
        val_times = set(unique_times[cut:])
        train = train_full[train_full[time_col].isin(train_times)].reset_index(drop=True)
        val = train_full[train_full[time_col].isin(val_times)].reset_index(drop=True)
    else:
        rng = np.random.default_rng(random_seed)
        perm = rng.permutation(len(train_full))
        cut = int(len(train_full) * (1 - val_ratio))
        train_idx = perm[:cut]
        val_idx = perm[cut:]
        train = train_full.iloc[train_idx].reset_index(drop=True)
        val = train_full.iloc[val_idx].reset_index(drop=True)

    return train, val

def _validate_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    oot: pd.DataFrame,
    target: str,
    warnings: List[str],
) -> None:
    for name, sub in (("train", train), ("val", val), ("oot", oot)):
        n = len(sub)
        if n < 10:
            raise ValueError(
                f"{name} 集合仅 {n} 条样本，过小无法训练/评估；请增大对应比例或检查 filter"
            )
        if n < 100:
            warnings.append(f"{name} 仅 {n} 条样本，建议增大该集合")
        if target in sub.columns:
            pos = int((sub[target] == 1).sum())
            neg = int((sub[target] == 0).sum())
            if pos == 0 or neg == 0:
                raise ValueError(
                    f"{name} 集合正/负样本失衡：pos={pos}, neg={neg}；请调整比例或 filter"
                )

    # val 早停稳定性
    val_n = len(val)
    val_pos = int((val[target] == 1).sum()) if target in val.columns else 0
    if val_n < 500 or val_pos < 30:
        warnings.append(
            f"val 样本量偏小（n={val_n}, pos={val_pos}），早停信号可能不稳定，"
            "建议加大 val_ratio 或换数据"
        )

    # 小数据提示
    train_full_n = len(train) + len(val)
    if train_full_n < 10000:
        warnings.append(
            f"train_full 仅 {train_full_n} 条，小数据场景建议未来切换到 CV 模式（本次未实现）"
        )

def _build_meta(
    *,
    strategy: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    oot: pd.DataFrame,
    target: str,
    time_col: Optional[str],
    oot_boundary: str,
    train_filter_used: Optional[str],
    oot_filter_used: Optional[str],
    random_seed: int,
    warnings: List[str],
) -> Dict[str, Any]:
    total = len(train) + len(val) + len(oot)
    return {
        "split_strategy": strategy,
        "ratios": {
            "train": round(len(train) / max(total, 1), 4),
            "val": round(len(val) / max(total, 1), 4),
            "oot": round(len(oot) / max(total, 1), 4),
        },
        "sample_counts": {
            "train": len(train),
            "val": len(val),
            "oot": len(oot),
        },
        "pos_rates": {
            "train": _pos_rate(train, target),
            "val": _pos_rate(val, target),
            "oot": _pos_rate(oot, target),
        },
        "time_col_used": time_col if strategy in ("time", "explicit") else None,
        "oot_boundary": oot_boundary,
        "train_filter_used": train_filter_used,
        "oot_filter_used": oot_filter_used,
        "random_seed": random_seed,
        "warnings": list(warnings),
    }

def _pos_rate(df: pd.DataFrame, target: str) -> float:
    if target not in df.columns or len(df) == 0:
        return 0.0
    return round(float((df[target] == 1).mean()), 6)

def _summary_line(
    strategy: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    oot: pd.DataFrame,
    target: str,
    oot_boundary: str,
) -> str:
    return (
        f"[split] strategy={strategy} | "
        f"train={len(train)}(pos={_pos_rate(train, target):.2%}) "
        f"val={len(val)}(pos={_pos_rate(val, target):.2%}) "
        f"oot={len(oot)}(pos={_pos_rate(oot, target):.2%}) | "
        f"oot_boundary={oot_boundary}"
    )

# ============================================================
# 报告渲染辅助（供各 skill 报告统一插入"数据切分"小节）
# ============================================================

def render_split_section(meta: Dict[str, Any]) -> str:
    """根据 meta 渲染 Markdown 的"数据切分"小节。"""
    if not meta:
        return ""
    cnt = meta["sample_counts"]
    pos = meta["pos_rates"]
    ratios = meta["ratios"]
    lines = [
        "## 数据切分",
        "",
        "- 训练范式：经典三段式 train / val / oot",
        f"- 切分策略：{_strategy_zh(meta['split_strategy'])}",
        f"- OOT 边界：{meta.get('oot_boundary', '-')}",
        (
            f"- 样本量：train={cnt['train']} (pos={pos['train']:.2%}) | "
            f"val={cnt['val']} (pos={pos['val']:.2%}) | "
            f"oot={cnt['oot']} (pos={pos['oot']:.2%})"
        ),
        (
            f"- 比例：{ratios['train']:.0%}/{ratios['val']:.0%}/{ratios['oot']:.0%}，"
            f"随机种子 {meta.get('random_seed', '-')}"
        ),
    ]
    if meta.get("warnings"):
        lines.append("")
        lines.append("> 切分警告：")
        for w in meta["warnings"]:
            lines.append(f"> - {w}")
    lines.append("")
    return "\n".join(lines)

def _strategy_zh(s: str) -> str:
    return {
        "explicit": "用户显式 filter",
        "time": "按时间字段排序",
        "random": "随机切分",
    }.get(s, s)
