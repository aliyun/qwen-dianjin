# -*- coding: utf-8 -*-
"""
lr_cli — LR 评分卡建模脚本的 CLI 参数定义。

设计对标 xgb_cli.py，为 lr-modeling skill 提供参数 single source of truth。
通用参数（data_path/target/time_col/filters）与 XGB 保持一致。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

def build_parser(description: str = "") -> argparse.ArgumentParser:
    """构建 LR 建模 CLI 的 ArgumentParser。"""
    parser = argparse.ArgumentParser(
        description=description or "LR ScoreCard Modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── 通用数据参数（与 xgb_cli _COMMON 保持一致） ──
    parser.add_argument("-d", "--data_path", required=True, help="数据文件路径（parquet/csv）")
    parser.add_argument("-t", "--target", required=True, help="目标变量列名（0/1 二分类）")
    parser.add_argument("--time_col", default="busi_dt", help="时间列名")
    parser.add_argument("--train_filter", default=None, help="训练集筛选条件（pandas query）")
    parser.add_argument("--oot_filter", default=None, help="OOT 集筛选条件")
    parser.add_argument("--oot_ratio", type=float, default=0.20, help="自动切分时 OOT 占比")
    parser.add_argument("--val_ratio", type=float, default=0.25, help="从 train_full 中切 val 的占比")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子")
    parser.add_argument("--exclude_cols", default="", help="排除列，逗号分隔")

    # ── LR 特有参数 ──
    parser.add_argument("--features", default=None, help="指定特征列表，逗号分隔；不传则自动按 IV 筛选")
    parser.add_argument("--max_n_bins", type=int, default=8, help="WoE 分箱最大箱数")
    parser.add_argument("--min_bin_size", type=float, default=0.05, help="最小分箱比例")
    parser.add_argument("--iv_threshold", type=float, default=0.02, help="IV 筛选阈值")
    parser.add_argument("--regularization", default="l2", choices=["l1", "l2", "elasticnet"],
                        help="正则化类型")
    parser.add_argument("--C", type=float, default=1.0, help="正则化强度（越小正则化越强）")
    parser.add_argument("--max_iter", type=int, default=1000, help="LR 最大迭代次数")

    # ── 评分卡转换参数 ──
    parser.add_argument("--base_score", type=int, default=600, help="评分卡基础分")
    parser.add_argument("--pdo", type=int, default=50, help="评分卡 PDO（分数翻倍点）")
    parser.add_argument("--base_odds", type=float, default=50.0, help="基础 Odds（好坏比）")

    # ── 输出参数 ──
    parser.add_argument("--output_dir", default=None, help="产物输出目录（默认 ./outputs/<timestamp>）")
    parser.add_argument("--model_name", default=None, help="模型名称（不含扩展名）")
    parser.add_argument("--report_output", default=None, help="报告输出路径")
    parser.add_argument("--config", default=None, help="JSON 配置文件路径")

    return parser

def parse_args(
    parser: argparse.ArgumentParser,
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """parse_args + --config 合并。"""
    import sys
    if argv is None:
        argv = sys.argv[1:]

    parsed = parser.parse_args(argv)

    # --config 合并
    if parsed.config:
        config_path = Path(parsed.config)
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
            # 显式命令行优先：只补充 parsed 中为 None/default 的字段
            explicit_keys = set()
            for tok in argv:
                if tok.startswith("--"):
                    explicit_keys.add(tok[2:].split("=", 1)[0])
            for key, value in config.items():
                if hasattr(parsed, key) and key not in explicit_keys:
                    if isinstance(value, (dict, list)):
                        setattr(parsed, key, json.dumps(value, ensure_ascii=False))
                    else:
                        setattr(parsed, key, value)

    return parsed

__all__ = ["build_parser", "parse_args"]
