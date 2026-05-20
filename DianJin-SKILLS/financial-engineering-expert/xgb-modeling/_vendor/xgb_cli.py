# -*- coding: utf-8 -*-
"""
xgb_cli — XGB 域脚本的 CLI 参数单一来源 (single source of truth)。

替代 modeling.py / tuner.py / sub_trainer.py / stacker.py / comparator.py 各自
重复声明的 14-22 个 argparse 参数。

设计目标：
1. **统一命名**
   - 全面使用 --val_filter（删除所有 --test_filter 遗留别名）
   - 统一 --model_name / --output_dir
   - 统一 --auto 为 flag（替代混乱的 --auto_tune true|false 字符串）
2. **单一来源**
   - 所有 XGB 脚本通过 build_parser(domain) 获取 argparse
   - ArgSpec 暴露给 SKILL.md 生成器 / 测试做一致性校验
3. **复杂 JSON 参数免转义**
   - 统一 --config <json_file>：parse 完命令行后从 JSON 文件合并，命令行优先
   - 配合 python_executor 的 arguments_json，LLM 侧只需传 JSON 对象

用法：
    from xgb_cli import build_parser, parse_args

    parser = build_parser("modeling")
    args = parse_args(parser)   # 自动处理 --config 合并
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# 参数 spec 元数据（供 SKILL.md 生成器 + 测试引用）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArgSpec:
    name: str                       # --data_path
    required: bool = False
    default: Any = None
    type_: str = "str"              # "str" | "int" | "float" | "bool_flag" | "json"
    choices: Optional[Sequence[str]] = None
    channel: str = "cli"            # "cli" | "json"（复杂嵌套 JSON 推荐走 --config）
    help: str = ""
    short: Optional[str] = None

# 通用参数 — 所有 domain 共享
_COMMON: List[ArgSpec] = [
    ArgSpec("--data_path", required=True, short="-d", help="数据文件路径（parquet/csv）"),
    ArgSpec("--target", required=True, short="-t", help="目标变量列名（0/1 二分类）"),
    ArgSpec("--time_col", default="busi_dt", help="时间列名"),
    ArgSpec("--train_filter", help="训练集筛选条件（pandas query）"),
    ArgSpec("--val_filter", help="验证集 val 筛选条件；不传则从 train_full 按 val_ratio 切出"),
    ArgSpec("--oot_filter", help="OOT 集筛选条件；不传则按 oot_ratio 按时间切出"),
    ArgSpec("--oot_ratio", type_="float", default=0.20, help="自动切分时 OOT 占比"),
    ArgSpec("--val_ratio", type_="float", default=0.25, help="从 train_full 中切 val 的占比"),
    ArgSpec("--random_seed", type_="int", default=42, help="随机种子"),
    ArgSpec("--exclude_cols", default="", help="排除列，逗号分隔"),
    ArgSpec("--config", channel="json",
            help="JSON 配置文件路径；脚本读入后合并到命令行默认值（命令行显式传值优先）"),
]

# 各 domain 独有参数
_EXTRA: Dict[str, List[ArgSpec]] = {
    "modeling": [
        ArgSpec("--features", help="指定单套特征，逗号分隔"),
        ArgSpec("--feature_sets", channel="json",
                help="JSON 多方案（建议放到 --config 的 feature_sets 字段）"),
        ArgSpec("--feature_scheme", default="",
                choices=("full", "decorr", "high_iv", "stable"),
                help="指定单套自动筛选方案：full(全量入模)/decorr(去共线性)/high_iv(高预测力)/stable(稳定性优先)；不传或传空则走 --auto_select 全量对比"),
        ArgSpec("--auto_select", type_="bool_flag", default=True,
                help="自动特征筛选（四大算子全量对比；指定 --feature_scheme 时忽略本参数）"),
        ArgSpec("--baseline_filter", help="PSI 基准条件"),
        ArgSpec("--comparison_filter", help="PSI 对比条件"),
        ArgSpec("--sample_strategy", default="auto_weight",
                choices=("auto_weight", "undersample", "none"),
                help="样本不均衡处理策略"),
        ArgSpec("--model_name", help="模型名称（不含扩展名，默认自动生成）"),
        ArgSpec("--report_output", help="报告输出名称（可不含扩展名）"),
    ],
    "tuning": [
        ArgSpec("--features", required=True, short="-f", help="特征列表，逗号分隔"),
        ArgSpec("--params", short="-p", channel="json",
                help="当前参数（JSON；推荐放 --config.params）"),
        ArgSpec("--baseline", short="-b", channel="json",
                help="基线参数（JSON；推荐放 --config.baseline）"),
        ArgSpec("--round", type_="int", default=0, short="-r", help="当前轮次"),
        ArgSpec("--prev_val_metric", type_="float", help="上一轮 val 指标，用于收敛判断"),
        ArgSpec("--max_rounds", type_="int", default=5, help="最大调优轮数"),
        ArgSpec("--auto", type_="bool_flag", default=False, help="启用自动调优循环"),
        ArgSpec("--metric", default="ks", choices=("auc", "ks"), help="优化指标"),
        ArgSpec("--model_name", help="模型名称（不含扩展名）"),
        ArgSpec("--report_output", short="-o", help="报告输出路径"),
    ],
    "deepmodel-sub": [
        ArgSpec("--segments", channel="json",
                help="分群条件 JSON；推荐通过 --config.segments 传入"),
        ArgSpec("--segment_col", help="按列唯一值自动分群（与 segments 二选一）"),
        ArgSpec("--features", help="全局特征，逗号分隔（不传则自动筛选）"),
        ArgSpec("--features_per_segment", channel="json",
                help="分群差异化特征 JSON；推荐 --config.features_per_segment"),
        ArgSpec("--sample_weight_col", help="样本权重列名"),
        ArgSpec("--pos_weight_per_segment", channel="json",
                help="各分群正样本权重 JSON；推荐 --config.pos_weight_per_segment"),
        ArgSpec("--auto", type_="bool_flag", default=False,
                help="对 OOT Gap > 0.05 的分群自动调参"),
        ArgSpec("--output_dir", help="子模型保存目录"),
    ],
    "deepmodel-stack": [
        ArgSpec("--submodel_paths", required=True, channel="json",
                help="子模型路径 JSON 列表；推荐 --config.submodel_paths"),
        ArgSpec("--segments", channel="json", help="与 sub_trainer 保持一致的分群条件"),
        ArgSpec("--segment_col", help="分群列（与 segments 二选一）"),
        ArgSpec("--cv_folds", type_="int", default=5, help="OOF 交叉验证折数"),
        ArgSpec("--output_dir", help="Meta-learner 保存目录"),
    ],
    "deepmodel-compare": [
        ArgSpec("--stack_model_path", help="Stacking 模型路径（可选）"),
        ArgSpec("--submodel_paths", required=True, channel="json",
                help="子模型路径 JSON 列表；推荐 --config.submodel_paths"),
        ArgSpec("--segments", channel="json", help="分群条件"),
        ArgSpec("--segment_col", help="分群列"),
        ArgSpec("--baseline_model_path", help="单模型基线路径（可选）"),
        ArgSpec("--extra_baseline_algos", default="",
                help="额外多算法单模型基线，逗号分隔（可选值：lr,dnn）。"
                     "启用后会用同切分跑对应算法基线，扩展最终对比表。"),
        ArgSpec("--output_dir", help="产物保存目录"),
    ],
}

DOMAINS = tuple(_EXTRA.keys())

def get_arg_specs(domain: str) -> List[ArgSpec]:
    if domain not in _EXTRA:
        raise ValueError(f"Unknown domain: {domain!r}. Valid: {DOMAINS}")
    return _COMMON + _EXTRA[domain]

# ---------------------------------------------------------------------------
# argparse builder
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "str": str,
    "int": int,
    "float": float,
}

def _add_spec(parser: argparse.ArgumentParser, spec: ArgSpec) -> None:
    kwargs: Dict[str, Any] = {"help": spec.help}
    names = [spec.name]
    if spec.short:
        names.append(spec.short)

    if spec.type_ == "bool_flag":
        # --flag（action=store_true）；默认 False。若 default=True，用 store_false + 反向命名不可读，
        # 简单处理：仍然 store_true，true 成为启用开关；默认 True 的 flag 退化为 bool 字符串可读参数
        if spec.default:
            # 例：--auto_select 默认 true，允许 --auto_select false 关闭
            kwargs["type"] = lambda x: str(x).lower() not in ("false", "0", "no")
            kwargs["default"] = True
            kwargs["nargs"] = "?"
            kwargs["const"] = True
        else:
            kwargs["action"] = "store_true"
            kwargs["default"] = False
    else:
        kwargs["type"] = _TYPE_MAP.get(spec.type_, str)
        if spec.default is not None:
            kwargs["default"] = spec.default
        if spec.required:
            kwargs["required"] = True
        if spec.choices:
            kwargs["choices"] = list(spec.choices)

    parser.add_argument(*names, **kwargs)

def build_parser(domain: str, description: str = "") -> argparse.ArgumentParser:
    """构建指定 domain 的 ArgumentParser。"""
    parser = argparse.ArgumentParser(
        description=description or f"XGB CLI - {domain}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for spec in get_arg_specs(domain):
        _add_spec(parser, spec)
    return parser

# ---------------------------------------------------------------------------
# --config 合并逻辑
# ---------------------------------------------------------------------------

def _load_config_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"--config 指向的文件不存在: {path}")
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"--config 文件 JSON 解析失败: {e}")
    if not isinstance(data, dict):
        raise ValueError("--config 文件必须是 JSON 对象（顶层 dict）")
    return data

def _merge_config(
    parsed: argparse.Namespace,
    parser: argparse.ArgumentParser,
    explicit_args: Sequence[str],
) -> argparse.Namespace:
    """把 --config JSON 文件合并到 parsed；显式命令行参数优先。

    判定"显式传入"的方式：扫描 explicit_args 中是否出现 `--name` 或 `--name=...`。
    未在命令行出现的参数，允许被 config 覆盖（包括 None 与 default 值）。
    """
    if not getattr(parsed, "config", None):
        return parsed

    config = _load_config_file(parsed.config)
    explicit_set = set()
    for tok in explicit_args:
        if tok.startswith("--"):
            key = tok[2:].split("=", 1)[0]
            explicit_set.add(key)
        elif tok.startswith("-") and len(tok) == 2:
            # 短选项暂不做反查，保守跳过（XGB 域的 short 都有同名 long 选项）
            pass

    for key, value in config.items():
        # 仅合并 parser 认识的参数名
        if not hasattr(parsed, key):
            continue
        if key in explicit_set:
            continue
        # 复杂对象（dict/list）需要 JSON 序列化保留原样，脚本侧按需再 loads。
        # 对于 parser 声明为 str 的字段（如 --segments），我们将对象转为 JSON 字符串；
        # 对于简单标量直接赋值。
        if isinstance(value, (dict, list)):
            setattr(parsed, key, json.dumps(value, ensure_ascii=False))
        else:
            setattr(parsed, key, value)
    return parsed

def parse_args(
    parser: argparse.ArgumentParser,
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """parse_args + --config 合并。"""
    if argv is None:
        import sys
        argv = sys.argv[1:]
    parsed = parser.parse_args(argv)
    return _merge_config(parsed, parser, argv)

__all__ = [
    "ArgSpec",
    "DOMAINS",
    "get_arg_specs",
    "build_parser",
    "parse_args",
]
