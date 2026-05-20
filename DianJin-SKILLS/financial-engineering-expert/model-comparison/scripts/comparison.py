# -*- coding: utf-8 -*-
"""多模型效果对比 — 主入口（薄编排）

设计一张图：

    load_split  ──►  trainers.*  ──►  analysis.*  ──►  recommend.*  ──►  reporting.build_report
                     (xgb/lr/dnn)     (enrich/psi/...)   (scenario)         (倒金字塔)

主函数只负责：解析 CLI → 切分数据 → 并列跑训练 → 补齐指标/稳定性 → 场景化推荐 → 装配报告。

"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# ── 共享基础 ─────────────────────────────────────────────────
# Vendor bootstrap: 本 skill 独立运行，无需外部 _runtime
_SCRIPT_DIR = Path(__file__).resolve().parent
_SKILL_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SKILL_DIR / "_vendor"))

from data_utils import load_and_split_data  # noqa: E402
from result_emitter import ResultEmitter  # noqa: E402
from safety_gate import SafetyGate  # noqa: E402

# scripts 目录的子包
sys.path.insert(0, str(_SCRIPT_DIR))
from analysis.metrics import enrich_metrics  # noqa: E402
from analysis.stability import compute_monthly_performance, compute_psi  # noqa: E402
from recommend import get_scenario, compute_facts  # noqa: E402
from reporting import build_report  # noqa: E402
from trainers import REGISTRY, TrainResult  # noqa: E402
from tuned_params import load_tuned_params  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── CLI ────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="多算法对比（XGB / LR+WoE / DNN）")
    p.add_argument("-d", "--data_path", required=True)
    p.add_argument("-t", "--target", required=True)
    p.add_argument("--time_col", default="busi_dt")
    p.add_argument("--train_filter", default=None)
    p.add_argument("--oot_filter", default=None)
    p.add_argument("--oot_ratio", type=float, default=0.20)
    p.add_argument("--val_ratio", type=float, default=0.25)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--exclude_cols", default="")
    p.add_argument("--features", default=None)
    p.add_argument("--algorithms", default="xgb,lr,dnn", help="逗号分隔：xgb,lr,dnn")
    p.add_argument("--scenario", default="general",
                   help="场景键：general / scorecard / fraud / stability_first")
    p.add_argument("--model_name", default=None)
    p.add_argument("--report_output", default=None)
    p.add_argument("--output_dir", default=None, help="产物输出目录")
    p.add_argument("--config", default=None, help="覆盖各算法超参的 JSON 文件")
    p.add_argument("--use_tuned", action="store_true",
                   help="自动从 models 目录加载最新 tuning 最优参数")
    p.add_argument("--tuned_params_file", default=None,
                   help='显式指定 {"xgb":{...},"lr":{...},"dnn":{...}} JSON')
    return p

# ── 工具函数 ───────────────────────────────────────────────
def _parse_algorithms(raw: str) -> list:
    keys = [a.strip().lower() for a in raw.split(",") if a.strip()]
    keys = [k for k in keys if k in REGISTRY]
    return keys or list(REGISTRY.keys())

def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        logger.warning("config 文件不存在: %s", path)
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _merge_tuned(
    base_config: Dict[str, Any],
    tuned: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """base_config 优先于 tuned（手工配置覆盖历史最优）。"""
    merged = {algo: dict(params) for algo, params in tuned.items() if params}
    for algo, params in base_config.items():
        if algo in merged:
            merged[algo].update(params)
        else:
            merged[algo] = params
    return merged

def _save_model_bundle(models_dir: Path, name: str, result: TrainResult, emitter: ResultEmitter) -> None:
    """按算法类型持久化产物，并登记到 emitter。"""
    short = result.short
    if short == "xgb":
        path = models_dir / f"{name}_xgb.json"
        result.model_artifact.save_model(str(path))
        emitter.add_file(path, role="model", meta={"algorithm": "xgb"})
    elif short == "lr":
        import joblib
        path = models_dir / f"{name}_lr.joblib"
        joblib.dump(result.model_artifact, path)
        emitter.add_file(path, role="model", meta={"algorithm": "lr"})
    elif short == "dnn":
        import torch
        path = models_dir / f"{name}_dnn.pt"
        bundle = dict(result.model_artifact)
        # 只存 state_dict + 元信息（避免 torch 模块序列化问题）
        torch.save({
            "state_dict": bundle["model"].state_dict(),
            "hidden_dims": bundle["hidden_dims"],
            "dropout": bundle["dropout"],
            "model_features": bundle["model_features"],
            "imputer": {
                "medians": bundle["imputer"].medians_,
                "indicator_cols": bundle["imputer"].indicator_cols_,
            },
        }, path)
        emitter.add_file(path, role="model", meta={"algorithm": "dnn"})

# ── 主流程 ─────────────────────────────────────────────────
def main() -> int:
    args = _build_parser().parse_args()

    # ── output_dir ──
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"./outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    emitter = ResultEmitter(skill="model-comparison", output_dir=output_dir)

    logger.info("=" * 60)
    logger.info("多模型效果对比 START")
    logger.info("=" * 60)

    algorithms = _parse_algorithms(args.algorithms)
    logger.info("对比算法：%s | 场景：%s", algorithms, args.scenario)

    # 1. 数据切分（一次切分，所有算法共享）
    exclude_cols = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]
    split = load_and_split_data(
        data_path=args.data_path,
        target=args.target,
        time_col=args.time_col,
        train_filter=args.train_filter,
        oot_filter=args.oot_filter,
        oot_ratio=args.oot_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed,
        return_features=True,
        exclude_cols=exclude_cols,
        return_meta=True,
    )
    train_data, val_data, oot_data = split.train, split.val, split.oot
    features = [f.strip() for f in args.features.split(",")] if args.features else split.features
    logger.info(
        "数据切分：train=%d val=%d oot=%d 特征=%d",
        len(train_data), len(val_data), len(oot_data), len(features),
    )

    # 2. SafetyGate（软提示）
    gate = SafetyGate()
    gate_result = gate.check(train_data, val_data, oot_data, target_col=args.target, features=features)
    if not gate_result.passed:
        logger.warning("SafetyGate: %s", '; '.join(gate_result.blocking_reasons))

    # 3. 组装各算法的超参（tuned → explicit config）
    base_config = _load_config(args.config)
    tuned = {}
    if args.use_tuned or args.tuned_params_file:
        models_dir = Path(args.data_path).parent.parent / "models"
        tuned = load_tuned_params(models_dir=models_dir, tuned_params_file=args.tuned_params_file)
    algo_config = _merge_tuned(base_config, tuned)

    # 参数来源可视化：温和提示 Agent 哪些算法还没有调参产物
    if args.use_tuned:
        missing = [a for a in algorithms if a not in tuned or not tuned[a]]
        if missing:
            logger.warning(
                "[--use_tuned] 未找到调参产物的算法：%s。"
                "这些算法将走默认超参。如需公平对比，请先依次调用 "
                "%s 产出调参最优参数后重跑本技能。",
                missing,
                " / ".join(f"{m}-tuning" for m in missing),
            )
        for algo in algorithms:
            src = ("explicit-config" if algo in base_config else
                   ("tuned" if tuned.get(algo) else "default"))
            logger.info("[%s] 超参来源：%s", algo.upper(), src)

    # 4. 多路训练（失败容忍，其他算法继续）
    results: Dict[str, TrainResult] = {}
    for algo in algorithms:
        Trainer = REGISTRY[algo]
        try:
            res = Trainer().train(train_data, val_data, oot_data, features, args.target, algo_config)
            if res is None:
                logger.warning("[%s] 训练返回 None，跳过", algo.upper())
                continue
            # 指标补全 + PSI + 分月 KS
            enrich_metrics(res, train_data[args.target].values,
                           val_data[args.target].values,
                           oot_data[args.target].values)
            res.psi = compute_psi(res.prob_train, res.prob_oot)
            res.monthly_ks = compute_monthly_performance(
                oot_data, args.target, res.prob_oot, time_col=args.time_col,
            )
            results[algo] = res
        except Exception as e:
            logger.error("[%s] 训练失败: %s", algo.upper(), e)
            traceback.print_exc()

    if not results:
        logger.error("所有算法训练失败，退出")
        return 1

    # 5. 客观事实计算（不做主观打分/推荐，仅输出 Pareto 前沿与原始指标）
    scenario = get_scenario(args.scenario)
    facts = compute_facts(results, scenario)
    logger.info("事实陈述：%s", facts.summary)
    for line in facts.facts_for_llm:
        logger.info("  · %s", line)

    # 6. 报告装配
    report_content = build_report(
        data_path=args.data_path,
        target=args.target,
        train_data=train_data,
        val_data=val_data,
        oot_data=oot_data,
        results=results,
        facts=facts,
        split_meta=split.meta,
    )

    # 7. 持久化
    model_name = args.model_name or f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_path = output_dir / f"{model_name}_report.md"
    report_path.write_text(report_content, encoding='utf-8')
    emitter.add_report(report_path)

    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for res in results.values():
        try:
            _save_model_bundle(models_dir, model_name, res, emitter)
        except Exception as e:
            logger.error("[%s] 模型保存失败: %s", res.short.upper(), e)

    # 8. State & Artifact（不记录主观最优，仅记录 Pareto 前沿与 OOT AUC 最高者作为事实）
    top_auc_short = facts.algos[0].short  # facts.algos 已按 OOT AUC 降序
    top_auc_res = results[top_auc_short]
    emitter.update_state({
        "last_model_type": "comparison",
        "last_model_name": model_name,
        "last_pareto_frontier": facts.pareto_frontier,
        "last_top_oot_auc_algorithm": top_auc_short,
        "last_scenario": scenario.key,
        "last_oot_auc": top_auc_res.oot.auc,
        "last_oot_ks": top_auc_res.oot.ks,
        "last_oot_psi": top_auc_res.psi,
        "comparison_algorithms": list(results.keys()),
    })
    emitter.add_artifact("model-comparison", "多模型效果对比", {
        "scenario": {
            "key": scenario.key,
            "name": scenario.name,
            "description": scenario.description,
        },
        "algorithms": [{
            "short": f.short,
            "algorithm": f.algorithm,
            "n_features": f.n_features,
            "interpretability": f.interpretability,
            "oot_auc": f.oot_auc,
            "oot_ks": f.oot_ks,
            "oot_gini": f.oot_gini,
            "oot_bcr10": f.oot_bcr10,
            "oot_brier": f.oot_brier,
            "ks_gap": f.ks_gap,
            "psi": f.psi,
            "is_pareto_optimal": f.is_pareto_optimal,
            "overfit_label": f.overfit_label,
            "psi_label": f.psi_label,
        } for f in facts.algos],
        "pareto_frontier": facts.pareto_frontier,
        "facts_for_llm": facts.facts_for_llm,
        "summary": facts.summary,
    })
    emitter.set_summary(f"多模型对比完成: Pareto={facts.pareto_frontier}, 最优 OOT AUC={top_auc_res.oot.auc:.4f}")
    emitter.update_metrics({"top_oot_auc": top_auc_res.oot.auc, "top_oot_ks": top_auc_res.oot.ks})
    emitter.write(output_dir / "result.json")

    logger.info("=" * 60)
    logger.info(
        "多模型对比完成 ✓ Pareto 前沿=%s 报告=%s",
        facts.pareto_frontier or ["无"],
        report_path,
    )
    logger.info("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
