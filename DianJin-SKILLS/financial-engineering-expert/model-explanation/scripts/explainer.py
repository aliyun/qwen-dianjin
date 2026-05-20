# -*- coding: utf-8 -*-
"""
XGBoost 模型解释器 (portable)

基于 SHAP 对 XGBoost 模型做全局/单样本/特征交互分析，生成 Markdown 报告 + PNG 附件。

Usage:
    python explainer.py --model_path ./models/my_model.json \
                        --data_path ./data.parquet --target y_label \
                        --output_dir ./outputs/explain_run
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Vendor bootstrap: 本 skill 独立运行，无需外部 _runtime
_SCRIPT_DIR = Path(__file__).resolve().parent
_SKILL_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SKILL_DIR / "_vendor"))

from result_emitter import ResultEmitter  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class ModelExplainer:
    """XGBoost 模型解释器，基于 SHAP。"""

    def __init__(self, model_path: str, data_path: str, target: str, features: List[str]):
        self.model_path = model_path
        self.data_path = data_path
        self.target = target
        self.features = features

        self.model = None
        self.data: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.explainer = None
        self.shap_values = None

        self._load_model()
        self._load_data()
        self._init_shap()

    def _fix_base_score_in_model_file(self, model_path: str) -> str:
        """修复 XGBoost>=1.7 中 base_score 的数组格式（SHAP 要求标量）。"""
        try:
            with open(model_path, "r", encoding="utf-8") as f:
                content = f.read()
            array_pattern = r'"base_score":\s*"\[([^\]]+)\]"'
            match = re.search(array_pattern, content)
            if not match:
                return model_path
            value_str = match.group(1)
            scalar_value = float(value_str)
            fixed_content = re.sub(array_pattern, f'"base_score":"{scalar_value}"', content)
            logger.info(f"修复 base_score: [{value_str}] -> {scalar_value}")
            fd, temp_path = tempfile.mkstemp(suffix=".json", prefix="shap_fixed_")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(fixed_content)
            return temp_path
        except Exception as e:
            logger.warning(f"修复 base_score 失败 (用原文件): {e}")
            return model_path

    def _load_model(self) -> None:
        import xgboost as xgb
        fixed_path = self._fix_base_score_in_model_file(self.model_path)
        self.model = xgb.Booster()
        self.model.load_model(fixed_path)
        if fixed_path != self.model_path:
            try:
                os.unlink(fixed_path)
            except Exception:
                pass
        logger.info(f"模型已加载: {self.model_path}")

    def _load_data(self) -> None:
        path = Path(self.data_path)
        if path.suffix == ".parquet":
            self.data = pd.read_parquet(path)
        else:
            self.data = pd.read_csv(path)
        self.data = self.data[self.data[self.target].isin([0, 1])].reset_index(drop=True)
        self.X = self.data[self.features]
        logger.info(f"数据已加载: {len(self.data)} 样本, {len(self.features)} 特征")

    def _init_shap(self) -> None:
        import shap
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(self.X)
        logger.info("SHAP 解释器初始化完成")

    def global_explanation(self, top_n: int = 20) -> Dict[str, Any]:
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            "feature": self.features,
            "mean_shap": mean_shap,
        }).sort_values("mean_shap", ascending=False)
        return {
            "feature_importance": feature_importance.to_dict("records"),
            "top_n": top_n,
        }

    def sample_explanation(
        self,
        sample_id: Optional[int] = None,
        sample_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        if sample_filter:
            matched = self.data.query(sample_filter)
            if len(matched) == 0:
                raise ValueError(f"未找到匹配样本: {sample_filter}")
            idx = matched.index[0]
            sample_desc = f"filter:{sample_filter}"
        elif sample_id is not None:
            if sample_id < 0 or sample_id >= len(self.data):
                raise ValueError(f"样本索引越界: {sample_id}")
            idx = sample_id
            sample_desc = f"index:{sample_id}"
        else:
            pos = self.data[self.data[self.target] == 1].index
            idx = int(pos[0]) if len(pos) > 0 else 0
            sample_desc = f"index:{idx}(auto)"

        sample_X = self.X.iloc[idx:idx + 1]
        sample_shap = self.shap_values[idx]

        import xgboost as xgb
        dmatrix = xgb.DMatrix(sample_X)
        pred_prob = float(self.model.predict(dmatrix)[0])

        abs_shap = np.abs(sample_shap)
        top_indices = np.argsort(abs_shap)[-10:][::-1]
        top_contributors = [
            {
                "feature": self.features[i],
                "value": float(sample_X.values[0][i]),
                "shap_value": float(sample_shap[i]),
                "direction": "positive" if sample_shap[i] > 0 else "negative",
            }
            for i in top_indices
        ]
        return {
            "sample_idx": int(idx),
            "sample_desc": sample_desc,
            "actual_label": int(self.data.iloc[idx][self.target]),
            "predicted_prob": pred_prob,
            "base_value": float(self.explainer.expected_value),
            "top_contributors": top_contributors,
        }

    def interaction_analysis(self, feature_pair: Tuple[str, str]) -> Dict[str, Any]:
        feat1, feat2 = feature_pair
        if feat1 not in self.features or feat2 not in self.features:
            raise ValueError(f"特征不存在: {feat1} 或 {feat2}")
        return {"feature1": feat1, "feature2": feat2}

# ────────────────────────────────────────────────────────────────────
# 图表生成
# ────────────────────────────────────────────────────────────────────

def generate_shap_charts(
    global_result: Dict[str, Any],
    sample_result: Optional[Dict[str, Any]],
    interaction_results: List[Dict[str, Any]],
    features: List[str],
    shap_values,
    X: pd.DataFrame,
    explainer,
    assets_dir: Path,
) -> Dict[str, Dict[str, str]]:
    """生成 SHAP 图表并保存到 assets_dir，返回 {name: {filename, path}}。"""
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assets_dir.mkdir(parents=True, exist_ok=True)
    charts: Dict[str, Dict[str, str]] = {}

    def _save(name: str) -> Dict[str, str]:
        fname = f"{name}.png"
        out = assets_dir / fname
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight", format="png")
        plt.close()
        return {"filename": fname, "path": str(out)}

    # 1. Summary Plot (beeswarm)
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=features,
                          max_display=global_result["top_n"], show=False)
        charts["summary_plot"] = _save("shap_summary")
        logger.info("Generated SHAP summary plot")
    except Exception as e:
        logger.warning(f"summary plot 失败: {e}")

    # 2. Bar Plot
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=features, plot_type="bar",
                          max_display=global_result["top_n"], show=False)
        charts["bar_plot"] = _save("shap_bar")
        logger.info("Generated SHAP bar plot")
    except Exception as e:
        logger.warning(f"bar plot 失败: {e}")

    # 3. Waterfall for sample
    if sample_result:
        try:
            idx = sample_result["sample_idx"]
            sample_shap = shap_values[idx]
            sample_X = X.iloc[idx:idx + 1]
            plt.figure(figsize=(12, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=sample_shap,
                    base_values=explainer.expected_value,
                    data=sample_X.values[0],
                    feature_names=features,
                ),
                max_display=15, show=False,
            )
            charts["waterfall_plot"] = _save("shap_waterfall")
            logger.info("Generated waterfall plot")
        except Exception as e:
            logger.warning(f"waterfall plot 失败: {e}")

    # 4. Dependence plots
    for r in interaction_results:
        try:
            f1, f2 = r["feature1"], r["feature2"]
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(f1, shap_values, X, feature_names=features,
                                 interaction_index=f2, show=False)
            charts[f"dependence_{f1}_{f2}"] = _save(f"shap_dependence_{f1}_{f2}")
            logger.info(f"Generated dependence plot: {f1} vs {f2}")
        except Exception as e:
            logger.warning(f"dependence plot 失败: {e}")

    return charts

def generate_report(
    explainer_obj: ModelExplainer,
    global_result: Dict[str, Any],
    sample_result: Optional[Dict[str, Any]],
    interaction_results: List[Dict[str, Any]],
    charts: Dict[str, Dict[str, str]],
    assets_rel: str = "assets",
) -> str:
    lines: List[str] = []
    lines.append("# 模型解释报告\n")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. 模型概览
    lines.append("## 1. 模型概览\n")
    lines.append("| 项目 | 内容 |")
    lines.append("|------|------|")
    lines.append(f"| 模型路径 | `{explainer_obj.model_path}` |")
    lines.append(f"| 数据路径 | `{explainer_obj.data_path}` |")
    lines.append(f"| 目标变量 | {explainer_obj.target} |")
    lines.append(f"| 特征数量 | {len(explainer_obj.features)} |")
    lines.append(f"| 样本规模 | {len(explainer_obj.data):,} |")
    lines.append(f"| 正样本率 | {explainer_obj.data[explainer_obj.target].mean():.2%} |")
    lines.append("")

    # 2. 全局解释
    lines.append("## 2. 全局特征解释\n")
    lines.append("### 2.1 SHAP Summary\n")
    if "summary_plot" in charts:
        lines.append(f"![SHAP Summary]({assets_rel}/{charts['summary_plot']['filename']})\n")

    lines.append(f"### 2.2 平均绝对 SHAP 值 (Top {global_result['top_n']})\n")
    if "bar_plot" in charts:
        lines.append(f"![SHAP Bar]({assets_rel}/{charts['bar_plot']['filename']})\n")

    lines.append("### 2.3 特征重要性表格\n")
    lines.append("| 排名 | 特征 | 平均 SHAP 值 | 重要性等级 |")
    lines.append("|------|------|-------------|-----------|")
    for i, feat in enumerate(global_result["feature_importance"][:global_result["top_n"]], 1):
        ms = feat["mean_shap"]
        level = "极高" if ms > 0.1 else ("高" if ms > 0.05 else ("中" if ms > 0.02 else "低"))
        lines.append(f"| {i} | {feat['feature']} | {ms:.4f} | {level} |")
    lines.append("")

    # 2.4 总结
    feat_list = global_result["feature_importance"][:global_result["top_n"]]
    total_shap = sum(f["mean_shap"] for f in feat_list)
    if total_shap > 0 and len(feat_list) >= 3:
        top3_pct = sum(f["mean_shap"] for f in feat_list[:3]) / total_shap * 100
        lines.append("### 2.4 全局解释总结\n")
        lines.append(f"- 前 3 个特征贡献了 **{top3_pct:.1f}%** 的总预测力")
        lines.append(f"- 最重要特征：**{feat_list[0]['feature']}**（SHAP={feat_list[0]['mean_shap']:.4f}）")
        lines.append("")

    # 3. 单样本解释
    if sample_result:
        lines.append("## 3. 单样本解释\n")
        lines.append(f"**样本标识**: {sample_result['sample_desc']}  ")
        lines.append(f"**样本索引**: {sample_result['sample_idx']}\n")
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| 实际标签 | {sample_result['actual_label']} |")
        lines.append(f"| 预测概率 | {sample_result['predicted_prob']:.4f} |")
        lines.append(f"| 基线值 | {sample_result['base_value']:.4f} |")
        lines.append("")
        if "waterfall_plot" in charts:
            lines.append("### 3.1 瀑布图\n")
            lines.append(f"![Waterfall]({assets_rel}/{charts['waterfall_plot']['filename']})\n")
        lines.append("### 3.2 主要贡献特征\n")
        lines.append("| 特征 | 样本值 | SHAP 值 | 影响方向 |")
        lines.append("|------|--------|---------|----------|")
        for c in sample_result["top_contributors"][:10]:
            direction = "增加预测" if c["direction"] == "positive" else "降低预测"
            lines.append(f"| {c['feature']} | {c['value']:.4f} | {c['shap_value']:.4f} | {direction} |")
        lines.append("")

    # 4. 交互
    if interaction_results:
        lines.append("## 4. 特征交互分析\n")
        for idx, r in enumerate(interaction_results, 1):
            lines.append(f"### 4.{idx} {r['feature1']} vs {r['feature2']}\n")
            key = f"dependence_{r['feature1']}_{r['feature2']}"
            if key in charts:
                lines.append(f"![Dependence]({assets_rel}/{charts[key]['filename']})\n")

    # 5. 总结
    if len(feat_list) >= 3:
        top_feats = [f["feature"] for f in feat_list[:3]]
        lines.append("## 5. 模型解释总结\n")
        lines.append(f"模型主要决策依据：**{top_feats[0]}**、**{top_feats[1]}**、**{top_feats[2]}**。")

    return "\n".join(lines)

# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────

def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _resolve_features_from_meta(model_path: str) -> List[str]:
    """从 <model_path>_meta.json 自动加载特征列表（xgb-modeling 产物约定）。"""
    mp = Path(model_path)
    # 尝试 <name>_meta.json / <name>.meta.json
    candidates = [
        mp.with_name(mp.stem + "_meta.json"),
        mp.with_suffix(".meta.json"),
    ]
    for c in candidates:
        if c.is_file():
            try:
                meta = json.loads(c.read_text(encoding="utf-8"))
                feats = meta.get("feature_names") or meta.get("features") or []
                if feats:
                    logger.info(f"从元数据加载特征: {c} ({len(feats)} 个)")
                    return list(feats)
            except Exception as e:
                logger.warning(f"读取 meta 失败 {c}: {e}")
    return []

def main() -> int:
    parser = argparse.ArgumentParser(description="XGBoost 模型解释 (portable)")
    parser.add_argument("--model_path", help="XGBoost 模型文件路径 (.json)")
    parser.add_argument("--data_path", help="数据文件路径 (parquet/csv)")
    parser.add_argument("--target", help="目标变量列名")
    parser.add_argument("--features", help="特征列表，逗号分隔（不传则从 *_meta.json 自动读取）")
    parser.add_argument("--sample_id", type=int, help="单样本解释：样本索引")
    parser.add_argument("--sample_filter", help="单样本解释：pandas query")
    parser.add_argument("--top_n", type=int, default=20, help="Top N 特征")
    parser.add_argument("--interaction_features", help="交互分析特征对，如 f1,f2")
    parser.add_argument("--output_dir", default=None, help="产物目录；默认 ./outputs/<ts>")
    parser.add_argument("--output_name", default="model_explanation_report", help="报告基名")
    parser.add_argument("--config", default=None, help="JSON 配置文件路径（命令行优先）")
    parsed = parser.parse_args()

    cfg = _load_config(parsed.config)
    model_path = parsed.model_path or cfg.get("model_path")
    data_path = parsed.data_path or cfg.get("data_path")
    target = parsed.target or cfg.get("target")
    if not model_path:
        print("ERROR: --model_path 必填")
        return 2
    if not data_path:
        print("ERROR: --data_path 必填")
        return 2
    if not target:
        print("ERROR: --target 必填")
        return 2

    # 特征
    if parsed.features:
        features = [x.strip() for x in parsed.features.split(",") if x.strip()]
    elif cfg.get("features"):
        feats_cfg = cfg["features"]
        features = feats_cfg if isinstance(feats_cfg, list) else [x.strip() for x in str(feats_cfg).split(",")]
    else:
        features = _resolve_features_from_meta(model_path)
    if not features:
        print("ERROR: 特征列表未指定；请传 --features 或确保模型旁存在 *_meta.json")
        return 2

    # output_dir
    if parsed.output_dir:
        output_dir = Path(parsed.output_dir).resolve()
    elif cfg.get("output_dir"):
        output_dir = Path(cfg["output_dir"]).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (Path.cwd() / "outputs" / ts).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = parsed.output_name or "model_explanation_report"

    # 交互对
    interaction_pair: Optional[Tuple[str, str]] = None
    if parsed.interaction_features:
        parts = [p.strip() for p in parsed.interaction_features.split(",")]
        if len(parts) == 2:
            interaction_pair = (parts[0], parts[1])

    emitter = ResultEmitter(skill="model-explanation", output_dir=output_dir)

    try:
        explainer_obj = ModelExplainer(
            model_path=model_path,
            data_path=data_path,
            target=target,
            features=features,
        )
        global_result = explainer_obj.global_explanation(top_n=parsed.top_n)

        sample_result = None
        if parsed.sample_id is not None or parsed.sample_filter:
            sample_result = explainer_obj.sample_explanation(
                sample_id=parsed.sample_id,
                sample_filter=parsed.sample_filter,
            )

        interaction_results: List[Dict[str, Any]] = []
        if interaction_pair:
            interaction_results.append(explainer_obj.interaction_analysis(interaction_pair))

        assets_dir = output_dir / "assets"
        charts = generate_shap_charts(
            global_result, sample_result, interaction_results,
            features, explainer_obj.shap_values, explainer_obj.X, explainer_obj.explainer,
            assets_dir,
        )

        # 报告
        report_md = generate_report(
            explainer_obj, global_result, sample_result,
            interaction_results, charts, assets_rel="assets",
        )
        report_path = output_dir / f"{output_name}.md"
        report_path.write_text(report_md, encoding="utf-8")

        # shap_data.json
        shap_data = {
            "type": "shap_explanation",
            "model_path": model_path,
            "data_path": data_path,
            "target": target,
            "feature_count": len(features),
            "sample_count": len(explainer_obj.data),
            "global": {
                "feature_importance": global_result["feature_importance"][:global_result["top_n"]],
                "top_n": global_result["top_n"],
            },
            "sample": sample_result,
        }
        shap_data_path = output_dir / "shap_data.json"
        shap_data_path.write_text(
            json.dumps(shap_data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        # 登记产物
        emitter.add_report(str(report_path))
        emitter.add_file(str(shap_data_path), role="asset", meta={"type": "shap_data"})
        for key, info in charts.items():
            emitter.add_asset(info["path"], title=key)

        top_feats = global_result["feature_importance"][:min(global_result["top_n"], 20)]
        emitter.update_metrics({
            "n_features": len(features),
            "n_samples": int(len(explainer_obj.data)),
            "top1_shap": round(float(top_feats[0]["mean_shap"]), 6) if top_feats else None,
        })
        emitter.update_state({
            "explanation": {
                "method": "shap",
                "feature_count": len(features),
                "top_features": [
                    {"name": f["feature"], "importance": round(float(f["mean_shap"]), 4)}
                    for f in top_feats[:10]
                ],
            },
        })
        top_name = top_feats[0]["feature"] if top_feats else "-"
        emitter.set_summary(
            f"model-explanation 完成：{len(features)} 特征 / {len(explainer_obj.data)} 样本，"
            f"Top 特征「{top_name}」，生成 {len(charts)} 张图表。"
        )
        result_path = emitter.write(output_dir / "result.json")

        # stdout 摘要
        print("\n=== SHAP 模型解释结果 ===")
        print(f"特征数量: {len(features)} | 样本数量: {len(explainer_obj.data)}")
        print("\nTop 5 全局特征重要性:")
        for i, f in enumerate(top_feats[:5], 1):
            print(f"  {i}. {f['feature']}: {f['mean_shap']:.4f}")
        if sample_result:
            print(f"\n单样本（idx={sample_result['sample_idx']}）: "
                  f"label={sample_result['actual_label']}, prob={sample_result['predicted_prob']:.4f}")
        print(f"\n图表数量: {len(charts)}")
        print(f"report: {report_path}")
        print(f"result: {result_path}")
        return 0

    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        try:
            emitter.set_status("failed")
            emitter.set_summary(f"执行失败：{type(e).__name__}: {e}")
            fail_path = emitter.write(output_dir / "result.json")
            print(f"FAILED | {type(e).__name__}: {e}")
            print(f"result: {fail_path}")
        except Exception:
            pass
        return 1

if __name__ == "__main__":
    sys.exit(main())
