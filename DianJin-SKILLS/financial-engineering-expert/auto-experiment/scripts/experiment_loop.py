# -*- coding: utf-8 -*-
"""
自主实验循环引擎 v2
支持特征组级别的自主探索: 组发现 → 组评估 → 组叠加 → 组消融 → 精细筛选

借鉴 karpathy/autoresearch 核心思想: Try → Measure → Keep/Discard → Repeat

"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 添加项目路径以导入本 skill 内其他模块
SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR / 'scripts'))

from significance import SignificanceTester
from session_tracker import SessionTracker
from hypothesis_generator import HypothesisGenerator, FeatureGroupDiscovery, create_cross_feature

# xgb_core 由上层 run_experiment.py 的 _vendor bootstrap 暴露
from xgb_core import XGBTrainer, ModelEvaluator
from report_artifact import render_ks_curve, render_roc_curve

class ExperimentResult:
    """单次实验结果"""

    def __init__(
        self,
        round_num: int,
        hypothesis: Dict[str, Any],
        metrics: Dict[str, float],
        improvement: float,
        confidence: float,
        is_significant: bool,
        decision: str,
        reason: str,
        features: List[str],
        feature_importance: Dict[str, float] = None,
    ):
        self.round_num = round_num
        self.hypothesis = hypothesis
        self.metrics = metrics
        self.improvement = improvement
        self.confidence = confidence
        self.is_significant = is_significant
        self.decision = decision
        self.reason = reason
        self.features = features
        self.feature_importance = feature_importance or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'round': self.round_num,
            'hypothesis': self.hypothesis.get('description', ''),
            'metrics': self.metrics,
            'improvement': self.improvement,
            'confidence': self.confidence,
            'is_significant': self.is_significant,
            'decision': self.decision,
            'reason': self.reason,
            'n_features': len(self.features),
        }

class AutoExperimentLoop:
    """
    自主实验循环引擎 v2

    核心流程:
    1. 数据概览 & 特征分组发现
    2. 建立全量基线
    3. Phase 1: 特征组独立评估
    4. Phase 2: 组间增量叠加
    5. Phase 3: 组级消融分析
    6. Phase 4: 组内精细筛选
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target: str,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        oot_idx: np.ndarray = None,
        metric: str = 'ks',
        direction: str = 'maximize',
        significance_threshold: float = 2.0,
        session_tracker: SessionTracker = None,
        verbose: bool = True,
    ):
        self.data = data
        self.target = target
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.oot_idx = oot_idx
        self.metric = metric.lower()
        self.direction = direction
        self.verbose = verbose

        # 准备数据
        self.y = data[target].values
        self.all_features = [c for c in data.columns
                             if c != target and data[c].dtype in ['int64', 'float64']]

        # 初始化组件
        self.sig_tester = SignificanceTester(threshold=significance_threshold)
        self.session = session_tracker

        # 状态
        self.current_features: List[str] = []
        self.current_best_metric: float = 0.0
        self.baseline_metric: float = 0.0
        self.results: List[ExperimentResult] = []
        self.current_trainer: XGBTrainer = None

    def run_loop(
        self,
        exploration: str,
        max_rounds: int = 5,
        baseline_features: List[str] = None,
        progress_callback: Callable[[int, int, str], None] = None,
    ) -> Dict[str, Any]:
        """执行自主实验循环"""

        # ── Step 0: 数据概览 ──
        self._print_data_overview(exploration, max_rounds)

        # ── Step 1: 特征分组发现 ──
        groups = FeatureGroupDiscovery.discover_groups(self.all_features)
        self._print_group_discovery(groups)

        # ── Step 2: 建立全量基线 ──
        self._log("\n## 建立基线模型\n")
        baseline_result = self._establish_baseline(baseline_features)
        self.baseline_metric = baseline_result['val'][self.metric]
        self.current_best_metric = self.baseline_metric
        self.sig_tester.add_observation(self.baseline_metric)
        self._print_baseline(baseline_result)

        # 初始化会话
        if self.session:
            self.session.init_session(
                exploration=exploration,
                metric=self.metric,
                direction=self.direction,
                baseline_metric=self.baseline_metric,
                baseline_features=self.current_features.copy(),
            )

        # 初始化假设生成器
        tried = [r.hypothesis.get('description', '') for r in self.results]
        hypothesis_gen = HypothesisGenerator(
            available_features=self.all_features,
            current_features=self.current_features,
            exploration_direction=exploration,
            tried_hypotheses=tried,
        )

        # ── Step 3: 循环实验 ──
        self._log(f"\n## 开始自主探索实验循环 (最多 {max_rounds} 轮)\n")
        kept_features: List[str] = []
        discarded_features: List[str] = []

        for round_num in range(1, max_rounds + 1):
            if progress_callback:
                progress_callback(round_num, max_rounds, f"Round {round_num}")

            # 生成假设
            hypothesis = hypothesis_gen.generate_next_hypothesis()
            if hypothesis is None:
                self._log("\n> 所有探索阶段已完成，无更多假设可尝试。\n")
                break

            prev_metric = self.current_best_metric

            # 执行实验
            result = self._run_single_experiment(round_num, hypothesis)
            self.results.append(result)

            # 记录 Phase 1 组级结果到生成器
            if hypothesis.get('type') == 'group_isolation':
                val_score = result.metrics.get('val', {}).get(self.metric, 0)
                hypothesis_gen.record_group_result(
                    hypothesis.get('group_name', ''), val_score
                )

            # 记录到会话
            if self.session:
                self.session.log_experiment(
                    round_num=round_num,
                    hypothesis=hypothesis['description'],
                    features=result.features,
                    metric_value=result.metrics.get('val', {}).get(self.metric, 0),
                    improvement=result.improvement,
                    confidence=result.confidence,
                    decision=result.decision,
                    reason=result.reason,
                )

            metric_val = result.metrics.get('val', {}).get(self.metric, 0)

            if result.decision == 'KEEP':
                self.current_features = result.features
                self.current_best_metric = metric_val
                kept_features.extend(hypothesis.get('added', []))
                hypothesis_gen.update_current_features(self.current_features)
            else:
                discarded_features.extend(hypothesis.get('added', []))

            self.sig_tester.add_observation(metric_val)

            # === 输出本轮结构化 Markdown ===
            self._print_round_detail(
                round_num=round_num,
                max_rounds=max_rounds,
                hypothesis=hypothesis,
                result=result,
                prev_metric=prev_metric,
            )

        # ── Step 4: 汇总报告 ──
        total_improvement = self.current_best_metric - self.baseline_metric
        improvement_pct = (total_improvement / self.baseline_metric * 100
                           if self.baseline_metric else 0)
        self._print_final_summary(
            kept_features, discarded_features,
            total_improvement, improvement_pct,
        )

        suggestions = self._generate_suggestions(hypothesis_gen)
        if suggestions and self.session:
            self.session.add_pending(suggestions)

        # 生成最终模型的可视化 artifact（KS/ROC已由 shared/report_artifact 统一渲染）
        visualization = {}
        if self.oot_idx is not None and len(self.oot_idx) > 0 and self.current_trainer is not None:
            X_oot = self.data.iloc[self.oot_idx][self.current_features]
            y_oot = self.y[self.oot_idx]
            prob_oot = self.current_trainer.predict_proba(X_oot)

            oot_ks = float(ModelEvaluator.calculate_ks(y_oot, prob_oot))
            oot_auc = float(ModelEvaluator.evaluate(y_oot, prob_oot)['auc'])

            visualization = {
                'ks_curve_md': render_ks_curve(y_oot, prob_oot, prefix="OOT "),
                'roc_curve_md': render_roc_curve(y_oot, prob_oot, prefix="OOT "),
                'oot_ks': oot_ks,
                'oot_auc': oot_auc,
            }

        return {
            'baseline': {
                'metric': self.baseline_metric,
                'features': baseline_result.get('features', []),
            },
            'final': {
                'metric': self.current_best_metric,
                'features': self.current_features,
            },
            'improvement': total_improvement,
            'improvement_pct': improvement_pct,
            'results': [r.to_dict() for r in self.results],
            'kept_features': kept_features,
            'discarded_features': discarded_features,
            'suggestions': suggestions,
            'stats': self.sig_tester.get_summary(),
            'visualization': visualization,
        }

    # ════════════════════════════════════════
    #  模型训练与评估
    # ════════════════════════════════════════

    def _establish_baseline(self, baseline_features: List[str] = None) -> Dict[str, Any]:
        """建立基线模型（默认使用全量特征）"""
        if baseline_features:
            features = [f for f in baseline_features if f in self.all_features]
        else:
            # 默认使用全量特征建立基线
            features = self.all_features[:]

        self.current_features = features

        X_train = self.data.iloc[self.train_idx][features]
        y_train = self.y[self.train_idx]
        X_val = self.data.iloc[self.val_idx][features]
        y_val = self.y[self.val_idx]

        trainer = XGBTrainer()
        # early stopping 严格使用 val，禁止用 oot
        trainer.train(X_train, y_train, X_val, y_val)
        self.current_trainer = trainer

        train_prob = trainer.predict_proba(X_train)
        val_prob = trainer.predict_proba(X_val)

        train_metrics = ModelEvaluator.evaluate(y_train, train_prob)
        val_metrics = ModelEvaluator.evaluate(y_val, val_prob)

        result = {
            'features': features,
            'train': train_metrics,
            'val': val_metrics,
        }

        if self.oot_idx is not None and len(self.oot_idx) > 0:
            X_oot = self.data.iloc[self.oot_idx][features]
            y_oot = self.y[self.oot_idx]
            oot_prob = trainer.predict_proba(X_oot)
            result['oot'] = ModelEvaluator.evaluate(y_oot, oot_prob)

        return result

    def _run_single_experiment(
        self, round_num: int, hypothesis: Dict[str, Any]
    ) -> ExperimentResult:
        """执行单次实验"""
        features = hypothesis['features']

        # 处理交叉特征
        data_copy = self.data
        if hypothesis.get('type') == 'cross_feature':
            data_copy = self.data.copy()
            cross_cols = hypothesis.get('cross_cols', [])
            if len(cross_cols) == 2:
                cross_name = f"{cross_cols[0]}_x_{cross_cols[1]}"
                data_copy[cross_name] = create_cross_feature(
                    data_copy, cross_cols[0], cross_cols[1], 'multiply'
                )

        valid_features = [f for f in features if f in data_copy.columns]
        if not valid_features:
            return ExperimentResult(
                round_num=round_num, hypothesis=hypothesis,
                metrics={'train': {}, 'val': {}},
                improvement=0, confidence=0, is_significant=False,
                decision='DISCARD', reason='无有效特征',
                features=self.current_features,
            )

        X_train = data_copy.iloc[self.train_idx][valid_features]
        y_train = self.y[self.train_idx]
        X_val = data_copy.iloc[self.val_idx][valid_features]
        y_val = self.y[self.val_idx]

        try:
            trainer = XGBTrainer()
            # early stopping 严格使用 val，禁止用 oot
            trainer.train(X_train, y_train, X_val, y_val)

            train_prob = trainer.predict_proba(X_train)
            val_prob = trainer.predict_proba(X_val)

            train_metrics = ModelEvaluator.evaluate(y_train, train_prob)
            val_metrics = ModelEvaluator.evaluate(y_val, val_prob)

            metrics = {'train': train_metrics, 'val': val_metrics}

            if self.oot_idx is not None and len(self.oot_idx) > 0:
                X_oot = data_copy.iloc[self.oot_idx][valid_features]
                y_oot = self.y[self.oot_idx]
                oot_prob = trainer.predict_proba(X_oot)
                metrics['oot'] = ModelEvaluator.evaluate(y_oot, oot_prob)

        except Exception as e:
            return ExperimentResult(
                round_num=round_num, hypothesis=hypothesis,
                metrics={'train': {}, 'val': {}},
                improvement=0, confidence=0, is_significant=False,
                decision='DISCARD', reason=f'训练失败: {str(e)}',
                features=self.current_features,
            )

        new_metric = val_metrics[self.metric]
        improvement = new_metric - self.current_best_metric

        is_sig, confidence, color = self.sig_tester.is_significant(improvement)

        if self.direction == 'maximize':
            should_keep = improvement > 0 and is_sig
        else:
            should_keep = improvement < 0 and is_sig

        if should_keep:
            decision = 'KEEP'
            reason = f'改进显著 ({color} {confidence:.1f}x)'
            self.current_trainer = trainer
        else:
            decision = 'DISCARD'
            if improvement <= 0:
                reason = f'指标下降或无改进 ({improvement:+.4f})'
            else:
                reason = f'改进不显著 ({color} {confidence:.1f}x)'

        return ExperimentResult(
            round_num=round_num, hypothesis=hypothesis,
            metrics=metrics, improvement=improvement,
            confidence=confidence, is_significant=is_sig,
            decision=decision, reason=reason,
            features=valid_features,
            feature_importance=trainer.get_feature_importance() if trainer else {},
        )

    # ════════════════════════════════════════
    #  结构化输出
    # ════════════════════════════════════════

    def _print_data_overview(self, exploration: str, max_rounds: int) -> None:
        """输出数据概览"""
        n_rows = len(self.data)
        n_feats = len(self.all_features)
        n_train = len(self.train_idx)
        n_val = len(self.val_idx)
        n_oot = len(self.oot_idx) if self.oot_idx is not None else 0
        pos_rate = self.y.mean() * 100

        md = f"""# 自主实验循环报告

## 数据概览

| 项目 | 值 |
|------|-----|
| 总样本数 | {n_rows:,} |
| 数值特征数 | {n_feats} |
| 训练集 train | {n_train:,} ({n_train/n_rows*100:.1f}%) |
| 验证集 val | {n_val:,} ({n_val/n_rows*100:.1f}%) |
| OOT集 | {n_oot:,} ({n_oot/n_rows*100:.1f}%) |
| 正样本率 | {pos_rate:.2f}% |
| 优化指标 | {self.metric.upper()} ({self.direction}) |
| 探索方向 | {exploration} |
| 最大轮数 | {max_rounds} |
"""
        print(md)

    def _print_group_discovery(self, groups: Dict[str, List[str]]) -> None:
        """输出特征分组发现结果"""
        md = f"## 特征分组发现\n\n共发现 **{len(groups)}** 个特征组:\n\n"
        md += FeatureGroupDiscovery.format_group_summary(groups)
        md += "\n"
        print(md)

    def _print_baseline(self, baseline_result: Dict) -> None:
        """输出基线模型结果"""
        train = baseline_result.get('train', {})
        val = baseline_result.get('val', {})
        oot = baseline_result.get('oot', {})

        md = f"""**基线模型** (全量 {len(self.current_features)} 个特征):

| 指标 | Train | Val |"""
        if oot:
            md += " OOT |"
        md += "\n|------|-------|------|"
        if oot:
            md += "------|"
        md += f"\n| AUC | {train.get('auc',0):.4f} | {val.get('auc',0):.4f} |"
        if oot:
            md += f" {oot.get('auc',0):.4f} |"
        md += f"\n| KS | {train.get('ks',0):.4f} | {val.get('ks',0):.4f} |"
        if oot:
            md += f" {oot.get('ks',0):.4f} |"

        # 过拟合诊断 (train vs val)
        train_m = train.get(self.metric, 0)
        val_m = val.get(self.metric, 0)
        gap = train_m - val_m
        if gap > 0.05:
            md += f"\n\n> **过拟合风险**: Train-Val gap = {gap:.4f}，需关注泛化性"
        md += "\n"
        print(md)

    def _print_round_detail(
        self,
        round_num: int,
        max_rounds: int,
        hypothesis: Dict[str, Any],
        result: ExperimentResult,
        prev_metric: float,
    ) -> None:
        """输出每轮结构化 Markdown"""
        metric_upper = self.metric.upper()
        val_m = result.metrics.get('val', {})
        train_m = result.metrics.get('train', {})
        oot_m = result.metrics.get('oot', {})
        val_score = val_m.get(self.metric, 0)

        decision_icon = "KEEP" if result.decision == 'KEEP' else "DISCARD"
        decision_emoji = "+" if result.decision == 'KEEP' else "-"

        # 置信度
        conf = result.confidence
        if conf >= 2.0:
            conf_label = f"[PASS] {conf:.1f}x MAD (>=2.0) -> 改进显著"
        elif conf >= 1.0:
            conf_label = f"[EDGE] {conf:.1f}x MAD (1.0-2.0) -> 边缘改进"
        else:
            conf_label = f"[FAIL] {conf:.1f}x MAD (<1.0) -> 噪声范围内"

        def delta(v):
            return f"+{v:.4f}" if v > 0 else f"{v:.4f}"

        hyp_type = hypothesis.get('type', '')
        n_feats = len(result.features)

        md = f"""
---
### Round {round_num}/{max_rounds}

**{hypothesis.get('strategy', 'Unknown')}**

**探索逻辑**:
{hypothesis.get('reasoning', '(无)')}

**本轮假设**: {hypothesis['description']}
"""

        # 新增/移除特征展示
        added = hypothesis.get('added', [])
        removed = hypothesis.get('removed', [])
        if added:
            md += f"\n**新增特征** ({len(added)}个):\n"
            for feat in added[:10]:
                md += f"- `{feat}`\n"
            if len(added) > 10:
                md += f"- ... (+{len(added)-10}个)\n"
        if removed:
            md += f"\n**移除特征** ({len(removed)}个):\n"
            for feat in removed[:10]:
                md += f"- ~~`{feat}`~~\n"
            if len(removed) > 10:
                md += f"- ... (+{len(removed)-10}个)\n"

        # 指标结果
        md += f"""
**模型评估** (共 {n_feats} 个特征):

| 指标 | Train | Val | 基线Val | 变化 |
|------|-------|------|----------|------|
| AUC | {train_m.get('auc',0):.4f} | {val_m.get('auc',0):.4f} | {prev_metric:.4f} | {delta(result.improvement)} |
| KS  | {train_m.get('ks',0):.4f} | {val_m.get('ks',0):.4f} | - | - |
"""

        if oot_m:
            md += f"""
**OOT验证**:
| 指标 | OOT值 |
|------|-------|
| AUC | {oot_m.get('auc',0):.4f} |
| KS  | {oot_m.get('ks',0):.4f} |
"""

        # 过拟合检查（train vs val）
        train_val = train_m.get(self.metric, 0)
        gap = train_val - val_score
        if gap > 0.05:
            md += f"\n> Train-Val gap = {gap:.4f}，存在过拟合风险\n"

        # Top-5 特征重要性
        if result.feature_importance:
            sorted_imp = sorted(
                result.feature_importance.items(),
                key=lambda x: x[1], reverse=True
            )[:5]
            md += "\n**Top-5 特征重要性**:\n"
            md += "| # | 特征 | 重要性 |\n|---|------|--------|\n"
            for rank, (feat, imp) in enumerate(sorted_imp, 1):
                new_tag = " [NEW]" if feat in added else ""
                md += f"| {rank} | `{feat}`{new_tag} | {imp:.4f} |\n"

        md += f"""
**显著性检验**: {conf_label}

**决策**: [{decision_emoji}] {decision_icon} - {result.reason}

**当前最优特征集** ({len(self.current_features)}个特征, {metric_upper}={self.current_best_metric:.4f})
"""
        print(md)

    def _print_final_summary(
        self,
        kept: List[str],
        discarded: List[str],
        total_imp: float,
        imp_pct: float,
    ) -> None:
        """输出最终汇总"""
        md = f"""
---
## 实验汇总

| 项目 | 值 |
|------|-----|
| 总轮次 | {len(self.results)} |
| 基线 {self.metric.upper()} | {self.baseline_metric:.4f} |
| 最终 {self.metric.upper()} | {self.current_best_metric:.4f} |
| 总改进 | {total_imp:+.4f} ({imp_pct:+.2f}%) |
| 最终特征数 | {len(self.current_features)} |

### 实验总览

| Round | 阶段 | 假设 | {self.metric.upper()} | 变化 | 决策 |
|-------|------|------|----|------|------|
"""
        for r in self.results:
            dec = "KEEP" if r.decision == 'KEEP' else "DISCARD"
            val_score = r.metrics.get('val', {}).get(self.metric, 0)
            strategy = r.hypothesis.get('strategy', '')[:20]
            desc = r.hypothesis.get('description', '')[:25]
            md += f"| {r.round_num} | {strategy} | {desc} | {val_score:.4f} | {r.improvement:+.4f} | {dec} |\n"

        if kept:
            md += "\n**保留的特征/组**:\n"
            for f in kept[:20]:
                md += f"- {f}\n"
            if len(kept) > 20:
                md += f"- ... (+{len(kept)-20}个)\n"

        if discarded:
            md += "\n**丢弃的特征/组**:\n"
            for f in discarded[:20]:
                md += f"- ~~{f}~~\n"
            if len(discarded) > 20:
                md += f"- ... (+{len(discarded)-20}个)\n"

        md += "\n"
        print(md)

    # ════════════════════════════════════════
    #  建议生成
    # ════════════════════════════════════════

    def _generate_suggestions(self, hypothesis_gen: HypothesisGenerator) -> List[str]:
        """生成下一步建议"""
        suggestions = []

        remaining = hypothesis_gen.get_remaining_candidates()
        if remaining:
            suggestions.append(f"尚有未探索的特征组: {', '.join(remaining[:3])}")

        if self.current_trainer and self.current_trainer.feature_importance_:
            top_features = list(self.current_trainer.get_feature_importance().keys())[:3]
            for feat in top_features:
                prefix = feat.split('_')[0]
                suggestions.append(f"基于 {feat} 的重要性，可深入探索 {prefix} 组的衍生特征")

        return suggestions[:5]

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
