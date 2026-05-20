# -*- coding: utf-8 -*-
"""
分群自主探索引擎
融合 autoresearch 思想：Try → Measure → Keep/Discard → Repeat

"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 添加路径（同 skill 内模块：segment_strategies / segment_hypothesis / model_merger）
SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR / 'scripts'))
sys.path.insert(0, str(SKILL_DIR.parent / 'auto-experiment' / 'scripts'))
# xgb_core 由 python_executor 预置 PYTHONPATH=backend/libs 暴露，不再需 sys.path.insert

from segment_strategies import BaseSegmenter, RuleSegmenter, ClusterSegmenter, TreeSegmenter
from segment_hypothesis import SegmentHypothesis, SegmentHypothesisGenerator
from model_merger import (
    SegmentModelTrainer,
    SegmentModelResult,
    RouteMerger,
    evaluate_segment_strategy,
)
from xgb_core import XGBTrainer, ModelEvaluator
from significance import SignificanceTester

class SegmentExplorationResult:
    """单轮探索结果"""
    
    def __init__(
        self,
        round_num: int,
        hypothesis: SegmentHypothesis,
        segment_info: Dict[str, Any],
        merged_metrics: Dict[str, Any],
        improvement: float,
        confidence: float,
        is_significant: bool,
        decision: str,
        reason: str,
        segment_models: Dict[int, SegmentModelResult] = None,
    ):
        self.round_num = round_num
        self.hypothesis = hypothesis
        self.segment_info = segment_info
        self.merged_metrics = merged_metrics
        self.improvement = improvement
        self.confidence = confidence
        self.is_significant = is_significant
        self.decision = decision
        self.reason = reason
        self.segment_models = segment_models
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'round': self.round_num,
            'strategy_type': self.hypothesis.strategy_type if self.hypothesis else 'baseline',
            'description': self.hypothesis.description if self.hypothesis else '基线',
            'params': self.hypothesis.params if self.hypothesis else {},
            'n_segments': self.segment_info.get('n_segments', 1),
            'val_auc': self.merged_metrics.get('val_auc', 0),
            'val_ks': self.merged_metrics.get('val_ks', 0),
            'oot_auc': self.merged_metrics.get('oot_auc'),
            'oot_ks': self.merged_metrics.get('oot_ks'),
            'improvement': self.improvement,
            'confidence': self.confidence,
            'decision': self.decision,
            'reason': self.reason,
            'segment_details': self.merged_metrics.get('segment_details', []),
        }

class SegmentAutoExplorer:
    """
    分群自主探索引擎
    
    核心流程：
    1. 建立基线（不分群的单一模型）
    2. 生成分群假设
    3. 训练分群子模型
    4. 评估整体效果
    5. 决策（Keep 或 Discard）
    6. 重复
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        target: str,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        oot_idx: np.ndarray = None,
        features: List[str] = None,
        metric: str = 'auc',
        significance_threshold: float = 2.0,
        min_segment_ratio: float = 0.05,
        verbose: bool = True,
    ):
        """初始化探索引擎

        Args:
            data: 完整数据集
            target: 目标变量列名
            train_idx: 训练集索引
            val_idx: 验证集 val 索引（用于 early stopping 与决策）
            oot_idx: OOT索引（仅用于最终评估）
            features: 特征列表
            metric: 优化指标
            significance_threshold: 显著性阈值
            min_segment_ratio: 最小分群占比
            verbose: 是否输出详细日志
        """
        self.data = data
        self.target = target
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.oot_idx = oot_idx
        self.metric = metric.lower()
        self.min_segment_ratio = min_segment_ratio
        self.verbose = verbose
        
        # 确定特征
        if features:
            self.features = [f for f in features if f in data.columns]
        else:
            self.features = [c for c in data.columns 
                           if c != target and data[c].dtype in ['int64', 'float64']]
        
        self.y = data[target].values
        
        # 初始化组件
        self.sig_tester = SignificanceTester(threshold=significance_threshold)
        self.model_trainer = SegmentModelTrainer(features=self.features)
        self.merger = RouteMerger()
        
        # 状态
        self.baseline_metric: float = 0.0
        self.current_best_metric: float = 0.0
        self.current_best_result: SegmentExplorationResult = None
        self.results: List[SegmentExplorationResult] = []
    
    def run_exploration(
        self,
        max_rounds: int = 5,
        user_rules: Dict[str, str] = None,
        progress_callback: Callable[[int, int, str], None] = None,
    ) -> Dict[str, Any]:
        """
        执行分群自主探索
        
        Args:
            max_rounds: 最大探索轮数
            user_rules: 用户指定的分群规则（作为候选方案之一）
            progress_callback: 进度回调
        
        Returns:
            探索结果摘要
        """
        self._log("=" * 60)
        self._log("🚀 启动分群自主探索")
        self._log(f"   最大轮数: {max_rounds}")
        self._log(f"   优化指标: {self.metric.upper()}")
        self._log(f"   最小分群占比: {self.min_segment_ratio:.1%}")
        self._log("=" * 60)
        
        # 初始化假设生成器
        hypothesis_gen = SegmentHypothesisGenerator(
            data=self.data,
            target=self.target,
            user_rules=user_rules,
            candidate_features=self.features[:10],
            tried_hypotheses=[],
        )
        
        # 执行探索循环
        for round_num in range(1, max_rounds + 1):
            self._log(f"\n{'━' * 40}")
            self._log(f"🔬 Round {round_num}/{max_rounds}")
            
            if progress_callback:
                progress_callback(round_num, max_rounds, f"Round {round_num}")
            
            # 生成假设
            hypothesis = hypothesis_gen.generate_next(
                history=[r.to_dict() for r in self.results]
            )
            
            if hypothesis is None:
                self._log("   ⚠️ 没有更多假设可尝试，提前结束")
                break
            
            self._log(f"   假设: {hypothesis.description}")
            
            # 执行单轮探索
            result = self._run_single_round(round_num, hypothesis)
            self.results.append(result)
            
            # 输出结果
            metric_val = result.merged_metrics.get(f'val_{self.metric}', 0)
            self._log(f"   结果: {self.metric.upper()} = {metric_val:.4f} ({result.improvement:+.4f})")
            self._log(f"   分群数: {result.segment_info.get('n_segments', 1)}")
            
            if result.merged_metrics.get('min_segment_ratio', 1) < self.min_segment_ratio:
                self._log(f"   ⚠️ 最小分群占比 {result.merged_metrics.get('min_segment_ratio', 0):.1%} < {self.min_segment_ratio:.1%}")
            
            if result.decision == 'KEEP':
                self._log(f"   决策: ✅ KEEP - {result.reason}")
                self.current_best_metric = metric_val
                self.current_best_result = result
            else:
                self._log(f"   决策: ❌ DISCARD - {result.reason}")
            
            # 添加观测值
            self.sig_tester.add_observation(metric_val)
        
        # 生成最终报告
        return self._generate_final_report()
    
    def _run_single_round(
        self,
        round_num: int,
        hypothesis: SegmentHypothesis,
    ) -> SegmentExplorationResult:
        """执行单轮探索"""
        
        # 基线情况（不分群）
        if hypothesis.strategy_type == 'baseline' or hypothesis.segmenter is None:
            return self._run_baseline(round_num, hypothesis)
        
        try:
            # 1. 应用分群策略
            segmenter = hypothesis.segmenter
            segment_labels = segmenter.fit_predict(self.data, self.y)
            segment_info = segmenter.get_segment_info()
            
            # 2. 训练分群子模型
            segment_models = self.model_trainer.train_segments(
                data=self.data,
                target=self.target,
                segment_labels=segment_labels,
                segment_names=segment_info['segment_names'],
                train_idx=self.train_idx,
                val_idx=self.val_idx,
                oot_idx=self.oot_idx,
            )

            if not segment_models:
                return SegmentExplorationResult(
                    round_num=round_num,
                    hypothesis=hypothesis,
                    segment_info={'n_segments': 0},
                    merged_metrics={'val_auc': 0, 'val_ks': 0},
                    improvement=0,
                    confidence=0,
                    is_significant=False,
                    decision='DISCARD',
                    reason='无有效分群模型',
                )

            # 3. 评估整体效果
            merged_metrics = evaluate_segment_strategy(
                data=self.data,
                target=self.target,
                segment_labels=segment_labels,
                segment_models=segment_models,
                features=self.features,
                val_idx=self.val_idx,
                oot_idx=self.oot_idx,
                merger=self.merger,
            )

            # 4. 计算改进和显著性
            metric_val = merged_metrics.get(f'val_{self.metric}', 0)
            improvement = metric_val - self.current_best_metric
            is_sig, confidence, color = self.sig_tester.is_significant(improvement)
            
            # 5. 检查分群有效性
            min_ratio = merged_metrics.get('min_segment_ratio', 0)
            segment_psi = merged_metrics.get('segment_psi', 0)
            
            # 6. 决策
            if improvement <= 0:
                decision = 'DISCARD'
                reason = '指标未提升'
            elif min_ratio < self.min_segment_ratio:
                decision = 'DISCARD'
                reason = f'分群过小({min_ratio:.1%})'
            elif segment_psi > 0.25:
                decision = 'DISCARD'
                reason = f'分群不稳定(PSI={segment_psi:.3f})'
            elif not is_sig and len(self.results) >= 3:
                decision = 'DISCARD'
                reason = f'改进不显著({color} {confidence:.1f}×)'
            else:
                decision = 'KEEP'
                reason = f'改进显著({color} {confidence:.1f}×)' if is_sig else '早期探索保留'
            
            return SegmentExplorationResult(
                round_num=round_num,
                hypothesis=hypothesis,
                segment_info=segment_info,
                merged_metrics=merged_metrics,
                improvement=improvement,
                confidence=confidence,
                is_significant=is_sig,
                decision=decision,
                reason=reason,
                segment_models=segment_models,
            )
            
        except Exception as e:
            return SegmentExplorationResult(
                round_num=round_num,
                hypothesis=hypothesis,
                segment_info={'n_segments': 0, 'error': str(e)},
                merged_metrics={'val_auc': 0, 'val_ks': 0},
                improvement=0,
                confidence=0,
                is_significant=False,
                decision='DISCARD',
                reason=f'执行失败: {str(e)}',
            )
    
    def _run_baseline(
        self,
        round_num: int,
        hypothesis: SegmentHypothesis,
    ) -> SegmentExplorationResult:
        """运行基线模型（不分群）"""
        
        # 训练单一模型
        X_train = self.data.iloc[self.train_idx][self.features]
        y_train = self.y[self.train_idx]
        X_val = self.data.iloc[self.val_idx][self.features]
        y_val = self.y[self.val_idx]

        trainer = XGBTrainer()
        # early stopping 严格使用 val，禁止用 oot
        trainer.train(X_train, y_train, X_val, y_val)

        # 评估
        val_prob = trainer.predict_proba(X_val)
        val_metrics = ModelEvaluator.evaluate(y_val, val_prob)

        merged_metrics = {
            'val_auc': val_metrics['auc'],
            'val_ks': val_metrics['ks'],
            'n_segments': 1,
            'min_segment_ratio': 1.0,
        }

        # OOT 评估
        if self.oot_idx is not None and len(self.oot_idx) > 0:
            X_oot = self.data.iloc[self.oot_idx][self.features]
            y_oot = self.y[self.oot_idx]
            oot_prob = trainer.predict_proba(X_oot)
            oot_metrics = ModelEvaluator.evaluate(y_oot, oot_prob)
            merged_metrics['oot_auc'] = oot_metrics['auc']
            merged_metrics['oot_ks'] = oot_metrics['ks']

        # 设置基线
        metric_val = merged_metrics.get(f'val_{self.metric}', 0)
        self.baseline_metric = metric_val
        self.current_best_metric = metric_val
        self.sig_tester.add_observation(metric_val)
        
        return SegmentExplorationResult(
            round_num=round_num,
            hypothesis=hypothesis,
            segment_info={'n_segments': 1, 'segment_names': ['全量']},
            merged_metrics=merged_metrics,
            improvement=0,
            confidence=0,
            is_significant=False,
            decision='KEEP',
            reason='基线模型',
        )
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """生成最终报告"""
        self._log(f"\n{'=' * 60}")
        self._log("📋 分群探索完成报告")
        self._log("=" * 60)
        
        # 统计
        kept_results = [r for r in self.results if r.decision == 'KEEP']
        discarded_results = [r for r in self.results if r.decision == 'DISCARD']
        
        # 最优结果
        best = self.current_best_result
        if best:
            best_metric = best.merged_metrics.get(f'val_{self.metric}', 0)
            total_improvement = best_metric - self.baseline_metric
            improvement_pct = total_improvement / self.baseline_metric * 100 if self.baseline_metric else 0
            
            self._log(f"\n🏆 最优方案: {best.hypothesis.description if best.hypothesis else '基线'}")
            self._log(f"   基线 → 最优: {self.baseline_metric:.4f} → {best_metric:.4f}")
            self._log(f"   总改进: {total_improvement:+.4f} ({improvement_pct:+.2f}%)")
            self._log(f"   分群数: {best.segment_info.get('n_segments', 1)}")
            
            # 分群详情
            if best.segment_info.get('segment_names'):
                self._log(f"   分群: {', '.join(best.segment_info['segment_names'])}")
        
        self._log(f"\n📊 探索统计:")
        self._log(f"   总轮数: {len(self.results)}")
        self._log(f"   保留: {len(kept_results)}")
        self._log(f"   丢弃: {len(discarded_results)}")
        
        return {
            'baseline_metric': self.baseline_metric,
            'best_metric': best.merged_metrics.get(f'val_{self.metric}', 0) if best else self.baseline_metric,
            'best_strategy': best.hypothesis.description if best and best.hypothesis else '基线',
            'best_segment_info': best.segment_info if best else {},
            'best_segment_models': best.segment_models if best else None,
            'total_improvement': (best.merged_metrics.get(f'val_{self.metric}', 0) - self.baseline_metric) if best else 0,
            'n_rounds': len(self.results),
            'n_kept': len(kept_results),
            'n_discarded': len(discarded_results),
            'results': [r.to_dict() for r in self.results],
            'stats': self.sig_tester.get_summary(),
        }
    
    def _log(self, message: str) -> None:
        """输出日志"""
        if self.verbose:
            print(message)
