# -*- coding: utf-8 -*-
"""
分群假设生成器模块
AI 自主生成分群策略假设

"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from segment_strategies import (
    BaseSegmenter,
    RuleSegmenter,
    ClusterSegmenter,
    TreeSegmenter,
    create_segmenter,
)

class SegmentHypothesis:
    """分群假设"""
    
    def __init__(
        self,
        strategy_type: str,
        description: str,
        segmenter: BaseSegmenter,
        params: Dict[str, Any] = None,
    ):
        self.strategy_type = strategy_type
        self.description = description
        self.segmenter = segmenter
        self.params = params or {}
    
    def __repr__(self):
        return f"SegmentHypothesis({self.description})"

class SegmentHypothesisGenerator:
    """
    分群假设生成器
    
    智能生成分群策略假设，支持：
    1. 用户规则作为基线
    2. 聚类分群（不同K值）
    3. 决策树分群（不同深度、不同特征）
    4. 混合策略
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        target: str,
        user_rules: Dict[str, str] = None,
        candidate_features: List[str] = None,
        tried_hypotheses: List[str] = None,
    ):
        """
        初始化假设生成器
        
        Args:
            data: 数据集（用于分析特征）
            target: 目标变量列名
            user_rules: 用户指定的分群规则
            candidate_features: 候选分群特征
            tried_hypotheses: 已尝试的假设描述列表
        """
        self.data = data
        self.target = target
        self.user_rules = user_rules
        self.tried: Set[str] = set(tried_hypotheses or [])
        
        # 分析数据，确定候选特征
        if candidate_features:
            self.candidate_features = candidate_features
        else:
            self.candidate_features = self._analyze_candidate_features()
        
        # 假设队列
        self.hypothesis_queue: List[SegmentHypothesis] = []
        self._build_initial_queue()
    
    def _analyze_candidate_features(self) -> List[str]:
        """分析数据，找出适合做分群的特征"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 排除目标变量
        if self.target in numeric_cols:
            numeric_cols.remove(self.target)
        
        # 选择方差较大、非常数的特征
        candidates = []
        for col in numeric_cols:
            n_unique = self.data[col].nunique()
            if n_unique > 1 and n_unique < len(self.data) * 0.9:  # 非常数且非唯一
                candidates.append(col)
        
        # 按特征名排序，优先选择常见的分群特征
        priority_keywords = ['age', 'income', 'score', 'credit', 'amount', 'balance']
        
        def sort_key(col):
            col_lower = col.lower()
            for i, kw in enumerate(priority_keywords):
                if kw in col_lower:
                    return (i, col)
            return (len(priority_keywords), col)
        
        candidates.sort(key=sort_key)
        return candidates[:10]  # 最多取10个候选特征
    
    def _build_initial_queue(self) -> None:
        """构建初始假设队列"""
        # 1. 基线（不分群）
        self.hypothesis_queue.append(SegmentHypothesis(
            strategy_type='baseline',
            description='基线（不分群）',
            segmenter=None,
            params={},
        ))
        
        # 2. 用户规则（如果有）
        if self.user_rules:
            self.hypothesis_queue.append(SegmentHypothesis(
                strategy_type='rule',
                description=f"用户规则分群: {' | '.join(self.user_rules.keys())}",
                segmenter=RuleSegmenter(self.user_rules),
                params={'rules': self.user_rules},
            ))
        
        # 3. 决策树分群（不同深度）
        for depth in [2, 3]:
            self.hypothesis_queue.append(SegmentHypothesis(
                strategy_type='tree',
                description=f"决策树分群(depth={depth})",
                segmenter=TreeSegmenter(max_depth=depth),
                params={'max_depth': depth},
            ))
        
        # 4. 聚类分群（不同K值）
        for k in [2, 3, 4]:
            self.hypothesis_queue.append(SegmentHypothesis(
                strategy_type='cluster',
                description=f"K-Means聚类(K={k})",
                segmenter=ClusterSegmenter(n_clusters=k),
                params={'n_clusters': k},
            ))
        
        # 5. 特定特征的决策树
        if self.candidate_features:
            for feat in self.candidate_features[:3]:
                self.hypothesis_queue.append(SegmentHypothesis(
                    strategy_type='tree',
                    description=f"决策树分群(特征={feat})",
                    segmenter=TreeSegmenter(max_depth=2, features=[feat]),
                    params={'max_depth': 2, 'features': [feat]},
                ))
    
    def generate_next(self, history: List[Dict] = None) -> Optional[SegmentHypothesis]:
        """
        生成下一个假设
        
        Args:
            history: 历史实验结果，用于智能调整
        
        Returns:
            下一个假设，或 None（如果没有更多假设）
        """
        # 从队列中获取
        while self.hypothesis_queue:
            hypothesis = self.hypothesis_queue.pop(0)
            
            # 检查是否已尝试过
            if hypothesis.description in self.tried:
                continue
            
            self.tried.add(hypothesis.description)
            return hypothesis
        
        # 队列空了，尝试基于历史结果生成新假设
        if history:
            new_hypothesis = self._generate_from_history(history)
            if new_hypothesis and new_hypothesis.description not in self.tried:
                self.tried.add(new_hypothesis.description)
                return new_hypothesis
        
        return None
    
    def _generate_from_history(self, history: List[Dict]) -> Optional[SegmentHypothesis]:
        """基于历史结果生成新假设"""
        if not history:
            return None
        
        # 找出最佳结果
        best = max(history, key=lambda x: x.get('val_ks', x.get('val_auc', 0)))
        best_type = best.get('strategy_type', '')
        best_params = best.get('params', {})
        
        # 根据最佳结果类型生成变体
        if best_type == 'tree':
            # 尝试增加深度
            current_depth = best_params.get('max_depth', 2)
            if current_depth < 4:
                new_depth = current_depth + 1
                desc = f"决策树分群(depth={new_depth})"
                if desc not in self.tried:
                    return SegmentHypothesis(
                        strategy_type='tree',
                        description=desc,
                        segmenter=TreeSegmenter(max_depth=new_depth),
                        params={'max_depth': new_depth},
                    )
        
        elif best_type == 'cluster':
            # 尝试调整K值
            current_k = best_params.get('n_clusters', 3)
            for new_k in [current_k - 1, current_k + 1]:
                if 2 <= new_k <= 6:
                    desc = f"K-Means聚类(K={new_k})"
                    if desc not in self.tried:
                        return SegmentHypothesis(
                            strategy_type='cluster',
                            description=desc,
                            segmenter=ClusterSegmenter(n_clusters=new_k),
                            params={'n_clusters': new_k},
                        )
        
        # 尝试混合策略
        if best_type == 'tree' and 'cluster' not in str(self.tried):
            # 决策树效果好，尝试聚类作为对比
            desc = "K-Means聚类(K=3)"
            if desc not in self.tried:
                return SegmentHypothesis(
                    strategy_type='cluster',
                    description=desc,
                    segmenter=ClusterSegmenter(n_clusters=3),
                    params={'n_clusters': 3},
                )
        
        return None
    
    def add_hypothesis(self, hypothesis: SegmentHypothesis) -> None:
        """手动添加假设到队列"""
        if hypothesis.description not in self.tried:
            self.hypothesis_queue.append(hypothesis)
    
    def get_exploration_summary(self) -> str:
        """获取探索进度摘要"""
        return f"""分群探索进度:
- 已尝试: {len(self.tried)} 种策略
- 队列剩余: {len(self.hypothesis_queue)} 种
- 候选特征: {', '.join(self.candidate_features[:5])}
"""

def suggest_segment_features(
    data: pd.DataFrame,
    target: str,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    推荐适合做分群的特征
    
    基于特征与目标变量的关联性和分布特点
    
    Args:
        data: 数据
        target: 目标变量
        top_k: 返回前K个
    
    Returns:
        [(特征名, 得分), ...]
    """
    scores = []
    y = data[target].values
    
    for col in data.select_dtypes(include=[np.number]).columns:
        if col == target:
            continue
        
        try:
            x = data[col].fillna(data[col].median()).values
            
            # 计算与目标的相关性
            corr = np.abs(np.corrcoef(x, y)[0, 1])
            if np.isnan(corr):
                corr = 0
            
            # 计算分箱后的 IV 近似
            n_bins = min(10, len(np.unique(x)))
            if n_bins < 2:
                iv_approx = 0
            else:
                bins = pd.qcut(x, q=n_bins, duplicates='drop')
                df_temp = pd.DataFrame({'bin': bins, 'y': y})
                grouped = df_temp.groupby('bin')['y'].agg(['sum', 'count'])
                grouped['bad'] = grouped['sum']
                grouped['good'] = grouped['count'] - grouped['sum']
                
                total_bad = grouped['bad'].sum()
                total_good = grouped['good'].sum()
                
                if total_bad > 0 and total_good > 0:
                    grouped['bad_rate'] = grouped['bad'] / total_bad
                    grouped['good_rate'] = grouped['good'] / total_good
                    grouped['woe'] = np.log((grouped['bad_rate'] + 0.001) / (grouped['good_rate'] + 0.001))
                    grouped['iv'] = (grouped['bad_rate'] - grouped['good_rate']) * grouped['woe']
                    iv_approx = grouped['iv'].sum()
                else:
                    iv_approx = 0
            
            # 综合得分
            score = corr * 0.3 + min(iv_approx, 1.0) * 0.7
            scores.append((col, score))
        
        except Exception:
            continue
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
