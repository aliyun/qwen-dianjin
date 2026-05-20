# -*- coding: utf-8 -*-
"""
子模型汇总器模块
将多个分群子模型汇总为整体预测能力

"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 添加路径以导入本 skill 内其他模块（如需）
SKILL_DIR = Path(__file__).parent.parent

# xgb_core 由 python_executor 预置 PYTHONPATH=backend/libs 暴露
from xgb_core import XGBTrainer, ModelEvaluator

class SegmentModelResult:
    """单个分群模型的结果"""
    
    def __init__(
        self,
        segment_id: int,
        segment_name: str,
        trainer: XGBTrainer,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        oot_metrics: Dict[str, float] = None,
        sample_count: int = 0,
        sample_ratio: float = 0.0,
    ):
        self.segment_id = segment_id
        self.segment_name = segment_name
        self.trainer = trainer
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.oot_metrics = oot_metrics
        self.sample_count = sample_count
        self.sample_ratio = sample_ratio

    def to_dict(self) -> Dict[str, Any]:
        return {
            'segment_id': self.segment_id,
            'segment_name': self.segment_name,
            'sample_count': self.sample_count,
            'sample_ratio': self.sample_ratio,
            'train_auc': self.train_metrics.get('auc', 0),
            'val_auc': self.val_metrics.get('auc', 0),
            'oot_auc': self.oot_metrics.get('auc', 0) if self.oot_metrics else None,
            'train_ks': self.train_metrics.get('ks', 0),
            'val_ks': self.val_metrics.get('ks', 0),
            'oot_ks': self.oot_metrics.get('ks', 0) if self.oot_metrics else None,
        }

class SegmentModelTrainer:
    """
    分群模型训练器
    
    为每个分群独立训练子模型
    """
    
    def __init__(
        self,
        features: List[str] = None,
        xgb_params: Dict[str, Any] = None,
    ):
        """
        初始化分群模型训练器
        
        Args:
            features: 特征列表（None则使用所有数值特征）
            xgb_params: XGBoost 参数
        """
        self.features = features
        self.xgb_params = xgb_params or {}
        self.segment_models: Dict[int, SegmentModelResult] = {}
    
    def train_segments(
        self,
        data: pd.DataFrame,
        target: str,
        segment_labels: np.ndarray,
        segment_names: List[str],
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        oot_idx: np.ndarray = None,
    ) -> Dict[int, SegmentModelResult]:
        """为每个分群训练子模型

        Args:
            data: 完整数据
            target: 目标变量列名
            segment_labels: 分群标签
            segment_names: 分群名称列表
            train_idx: 训练集索引
            val_idx: 验证集 val 索引（用于 early stopping 与诊断）
            oot_idx: OOT索引（可选，仅用于最终评估）
        """
        # 确定特征
        if self.features:
            features = [f for f in self.features if f in data.columns]
        else:
            features = [c for c in data.columns
                       if c != target and data[c].dtype in ['int64', 'float64']]

        y = data[target].values
        unique_segments = np.unique(segment_labels)
        total_samples = len(data)

        for seg_id in unique_segments:
            seg_name = segment_names[seg_id] if seg_id < len(segment_names) else f"分群_{seg_id}"

            # 获取该分群的数据索引
            seg_mask = (segment_labels == seg_id)
            seg_train_idx = train_idx[seg_mask[train_idx]]
            seg_val_idx = val_idx[seg_mask[val_idx]]

            if len(seg_train_idx) < 50:
                print(f"  ⚠️ 分群 {seg_name} 训练样本不足({len(seg_train_idx)}), 跳过")
                continue

            # 准备数据
            X_train = data.iloc[seg_train_idx][features]
            y_train = y[seg_train_idx]
            X_val = data.iloc[seg_val_idx][features]
            y_val = y[seg_val_idx]

            # 训练模型（early stopping 严格使用 val，禁止用 oot）
            trainer = XGBTrainer(self.xgb_params)
            trainer.train(X_train, y_train, X_val, y_val)

            # 评估
            train_prob = trainer.predict_proba(X_train)
            val_prob = trainer.predict_proba(X_val)

            train_metrics = ModelEvaluator.evaluate(y_train, train_prob)
            val_metrics = ModelEvaluator.evaluate(y_val, val_prob)

            # OOT 评估
            oot_metrics = None
            if oot_idx is not None and len(oot_idx) > 0:
                seg_oot_idx = oot_idx[seg_mask[oot_idx]]
                if len(seg_oot_idx) > 0:
                    X_oot = data.iloc[seg_oot_idx][features]
                    y_oot = y[seg_oot_idx]
                    oot_prob = trainer.predict_proba(X_oot)
                    oot_metrics = ModelEvaluator.evaluate(y_oot, oot_prob)

            # 计算样本占比
            sample_count = seg_mask.sum()
            sample_ratio = sample_count / total_samples

            self.segment_models[seg_id] = SegmentModelResult(
                segment_id=seg_id,
                segment_name=seg_name,
                trainer=trainer,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                oot_metrics=oot_metrics,
                sample_count=sample_count,
                sample_ratio=sample_ratio,
            )

        return self.segment_models

class ModelMerger:
    """
    模型汇总器基类
    """
    
    def __init__(self, strategy: str = 'route'):
        self.strategy = strategy
    
    def merge_predictions(
        self,
        data: pd.DataFrame,
        segment_labels: np.ndarray,
        segment_models: Dict[int, SegmentModelResult],
        features: List[str],
    ) -> np.ndarray:
        """汇总各分群预测结果"""
        raise NotImplementedError

class RouteMerger(ModelMerger):
    """
    路由汇总器
    
    根据样本所属分群，路由到对应子模型预测
    """
    
    def __init__(self):
        super().__init__(strategy='route')
    
    def merge_predictions(
        self,
        data: pd.DataFrame,
        segment_labels: np.ndarray,
        segment_models: Dict[int, SegmentModelResult],
        features: List[str],
    ) -> np.ndarray:
        """
        路由预测
        
        每个样本根据其分群标签，使用对应子模型预测
        """
        n = len(data)
        predictions = np.zeros(n)
        
        for seg_id, model_result in segment_models.items():
            mask = (segment_labels == seg_id)
            if mask.sum() == 0:
                continue
            
            X_seg = data.loc[mask, features]
            predictions[mask] = model_result.trainer.predict_proba(X_seg)
        
        return predictions

class StackingMerger(ModelMerger):
    """
    Stacking 汇总器
    
    使用所有子模型预测结果作为特征，训练一个元模型
    """
    
    def __init__(self):
        super().__init__(strategy='stacking')
        self.meta_model: XGBTrainer = None
    
    def fit(
        self,
        data: pd.DataFrame,
        y: np.ndarray,
        segment_models: Dict[int, SegmentModelResult],
        features: List[str],
        train_idx: np.ndarray,
    ) -> 'StackingMerger':
        """
        训练元模型
        
        Args:
            data: 数据
            y: 目标变量
            segment_models: 分群子模型
            features: 特征列表
            train_idx: 训练索引
        """
        # 生成子模型预测作为元特征
        meta_features = []
        for seg_id, model_result in sorted(segment_models.items()):
            X = data.iloc[train_idx][features]
            prob = model_result.trainer.predict_proba(X)
            meta_features.append(prob)
        
        X_meta = np.column_stack(meta_features)
        y_train = y[train_idx]
        
        # 训练元模型
        self.meta_model = XGBTrainer({'max_depth': 2, 'n_estimators': 50})
        self.meta_model.train(
            pd.DataFrame(X_meta, columns=[f'seg_{i}' for i in range(len(meta_features))]),
            y_train,
        )
        
        return self
    
    def merge_predictions(
        self,
        data: pd.DataFrame,
        segment_labels: np.ndarray,
        segment_models: Dict[int, SegmentModelResult],
        features: List[str],
    ) -> np.ndarray:
        """Stacking 预测"""
        if self.meta_model is None:
            raise ValueError("请先调用 fit() 训练元模型")
        
        # 生成元特征
        meta_features = []
        for seg_id, model_result in sorted(segment_models.items()):
            prob = model_result.trainer.predict_proba(data[features])
            meta_features.append(prob)
        
        X_meta = np.column_stack(meta_features)
        return self.meta_model.predict_proba(
            pd.DataFrame(X_meta, columns=[f'seg_{i}' for i in range(len(meta_features))])
        )

def evaluate_segment_strategy(
    data: pd.DataFrame,
    target: str,
    segment_labels: np.ndarray,
    segment_models: Dict[int, SegmentModelResult],
    features: List[str],
    val_idx: np.ndarray,
    oot_idx: np.ndarray = None,
    merger: ModelMerger = None,
) -> Dict[str, Any]:
    """评估分群策略效果

    Args:
        data: 数据
        target: 目标变量
        segment_labels: 分群标签
        segment_models: 分群子模型
        features: 特征列表
        val_idx: 验证集 val 索引
        oot_idx: OOT索引
        merger: 汇总器
    """
    if merger is None:
        merger = RouteMerger()

    y = data[target].values

    # Val 集评估
    val_labels = segment_labels[val_idx]
    val_data = data.iloc[val_idx]
    val_pred = merger.merge_predictions(val_data, val_labels, segment_models, features)
    val_metrics = ModelEvaluator.evaluate(y[val_idx], val_pred)

    result = {
        'val_auc': val_metrics['auc'],
        'val_ks': val_metrics['ks'],
        'n_segments': len(segment_models),
        'segment_details': [m.to_dict() for m in segment_models.values()],
    }

    # OOT 评估
    if oot_idx is not None and len(oot_idx) > 0:
        oot_labels = segment_labels[oot_idx]
        oot_data = data.iloc[oot_idx]
        oot_pred = merger.merge_predictions(oot_data, oot_labels, segment_models, features)
        oot_metrics = ModelEvaluator.evaluate(y[oot_idx], oot_pred)
        result['oot_auc'] = oot_metrics['auc']
        result['oot_ks'] = oot_metrics['ks']

    # 计算分群覆盖率
    segment_ratios = [m.sample_ratio for m in segment_models.values()]
    result['min_segment_ratio'] = min(segment_ratios) if segment_ratios else 0
    result['max_segment_ratio'] = max(segment_ratios) if segment_ratios else 0

    # 计算分群稳定性（val 与 OOT 间分群比例的 PSI）
    if oot_idx is not None and len(oot_idx) > 0:
        val_dist = np.bincount(segment_labels[val_idx], minlength=len(segment_models)) / len(val_idx)
        oot_dist = np.bincount(segment_labels[oot_idx], minlength=len(segment_models)) / len(oot_idx)

        # 计算 PSI
        psi = 0
        for i in range(len(val_dist)):
            if val_dist[i] > 0 and oot_dist[i] > 0:
                psi += (val_dist[i] - oot_dist[i]) * np.log(val_dist[i] / oot_dist[i])
        result['segment_psi'] = psi

    return result
