# -*- coding: utf-8 -*-
"""
分群策略模块
支持三种分群方式：规则分群、聚类分群、决策树分群

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

class BaseSegmenter(ABC):
    """分群策略基类"""
    
    def __init__(self, name: str = "base"):
        self.name = name
        self.segment_labels: np.ndarray = None
        self.segment_names: List[str] = []
        self.segment_rules: Dict[str, str] = {}  # 用于解释
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray = None) -> 'BaseSegmenter':
        """拟合分群策略"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测分群标签"""
        pass
    
    def fit_predict(self, X: pd.DataFrame, y: np.ndarray = None) -> np.ndarray:
        """拟合并预测"""
        self.fit(X, y)
        return self.predict(X)
    
    def get_segment_info(self) -> Dict[str, Any]:
        """获取分群信息"""
        return {
            'name': self.name,
            'n_segments': len(self.segment_names),
            'segment_names': self.segment_names,
            'segment_rules': self.segment_rules,
        }
    
    def get_description(self) -> str:
        """获取分群策略描述"""
        return f"{self.name}: {len(self.segment_names)} 群"

class RuleSegmenter(BaseSegmenter):
    """
    规则分群器
    
    使用用户指定的规则（pandas query 语法）拆分客群
    """
    
    def __init__(self, rules: Dict[str, str]):
        """
        初始化规则分群器
        
        Args:
            rules: 分群规则字典 {"群名": "pandas query 条件"}
                   例如: {"年轻": "age < 30", "中年": "age >= 30 and age < 50"}
        """
        super().__init__(name="规则分群")
        self.rules = rules
        self.segment_names = list(rules.keys())
        self.segment_rules = rules.copy()
    
    def fit(self, X: pd.DataFrame, y: np.ndarray = None) -> 'RuleSegmenter':
        """规则分群不需要拟合"""
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """应用规则分群"""
        labels = np.full(len(X), -1)  # 默认-1表示未分配
        
        for idx, (seg_name, rule) in enumerate(self.rules.items()):
            try:
                mask = X.eval(rule)
                labels[mask] = idx
            except Exception as e:
                print(f"规则 '{rule}' 执行失败: {e}")
        
        # 未分配的样本归入 "其他" 类别
        if (labels == -1).any():
            self.segment_names = list(self.rules.keys()) + ["其他"]
            labels[labels == -1] = len(self.rules)
        
        self.segment_labels = labels
        return labels
    
    def get_description(self) -> str:
        rules_str = " | ".join(self.rules.values())
        return f"规则分群: {rules_str}"

class ClusterSegmenter(BaseSegmenter):
    """
    聚类分群器
    
    使用 K-Means 等聚类算法自动发现客群
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        features: List[str] = None,
        algorithm: str = 'kmeans',
        random_state: int = 42,
    ):
        """
        初始化聚类分群器
        
        Args:
            n_clusters: 聚类数
            features: 用于聚类的特征列表（None则使用所有数值特征）
            algorithm: 聚类算法 ('kmeans')
            random_state: 随机种子
        """
        super().__init__(name=f"聚类分群(K={n_clusters})")
        self.n_clusters = n_clusters
        self.features = features
        self.algorithm = algorithm
        self.random_state = random_state
        self.model = None
        self.feature_names_: List[str] = []
        self.cluster_centers_: np.ndarray = None
    
    def fit(self, X: pd.DataFrame, y: np.ndarray = None) -> 'ClusterSegmenter':
        """拟合聚类模型"""
        # 选择特征
        if self.features:
            feature_cols = [f for f in self.features if f in X.columns]
        else:
            feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        self.feature_names_ = feature_cols
        X_cluster = X[feature_cols].fillna(X[feature_cols].median())
        
        # 标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        self.scaler_ = scaler
        
        # 聚类
        if self.algorithm == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
            self.model.fit(X_scaled)
            self.cluster_centers_ = self.model.cluster_centers_
        
        # 生成分群名称
        self.segment_names = [f"聚类_{i+1}" for i in range(self.n_clusters)]
        
        # 生成分群规则描述（基于聚类中心）
        self._generate_cluster_rules(X_cluster)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测聚类标签"""
        if not self.is_fitted:
            raise ValueError("请先调用 fit()")
        
        X_cluster = X[self.feature_names_].fillna(X[self.feature_names_].median())
        X_scaled = self.scaler_.transform(X_cluster)
        labels = self.model.predict(X_scaled)
        self.segment_labels = labels
        return labels
    
    def _generate_cluster_rules(self, X: pd.DataFrame) -> None:
        """根据聚类中心生成规则描述"""
        if self.cluster_centers_ is None:
            return
        
        # 找出每个聚类的主要特征
        for i in range(self.n_clusters):
            center = self.cluster_centers_[i]
            # 找出距离均值最远的特征
            deviations = []
            for j, feat in enumerate(self.feature_names_):
                feat_mean = X[feat].mean()
                feat_std = X[feat].std()
                if feat_std > 0:
                    z_score = (center[j] * feat_std + feat_mean - feat_mean) / feat_std
                    deviations.append((feat, z_score, center[j]))
            
            # 按偏离程度排序，取前2个特征描述
            deviations.sort(key=lambda x: abs(x[1]), reverse=True)
            desc_parts = []
            for feat, z, val in deviations[:2]:
                direction = "高" if z > 0 else "低"
                desc_parts.append(f"{feat}{direction}")
            
            self.segment_rules[self.segment_names[i]] = " & ".join(desc_parts) if desc_parts else "平均"
    
    def get_description(self) -> str:
        return f"K-Means聚类(K={self.n_clusters})"

class TreeSegmenter(BaseSegmenter):
    """
    决策树分群器
    
    使用决策树有监督地找最优分割点
    """
    
    def __init__(
        self,
        max_depth: int = 2,
        features: List[str] = None,
        min_samples_leaf: int = 100,
        random_state: int = 42,
    ):
        """
        初始化决策树分群器
        
        Args:
            max_depth: 树深度（控制分群数，depth=2 约 4 群）
            features: 用于分割的特征（None则使用所有数值特征）
            min_samples_leaf: 叶子节点最小样本数
            random_state: 随机种子
        """
        super().__init__(name=f"决策树分群(depth={max_depth})")
        self.max_depth = max_depth
        self.features = features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model: DecisionTreeClassifier = None
        self.feature_names_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y: np.ndarray = None) -> 'TreeSegmenter':
        """拟合决策树"""
        if y is None:
            raise ValueError("决策树分群需要目标变量 y")
        
        # 选择特征
        if self.features:
            feature_cols = [f for f in self.features if f in X.columns]
        else:
            feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        self.feature_names_ = feature_cols
        X_tree = X[feature_cols].fillna(X[feature_cols].median())
        
        # 训练决策树
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.model.fit(X_tree, y)
        
        # 获取叶子节点作为分群
        leaf_ids = self.model.apply(X_tree)
        unique_leaves = np.unique(leaf_ids)
        
        # 创建叶子ID到分群索引的映射
        self.leaf_to_segment_ = {leaf: idx for idx, leaf in enumerate(unique_leaves)}
        
        # 生成分群名称和规则
        self.segment_names = [f"分群_{i+1}" for i in range(len(unique_leaves))]
        self._extract_tree_rules(X_tree)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测分群标签"""
        if not self.is_fitted:
            raise ValueError("请先调用 fit()")
        
        X_tree = X[self.feature_names_].fillna(X[self.feature_names_].median())
        leaf_ids = self.model.apply(X_tree)
        
        # 转换为分群索引
        labels = np.array([self.leaf_to_segment_.get(l, 0) for l in leaf_ids])
        self.segment_labels = labels
        return labels
    
    def _extract_tree_rules(self, X: pd.DataFrame) -> None:
        """从决策树提取分群规则"""
        tree = self.model.tree_
        feature_names = self.feature_names_
        
        def get_leaf_rules(node_id, rules=[]):
            """递归提取到达每个叶子的规则"""
            if tree.children_left[node_id] == tree.children_right[node_id]:
                # 叶子节点
                return [(node_id, rules.copy())]
            
            feature = feature_names[tree.feature[node_id]]
            threshold = tree.threshold[node_id]
            
            left_rules = rules + [f"{feature} <= {threshold:.2f}"]
            right_rules = rules + [f"{feature} > {threshold:.2f}"]
            
            left_results = get_leaf_rules(tree.children_left[node_id], left_rules)
            right_results = get_leaf_rules(tree.children_right[node_id], right_rules)
            
            return left_results + right_results
        
        leaf_rules = get_leaf_rules(0)
        
        for leaf_id, rules in leaf_rules:
            if leaf_id in self.leaf_to_segment_:
                seg_idx = self.leaf_to_segment_[leaf_id]
                seg_name = self.segment_names[seg_idx]
                self.segment_rules[seg_name] = " & ".join(rules) if rules else "默认"
    
    def get_description(self) -> str:
        n_segments = len(self.segment_names) if self.segment_names else "?"
        return f"决策树分群(depth={self.max_depth}, {n_segments}群)"

class ColumnSegmenter(BaseSegmenter):
    """
    列分群器
    
    直接使用数据中已有的分群列
    """
    
    def __init__(self, column: str):
        """
        初始化列分群器
        
        Args:
            column: 分群列名
        """
        super().__init__(name=f"列分群({column})")
        self.column = column
    
    def fit(self, X: pd.DataFrame, y: np.ndarray = None) -> 'ColumnSegmenter':
        """从列中获取分群信息"""
        if self.column not in X.columns:
            raise ValueError(f"分群列 '{self.column}' 不存在")
        
        unique_values = X[self.column].unique()
        self.value_to_idx_ = {v: i for i, v in enumerate(unique_values)}
        self.segment_names = [str(v) for v in unique_values]
        self.segment_rules = {str(v): f"{self.column} == {repr(v)}" for v in unique_values}
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """应用列分群"""
        if not self.is_fitted:
            raise ValueError("请先调用 fit()")
        
        labels = X[self.column].map(self.value_to_idx_).fillna(-1).astype(int).values
        self.segment_labels = labels
        return labels

def create_segmenter(
    strategy: str,
    **kwargs
) -> BaseSegmenter:
    """
    工厂函数：创建分群器
    
    Args:
        strategy: 策略类型 ('rule', 'cluster', 'tree', 'column')
        **kwargs: 策略特定参数
    
    Returns:
        对应的分群器实例
    """
    if strategy == 'rule':
        return RuleSegmenter(rules=kwargs.get('rules', {}))
    elif strategy == 'cluster':
        return ClusterSegmenter(
            n_clusters=kwargs.get('n_clusters', 3),
            features=kwargs.get('features'),
            random_state=kwargs.get('random_state', 42),
        )
    elif strategy == 'tree':
        return TreeSegmenter(
            max_depth=kwargs.get('max_depth', 2),
            features=kwargs.get('features'),
            min_samples_leaf=kwargs.get('min_samples_leaf', 100),
            random_state=kwargs.get('random_state', 42),
        )
    elif strategy == 'column':
        return ColumnSegmenter(column=kwargs.get('column'))
    else:
        raise ValueError(f"未知的分群策略: {strategy}")
