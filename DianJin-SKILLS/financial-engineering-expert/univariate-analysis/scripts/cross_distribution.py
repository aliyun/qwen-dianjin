"""
交叉分布模块 - 计算两个特征的联合分布
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

class CrossDistributionCalculator:
    """交叉分布计算器"""
    
    def __init__(self, n_bins: int = 5, method: str = 'quantile'):
        """
        初始化交叉分布计算器
        
        Args:
            n_bins: 每个特征的分箱数量，默认5（交叉后共25格）
            method: 分箱方式，'quantile'(等频) 或 'distance'(等距)
        """
        self.n_bins = n_bins
        self.method = method
    
    def calculate(
        self, 
        df: pd.DataFrame, 
        feature1: str, 
        feature2: str,
        target: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        计算两个特征的交叉分布
        
        Args:
            df: 数据框
            feature1: 第一个特征（行）
            feature2: 第二个特征（列）
            target: 目标变量列名（可选，用于计算各格正样本率）
            
        Returns:
            交叉分布结果字典
        """
        result = {
            'feature1': feature1,
            'feature2': feature2,
            'cross_table': None,
            'cross_table_ratio': None,
            'status': 'success'
        }
        
        if feature1 not in df.columns or feature2 not in df.columns:
            result['status'] = 'feature_not_found'
            return result
        
        series1 = df[feature1]
        series2 = df[feature2]
        
        # 对两个特征分别分箱
        bins1, labels1 = self._create_bins(series1, feature1)
        bins2, labels2 = self._create_bins(series2, feature2)
        
        # 生成分箱后的列
        df_temp = df.copy()
        df_temp['_bin1'] = pd.cut(series1, bins=bins1, labels=labels1, include_lowest=True)
        df_temp['_bin2'] = pd.cut(series2, bins=bins2, labels=labels2, include_lowest=True)
        
        # 处理缺失值
        df_temp['_bin1'] = df_temp['_bin1'].cat.add_categories(['Missing']).fillna('Missing')
        df_temp['_bin2'] = df_temp['_bin2'].cat.add_categories(['Missing']).fillna('Missing')
        
        # 生成交叉表（计数）
        cross_count = pd.crosstab(
            df_temp['_bin1'], 
            df_temp['_bin2'], 
            margins=True, 
            margins_name='合计'
        )
        
        # 生成交叉表（占比）
        total = len(df)
        cross_ratio = cross_count.apply(lambda x: x / total).map(lambda x: '{:.2%}'.format(x))
        
        result['cross_table'] = cross_count
        result['cross_table_ratio'] = cross_ratio
        
        # 如果有目标变量，计算各格正样本率
        if target and target in df.columns:
            cross_target = pd.crosstab(
                df_temp['_bin1'], 
                df_temp['_bin2'], 
                values=df[target],
                aggfunc='mean'
            ).map(lambda x: '{:.2%}'.format(x) if pd.notna(x) else '-')
            result['cross_table_target_rate'] = cross_target
        
        return result
    
    def _create_bins(self, series: pd.Series, feature_name: str) -> Tuple[List[float], List[str]]:
        """
        为单个特征创建分箱边界和标签
        
        Args:
            series: 特征数据
            feature_name: 特征名称
            
        Returns:
            (分箱边界列表, 分箱标签列表)
        """
        valid_series = series.dropna()
        
        if len(valid_series) == 0:
            return [0, 1], ['[0, 1]']
        
        # 根据方法计算分箱边界
        if self.method == 'quantile':
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            bin_edges = valid_series.quantile(quantiles).values
            bin_edges = np.unique(bin_edges)
        else:
            min_val = valid_series.min()
            max_val = valid_series.max()
            if min_val == max_val:
                bin_edges = [min_val - 0.5, max_val + 0.5]
            else:
                bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
        
        # 扩展边界以包含所有值
        bin_edges = list(bin_edges)
        bin_edges[0] = bin_edges[0] - 0.001
        bin_edges[-1] = bin_edges[-1] + 0.001
        
        # 生成标签
        labels = []
        for i in range(len(bin_edges) - 1):
            left = bin_edges[i] + 0.001 if i == 0 else bin_edges[i]
            right = bin_edges[i + 1] - 0.001 if i == len(bin_edges) - 2 else bin_edges[i + 1]
            labels.append('[{:.2f}, {:.2f})'.format(left, right))
        
        return bin_edges, labels

def format_cross_table_markdown(
    result: Dict[str, Any],
    include_ratio: bool = True,
    include_target_rate: bool = False
) -> str:
    """
    将交叉分布结果格式化为 Markdown 表格
    
    Args:
        result: calculate() 方法返回的结果字典
        include_ratio: 是否包含占比表
        include_target_rate: 是否包含正样本率表
        
    Returns:
        Markdown 格式的表格字符串
    """
    if result['status'] != 'success':
        return '交叉分布计算失败: {}'.format(result['status'])
    
    feature1 = result['feature1']
    feature2 = result['feature2']
    
    sections = []
    
    # 计数表
    sections.append('**{} × {} 交叉分布（计数）**\n'.format(feature1, feature2))
    sections.append(result['cross_table'].to_markdown())
    
    # 占比表
    if include_ratio and result.get('cross_table_ratio') is not None:
        sections.append('\n\n**{} × {} 交叉分布（占比）**\n'.format(feature1, feature2))
        sections.append(result['cross_table_ratio'].to_markdown())
    
    # 正样本率表
    if include_target_rate and result.get('cross_table_target_rate') is not None:
        sections.append('\n\n**{} × {} 交叉分布（正样本率）**\n'.format(feature1, feature2))
        sections.append(result['cross_table_target_rate'].to_markdown())
    
    return '\n'.join(sections)
