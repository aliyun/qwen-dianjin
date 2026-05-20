"""
分箱模块 - 支持等频分箱和等距分箱
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

class BinningCalculator:
    """分箱计算器"""
    
    def __init__(self, n_bins: int = 10, method: str = 'quantile'):
        """
        初始化分箱计算器
        
        Args:
            n_bins: 分箱数量，默认10
            method: 分箱方式，'quantile'(等频) 或 'distance'(等距)
        """
        self.n_bins = n_bins
        self.method = method
    
    def calculate_distribution(self, series: pd.Series, feature_name: str) -> Dict[str, Any]:
        """
        计算单特征的分箱分布
        
        Args:
            series: 特征数据
            feature_name: 特征名称
            
        Returns:
            分布信息字典，包含分箱表和统计信息
        """
        result = {
            'feature': feature_name,
            'method': 'quantile' if self.method == 'quantile' else 'distance',
            'n_bins': self.n_bins,
            'distribution_table': None,
            'special_values': {}
        }
        
        # 处理特殊值
        total_count = len(series)
        missing_mask = series.isna()
        missing_count = missing_mask.sum()
        
        # 非缺失数据
        valid_series = series[~missing_mask]
        
        if len(valid_series) == 0:
            result['distribution_table'] = pd.DataFrame({
                'bin': ['Missing'],
                'count': [missing_count],
                'ratio': ['100.00%'],
                'cumulative_ratio': ['100.00%']
            })
            return result
        
        # 检查是否为数值类型
        if not pd.api.types.is_numeric_dtype(valid_series):
            # 非数值类型：按值频次统计
            return self._calculate_categorical_distribution(series, feature_name, total_count, missing_count)
        
        # 零值处理
        zero_mask = valid_series == 0
        zero_count = zero_mask.sum()
        zero_ratio = zero_count / total_count
        
        # 负数处理
        negative_mask = valid_series < 0
        negative_count = negative_mask.sum()
        
        # 正数数据用于分箱
        positive_mask = valid_series > 0
        positive_series = valid_series[positive_mask]
        
        # 构建分箱表
        bins_data = []
        cumulative_count = 0
        
        # 1. 缺失值箱
        if missing_count > 0:
            cumulative_count += missing_count
            bins_data.append({
                'bin': 'Missing',
                'count': missing_count,
                'ratio': missing_count / total_count,
                'cumulative_ratio': cumulative_count / total_count
            })
        
        # 2. 负数箱
        if negative_count > 0:
            cumulative_count += negative_count
            neg_min = valid_series[negative_mask].min()
            bins_data.append({
                'bin': '(-inf, 0)',
                'count': negative_count,
                'ratio': negative_count / total_count,
                'cumulative_ratio': cumulative_count / total_count
            })
        
        # 3. 零值箱（当占比 > 5% 时单独分箱）
        if zero_ratio > 0.05:
            cumulative_count += zero_count
            bins_data.append({
                'bin': '[0, 0]',
                'count': zero_count,
                'ratio': zero_count / total_count,
                'cumulative_ratio': cumulative_count / total_count
            })
            result['special_values']['zero_separate'] = True
        else:
            # 零值并入正数处理
            positive_series = valid_series[valid_series >= 0]
            result['special_values']['zero_separate'] = False
        
        # 4. 正数分箱
        if len(positive_series) > 0:
            if self.method == 'quantile':
                bin_edges = self._quantile_bins(positive_series)
            else:
                bin_edges = self._distance_bins(positive_series)
            
            # 按分箱边界统计
            for i in range(len(bin_edges) - 1):
                left = bin_edges[i]
                right = bin_edges[i + 1]
                
                if i == len(bin_edges) - 2:
                    # 最后一箱包含右边界
                    mask = (positive_series >= left) & (positive_series <= right)
                    bin_label = '[{:.2f}, {:.2f}]'.format(left, right)
                else:
                    mask = (positive_series >= left) & (positive_series < right)
                    bin_label = '[{:.2f}, {:.2f})'.format(left, right)
                
                bin_count = mask.sum()
                if bin_count > 0:
                    cumulative_count += bin_count
                    bins_data.append({
                        'bin': bin_label,
                        'count': bin_count,
                        'ratio': bin_count / total_count,
                        'cumulative_ratio': cumulative_count / total_count
                    })
        
        # 转换为 DataFrame
        df = pd.DataFrame(bins_data)
        df['ratio'] = df['ratio'].apply(lambda x: '{:.2%}'.format(x))
        df['cumulative_ratio'] = df['cumulative_ratio'].apply(lambda x: '{:.2%}'.format(x))
        
        result['distribution_table'] = df
        result['special_values']['missing_count'] = missing_count
        result['special_values']['negative_count'] = negative_count
        result['special_values']['zero_count'] = zero_count
        
        return result
    
    def _quantile_bins(self, series: pd.Series) -> List[float]:
        """
        计算等频分箱边界
        
        Args:
            series: 数据序列（仅正数）
            
        Returns:
            分箱边界列表
        """
        # 计算分位数
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        bin_edges = series.quantile(quantiles).values
        
        # 去重（处理分位数重复的情况）
        bin_edges = np.unique(bin_edges)
        
        return bin_edges.tolist()
    
    def _distance_bins(self, series: pd.Series) -> List[float]:
        """
        计算等距分箱边界
        
        Args:
            series: 数据序列（仅正数）
            
        Returns:
            分箱边界列表
        """
        min_val = series.min()
        max_val = series.max()
        
        if min_val == max_val:
            return [min_val, max_val]
        
        bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
        return bin_edges.tolist()
    
    def _calculate_categorical_distribution(
        self, 
        series: pd.Series, 
        feature_name: str,
        total_count: int,
        missing_count: int
    ) -> Dict[str, Any]:
        """
        计算分类特征的分布
        
        Args:
            series: 特征数据
            feature_name: 特征名称
            total_count: 总样本数
            missing_count: 缺失数
            
        Returns:
            分布信息字典
        """
        result = {
            'feature': feature_name,
            'method': 'categorical',
            'n_bins': None,
            'distribution_table': None,
            'special_values': {'missing_count': missing_count}
        }
        
        bins_data = []
        cumulative_count = 0
        
        # 缺失值箱
        if missing_count > 0:
            cumulative_count += missing_count
            bins_data.append({
                'bin': 'Missing',
                'count': missing_count,
                'ratio': missing_count / total_count,
                'cumulative_ratio': cumulative_count / total_count
            })
        
        # 按频次统计（取前 n_bins 个）
        valid_series = series.dropna()
        value_counts = valid_series.value_counts().head(self.n_bins)
        
        for value, count in value_counts.items():
            cumulative_count += count
            bins_data.append({
                'bin': str(value),
                'count': count,
                'ratio': count / total_count,
                'cumulative_ratio': cumulative_count / total_count
            })
        
        # 剩余的合并为 "Other"
        other_count = len(valid_series) - value_counts.sum()
        if other_count > 0:
            cumulative_count += other_count
            bins_data.append({
                'bin': 'Other',
                'count': other_count,
                'ratio': other_count / total_count,
                'cumulative_ratio': cumulative_count / total_count
            })
        
        # 转换为 DataFrame
        df = pd.DataFrame(bins_data)
        df['ratio'] = df['ratio'].apply(lambda x: '{:.2%}'.format(x))
        df['cumulative_ratio'] = df['cumulative_ratio'].apply(lambda x: '{:.2%}'.format(x))
        
        result['distribution_table'] = df
        return result
    
    def calculate_iv_distribution(
        self, 
        series: pd.Series, 
        target: pd.Series, 
        feature_name: str
    ) -> Dict[str, Any]:
        """
        计算带目标变量的 IV 分箱分布
        
        Args:
            series: 特征数据
            target: 目标变量（0/1）
            feature_name: 特征名称
            
        Returns:
            IV 分布信息字典
        """
        result = {
            'feature': feature_name,
            'iv_value': 0.0,
            'iv_table': None,
            'positive_rates': [],  # 原始正样本率数值，用于前端图表
            'status': 'success'
        }
        
        try:
            # 使用 optbinning 进行最优分箱
            from optbinning import OptimalBinning
            
            # 准备数据
            x = series.values
            y = target.values
            
            # 过滤缺失值和非0/1目标变量
            valid_mask = ~pd.isna(x) & ~pd.isna(y) & ((y == 0) | (y == 1))
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            
            # 过滤非数值类型特征
            if not pd.api.types.is_numeric_dtype(series):
                result['status'] = 'non_numeric_feature'
                return result
            
            # 转换为数值数组
            x_valid = x_valid.astype(float)
            y_valid = y_valid.astype(int)
            
            if len(x_valid) < 100:
                result['status'] = 'insufficient_data'
                return result
            
            # 最优分箱
            optb = OptimalBinning(
                name=feature_name,
                dtype='numerical',
                max_n_bins=self.n_bins,
                min_bin_size=0.05
            )
            optb.fit(x_valid, y_valid)
            
            # 获取分箱表
            binning_table = optb.binning_table.build()
            
            # 处理分箱表格式
            iv_table = binning_table[['Bin', 'Count', 'Count (%)', 'Event', 'Event rate', 'WoE', 'IV']].copy()
            iv_table.columns = ['bin', 'count', 'ratio', 'positive', 'positive_rate', 'woe', 'iv']
            
            # 格式化
            def safe_format_percent(x, divisor=1):
                try:
                    if pd.isna(x) or x == '':
                        return '-'
                    return '{:.2%}'.format(float(x) / divisor)
                except (ValueError, TypeError):
                    return '-'
            
            def safe_format_float(x, decimals=4):
                try:
                    if pd.isna(x) or x == '':
                        return '-'
                    return '{:.{}f}'.format(float(x), decimals)
                except (ValueError, TypeError):
                    return '-'
            
            # 保存原始正样本率数值（用于前端 artifact）
            result['positive_rates'] = binning_table['Event rate'].fillna(0).tolist()
            
            iv_table['ratio'] = iv_table['ratio'].apply(lambda x: safe_format_percent(x, 100))
            iv_table['positive_rate'] = iv_table['positive_rate'].apply(safe_format_percent)
            iv_table['woe'] = iv_table['woe'].apply(safe_format_float)
            iv_table['iv'] = iv_table['iv'].apply(safe_format_float)
            
            result['iv_value'] = optb.binning_table.iv
            result['iv_table'] = iv_table
            
        except ImportError:
            result['status'] = 'optbinning_not_installed'
        except Exception as e:
            result['status'] = 'error'
            result['error_message'] = str(e)
        
        return result

def calculate_basic_statistics(series: pd.Series, feature_name: str) -> Dict[str, Any]:
    """
    计算基础统计信息
    
    Args:
        series: 特征数据
        feature_name: 特征名称
        
    Returns:
        统计信息字典
    """
    total_count = len(series)
    missing_count = series.isna().sum()
    valid_series = series.dropna()
    
    stats = {
        'feature': feature_name,
        'count': total_count,
        'missing_count': missing_count,
        'missing_rate': '{:.2%}'.format(missing_count / total_count) if total_count > 0 else '-',
        'cardinality': valid_series.nunique(),
    }
    
    if len(valid_series) > 0 and pd.api.types.is_numeric_dtype(valid_series):
        stats.update({
            'mean': '{:.4f}'.format(valid_series.mean()),
            'median': '{:.4f}'.format(valid_series.median()),
            'std': '{:.4f}'.format(valid_series.std()),
            'min': '{:.4f}'.format(valid_series.min()),
            'max': '{:.4f}'.format(valid_series.max()),
            'skewness': '{:.4f}'.format(valid_series.skew()),
            'kurtosis': '{:.4f}'.format(valid_series.kurtosis()),
        })
    else:
        stats.update({
            'mean': '-',
            'median': '-',
            'std': '-',
            'min': '-',
            'max': '-',
            'skewness': '-',
            'kurtosis': '-',
        })
    
    return stats
