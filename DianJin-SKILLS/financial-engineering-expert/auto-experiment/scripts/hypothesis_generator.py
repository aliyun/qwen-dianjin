# -*- coding: utf-8 -*-
"""
特征假设生成器模块 v2
支持特征组自动发现和组级探索策略

探索策略优先级：
  Phase 1 - 特征组独立评估：逐组测试每个特征组的独立贡献
  Phase 2 - 组间增量叠加：在最佳单组基础上逐步叠加下一个组
  Phase 3 - 组级消融分析：从全量特征中逐组移除，量化每组边际贡献
  Phase 4 - 组内精细筛选：在最优组合内，逐个移除低贡献特征

"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np

class FeatureGroupDiscovery:
    """
    自动发现数据集中的特征分组。
    
    分组策略:
    1. 前缀聚合: firefly_xxx, umeng_xxx, mob3_xxx → 按前缀分组
    2. 层级前缀: 先按第一级前缀，若某组过大则拆分到第二级
    3. 最小组大小: < 3 个特征的组合并到 'other' 组
    """
    
    MIN_GROUP_SIZE = 3
    MAX_GROUP_SIZE = 50
    
    @staticmethod
    def discover_groups(features: List[str]) -> Dict[str, List[str]]:
        """
        自动发现特征分组
        
        Returns:
            {'firefly': ['firefly_score', ...], 'mob3': [...], ...}
        """
        if not features:
            return {}
        
        # Step 1: 按第一级前缀聚合
        prefix_groups: Dict[str, List[str]] = defaultdict(list)
        for feat in features:
            prefix = FeatureGroupDiscovery._extract_prefix(feat, level=1)
            prefix_groups[prefix].append(feat)
        
        # Step 2: 大组拆分到第二级前缀
        final_groups: Dict[str, List[str]] = {}
        for prefix, group_feats in prefix_groups.items():
            if len(group_feats) > FeatureGroupDiscovery.MAX_GROUP_SIZE:
                # 尝试按二级前缀拆分
                sub_groups: Dict[str, List[str]] = defaultdict(list)
                for feat in group_feats:
                    sub_prefix = FeatureGroupDiscovery._extract_prefix(feat, level=2)
                    sub_groups[sub_prefix].append(feat)
                for sp, sf in sub_groups.items():
                    final_groups[sp] = sf
            else:
                final_groups[prefix] = group_feats
        
        # Step 3: 小组合并到 'other'
        other = []
        cleaned = {}
        for gname, gfeats in final_groups.items():
            if len(gfeats) < FeatureGroupDiscovery.MIN_GROUP_SIZE:
                other.extend(gfeats)
            else:
                cleaned[gname] = sorted(gfeats)
        
        if other:
            cleaned['other'] = sorted(other)
        
        # 按组大小排序（大组优先探索）
        return dict(sorted(cleaned.items(), key=lambda x: len(x[1]), reverse=True))
    
    @staticmethod
    def _extract_prefix(feat: str, level: int = 1) -> str:
        """提取特征名的前缀"""
        parts = feat.split('_')
        if level == 1:
            return parts[0].lower() if parts else feat.lower()
        elif level == 2:
            return '_'.join(parts[:2]).lower() if len(parts) >= 2 else parts[0].lower()
        return feat.lower()
    
    @staticmethod
    def format_group_summary(groups: Dict[str, List[str]]) -> str:
        """格式化分组概览"""
        lines = ["| 特征组 | 特征数 | 示例特征 |", "|--------|--------|----------|"]
        for gname, gfeats in groups.items():
            examples = ', '.join(gfeats[:3])
            if len(gfeats) > 3:
                examples += f' ... (+{len(gfeats) - 3})'
            lines.append(f"| `{gname}` | {len(gfeats)} | {examples} |")
        return '\n'.join(lines)

class HypothesisGenerator:
    """
    特征假设生成器 v2
    
    支持四阶段渐进式探索:
      Phase 1: 特征组独立评估 (每组单独建模)
      Phase 2: 组间增量叠加 (最优组+次优组+...)
      Phase 3: 组级消融分析 (全量-某组)
      Phase 4: 组内精细筛选 (移除低贡献特征)
    """
    
    def __init__(
        self,
        available_features: List[str],
        current_features: List[str],
        exploration_direction: str,
        tried_hypotheses: List[str] = None,
    ):
        self.available_features = list(available_features)
        self.current_features = list(current_features)
        self.exploration_direction = exploration_direction
        self.tried_hypotheses: Set[str] = set(tried_hypotheses or [])
        
        # 自动发现特征分组
        self.feature_groups = FeatureGroupDiscovery.discover_groups(self.available_features)
        
        # 探索阶段状态
        self._phase = 1
        self._group_names = list(self.feature_groups.keys())
        self._group_iter_idx = 0
        
        # Phase 2 状态: 按组贡献排序后逐步叠加
        self._group_rankings: List[Tuple[str, float]] = []  # (group_name, metric)
        self._incremental_idx = 0
        self._incremental_base: List[str] = []
        
        # Phase 3 状态: 消融
        self._ablation_idx = 0
        
        # Phase 4 状态: 精细筛选
        self._refinement_features: List[str] = []
        self._refinement_idx = 0
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """返回发现的特征分组"""
        return self.feature_groups
    
    def record_group_result(self, group_name: str, metric: float) -> None:
        """记录某个特征组的独立评估结果，用于后续排序"""
        self._group_rankings.append((group_name, metric))
    
    def generate_next_hypothesis(self) -> Optional[Dict[str, Any]]:
        """
        生成下一个实验假设（自动推进阶段）
        """
        # Phase 1: 特征组独立评估
        if self._phase == 1:
            hyp = self._phase1_group_isolation()
            if hyp:
                return hyp
            # Phase 1 结束，准备 Phase 2
            self._prepare_phase2()
            self._phase = 2
        
        # Phase 2: 组间增量叠加
        if self._phase == 2:
            hyp = self._phase2_incremental_addition()
            if hyp:
                return hyp
            # Phase 2 结束，准备 Phase 3
            self._phase = 3
            self._ablation_idx = 0
        
        # Phase 3: 组级消融
        if self._phase == 3:
            hyp = self._phase3_group_ablation()
            if hyp:
                return hyp
            self._phase = 4
        
        # Phase 4: 组内精细筛选（可选，基于当前特征集）
        if self._phase == 4:
            hyp = self._phase4_feature_refinement()
            if hyp:
                return hyp
        
        return None
    
    # ──── Phase 1: 特征组独立评估 ────
    
    def _phase1_group_isolation(self) -> Optional[Dict[str, Any]]:
        """Phase 1: 逐组测试每个特征组的独立贡献"""
        while self._group_iter_idx < len(self._group_names):
            gname = self._group_names[self._group_iter_idx]
            self._group_iter_idx += 1
            
            gfeats = self.feature_groups[gname]
            hyp_key = f"phase1_group_{gname}"
            if hyp_key in self.tried_hypotheses:
                continue
            self.tried_hypotheses.add(hyp_key)
            
            n_total_groups = len(self._group_names)
            tested = self._group_iter_idx
            
            reasoning = (
                f"[Phase 1/{tested}/{n_total_groups}] 特征组独立评估\n"
                f"目标: 单独测试「{gname}」组的 {len(gfeats)} 个特征，"
                f"评估该组对目标变量的独立预测能力\n"
                f"逻辑: 先让每个特征组独立'上场'，获得各组的独立贡献排名，"
                f"避免组间相互干扰导致贡献被掩盖\n"
                f"探索方向: {self.exploration_direction}"
            )
            
            return {
                'type': 'group_isolation',
                'group_name': gname,
                'description': f'独立评估特征组「{gname}」({len(gfeats)}个特征)',
                'reasoning': reasoning,
                'strategy': f'Phase 1: 特征组独立评估 [{tested}/{n_total_groups}]',
                'features': gfeats,
                'added': gfeats,
                'removed': [],
            }
        return None
    
    # ──── Phase 2: 组间增量叠加 ────
    
    def _prepare_phase2(self) -> None:
        """根据 Phase 1 结果排序，准备增量叠加"""
        # 按指标降序排列
        self._group_rankings.sort(key=lambda x: x[1], reverse=True)
        self._incremental_idx = 0
        self._incremental_base = []
    
    def _phase2_incremental_addition(self) -> Optional[Dict[str, Any]]:
        """Phase 2: 从最强组开始，逐步叠加下一个组"""
        if not self._group_rankings:
            return None
        
        while self._incremental_idx < len(self._group_rankings):
            gname, prev_score = self._group_rankings[self._incremental_idx]
            self._incremental_idx += 1
            
            hyp_key = f"phase2_incr_{self._incremental_idx}"
            if hyp_key in self.tried_hypotheses:
                continue
            self.tried_hypotheses.add(hyp_key)
            
            gfeats = self.feature_groups.get(gname, [])
            new_features = self._incremental_base + gfeats
            
            step = self._incremental_idx
            total = len(self._group_rankings)
            
            if step == 1:
                reasoning = (
                    f"[Phase 2/{step}/{total}] 组间增量叠加\n"
                    f"起点: 选择 Phase 1 中表现最佳的组「{gname}」"
                    f"(独立评估指标 {prev_score:.4f}) 作为基座\n"
                    f"逻辑: 以最强单组为起点，确保基座本身就有良好预测力"
                )
            else:
                prev_groups = [g for g, _ in self._group_rankings[:step-1]]
                reasoning = (
                    f"[Phase 2/{step}/{total}] 组间增量叠加\n"
                    f"当前基座: {' + '.join(prev_groups)} ({len(self._incremental_base)}个特征)\n"
                    f"本轮叠加: 「{gname}」组 ({len(gfeats)}个特征, "
                    f"独立贡献排名第{step})\n"
                    f"逻辑: 在已有组合上叠加下一个最强组，观察边际增量是否显著"
                )
            
            # 更新累计基座（无论是否 KEEP，phase2 都要叠加来测试）
            self._incremental_base = new_features[:]
            
            return {
                'type': 'group_incremental',
                'group_name': gname,
                'description': f'在基座上叠加特征组「{gname}」(+{len(gfeats)}个)',
                'reasoning': reasoning,
                'strategy': f'Phase 2: 组间增量叠加 [{step}/{total}]',
                'features': new_features,
                'added': gfeats,
                'removed': [],
            }
        return None
    
    # ──── Phase 3: 组级消融 ────
    
    def _phase3_group_ablation(self) -> Optional[Dict[str, Any]]:
        """Phase 3: 从全量特征中逐组移除，量化边际贡献"""
        if not self._group_rankings:
            return None
        
        all_features = self.available_features[:]
        
        while self._ablation_idx < len(self._group_rankings):
            gname, _ = self._group_rankings[self._ablation_idx]
            self._ablation_idx += 1
            
            hyp_key = f"phase3_ablation_{gname}"
            if hyp_key in self.tried_hypotheses:
                continue
            self.tried_hypotheses.add(hyp_key)
            
            gfeats_set = set(self.feature_groups.get(gname, []))
            remaining = [f for f in all_features if f not in gfeats_set]
            
            if not remaining:
                continue
            
            step = self._ablation_idx
            total = len(self._group_rankings)
            
            reasoning = (
                f"[Phase 3/{step}/{total}] 组级消融分析\n"
                f"全量特征: {len(all_features)}个\n"
                f"移除组: 「{gname}」({len(gfeats_set)}个特征)\n"
                f"剩余特征: {len(remaining)}个\n"
                f"逻辑: 如果移除后指标大幅下降，说明该组不可替代；"
                f"若下降很小，说明其信息已被其他组覆盖"
            )
            
            return {
                'type': 'group_ablation',
                'group_name': gname,
                'description': f'消融分析: 移除特征组「{gname}」(-{len(gfeats_set)}个)',
                'reasoning': reasoning,
                'strategy': f'Phase 3: 组级消融分析 [{step}/{total}]',
                'features': remaining,
                'added': [],
                'removed': list(gfeats_set),
            }
        return None
    
    # ──── Phase 4: 组内精细筛选 ────
    
    def _phase4_feature_refinement(self) -> Optional[Dict[str, Any]]:
        """Phase 4: 在当前最优组合内，逐个尝试移除贡献最低的特征"""
        if not self._refinement_features:
            # 用当前特征初始化
            self._refinement_features = self.current_features[:]
            self._refinement_idx = 0
        
        if len(self._refinement_features) <= 3:
            return None  # 特征太少不再精简
        
        while self._refinement_idx < len(self._refinement_features):
            feat = self._refinement_features[self._refinement_idx]
            self._refinement_idx += 1
            
            hyp_key = f"phase4_remove_{feat}"
            if hyp_key in self.tried_hypotheses:
                continue
            self.tried_hypotheses.add(hyp_key)
            
            remaining = [f for f in self._refinement_features if f != feat]
            meaning = self._infer_feature_meaning(feat)
            
            reasoning = (
                f"[Phase 4] 组内精细筛选\n"
                f"当前特征数: {len(self._refinement_features)}\n"
                f"尝试移除: {feat} ({meaning})\n"
                f"逻辑: 逐个尝试移除，若指标不降或上升，"
                f"说明该特征冗余可剔除，从而精简模型"
            )
            
            return {
                'type': 'feature_removal',
                'description': f'精细筛选: 尝试移除「{feat}」({meaning})',
                'reasoning': reasoning,
                'strategy': 'Phase 4: 组内精细筛选',
                'features': remaining,
                'added': [],
                'removed': [feat],
            }
        return None
    
    # ──── 工具方法 ────
    
    def _infer_feature_meaning(self, feat: str) -> str:
        """推断特征含义"""
        parts = feat.lower().split('_')
        meaning_map = {
            'mob': '月份距今', 'overdue': '逾期', 'due': '到期', 'amt': '金额',
            'cnt': '次数', 'ratio': '比率', 'rate': '费率', 'bal': '余额',
            'avg': '均值', 'max': '最大值', 'min': '最小值', 'sum': '累计',
            'std': '标准差', 'credit': '信用', 'repay': '还款', 'pay': '支付',
            'loan': '贷款', 'limit': '额度', 'util': '利用率', 'days': '天数',
            'total': '合计', 'curr': '当前', 'hist': '历史', 'recent': '近期',
            'firefly': 'Firefly模型', 'umeng': '友盟', 'score': '评分',
            'model': '模型', 'sub': '子模型', 'pred': '预测',
        }
        desc_parts = []
        for p in parts:
            prefix = ''.join(c for c in p if c.isalpha())
            num = ''.join(c for c in p if c.isdigit())
            if prefix in meaning_map:
                label = meaning_map[prefix]
                if num:
                    label += num
                desc_parts.append(label)
            elif p:
                desc_parts.append(p)
        return '_'.join(desc_parts) if desc_parts else feat
    
    def update_current_features(self, new_features: List[str]) -> None:
        """更新当前特征集"""
        self.current_features = list(new_features)
        # Phase 4 需要用最新特征
        if self._phase == 4:
            self._refinement_features = list(new_features)
    
    def get_remaining_candidates(self) -> List[str]:
        """获取尚未探索的特征组"""
        remaining = []
        for gname in self._group_names:
            if f"phase1_group_{gname}" not in self.tried_hypotheses:
                remaining.append(gname)
        return remaining
    
    def get_exploration_summary(self) -> str:
        """获取探索进度摘要"""
        total_groups = len(self._group_names)
        tested = len([g for g in self._group_names
                       if f"phase1_group_{g}" in self.tried_hypotheses])
        return (
            f"探索进度:\n"
            f"- 当前阶段: Phase {self._phase}\n"
            f"- 特征组总数: {total_groups}\n"
            f"- 已评估组数: {tested}\n"
            f"- 当前特征数: {len(self.current_features)}\n"
        )

def create_cross_feature(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    method: str = 'multiply'
) -> pd.Series:
    """创建交叉特征"""
    if method == 'multiply':
        return df[col1] * df[col2]
    elif method == 'divide':
        return df[col1] / (df[col2] + 1e-10)
    elif method == 'add':
        return df[col1] + df[col2]
    elif method == 'subtract':
        return df[col1] - df[col2]
    else:
        raise ValueError(f"不支持的交叉方法: {method}")
