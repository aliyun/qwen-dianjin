# -*- coding: utf-8 -*-
"""
建模会话文档管理模块
借鉴 pi-autoresearch 的会话恢复机制

"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

class SessionTracker:
    """
    建模会话追踪器
    
    维护一个 Markdown 格式的活文档，记录：
    - 优化目标和基线
    - 当前最佳结果
    - 已验证有效的方向
    - 已验证无效的方向（避免重复）
    - 待探索方向
    
    新会话可以从该文档恢复上下文。
    """
    
    def __init__(self, session_file: str | Path):
        """
        初始化会话追踪器
        
        Args:
            session_file: 会话文档路径 (modeling_session.json)
        """
        self.session_file = Path(session_file)
        self.data: Dict[str, Any] = {
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'target_metric': 'ks',
            'direction': 'maximize',
            'baseline': {},
            'current_best': {},
            'effective': [],      # 已验证有效
            'ineffective': [],    # 已验证无效
            'pending': [],        # 待探索
            'history': [],        # 完整实验历史
        }
        
        # 如果文件存在，尝试加载
        self._load_if_exists()
    
    def _load_if_exists(self) -> None:
        """如果会话文件存在，加载数据"""
        if not self.session_file.exists():
            return
        
        # 尝试从同名的 .json 文件加载结构化数据
        json_file = self.session_file.with_suffix('.json')
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
    
    def init_session(
        self,
        exploration: str,
        metric: str = 'ks',
        direction: str = 'maximize',
        baseline_metric: float = 0.0,
        baseline_features: List[str] = None,
    ) -> None:
        """
        初始化新会话
        
        Args:
            exploration: 探索方向描述
            metric: 优化指标
            direction: 优化方向
            baseline_metric: 基线指标值
            baseline_features: 基线特征列表
        """
        self.data['exploration'] = exploration
        self.data['target_metric'] = metric
        self.data['direction'] = direction
        self.data['baseline'] = {
            'metric': baseline_metric,
            'features': baseline_features or [],
        }
        self.data['current_best'] = {
            'metric': baseline_metric,
            'features': baseline_features or [],
            'round': 0,
        }
        self.data['updated_at'] = datetime.now().isoformat()
        self._save()
    
    def log_experiment(
        self,
        round_num: int,
        hypothesis: str,
        features: List[str],
        metric_value: float,
        improvement: float,
        confidence: float,
        decision: str,  # 'KEEP' or 'DISCARD'
        reason: str = '',
    ) -> None:
        """
        记录一次实验结果
        
        Args:
            round_num: 轮次
            hypothesis: 本轮假设描述
            features: 特征列表
            metric_value: 指标值
            improvement: 相对改进
            confidence: 置信度
            decision: 决策 (KEEP/DISCARD)
            reason: 原因说明
        """
        experiment = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'hypothesis': hypothesis,
            'features': features,
            'metric': metric_value,
            'improvement': improvement,
            'confidence': confidence,
            'decision': decision,
            'reason': reason,
        }
        
        self.data['history'].append(experiment)
        
        # 更新有效/无效列表
        entry = {
            'round': round_num,
            'hypothesis': hypothesis,
            'metric': metric_value,
            'improvement': improvement,
        }
        
        if decision == 'KEEP':
            self.data['effective'].append(entry)
            # 更新当前最佳
            self.data['current_best'] = {
                'metric': metric_value,
                'features': features,
                'round': round_num,
            }
        else:
            self.data['ineffective'].append({
                **entry,
                'reason': reason,
            })
        
        self.data['updated_at'] = datetime.now().isoformat()
        self._save()
    
    def add_pending(self, suggestions: List[str]) -> None:
        """添加待探索方向"""
        self.data['pending'].extend(suggestions)
        self.data['updated_at'] = datetime.now().isoformat()
        self._save()
    
    def get_context_for_llm(self) -> str:
        """
        生成供 LLM 理解的上下文摘要
        
        Returns:
            结构化的上下文字符串
        """
        baseline = self.data.get('baseline', {})
        best = self.data.get('current_best', {})
        effective = self.data.get('effective', [])
        ineffective = self.data.get('ineffective', [])
        pending = self.data.get('pending', [])
        
        context = f"""
当前建模任务上下文:
- 探索方向: {self.data.get('exploration', '未指定')}
- 优化指标: {self.data.get('target_metric', 'auc')} ({self.data.get('direction', 'maximize')})
- 基线: {baseline.get('metric', 0):.4f}
- 当前最佳: {best.get('metric', 0):.4f} (Round {best.get('round', 0)})
- 已尝试 {len(effective) + len(ineffective)} 个方向，{len(effective)} 个有效

已验证有效的方向:
{self._format_list(effective, 'hypothesis', 'improvement')}

已验证无效的方向 (避免重复):
{self._format_list(ineffective, 'hypothesis', 'reason')}

待探索方向:
{chr(10).join(f'- {p}' for p in pending) if pending else '- 暂无'}
"""
        return context.strip()
    
    def _format_list(self, items: List[dict], key1: str, key2: str) -> str:
        """格式化列表项"""
        if not items:
            return '- 暂无'
        lines = []
        for item in items[-5:]:  # 只显示最近5条
            val2 = item.get(key2, '')
            if isinstance(val2, float):
                val2 = f'{val2:+.2%}'
            lines.append(f"- {item.get(key1, '')}: {val2}")
        return '\n'.join(lines)
    
    def _save(self) -> None:
        """保存会话数据"""
        # 确保目录存在
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 只保存 JSON 格式
        json_file = self.session_file.with_suffix('.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def get_history(self) -> List[Dict]:
        """获取完整实验历史"""
        return self.data.get('history', [])
    
    def get_current_best(self) -> Dict:
        """获取当前最佳结果"""
        return self.data.get('current_best', {})
    
    def get_baseline(self) -> Dict:
        """获取基线"""
        return self.data.get('baseline', {})
    
    def has_previous_session(self) -> bool:
        """检查是否有之前的会话"""
        return len(self.data.get('history', [])) > 0
