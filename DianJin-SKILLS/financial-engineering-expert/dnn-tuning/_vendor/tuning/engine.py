# -*- coding: utf-8 -*-
"""多算法调参引擎 — BaseTuningEngine 抽象基类。

提供两种工作形态：
  1. run_batch(n_trials)  — 一次性黑盒搜索
  2. run_round(diagnosis) — 诊断驱动的约束搜索

Optuna 缺失时自动回退到约束空间内的随机搜索。

子类实现：
  - XGBTuningEngine (xgb_engine.py)
  - LRTuningEngine (lr_engine.py)
  - DNNTuningEngine (dnn_engine.py)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from tuning.diagnose import DiagnosisResult

logger = logging.getLogger(__name__)

class BaseTuningEngine(ABC):
    """算法无关的调参引擎抽象基类。

    子类必须实现：
      - _evaluate(params) : 用给定参数训练并返回评估分数
      - _get_default_space() : 返回全量硬边界搜索空间
      - _build_constrained_space(diagnosis, tried_directions) : 构造诊断约束空间

    子类应覆写的类变量：
      - INT_PARAMS : 整数参数集合
      - LOG_PARAMS : 对数空间参数集合
      - CAT_PARAMS : 分类参数字典 {name: [choices]}
    """

    INT_PARAMS: Set[str] = set()
    LOG_PARAMS: Set[str] = set()
    CAT_PARAMS: Dict[str, List[Any]] = {}  # {param_name: [choice1, choice2, ...]}

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_eval: pd.DataFrame,
        y_eval: np.ndarray,
        base_params: Dict[str, Any],
        metric: str = "ks",
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        self.X_train = X_train
        self.y_train = np.asarray(y_train)
        self.X_eval = X_eval
        self.y_eval = np.asarray(y_eval)
        self.base_params = dict(base_params)
        self.metric = metric.lower()
        self.sample_weight = sample_weight
        self._call_count = 0

    # ------------------------------------------------------------------
    # 抽象方法（子类必须实现）
    # ------------------------------------------------------------------

    @abstractmethod
    def _evaluate(self, params: Dict[str, Any]) -> float:
        """用给定参数训练并返回评估分数（越高越好）。"""
        ...

    @abstractmethod
    def _get_default_space(self) -> Dict[str, Tuple[float, float]]:
        """返回全量硬边界搜索空间 {param: (lo, hi)}。"""
        ...

    @abstractmethod
    def _build_constrained_space(
        self,
        diagnosis: DiagnosisResult,
        tried_directions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """根据诊断结论构造约束搜索空间。"""
        ...

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def run_batch(
        self,
        n_trials: int = 30,
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """一次性贝叶斯搜索（黑盒），使用全量硬边界搜索空间。"""
        space = self._get_default_space()
        return self._optimize(space, n_trials)

    def run_round(
        self,
        diagnosis: DiagnosisResult,
        tried_directions: Optional[List[Dict[str, Any]]] = None,
        n_trials: int = 5,
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]], str]:
        """诊断驱动的单轮约束搜索。

        Returns:
            (best_params, best_score, trials, suggestion_text)
        """
        space = self._build_constrained_space(diagnosis, tried_directions)
        best_params, best_score, trials = self._optimize(space, n_trials)
        suggestion = self._format_suggestion(diagnosis, space, best_params)
        return best_params, best_score, trials, suggestion

    # ------------------------------------------------------------------
    # 核心：调用 Optuna（缺失时回退随机搜索）
    # ------------------------------------------------------------------

    def _optimize(
        self,
        space: Dict[str, Tuple[float, float]],
        n_trials: int,
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("optuna 未安装，回退到随机搜索")
            return self._random_search(space, n_trials)

        def objective(trial: "optuna.Trial") -> float:
            params = self._sample_params(trial, space)
            return self._evaluate(params)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = {**self.base_params, **study.best_params}
        metric_name = self.metric.upper()
        trials = [
            {"trial": t.number + 1, metric_name: t.value, "params": t.params}
            for t in study.trials
            if t.value is not None
        ]
        return best_params, float(study.best_value), trials

    def _sample_params(self, trial, space: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        sampled: Dict[str, Any] = {}
        for key, bounds in space.items():
            # 分类参数特殊处理
            if key in self.CAT_PARAMS:
                sampled[key] = trial.suggest_categorical(key, self.CAT_PARAMS[key])
                continue
            lo, hi = bounds
            if lo == hi:
                sampled[key] = int(lo) if key in self.INT_PARAMS else lo
                continue
            if key in self.INT_PARAMS:
                sampled[key] = trial.suggest_int(key, int(round(lo)), int(round(hi)))
            elif key in self.LOG_PARAMS:
                sampled[key] = trial.suggest_float(key, max(lo, 1e-6), hi, log=True)
            else:
                sampled[key] = trial.suggest_float(key, lo, hi)
        return {**self.base_params, **sampled}

    def _random_search(
        self,
        space: Dict[str, Tuple[float, float]],
        n_trials: int,
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        self._call_count += 1
        base_seed = int(self.base_params.get("random_state", 42))
        rng = np.random.RandomState(base_seed + self._call_count)
        best_score = -np.inf
        best_params: Optional[Dict[str, Any]] = None
        trials: List[Dict[str, Any]] = []
        metric_name = self.metric.upper()

        for i in range(n_trials):
            sampled: Dict[str, Any] = {}
            for key, bounds in space.items():
                # 分类参数
                if key in self.CAT_PARAMS:
                    sampled[key] = rng.choice(self.CAT_PARAMS[key])
                    continue
                lo, hi = bounds
                if lo == hi:
                    sampled[key] = int(lo) if key in self.INT_PARAMS else lo
                elif key in self.INT_PARAMS:
                    sampled[key] = int(rng.randint(int(round(lo)), int(round(hi)) + 1))
                elif key in self.LOG_PARAMS:
                    lo_l = np.log(max(lo, 1e-6))
                    hi_l = np.log(hi)
                    sampled[key] = float(np.exp(rng.uniform(lo_l, hi_l)))
                else:
                    sampled[key] = float(rng.uniform(lo, hi))
            params = {**self.base_params, **sampled}
            try:
                score = self._evaluate(params)
            except Exception as e:
                logger.warning(f"Trial {i + 1}/{n_trials} 失败: {e}")
                continue
            trials.append({"trial": i + 1, metric_name: score, "params": sampled})
            if score > best_score:
                best_score = score
                best_params = params

        if best_params is None:
            best_params = dict(self.base_params)
            best_score = 0.0
        return best_params, float(best_score), trials

    # ------------------------------------------------------------------
    # 辅助：本轮建议文案
    # ------------------------------------------------------------------

    def _format_suggestion(
        self,
        diagnosis: DiagnosisResult,
        space: Dict[str, Tuple[float, float]],
        best_params: Dict[str, Any],
    ) -> str:
        """生成一句人类可读的“下一步建议”。"""
        key_moves: List[str] = []
        for key, bounds in space.items():
            if key in self.CAT_PARAMS:
                v = best_params.get(key)
                if v is not None:
                    key_moves.append(f"{key}={v}")
                continue
            lo, hi = bounds
            v = best_params.get(key)
            if v is None:
                continue
            if isinstance(v, float):
                key_moves.append(f"{key}={v:.4g} (∈[{lo:.3g},{hi:.3g}])")
            else:
                key_moves.append(f"{key}={v} (∈[{lo:.3g},{hi:.3g}])")
        moves_str = "; ".join(key_moves[:5]) if key_moves else "无"
        return f"[{diagnosis.status}] 本轮约束搜索命中: {moves_str}"

# 向后兼容：保留 TuningEngine 别名
from tuning.xgb_engine import XGBTuningEngine  # noqa: E402
TuningEngine = XGBTuningEngine

__all__ = ["BaseTuningEngine", "TuningEngine", "XGBTuningEngine"]
