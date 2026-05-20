# -*- coding: utf-8 -*-
"""场景上下文 — 仅作为业务背景标签，不再承载权重/门禁。

历史版本里 Scenario 还带 weights（五维加权）和 gates（硬阈值淘汰），
现已统一改为"客观事实陈述 + LLM 推理"模式：
  - 不再有"加权综合分"
  - 不再有"门禁淘汰"
  - 场景仅作为给 LLM/人类阅读的业务上下文（描述这是评分卡 / 反欺诈 / 通用 等）

用户可通过 --scenario custom + 自定义描述传任意上下文。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Scenario:
    key: str
    name: str
    description: str

# ── 场景预设（仅作为上下文标签）──────────────────────────────
SCENARIOS: Dict[str, Scenario] = {
    "general": Scenario(
        key="general",
        name="通用",
        description="平衡效果、泛化、稳定、可解释；适合首次对比探索",
    ),
    "scorecard": Scenario(
        key="scorecard",
        name="评分卡 / 白盒合规",
        description="监管合规、可解释优先；LR + WoE 天然占优",
    ),
    "fraud": Scenario(
        key="fraud",
        name="反欺诈 / 高风险",
        description="极限捕获坏样本，容忍黑盒；OOT KS / BCR 是关键观测点",
    ),
    "stability_first": Scenario(
        key="stability_first",
        name="稳定优先",
        description="存量运维场景，分布漂移容忍度低；重点关注 PSI 与分月 KS",
    ),
}

def get_scenario(key: str) -> Scenario:
    """按 key 取场景；未知 key 退化到 general 并标注。"""
    s = SCENARIOS.get(key)
    if s is None:
        base = SCENARIOS["general"]
        return Scenario(
            key=key,
            name=f"{key}（未知，回退到通用）",
            description=base.description,
        )
    return s

def list_scenarios() -> List[Dict[str, str]]:
    return [
        {"key": s.key, "name": s.name, "description": s.description}
        for s in SCENARIOS.values()
    ]
