
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types import Audit, Quadrant, Trajectory

_ACTION = {
    Quadrant.SOLVED: "s2t",
    Quadrant.T2I: "t2i",
    Quadrant.I2T: "i2t",
    Quadrant.UNTEACHABLE: "neg",
}


@dataclass
class RoutingDecision:
    quadrant: Quadrant
    action: str
    reliable_side: Optional[str]
    teach_description: str
    notes: str = ""

    @property
    def trainable(self) -> bool:
        return self.action in ("t2i", "i2t", "neg", "s2t")


def route(traj: Trajectory, audit: Audit) -> RoutingDecision:
    q = audit.quadrant
    if q is Quadrant.T2I:
        return RoutingDecision(q, "t2i", "text", audit.text.description, audit.image.reason)
    if q is Quadrant.I2T:
        return RoutingDecision(q, "i2t", "image", audit.image.description, audit.text.reason)
    if q is Quadrant.UNTEACHABLE:
        return RoutingDecision(q, "neg", None, "", "double-fail: text-side unlikelihood")
    return RoutingDecision(q, "s2t", "image", "", "solved: consistency positive (recover T+ from images)")
