"""Structured, traceable decision log for the agent loop.

Every step records what was observed, what action was chosen and why, the
action parameters, and the result of acting. This trace is the evidence that
the agent's next action depends on actual visual output.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class Step:
    iteration: int
    phase: str                  # 'perceive' | 'evaluate' | 'choose' | 'act' | 'terminate'
    observation: Dict[str, Any] = field(default_factory=dict)
    action: str = ""
    action_params: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    result: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DecisionLog:
    def __init__(self) -> None:
        self._steps: List[Step] = []

    def add(self, step: Step) -> None:
        self._steps.append(step)

    @property
    def steps(self) -> List[Step]:
        return list(self._steps)

    def to_list(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._steps]

    def actions_taken(self) -> List[str]:
        return [s.action for s in self._steps if s.action]
