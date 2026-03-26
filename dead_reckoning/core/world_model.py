"""
WorldModel — the agent's internal representation of task state.

Like a ship's log: records last known position, heading, and speed.
The agent navigates forward from this without querying the LLM until
accumulated uncertainty demands a fix.
"""

from __future__ import annotations
import time
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Checkpoint:
    """A verified snapshot of the world at a known-good moment."""
    id: str
    timestamp: float
    step_index: int
    goal: str
    completed_steps: list[dict]
    env_state: dict[str, Any]
    predicted_next_steps: list[str]
    confidence_at_creation: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        payload = json.dumps({
            "goal": self.goal,
            "step_index": self.step_index,
            "env_state": self.env_state,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


class WorldModel:
    """
    Lightweight task-state model that evolves between LLM calls.

    The WorldModel does three things:
      1. Tracks what has been done and what remains.
      2. Predicts the next N steps deterministically when confidence is high.
      3. Signals when accumulated drift demands a real LLM re-invocation.

    Drift accumulates when:
      - Predicted steps don't match actual outcomes
      - Tools return unexpected results
      - Step count since last checkpoint exceeds max_steps_without_fix
    """

    def __init__(
        self,
        goal: str,
        max_steps_without_fix: int = 5,
        drift_decay: float = 0.15,
    ):
        self.goal = goal
        self.max_steps_without_fix = max_steps_without_fix
        self.drift_decay = drift_decay  # drift added per misprediction

        self.completed_steps: list[dict] = []
        self.predicted_next_steps: list[str] = []
        self.env_state: dict[str, Any] = {}

        self._accumulated_drift: float = 0.0
        self._steps_since_fix: int = 0
        self._checkpoints: list[Checkpoint] = []
        self._step_index: int = 0

    # ------------------------------------------------------------------ #
    #  State updates                                                       #
    # ------------------------------------------------------------------ #

    def record_step(self, action: str, result: Any, predicted: bool = False) -> None:
        """Record a completed step and update drift based on prediction accuracy."""
        self._step_index += 1
        self._steps_since_fix += 1

        step_record = {
            "index": self._step_index,
            "action": action,
            "result": result,
            "predicted": predicted,
            "timestamp": time.time(),
        }
        self.completed_steps.append(step_record)

        if predicted and self.predicted_next_steps:
            expected = self.predicted_next_steps.pop(0)
            if not self._actions_match(action, expected):
                self._accumulated_drift += self.drift_decay * 2
            else:
                # successful prediction slightly reduces drift
                self._accumulated_drift = max(0, self._accumulated_drift - 0.02)
        elif not predicted:
            # unpredicted step increases drift slightly
            self._accumulated_drift += self.drift_decay * 0.5

        self._accumulated_drift = min(1.0, self._accumulated_drift)

    def update_env(self, updates: dict[str, Any]) -> None:
        """Merge new environment observations into the world state."""
        self.env_state.update(updates)

    def set_predictions(self, steps: list[str]) -> None:
        """LLM-provided predictions for the next N steps."""
        self.predicted_next_steps = list(steps)
        self._accumulated_drift = max(0, self._accumulated_drift - 0.1)

    # ------------------------------------------------------------------ #
    #  Checkpoints                                                         #
    # ------------------------------------------------------------------ #

    def checkpoint(self, confidence: float) -> Checkpoint:
        """Snapshot current state. Called after each LLM fix."""
        cp = Checkpoint(
            id=f"cp_{len(self._checkpoints):04d}",
            timestamp=time.time(),
            step_index=self._step_index,
            goal=self.goal,
            completed_steps=list(self.completed_steps),
            env_state=dict(self.env_state),
            predicted_next_steps=list(self.predicted_next_steps),
            confidence_at_creation=confidence,
        )
        self._checkpoints.append(cp)
        self._steps_since_fix = 0
        self._accumulated_drift = max(0, self._accumulated_drift - 0.2)
        return cp

    def rollback(self, checkpoint_id: str) -> Checkpoint:
        """Restore state to a previous checkpoint."""
        for cp in reversed(self._checkpoints):
            if cp.id == checkpoint_id:
                self.completed_steps = list(cp.completed_steps)
                self.env_state = dict(cp.env_state)
                self.predicted_next_steps = list(cp.predicted_next_steps)
                self._step_index = cp.step_index
                self._accumulated_drift = 0.0
                self._steps_since_fix = 0
                return cp
        raise ValueError(f"Checkpoint {checkpoint_id!r} not found")

    def last_checkpoint(self) -> Checkpoint | None:
        return self._checkpoints[-1] if self._checkpoints else None

    # ------------------------------------------------------------------ #
    #  Drift / confidence                                                  #
    # ------------------------------------------------------------------ #

    @property
    def drift(self) -> float:
        """Current accumulated drift [0, 1]. Higher = less reliable."""
        step_drift = min(0.5, self._steps_since_fix / self.max_steps_without_fix * 0.5)
        return min(1.0, self._accumulated_drift + step_drift)

    @property
    def confidence(self) -> float:
        return 1.0 - self.drift

    def needs_fix(self, threshold: float = 0.35) -> bool:
        """True when drift has accumulated enough to warrant an LLM re-invocation."""
        return self.drift >= threshold or self._steps_since_fix >= self.max_steps_without_fix

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        """Compact context string for injecting into LLM prompts."""
        lines = [
            f"Goal: {self.goal}",
            f"Steps completed: {self._step_index}",
            f"Last {min(3, len(self.completed_steps))} actions: "
            + ", ".join(s["action"] for s in self.completed_steps[-3:]),
            f"Current env: {json.dumps(self.env_state, default=str)[:300]}",
            f"Drift: {self.drift:.2f} | Confidence: {self.confidence:.2f}",
        ]
        return "\n".join(lines)

    def _actions_match(self, actual: str, predicted: str) -> bool:
        a, p = actual.lower().strip(), predicted.lower().strip()
        if a == p:
            return True
        # fuzzy: check if key verb/noun overlap
        a_tokens = set(a.split())
        p_tokens = set(p.split())
        overlap = len(a_tokens & p_tokens) / max(len(p_tokens), 1)
        return overlap >= 0.6

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "step_index": self._step_index,
            "confidence": self.confidence,
            "drift": self.drift,
            "steps_since_fix": self._steps_since_fix,
            "completed_steps": self.completed_steps,
            "predicted_next_steps": self.predicted_next_steps,
            "env_state": self.env_state,
            "checkpoints": len(self._checkpoints),
        }
