"""
ConfidenceGate — the navigator's judgment call.

A ship's navigator uses dead reckoning until one of three things happens:
  1. They spot a known landmark and take a proper fix.
  2. They've been reckoning too long and drift is unacceptable.
  3. Something unexpected happens (current, storm, uncharted reef).

This module maps those three cases to the LLM agent context.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .world_model import WorldModel


class ExecutionMode(str, Enum):
    DETERMINISTIC = "deterministic"   # run without LLM
    FIX_REQUIRED  = "fix_required"    # invoke LLM to re-calibrate
    CHECKPOINT    = "checkpoint"      # LLM invoked + snapshot state


@dataclass
class GateDecision:
    mode: ExecutionMode
    reason: str
    confidence: float
    drift: float
    recommended_next: str | None = None  # from prediction queue, if any


class ConfidenceGate:
    """
    Evaluates the WorldModel and decides execution mode for each step.

    Usage:
        gate = ConfidenceGate(fix_threshold=0.35, checkpoint_interval=10)
        decision = gate.evaluate(world_model, proposed_action)
    """

    def __init__(
        self,
        fix_threshold: float = 0.35,
        hard_ceiling: float = 0.65,
        checkpoint_interval: int = 10,
        require_fix_on_tool_error: bool = True,
    ):
        """
        Args:
            fix_threshold:  Drift at which we request an LLM fix.
            hard_ceiling:   Drift at which we STOP and demand a fix regardless.
            checkpoint_interval: Every N steps, snapshot the world even if confident.
            require_fix_on_tool_error: Any tool error forces a fix.
        """
        self.fix_threshold = fix_threshold
        self.hard_ceiling = hard_ceiling
        self.checkpoint_interval = checkpoint_interval
        self.require_fix_on_tool_error = require_fix_on_tool_error
        self._steps_since_checkpoint = 0

    def evaluate(
        self,
        world: WorldModel,
        proposed_action: str | None = None,
        last_result: Any = None,
        tool_errored: bool = False,
    ) -> GateDecision:
        """
        Evaluate current world state and return an ExecutionMode decision.

        This is the core dead reckoning judgment:
          - If we have a good prediction and low drift → run deterministically.
          - If drift is building but still manageable → run but flag for upcoming fix.
          - If drift exceeds threshold or anomaly detected → stop, get a fix.
        """
        confidence = world.confidence
        drift = world.drift

        # Hard override: tool error always requires a fix
        if tool_errored and self.require_fix_on_tool_error:
            return GateDecision(
                mode=ExecutionMode.FIX_REQUIRED,
                reason="Tool error — reality diverged from model",
                confidence=confidence,
                drift=drift,
            )

        # Hard ceiling: drift too high to proceed at all
        if drift >= self.hard_ceiling:
            return GateDecision(
                mode=ExecutionMode.FIX_REQUIRED,
                reason=f"Drift ceiling hit ({drift:.2f} ≥ {self.hard_ceiling})",
                confidence=confidence,
                drift=drift,
            )

        # Scheduled checkpoint interval
        self._steps_since_checkpoint += 1
        if self._steps_since_checkpoint >= self.checkpoint_interval:
            self._steps_since_checkpoint = 0
            return GateDecision(
                mode=ExecutionMode.CHECKPOINT,
                reason=f"Scheduled checkpoint (every {self.checkpoint_interval} steps)",
                confidence=confidence,
                drift=drift,
            )

        # Soft fix threshold crossed
        if drift >= self.fix_threshold:
            return GateDecision(
                mode=ExecutionMode.FIX_REQUIRED,
                reason=f"Drift threshold reached ({drift:.2f} ≥ {self.fix_threshold})",
                confidence=confidence,
                drift=drift,
            )

        # We're navigating confidently — proceed deterministically
        next_predicted = (
            world.predicted_next_steps[0] if world.predicted_next_steps else None
        )
        return GateDecision(
            mode=ExecutionMode.DETERMINISTIC,
            reason=f"High confidence ({confidence:.2f}) — executing from world model",
            confidence=confidence,
            drift=drift,
            recommended_next=next_predicted,
        )

    def reset_checkpoint_counter(self) -> None:
        self._steps_since_checkpoint = 0
