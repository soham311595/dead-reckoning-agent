"""
DeadReckoningAgent — the main orchestrator.
"""

from __future__ import annotations
import time
import logging
from enum import Enum
from typing import Any, Callable, Generator
from dataclasses import dataclass

from .world_model import WorldModel, Checkpoint
from .confidence_gate import ConfidenceGate, ExecutionMode, GateDecision

logger = logging.getLogger("dead_reckoning")


class StopReason(str, Enum):
    TASK_COMPLETE  = "task_complete"
    MAX_STEPS      = "max_steps"
    NO_PREDICTIONS = "no_predictions"
    ADAPTER_ERROR  = "adapter_error"


@dataclass
class StepResult:
    step_index: int
    action: str
    result: Any
    mode: ExecutionMode
    confidence: float
    drift: float
    llm_call_made: bool
    duration_ms: float
    checkpoint_id: str | None = None
    task_complete: bool = False


@dataclass
class RunStats:
    total_steps: int = 0
    llm_calls: int = 0
    deterministic_steps: int = 0
    checkpoints: int = 0
    total_duration_ms: float = 0.0
    rollbacks: int = 0
    stop_reason: StopReason | None = None

    @property
    def llm_call_rate(self) -> float:
        return self.llm_calls / self.total_steps if self.total_steps else 0.0

    @property
    def savings_pct(self) -> float:
        return (self.deterministic_steps / self.total_steps * 100) if self.total_steps else 0.0

    def __str__(self) -> str:
        stop = f" | Stop: {self.stop_reason.value}" if self.stop_reason else ""
        return (
            f"Steps: {self.total_steps} | "
            f"LLM calls: {self.llm_calls} ({self.llm_call_rate:.0%}) | "
            f"Deterministic: {self.deterministic_steps} ({self.savings_pct:.0f}% saved) | "
            f"Checkpoints: {self.checkpoints}{stop}"
        )


class LLMAdapter:
    """
    Base adapter — subclass this to connect your LLM.

    get_fix returns a 4-tuple: (reasoning, predicted_steps, next_action, done)
      reasoning:       LLM chain-of-thought string
      predicted_steps: list of next step strings (can be empty)
      next_action:     the immediate action to execute (empty string if done)
      done:            True if the LLM considers the task complete
    """

    def get_fix(
        self, world: WorldModel, tools: dict[str, Callable]
    ) -> tuple[str, list[str], str, bool]:
        raise NotImplementedError

    def execute_action(
        self, action: str, tools: dict[str, Callable], env: dict[str, Any]
    ) -> tuple[Any, bool]:
        raise NotImplementedError


class DeadReckoningAgent:
    """
    An LLM agent that navigates tasks like a ship navigates open water.

    Calls the LLM only at decision points (fixes). Between fixes, steps
    execute deterministically from predictions — no LLM, no latency, no cost.

    Usage:
        agent = DeadReckoningAgent(adapter, goal="...", tools={...})
        for step in agent.run():
            print(step.action, step.mode)
        print(agent.stats)
    """

    def __init__(
        self,
        adapter: LLMAdapter,
        goal: str,
        tools: dict[str, Callable] | None = None,
        fix_threshold: float = 0.35,
        hard_ceiling: float = 0.65,
        max_steps_without_fix: int = 5,
        checkpoint_interval: int = 10,
        max_total_steps: int = 100,
        verbose: bool = False,
    ):
        self.adapter = adapter
        self.tools = tools or {}
        self.max_total_steps = max_total_steps
        self.verbose = verbose

        self.world = WorldModel(goal=goal, max_steps_without_fix=max_steps_without_fix)
        self.gate = ConfidenceGate(
            fix_threshold=fix_threshold,
            hard_ceiling=hard_ceiling,
            checkpoint_interval=checkpoint_interval,
        )
        self.stats = RunStats()
        self._step_history: list[StepResult] = []

    # ------------------------------------------------------------------ #
    #  Main run loop                                                       #
    # ------------------------------------------------------------------ #

    def run(self) -> Generator[StepResult, None, None]:
        """
        Execute the task. Yields StepResult for each step.

        Terminates with stats.stop_reason set to one of:
          TASK_COMPLETE  — LLM signalled done=True
          MAX_STEPS      — hit max_total_steps ceiling
          NO_PREDICTIONS — predictions exhausted even after a fresh fix
          ADAPTER_ERROR  — unrecoverable exception from adapter
        """
        t_start = time.perf_counter()

        self._log("⚓ Initial fix — establishing position")
        done, _ = yield from self._do_fix(initial=True)
        if self.stats.stop_reason == StopReason.ADAPTER_ERROR:
            self._finish(StopReason.ADAPTER_ERROR, t_start)
            return
        if done:
            self._finish(StopReason.TASK_COMPLETE, t_start)
            return

        _empty_prediction_strikes = 0

        while self.stats.total_steps < self.max_total_steps:
            decision = self.gate.evaluate(
                world=self.world,
                proposed_action=(
                    self.world.predicted_next_steps[0]
                    if self.world.predicted_next_steps else None
                ),
            )

            if decision.mode == ExecutionMode.DETERMINISTIC:
                if not decision.recommended_next:
                    # Bug #1 fix: never stop silently — force a fix instead
                    _empty_prediction_strikes += 1
                    self._log(
                        f"  ⚠ Predictions exhausted (strike {_empty_prediction_strikes}/2) "
                        "— forcing fix"
                    )
                    if _empty_prediction_strikes >= 2:
                        self._finish(StopReason.NO_PREDICTIONS, t_start)
                        return
                    done, _ = yield from self._do_fix()
                    if self.stats.stop_reason == StopReason.ADAPTER_ERROR:
                        self._finish(StopReason.ADAPTER_ERROR, t_start)
                        return
                    if done:
                        self._finish(StopReason.TASK_COMPLETE, t_start)
                        return
                    continue

                _empty_prediction_strikes = 0
                step = self._do_deterministic_step(decision)
                if step:
                    self._step_history.append(step)
                    self.stats.total_steps += 1
                    yield step

            elif decision.mode in (ExecutionMode.FIX_REQUIRED, ExecutionMode.CHECKPOINT):
                done, _ = yield from self._do_fix(
                    checkpoint=(decision.mode == ExecutionMode.CHECKPOINT)
                )
                if self.stats.stop_reason == StopReason.ADAPTER_ERROR:
                    self._finish(StopReason.ADAPTER_ERROR, t_start)
                    return
                if done:
                    self._finish(StopReason.TASK_COMPLETE, t_start)
                    return

        self._finish(StopReason.MAX_STEPS, t_start)

    # ------------------------------------------------------------------ #
    #  Step execution                                                      #
    # ------------------------------------------------------------------ #

    def _do_deterministic_step(self, decision: GateDecision) -> StepResult | None:
        action = decision.recommended_next
        if not action:
            return None

        self._log(f"  → [DET] {action}  conf={decision.confidence:.2f} drift={decision.drift:.2f}")
        t0 = time.perf_counter()

        try:
            result, errored = self.adapter.execute_action(action, self.tools, self.world.env_state)
        except Exception as e:
            result, errored = str(e), True

        self.world.record_step(action=action, result=result, predicted=True)
        if errored:
            self.world._accumulated_drift += 0.2
            self._log("  ✕ Tool error — drift bumped")

        self.stats.deterministic_steps += 1
        return StepResult(
            step_index=self.stats.total_steps + 1,
            action=action, result=result,
            mode=ExecutionMode.DETERMINISTIC,
            confidence=decision.confidence, drift=decision.drift,
            llm_call_made=False,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    def _do_fix(
        self, initial: bool = False, checkpoint: bool = False
    ) -> Generator[StepResult, None, tuple[bool, StepResult | None]]:
        """
        Invoke the LLM. Yields at most one StepResult (the immediate action).
        Returns (done, step_result) to the caller via generator return.
        """
        t0 = time.perf_counter()
        label = "Initial" if initial else ("Scheduled" if checkpoint else "Drift")
        self._log(f"  🔭 {label} fix  drift={self.world.drift:.2f}")

        try:
            reasoning, predicted_steps, next_action, done = self.adapter.get_fix(
                self.world, self.tools
            )
        except Exception as e:
            self._log(f"  ✕ Adapter error: {e}")
            self.stats.stop_reason = StopReason.ADAPTER_ERROR
            return False, None

        self.world.set_predictions(predicted_steps)
        cp = self.world.checkpoint(confidence=self.world.confidence)
        self.gate.reset_checkpoint_counter()
        self.stats.llm_calls += 1
        self.stats.checkpoints += 1
        self._log(f"  ✓ Fix  done={done}  preds={predicted_steps[:3]}")

        last_step: StepResult | None = None

        if next_action and not done:
            try:
                result, errored = self.adapter.execute_action(
                    next_action, self.tools, self.world.env_state
                )
            except Exception as e:
                result, errored = str(e), True

            self.world.record_step(action=next_action, result=result, predicted=False)
            self.stats.total_steps += 1

            last_step = StepResult(
                step_index=self.stats.total_steps,
                action=next_action, result=result,
                mode=ExecutionMode.FIX_REQUIRED,
                confidence=self.world.confidence, drift=self.world.drift,
                llm_call_made=True,
                duration_ms=(time.perf_counter() - t0) * 1000,
                checkpoint_id=cp.id,
                task_complete=done,
            )
            self._step_history.append(last_step)
            yield last_step

        return done, last_step

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _finish(self, reason: StopReason, t_start: float) -> None:
        self.stats.stop_reason = reason
        self.stats.total_duration_ms = (time.perf_counter() - t_start) * 1000
        self._log(f"🏁 {reason.value}  {self.stats}")

    def rollback_to_last_checkpoint(self) -> Checkpoint | None:
        cp = self.world.last_checkpoint()
        if cp:
            self.world.rollback(cp.id)
            self.stats.rollbacks += 1
            self._log(f"  ↩ Rolled back to {cp.id}")
        return cp

    def step_history(self) -> list[StepResult]:
        return list(self._step_history)

    def llm_call_log(self) -> list[StepResult]:
        return [s for s in self._step_history if s.llm_call_made]

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
        logger.debug(msg)
