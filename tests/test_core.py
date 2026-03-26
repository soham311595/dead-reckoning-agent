"""Tests for the Dead Reckoning Agent framework."""
import pytest
from dead_reckoning.core.world_model import WorldModel
from dead_reckoning.core.confidence_gate import ConfidenceGate, ExecutionMode
from dead_reckoning.core.agent import DeadReckoningAgent, LLMAdapter, StopReason
from dead_reckoning.adapters import _dispatch_action, _parse_fix_response


class TestWorldModel:
    def test_initial_state(self):
        wm = WorldModel(goal="test goal")
        assert wm.goal == "test goal"
        assert wm.confidence == 1.0
        assert wm.drift == 0.0

    def test_drift_accumulates_on_misprediction(self):
        wm = WorldModel(goal="test")
        wm.set_predictions(["step_a", "step_b"])
        wm.record_step("step_x", result="ok", predicted=True)
        assert wm.drift > 0

    def test_drift_decreases_on_correct_prediction(self):
        wm = WorldModel(goal="test")
        wm._accumulated_drift = 0.3
        wm.set_predictions(["analyze_file"])
        wm.record_step("analyze_file", result="ok", predicted=True)
        assert wm.drift < 0.3

    def test_checkpoint_resets_counter(self):
        wm = WorldModel(goal="test")
        for i in range(3):
            wm.record_step(f"step_{i}", result="ok")
        assert wm._steps_since_fix == 3
        wm.checkpoint(confidence=0.8)
        assert wm._steps_since_fix == 0

    def test_rollback_restores_state(self):
        wm = WorldModel(goal="test")
        wm.update_env({"files": ["a.py"]})
        cp = wm.checkpoint(confidence=0.9)
        wm.update_env({"files": ["a.py", "b.py"]})
        wm.record_step("write_file", result="ok")
        wm.rollback(cp.id)
        assert wm.env_state == {"files": ["a.py"]}
        assert wm._step_index == cp.step_index

    def test_needs_fix_after_max_steps(self):
        wm = WorldModel(goal="test", max_steps_without_fix=3)
        for i in range(3):
            wm.record_step(f"step_{i}", "ok")
        assert wm.needs_fix(threshold=0.35)

    def test_summary_returns_string(self):
        wm = WorldModel(goal="refactor auth")
        summary = wm.summary()
        assert "refactor auth" in summary
        assert "Drift" in summary


class TestConfidenceGate:
    def test_deterministic_when_low_drift(self):
        wm = WorldModel(goal="test")
        wm.set_predictions(["step_a"])
        gate = ConfidenceGate(fix_threshold=0.35)
        decision = gate.evaluate(wm)
        assert decision.mode == ExecutionMode.DETERMINISTIC

    def test_fix_required_on_tool_error(self):
        wm = WorldModel(goal="test")
        gate = ConfidenceGate()
        decision = gate.evaluate(wm, tool_errored=True)
        assert decision.mode == ExecutionMode.FIX_REQUIRED

    def test_fix_required_when_drift_high(self):
        wm = WorldModel(goal="test")
        wm._accumulated_drift = 0.4
        gate = ConfidenceGate(fix_threshold=0.35)
        decision = gate.evaluate(wm)
        assert decision.mode == ExecutionMode.FIX_REQUIRED

    def test_checkpoint_at_interval(self):
        wm = WorldModel(goal="test")
        gate = ConfidenceGate(checkpoint_interval=3)
        for _ in range(2):
            gate.evaluate(wm)
        decision = gate.evaluate(wm)
        assert decision.mode == ExecutionMode.CHECKPOINT

    def test_recommended_next_from_predictions(self):
        wm = WorldModel(goal="test")
        wm.set_predictions(["analyze_file(path='a.py')"])
        gate = ConfidenceGate()
        decision = gate.evaluate(wm)
        assert decision.recommended_next == "analyze_file(path='a.py')"

    def test_deterministic_with_no_predictions_returns_none_recommended(self):
        wm = WorldModel(goal="test")  # no predictions set
        gate = ConfidenceGate(fix_threshold=0.99)  # prevent FIX
        decision = gate.evaluate(wm)
        assert decision.mode == ExecutionMode.DETERMINISTIC
        assert decision.recommended_next is None


class TestStopReasons:
    def _make_agent(self, adapter, **kwargs):
        return DeadReckoningAgent(adapter, goal="test", **{"max_total_steps": 30, **kwargs})

    def test_task_complete_stop(self):
        calls = [0]
        class DoneAdapter(LLMAdapter):
            def get_fix(self, world, tools):
                calls[0] += 1
                return "ok", ["step_a"], "step_a", calls[0] >= 2
            def execute_action(self, action, tools, env):
                return "done", False

        agent = self._make_agent(DoneAdapter(), max_steps_without_fix=3)
        list(agent.run())
        assert agent.stats.stop_reason == StopReason.TASK_COMPLETE

    def test_no_predictions_stop(self):
        class NoPredAdapter(LLMAdapter):
            def get_fix(self, world, tools):
                return "ok", [], "step_a", False
            def execute_action(self, action, tools, env):
                return "done", False

        agent = self._make_agent(NoPredAdapter(), fix_threshold=0.35, max_steps_without_fix=3)
        list(agent.run())
        assert agent.stats.stop_reason == StopReason.NO_PREDICTIONS

    def test_max_steps_stop(self):
        class InfiniteAdapter(LLMAdapter):
            def get_fix(self, world, tools):
                return "ok", ["step_a", "step_b", "step_c"], "step_a", False
            def execute_action(self, action, tools, env):
                return "done", False

        agent = self._make_agent(InfiniteAdapter(), max_total_steps=8, max_steps_without_fix=10)
        list(agent.run())
        assert agent.stats.stop_reason == StopReason.MAX_STEPS

    def test_adapter_error_stop(self):
        class ErrorAdapter(LLMAdapter):
            def get_fix(self, world, tools):
                raise RuntimeError("boom")
            def execute_action(self, action, tools, env):
                return "done", False

        agent = self._make_agent(ErrorAdapter())
        list(agent.run())
        assert agent.stats.stop_reason == StopReason.ADAPTER_ERROR


class TestActionDispatcher:
    def setup_method(self):
        self.tools = {
            "read_file": lambda path: f"content:{path}",
            "write_file": lambda path, content="": f"wrote:{path}:{len(content)}",
            "run_tests": lambda path=".": {"passed": 12},
            "no_args": lambda: "result",
        }

    def test_single_kwarg_single_quotes(self):
        result, err = _dispatch_action("read_file(path='auth/login.py')", self.tools, {})
        assert result == "content:auth/login.py"
        assert not err

    def test_multiple_kwargs(self):
        result, err = _dispatch_action(
            "write_file(path='out.py', content='hello world')", self.tools, {}
        )
        assert result == "wrote:out.py:11"
        assert not err

    def test_no_args(self):
        result, err = _dispatch_action("no_args()", self.tools, {})
        assert result == "result"
        assert not err

    def test_bare_tool_name(self):
        result, err = _dispatch_action("no_args", self.tools, {})
        assert result == "result"

    def test_colon_format(self):
        result, err = _dispatch_action("read_file: auth/login.py", self.tools, {})
        assert result == "content:auth/login.py"

    def test_unknown_tool(self):
        result, err = _dispatch_action("nonexistent()", self.tools, {})
        assert "no tool matched" in result
        assert not err

    def test_prose_action_no_match(self):
        result, err = _dispatch_action("Read the file to understand it", self.tools, {})
        assert "no tool matched" in result


class TestParseFixResponse:
    def test_valid_json_with_done(self):
        import json
        resp = json.dumps({
            "reasoning": "all done",
            "done": True,
            "next_action": "",
            "predicted_steps": [],
            "confidence": 0.99,
        })
        r, preds, action, done = _parse_fix_response(resp)
        assert done is True
        assert action == ""

    def test_valid_json_not_done(self):
        import json
        resp = json.dumps({
            "reasoning": "keep going",
            "done": False,
            "next_action": "read_file(path='x.py')",
            "predicted_steps": ["write_file(path='x.py')"],
        })
        r, preds, action, done = _parse_fix_response(resp)
        assert not done
        assert action == "read_file(path='x.py')"
        assert preds == ["write_file(path='x.py')"]

    def test_markdown_fences_stripped(self):
        resp = '```json\n{"reasoning":"ok","done":false,"next_action":"step_a","predicted_steps":[]}\n```'
        r, preds, action, done = _parse_fix_response(resp)
        assert action == "step_a"
        assert not done
