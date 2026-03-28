"""
ClaudeCodeAdapter — uses Claude Code CLI as the LLM backend.

Requires Claude Code installed and authenticated:
    npm install -g @anthropic-ai/claude-code
    claude   # authenticate on first run

Key design decisions:
  - Each `claude -p` call is fully stateless — all context is embedded
    directly in the prompt, not via --system-prompt (which is unreliable
    across CLI versions).
  - Full completed-step history is included in every call so Claude
    never loses track of what has been done.
  - JSON is extracted from the response even if Claude wraps it in prose.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from typing import Any, Callable

from dead_reckoning.core.agent import LLMAdapter
from dead_reckoning.core.world_model import WorldModel
from dead_reckoning.adapters import _parse_fix_response, _dispatch_action


# ------------------------------------------------------------------ #
#  Prompt templates — embedded directly in the user prompt             #
#  (not via --system-prompt, which is unreliable across CLI versions)  #
# ------------------------------------------------------------------ #

_DR_PROMPT_TEMPLATE = """You are the navigation module of a Dead Reckoning Agent completing a task.

INSTRUCTIONS (follow exactly):
- Respond with ONLY a JSON object. No prose before or after. No markdown fences.
- Format: {{"reasoning": "1-2 sentences", "done": false, "next_action": "tool_name(param='value')", "predicted_steps": ["tool2()", "tool3()"], "confidence": 0.9}}
- next_action must be EXACTLY one of the available tool names with correct call syntax
- predicted_steps: list the next 2-3 tool calls that will likely follow
- Set done=true ONLY when ALL required steps of the goal are complete
- NEVER set done=true if steps_completed is 0
- Do NOT use bash, file tools, or web search — only the tools listed below

Available tools: {tool_names}

CURRENT STATE:
Goal: {goal}
Steps completed: {n_steps}
{history}
Drift: {drift:.2f} | Confidence: {confidence:.2f}
Environment: {env}

Respond with JSON only:"""


_REACT_PROMPT_TEMPLATE = """You are an agent completing a task step by step using API tools.

INSTRUCTIONS (follow exactly):
- Respond with ONLY a JSON object. No prose before or after. No markdown fences.
- Format: {{"thought": "what to do next", "action": "tool_name(param='value')", "done": false}}
- action must be EXACTLY one of the available tool names
- Set done=true ONLY after ALL required tools have been called
- NEVER set done=true if no tools have been called yet (steps_completed = 0)
- Do NOT repeat tools already called in completed_steps
- Do NOT use bash, file tools, or web search

Available tools: {tool_names}

TASK: {goal}

COMPLETED STEPS ({n_steps} so far):
{history}

What is the next action? Respond with JSON only:"""


# ------------------------------------------------------------------ #
#  Shared CLI runner                                                   #
# ------------------------------------------------------------------ #

def _run_claude(prompt: str, model: str, timeout: int) -> str:
    """
    Run `claude -p <prompt> --output-format json` and return the result text.
    Extracts Claude's response from the JSON envelope.
    Returns empty string on any failure.
    """
    cmd = [
        "claude",
        "-p", prompt,
        "--output-format", "json",
        "--model", model,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,  # prevents hang without a tty
        )
    except subprocess.TimeoutExpired:
        return ""
    except FileNotFoundError:
        raise RuntimeError(
            "Claude Code CLI not found.\n"
            "Install: npm install -g @anthropic-ai/claude-code\n"
            "Auth:    claude"
        )

    if proc.returncode != 0:
        return ""

    stdout = proc.stdout.strip()
    if not stdout:
        return ""

    # Claude Code returns a JSON envelope: {"result": "...", "cost_usd": ..., "usage": {...}}
    try:
        data = json.loads(stdout)
        return data.get("result", "")
    except json.JSONDecodeError:
        # Fall back to raw stdout if envelope parsing fails
        return stdout


def _track_usage(stdout: str) -> tuple[float, int, int]:
    """Extract cost_usd, input_tokens, output_tokens from Claude Code JSON envelope."""
    try:
        data = json.loads(stdout.strip())
        cost  = float(data.get("cost_usd", 0) or 0)
        usage = data.get("usage", {}) or {}
        return cost, int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)
    except Exception:
        return 0.0, 0, 0


def _build_history(completed_steps: list[dict], max_steps: int = 10) -> str:
    """Format completed steps as a readable history string."""
    if not completed_steps:
        return "(none)"
    # Show last max_steps to keep prompt size bounded
    recent = completed_steps[-max_steps:]
    lines = []
    for s in recent:
        result_preview = str(s.get("result", ""))[:80].replace("\n", " ")
        lines.append(f"  [{s['index']}] {s['action']} → {result_preview}")
    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Dead Reckoning adapter                                              #
# ------------------------------------------------------------------ #

class ClaudeCodeAdapter(LLMAdapter):
    """
    Dead Reckoning adapter that uses Claude Code CLI (-p headless mode).

    Every get_fix() embeds the full task context — goal, completed steps,
    available tools, drift — directly in the prompt. No --system-prompt flag,
    no state between calls.

    Args:
        model:         Claude model string (e.g. "claude-haiku-4-5")
        n_predictions: steps to predict ahead
        timeout:       seconds per CLI call before giving up
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        n_predictions: int = 4,
        timeout: int = 60,
    ):
        self.model         = model
        self.n_predictions = n_predictions
        self.timeout       = timeout
        self.total_cost_usd = 0.0
        self.input_tokens   = 0
        self.output_tokens  = 0
        self._call_count    = 0
        _verify_claude_installed()

    def get_fix(
        self,
        world: WorldModel,
        tools: dict[str, Callable],
    ) -> tuple[str, list[str], str, bool]:
        tool_names = ", ".join(list(tools.keys())[:25])
        history    = _build_history(world.completed_steps)

        prompt = _DR_PROMPT_TEMPLATE.format(
            tool_names=tool_names,
            goal=world.goal,
            n_steps=len(world.completed_steps),
            history=history,
            drift=world.drift,
            confidence=world.confidence,
            env=json.dumps(world.env_state, default=str)[:200],
        )

        # Run via Claude Code CLI
        cmd = ["claude", "-p", prompt, "--output-format", "json", "--model", self.model]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.timeout, stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return "timeout", [], "", False
        except FileNotFoundError:
            raise RuntimeError("Claude Code CLI not found. Install: npm install -g @anthropic-ai/claude-code")

        if proc.returncode != 0:
            return f"cli error: {proc.stderr[:100]}", [], "", False

        # Track usage
        cost, in_tok, out_tok = _track_usage(proc.stdout)
        self.total_cost_usd += cost
        self.input_tokens   += in_tok
        self.output_tokens  += out_tok
        self._call_count    += 1

        # Extract text from Claude Code envelope
        text = ""
        try:
            data = json.loads(proc.stdout.strip())
            text = data.get("result", "")
        except Exception:
            text = proc.stdout

        return _parse_fix_response(text)

    def execute_action(
        self,
        action: str,
        tools: dict[str, Callable],
        env: dict[str, Any],
    ) -> tuple[Any, bool]:
        return _dispatch_action(action, tools, env)

    def stats_str(self) -> str:
        return (
            f"Claude Code calls: {self._call_count} | "
            f"tokens in/out: {self.input_tokens}/{self.output_tokens} | "
            f"cost: ${self.total_cost_usd:.5f}"
        )


# ------------------------------------------------------------------ #
#  ReAct adapter                                                       #
# ------------------------------------------------------------------ #

class ClaudeCodeReActAdapter(LLMAdapter):
    """
    ReAct baseline via Claude Code CLI.

    Every call embeds the FULL step history in the prompt — since each
    `claude -p` subprocess is stateless, we can't rely on conversation
    memory. The history grows with each step (same as a proper ReAct loop).
    """

    def __init__(self, model: str, goal: str, timeout: int = 60):
        self.model   = model
        self.goal    = goal
        self.timeout = timeout
        self._tools_called = 0
        self.total_cost_usd = 0.0
        self.input_tokens   = 0
        self.output_tokens  = 0

    def get_fix(self, world: WorldModel, tools: dict) -> tuple[str, list[str], str, bool]:
        tool_names = ", ".join(list(tools.keys())[:25])
        history    = _build_history(world.completed_steps, max_steps=20)

        prompt = _REACT_PROMPT_TEMPLATE.format(
            tool_names=tool_names,
            goal=self.goal,
            n_steps=len(world.completed_steps),
            history=history,
        )

        cmd = ["claude", "-p", prompt, "--output-format", "json", "--model", self.model]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.timeout, stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return "timeout", [], "", False
        except FileNotFoundError:
            raise RuntimeError("Claude Code CLI not found.")

        if proc.returncode != 0:
            return f"cli error: {proc.stderr[:100]}", [], "", False

        cost, in_tok, out_tok = _track_usage(proc.stdout)
        self.total_cost_usd += cost
        self.input_tokens   += in_tok
        self.output_tokens  += out_tok

        text = ""
        try:
            data = json.loads(proc.stdout.strip())
            text = data.get("result", "")
        except Exception:
            text = proc.stdout

        # Parse response
        clean = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")

        # Find JSON — Claude sometimes adds prose before/after
        # Try direct parse first, then scan for first {...}
        parsed = None
        try:
            parsed = json.loads(clean)
        except Exception:
            match = re.search(r'\{[^{}]*"action"[^{}]*\}', clean, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except Exception:
                    pass

        if not parsed:
            m = re.search(r'"action"\s*:\s*"([^"]+)"', clean)
            return "parse error", [], m.group(1) if m else "", False

        action = parsed.get("action", "")
        done   = bool(parsed.get("done", False))

        # Block premature done
        if done and self._tools_called == 0:
            done = False
            if not action and tools:
                action = list(tools.keys())[0] + "()"

        return parsed.get("thought", ""), [], action, done

    def execute_action(self, action: str, tools: dict, env: dict) -> tuple[Any, bool]:
        result, errored = _dispatch_action(action, tools, env)
        if not errored and "no tool matched" not in str(result):
            self._tools_called += 1
        return result, errored


# ------------------------------------------------------------------ #
#  Helper                                                              #
# ------------------------------------------------------------------ #

def _verify_claude_installed() -> None:
    if shutil.which("claude") is None:
        raise RuntimeError(
            "Claude Code CLI not found.\n"
            "Install: npm install -g @anthropic-ai/claude-code\n"
            "Auth:    claude   (run once to log in)"
        )