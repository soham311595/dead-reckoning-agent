"""
Dead Reckoning Agent — Full Evaluation Harness
===============================================
Saves everything needed to write a paper:
  - Full LLM call traces (input, output, tokens, cost)
  - Full tool call traces (action, result, errored)
  - Per-task metrics CSV
  - Aggregate statistics with bootstrap confidence intervals
  - Human-readable paper report (Markdown)
  - Machine-readable summary JSON

Output structure:
  results/
  └── {run_id}/
      ├── traces/
      │   ├── G1_000_dead_reckoning.json   ← full trace per task
      │   ├── G1_000_react.json
      │   └── ...
      ├── metrics.csv                       ← per-task numbers
      ├── summary.json                      ← aggregate stats + CIs
      └── report.md                         ← paper-ready writeup

Run:
    # With Anthropic SDK (best quality)
    ANTHROPIC_API_KEY=sk-ant-... python3 benchmarks/eval.py \\
        --toolbench_dir ~/Desktop/AI\\ work/ToolBench \\
        --provider anthropic \\
        --model claude-haiku-4-5 \\
        --split all

    # With OpenRouter (free, slower)
    OPENROUTER_API_KEY=sk-or-... python3 benchmarks/eval.py \\
        --toolbench_dir ~/Desktop/AI\\ work/ToolBench \\
        --provider openrouter \\
        --model mistralai/mistral-small-3.1-24b-instruct:free \\
        --split all \\
        --delay 2.0

    # Resume interrupted run
    python3 benchmarks/eval.py ... --run_id 20260330_143022

Notes on metrics:
  - success_gt: ALL required APIs from api_list were called (ground truth)
  - success_llm: Claude self-reported done=true (unreliable, shown for comparison)
  - llm_reduction: (react_llm - dr_llm) / react_llm
  - We use bootstrap CIs (n=1000) for all reported percentages
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import random
import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dead_reckoning import DeadReckoningAgent
from dead_reckoning.adapters import _parse_fix_response, _dispatch_action
from dead_reckoning.core.agent import LLMAdapter, StopReason
from dead_reckoning.core.world_model import WorldModel

sys.path.insert(0, str(Path(__file__).parent))
from toolbench_data_example import (
    load_tasks,
    load_response_examples,
    build_tools_for_task,
    evaluate_success,
    sanitize_name,
)


# ================================================================== #
#  Trace dataclasses — capture EVERYTHING                              #
# ================================================================== #

@dataclass
class LLMCallTrace:
    """One LLM invocation."""
    call_index:      int
    step_index:      int
    prompt_summary:  str        # world.summary() sent to LLM
    raw_response:    str        # full text response from LLM
    reasoning:       str
    predicted_steps: list[str]
    next_action:     str
    done:            bool
    input_tokens:    int
    output_tokens:   int
    cost_usd:        float
    latency_s:       float
    error:           str = ""


@dataclass
class StepTrace:
    """One agent step."""
    step_index:    int
    mode:          str          # "llm_fix" | "deterministic"
    action:        str
    result:        str          # truncated tool output
    errored:       bool
    confidence:    float
    drift:         float
    llm_call_made: bool


@dataclass
class TaskTrace:
    """Complete trace for one task × method run."""
    # Identity
    task_id:         str
    split:           str
    goal:            str
    method:          str
    model:           str
    provider:        str
    run_id:          str
    timestamp:       str

    # Ground truth
    api_list:        list[str]   # required APIs from ToolBench
    toolbench_win:   Optional[bool]  # from answer/ json if available

    # Outcome
    success_gt:      bool        # all required APIs called
    success_llm:     bool        # Claude said done=true
    stop_reason:     str

    # Metrics
    total_steps:     int
    llm_calls:       int
    det_steps:       int
    savings_pct:     float
    tools_required:  int
    tools_called:    int
    tools_called_list: list[str]
    input_tokens:    int
    output_tokens:   int
    cost_usd:        float
    wall_time_s:     float

    # Full traces
    steps:           list[dict]      # StepTrace as dicts
    llm_calls_log:   list[dict]      # LLMCallTrace as dicts

    error:           str = ""


# ================================================================== #
#  Instrumented adapters that capture full traces                      #
# ================================================================== #

_DR_SYSTEM = """You are the navigation module of a Dead Reckoning Agent completing a task.

Output ONLY a JSON object — no prose, no markdown:
{{"reasoning": "1-2 sentences", "done": true/false, "next_action": "tool_name(param='value')", "predicted_steps": ["tool2()", "tool3()", "tool4()"], "confidence": 0.9}}

Rules:
- next_action MUST be exactly one of the available tool names
- predicted_steps: list ALL remaining tools you expect to call after next_action, in order
  If only 1 tool remains after next_action, still list it. Never return an empty predicted_steps unless done=true.
- Set done=true ONLY when every tool in the task has been called — check completed_steps carefully
- NEVER done=true if steps_completed is 0
- NEVER done=true if there are tools you haven't called yet
Available tools: {tool_names}"""

_REACT_SYSTEM = """You are an agent completing a task step by step.

Output ONLY a JSON object — no prose, no markdown:
{{"thought": "what to do next", "action": "tool_name(param='value')", "done": false}}

Rules:
- action MUST be exactly one of the available tool names
- NEVER done=true if no tools have been called yet
- Set done=true when ALL required tools have been called — check the completed list carefully
- Do NOT call a tool that already appears in COMPLETED TOOLS — that is a loop, stop it
- If all required tools are in COMPLETED TOOLS, set done=true immediately

Available tools: {tool_names}
Goal: {goal}
COMPLETED TOOLS SO FAR: {completed_tools}"""


class TracingAnthropicDRAdapter(LLMAdapter):
    def __init__(self, api_key: str, model: str, n_required: int = 1):
        import anthropic
        self.client      = anthropic.Anthropic(api_key=api_key)
        self.model       = model
        self.n_required  = n_required
        self._tools_called = 0
        self.llm_traces: list[LLMCallTrace] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self._call_idx = 0

    def get_fix(self, world, tools):
        tool_names = ", ".join(list(tools.keys())[:25])
        system = _DR_SYSTEM.format(tool_names=tool_names)
        prompt = world.summary()
        t0 = time.perf_counter()
        error = ""
        raw = ""
        in_tok = out_tok = 0
        result = ("", [], "", False)

        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=600,
                system=[{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
            in_tok  = resp.usage.input_tokens
            out_tok = resp.usage.output_tokens
            # Haiku: ~$0.00025/1K input, ~$0.00125/1K output
            cost = (in_tok * 0.00025 + out_tok * 0.00125) / 1000
            self.input_tokens  += in_tok
            self.output_tokens += out_tok
            self.cost_usd      += cost
            r, ps, na, dn = _parse_fix_response(raw)
            if dn and self._tools_called == 0:
                dn = False
                if not na and tools: na = list(tools.keys())[0] + "()"
            result = (r, ps, na, dn)
        except Exception as e:
            error = str(e)

        latency = time.perf_counter() - t0
        self.llm_traces.append(LLMCallTrace(
            call_index=self._call_idx,
            step_index=len(world.completed_steps),
            prompt_summary=prompt[:500],
            raw_response=raw[:1000],
            reasoning=result[0] if result else "",
            predicted_steps=result[1] if result else [],
            next_action=result[2] if result else "",
            done=result[3] if result else False,
            input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=cost if not error else 0.0,
            latency_s=round(latency, 3), error=error,
        ))
        self._call_idx += 1
        return result

    def execute_action(self, action, tools, env):
        result, errored = _dispatch_action(action, tools, env)
        if not errored and "no tool matched" not in str(result): self._tools_called += 1
        return result, errored


class TracingAnthropicReActAdapter(LLMAdapter):
    def __init__(self, api_key: str, model: str, goal: str, n_required: int = 1):
        import anthropic
        self.client      = anthropic.Anthropic(api_key=api_key)
        self.model       = model
        self.goal        = goal
        self.n_required  = n_required
        self._history: list[dict] = []
        self._tools_called = 0
        self.llm_traces: list[LLMCallTrace] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self._call_idx = 0

    def get_fix(self, world, tools):
        tool_names = ", ".join(list(tools.keys())[:25])
        completed = world.completed_steps
        completed_tools = list(set(s["action"].split("(")[0].strip() for s in completed)) if completed else []
        system = _REACT_SYSTEM.format(
            tool_names=tool_names,
            goal=self.goal,
            completed_tools=completed_tools if completed_tools else "none yet",
        )
        if completed:
            history_lines = "\n".join(
                f"  {i+1}. {s['action']} → success"
                for i, s in enumerate(completed[-10:])
            )
            user_msg = (
                f"Steps done ({len(completed)} total):\n{history_lines}\n\n"
                f"Tools already called: {completed_tools}\n"
                f"What is the next action? If all required tools are done, set done=true."
            )
        else:
            user_msg = f"Goal: {self.goal}\nNo steps yet. What is the first action?"

        self._history.append({"role": "user", "content": user_msg})
        t0 = time.perf_counter()
        error = ""
        raw = ""
        in_tok = out_tok = cost = 0
        action, done = "", False

        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=400,
                system=[{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}],
                messages=self._history,
            )
            raw = resp.content[0].text.strip()
            in_tok  = resp.usage.input_tokens
            out_tok = resp.usage.output_tokens
            cost = (in_tok * 0.00025 + out_tok * 0.00125) / 1000
            self.input_tokens  += in_tok
            self.output_tokens += out_tok
            self.cost_usd      += cost
            self._history.append({"role": "assistant", "content": raw})

            clean = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
            # Extract last valid JSON object (model may self-correct mid-response)
            d = None
            depth, start, candidates = 0, None, []
            for i, ch in enumerate(clean):
                if ch == "{":
                    if depth == 0: start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and start is not None:
                        candidates.append(clean[start:i+1])
                        start = None
            for blob in reversed(candidates):
                try:
                    parsed = json.loads(blob)
                    if any(k in parsed for k in ("action", "done")):
                        d = parsed; break
                except Exception:
                    continue
            if d is None:
                m = re.search(r'"action"\s*:\s*"([^"]+)"', clean)
                done_matches = re.findall(r'"done"\s*:\s*(true|false)', clean)
                d = {"action": m.group(1) if m else "", "done": bool(done_matches and done_matches[-1]=="true")}

            action = d.get("action", "")
            done   = bool(d.get("done", False))
            # Only block done=true if no tools have been called yet.
            if done and self._tools_called == 0:
                done = False
                if not action and tools:
                    action = list(tools.keys())[0] + "()"

        except Exception as e:
            error = str(e)

        latency = time.perf_counter() - t0
        thought = d.get("thought", "") if 'd' in dir() else ""
        self.llm_traces.append(LLMCallTrace(
            call_index=self._call_idx,
            step_index=len(world.completed_steps),
            prompt_summary=user_msg[:500],
            raw_response=raw[:1000],
            reasoning=thought,
            predicted_steps=[],
            next_action=action,
            done=done,
            input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=cost,
            latency_s=round(latency, 3), error=error,
        ))
        self._call_idx += 1
        return thought, [], action, done

    def execute_action(self, action, tools, env):
        result, errored = _dispatch_action(action, tools, env)
        if not errored and "no tool matched" not in str(result):
            self._tools_called += 1
        return result, errored


class TracingOpenRouterDRAdapter(LLMAdapter):
    """DR adapter via OpenRouter — for free models."""
    def __init__(self, api_key: str, model: str, n_required: int = 1):
        self.api_key     = api_key
        self.model       = model
        self.n_required  = n_required
        self._tools_called = 0
        self.llm_traces: list[LLMCallTrace] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self._call_idx = 0

    def get_fix(self, world, tools):
        from openai import OpenAI
        tool_names = ", ".join(list(tools.keys())[:25])
        system = _DR_SYSTEM.format(tool_names=tool_names)
        prompt = world.summary()
        client = OpenAI(api_key=self.api_key,
                        base_url="https://openrouter.ai/api/v1",
                        default_headers={"HTTP-Referer": "https://github.com/soham311595/dead-reckoning-agent"})
        t0 = time.perf_counter()
        raw = error = ""
        in_tok = out_tok = 0
        result = ("", [], "", False)
        try:
            resp = client.chat.completions.create(
                model=self.model, max_tokens=600,
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": prompt}],
            )
            msg = resp.choices[0].message
            raw = msg.content or getattr(msg, "reasoning", None) or ""
            if hasattr(resp, "usage") and resp.usage:
                in_tok  = resp.usage.prompt_tokens or 0
                out_tok = resp.usage.completion_tokens or 0
            self.input_tokens  += in_tok
            self.output_tokens += out_tok
            r, ps, na, dn = _parse_fix_response(raw)
            # Only block done=true if no tools have been called yet at all
            # (prevents quitting before doing anything).
            # Do NOT enforce n_required — the model correctly judges task completion.
            if dn and self._tools_called == 0:
                dn = False
                if not na and tools: na = list(tools.keys())[0] + "()"
            result = (r, ps, na, dn)
        except Exception as e:
            error = str(e)

        latency = time.perf_counter() - t0
        self.llm_traces.append(LLMCallTrace(
            call_index=self._call_idx, step_index=len(world.completed_steps),
            prompt_summary=prompt[:500], raw_response=raw[:1000],
            reasoning=result[0], predicted_steps=result[1],
            next_action=result[2], done=result[3],
            input_tokens=in_tok, output_tokens=out_tok, cost_usd=0.0,
            latency_s=round(latency, 3), error=error,
        ))
        self._call_idx += 1
        return result

    def execute_action(self, action, tools, env):
        result, errored = _dispatch_action(action, tools, env)
        if not errored and "no tool matched" not in str(result): self._tools_called += 1
        return result, errored


class TracingOpenRouterReActAdapter(LLMAdapter):
    def __init__(self, api_key: str, model: str, goal: str, n_required: int = 1):
        self.api_key     = api_key
        self.model       = model
        self.goal        = goal
        self.n_required  = n_required
        self._tools_called = 0
        self.llm_traces: list[LLMCallTrace] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self._call_idx = 0

    def get_fix(self, world, tools):
        from openai import OpenAI
        tool_names = ", ".join(list(tools.keys())[:25])
        completed = world.completed_steps
        completed_tools = list(set(s["action"].split("(")[0].strip() for s in completed)) if completed else []
        system = _REACT_SYSTEM.format(
            tool_names=tool_names,
            goal=self.goal,
            completed_tools=completed_tools if completed_tools else "none yet",
        )
        if completed:
            history_lines = "\n".join(
                f"  {i+1}. {s['action']} → success"
                for i, s in enumerate(completed[-10:])
            )
            user_msg = (
                f"Steps done ({len(completed)} total):\n{history_lines}\n\n"
                f"Tools already called: {completed_tools}\n"
                f"What is the next action? If all required tools are done, set done=true."
            )
        else:
            user_msg = f"Goal: {self.goal}\nFirst action?"
        client = OpenAI(api_key=self.api_key,
                        base_url="https://openrouter.ai/api/v1",
                        default_headers={"HTTP-Referer": "https://github.com/soham311595/dead-reckoning-agent"})
        t0 = time.perf_counter()
        raw = error = ""
        in_tok = out_tok = 0
        action, done = "", False
        thought = ""
        try:
            resp = client.chat.completions.create(
                model=self.model, max_tokens=400,
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": user_msg}],
            )
            msg = resp.choices[0].message
            raw = msg.content or getattr(msg, "reasoning", None) or ""
            if hasattr(resp, "usage") and resp.usage:
                in_tok  = resp.usage.prompt_tokens or 0
                out_tok = resp.usage.completion_tokens or 0
            self.input_tokens  += in_tok
            self.output_tokens += out_tok
            clean = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
            try:
                d = json.loads(clean)
            except Exception:
                m = re.search(r'"action"\s*:\s*"([^"]+)"', clean)
                d = {"action": m.group(1) if m else "", "done": False}
            action = d.get("action", "")
            done   = bool(d.get("done", False))
            thought = d.get("thought", "")
            # Only block done=true if no tools have been called yet.
            if done and self._tools_called == 0:
                done = False
                if not action and tools:
                    action = list(tools.keys())[0] + "()"
        except Exception as e:
            error = str(e)

        latency = time.perf_counter() - t0
        self.llm_traces.append(LLMCallTrace(
            call_index=self._call_idx, step_index=len(world.completed_steps),
            prompt_summary=user_msg[:500], raw_response=raw[:1000],
            reasoning=thought, predicted_steps=[], next_action=action, done=done,
            input_tokens=in_tok, output_tokens=out_tok, cost_usd=0.0,
            latency_s=round(latency, 3), error=error,
        ))
        self._call_idx += 1
        return thought, [], action, done

    def execute_action(self, action, tools, env):
        result, errored = _dispatch_action(action, tools, env)
        if not errored and "no tool matched" not in str(result):
            self._tools_called += 1
        return result, errored


# ================================================================== #
#  Statistics                                                          #
# ================================================================== #

def bootstrap_ci(values: list[float], n: int = 1000, ci: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, lower_ci, upper_ci) via bootstrap."""
    if not values:
        return 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, mean, mean
    rng = random.Random(42)
    boot_means = []
    for _ in range(n):
        sample = [rng.choice(values) for _ in values]
        boot_means.append(sum(sample) / len(sample))
    boot_means.sort()
    lo = int((1 - ci) / 2 * n)
    hi = int((1 + ci) / 2 * n)
    return mean, boot_means[lo], boot_means[hi]


def compute_stats(traces: list[TaskTrace]) -> dict:
    """Compute all statistics needed for a paper table."""
    by_method: dict[str, list[TaskTrace]] = defaultdict(list)
    for t in traces:
        by_method[t.method].append(t)

    stats = {}
    for method, rows in by_method.items():
        n = len(rows)
        gt_vals   = [float(r.success_gt)  for r in rows]
        llm_vals  = [float(r.success_llm) for r in rows]
        llm_calls = [float(r.llm_calls)   for r in rows]
        sav_vals  = [float(r.savings_pct) for r in rows]
        tok_vals  = [float(r.input_tokens + r.output_tokens) for r in rows]
        cost_vals = [float(r.cost_usd)    for r in rows]
        time_vals = [float(r.wall_time_s) for r in rows]
        tools_frac= [r.tools_called / max(r.tools_required, 1) for r in rows]

        gt_mean,   gt_lo,   gt_hi   = bootstrap_ci(gt_vals)
        llm_mean,  llm_lo,  llm_hi  = bootstrap_ci(llm_vals)
        lcall_mean,lcall_lo,lcall_hi = bootstrap_ci(llm_calls)
        sav_mean,  sav_lo,  sav_hi  = bootstrap_ci(sav_vals)
        tok_mean,  tok_lo,  tok_hi  = bootstrap_ci(tok_vals)
        tools_mean,tools_lo,tools_hi = bootstrap_ci(tools_frac)

        # Stop reason breakdown
        stop_counts: dict[str, int] = defaultdict(int)
        for r in rows:
            stop_counts[r.stop_reason] += 1

        # Per-split breakdown
        splits: dict[str, dict] = {}
        for split in sorted(set(r.split for r in rows)):
            split_rows = [r for r in rows if r.split == split]
            sg = sum(r.success_gt for r in split_rows) / len(split_rows) * 100
            sl = sum(r.llm_calls  for r in split_rows) / len(split_rows)
            splits[split] = {"n": len(split_rows), "gt_pct": round(sg, 1), "llm_per_task": round(sl, 2)}

        stats[method] = {
            "n": n,
            "gt_success_pct":   {"mean": round(gt_mean*100, 1),   "ci95_lo": round(gt_lo*100, 1),   "ci95_hi": round(gt_hi*100, 1)},
            "llm_success_pct":  {"mean": round(llm_mean*100, 1),  "ci95_lo": round(llm_lo*100, 1),  "ci95_hi": round(llm_hi*100, 1)},
            "llm_calls_per_task":{"mean": round(lcall_mean, 2),   "ci95_lo": round(lcall_lo, 2),    "ci95_hi": round(lcall_hi, 2)},
            "savings_pct":       {"mean": round(sav_mean, 1),     "ci95_lo": round(sav_lo, 1),      "ci95_hi": round(sav_hi, 1)},
            "tokens_per_task":   {"mean": round(tok_mean, 0),     "ci95_lo": round(tok_lo, 0),      "ci95_hi": round(tok_hi, 0)},
            "tools_coverage_pct":{"mean": round(tools_mean*100,1),"ci95_lo": round(tools_lo*100,1), "ci95_hi": round(tools_hi*100,1)},
            "total_cost_usd":    round(sum(cost_vals), 4),
            "avg_cost_usd":      round(sum(cost_vals)/n, 5) if n else 0,
            "avg_wall_time_s":   round(sum(time_vals)/n, 2) if n else 0,
            "stop_reasons":      dict(stop_counts),
            "per_split":         splits,
        }

    # Comparative stats
    if "dead_reckoning" in stats and "react" in stats:
        dr = by_method["dead_reckoning"]
        re = by_method["react"]
        dr_llm   = sum(t.llm_calls for t in dr) / len(dr)
        re_llm   = sum(t.llm_calls for t in re) / len(re)
        dr_tok   = sum(t.input_tokens+t.output_tokens for t in dr) / len(dr)
        re_tok   = sum(t.input_tokens+t.output_tokens for t in re) / len(re)
        dr_gt    = sum(t.success_gt for t in dr) / len(dr)
        re_gt    = sum(t.success_gt for t in re) / len(re)

        stats["_comparison"] = {
            "llm_call_reduction_pct":  round((1 - dr_llm/re_llm)*100, 1) if re_llm else 0,
            "token_reduction_pct":     round((1 - dr_tok/re_tok)*100, 1) if re_tok else 0,
            "gt_success_delta_pp":     round((dr_gt - re_gt)*100, 1),
            "dr_llm_per_task":         round(dr_llm, 2),
            "react_llm_per_task":      round(re_llm, 2),
            "dr_tokens_per_task":      round(dr_tok, 0),
            "react_tokens_per_task":   round(re_tok, 0),
        }
    return stats


# ================================================================== #
#  Report generator                                                    #
# ================================================================== #

def generate_report(traces: list[TaskTrace], stats: dict, run_id: str, args) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    comp = stats.get("_comparison", {})
    dr   = stats.get("dead_reckoning", {})
    re_  = stats.get("react", {})
    n    = dr.get("n", len(traces) // 2)

    lines = [
        f"# Dead Reckoning Agent — Evaluation Report",
        f"",
        f"**Run ID:** `{run_id}`  ",
        f"**Date:** {now}  ",
        f"**Model:** `{args.model}`  ",
        f"**Provider:** {args.provider}  ",
        f"**Tasks:** {n} × 2 methods = {n*2} total runs  ",
        f"**Splits:** {args.split}  ",
        f"",
        f"---",
        f"",
        f"## Main Results",
        f"",
        f"| Method | GT Success | 95% CI | LLM calls/task | 95% CI | Det. savings |",
        f"|--------|:----------:|:------:|:--------------:|:------:|:------------:|",
    ]

    for method, label in [("dead_reckoning", "**Dead Reckoning**"), ("react", "ReAct (baseline)")]:
        s = stats.get(method, {})
        if not s:
            continue
        gt   = s["gt_success_pct"]
        lc   = s["llm_calls_per_task"]
        sv   = s["savings_pct"]
        lines.append(
            f"| {label} | {gt['mean']}% | [{gt['ci95_lo']}%, {gt['ci95_hi']}%] "
            f"| {lc['mean']} | [{lc['ci95_lo']}, {lc['ci95_hi']}] "
            f"| {sv['mean']}% |"
        )

    lines += [
        f"",
        f"> GT Success = all required APIs from ToolBench `api_list` were called.  ",
        f"> 95% CIs computed via bootstrap (n=1000).  ",
        f"> n={n} tasks per method.",
        f"",
        f"### Comparative summary",
        f"",
    ]

    if comp:
        lines += [
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| LLM call reduction | **{comp.get('llm_call_reduction_pct', '?')}%** |",
            f"| Token reduction | **{comp.get('token_reduction_pct', '?')}%** |",
            f"| GT success delta | **{comp.get('gt_success_delta_pp', '?')} pp** |",
            f"| DR LLM calls/task | {comp.get('dr_llm_per_task', '?')} |",
            f"| ReAct LLM calls/task | {comp.get('react_llm_per_task', '?')} |",
            f"| DR tokens/task | {comp.get('dr_tokens_per_task', '?')} |",
            f"| ReAct tokens/task | {comp.get('react_tokens_per_task', '?')} |",
        ]

    lines += ["", "---", "", "## Per-Split Breakdown", ""]

    for method in ["dead_reckoning", "react"]:
        s = stats.get(method, {})
        if not s:
            continue
        lines.append(f"### {method}")
        lines.append(f"| Split | n | GT% | LLM/task |")
        lines.append(f"|-------|---|-----|----------|")
        for split, sv in sorted(s.get("per_split", {}).items()):
            lines.append(f"| {split} | {sv['n']} | {sv['gt_pct']}% | {sv['llm_per_task']} |")
        lines.append("")

    lines += ["---", "", "## Stop Reason Breakdown", ""]
    for method in ["dead_reckoning", "react"]:
        s = stats.get(method, {})
        if not s:
            continue
        lines.append(f"**{method}:**")
        for reason, count in sorted(s.get("stop_reasons", {}).items()):
            pct = round(count / s["n"] * 100)
            lines.append(f"- `{reason}`: {count} ({pct}%)")
        lines.append("")

    lines += [
        "---", "", "## Cost & Latency", "",
        f"| Method | Total cost | Cost/task | Avg wall time/task |",
        f"|--------|:----------:|:---------:|:-----------------:|",
    ]
    for method in ["dead_reckoning", "react"]:
        s = stats.get(method, {})
        if not s:
            continue
        lines.append(
            f"| {method} | ${s['total_cost_usd']} | ${s['avg_cost_usd']} | {s['avg_wall_time_s']}s |"
        )

    lines += [
        "", "---", "", "## Per-Task Results", "",
        "| Task | Split | Method | GT | LLM | Steps | LLM calls | Tools | Goal |",
        "|------|-------|--------|----|-----|-------|-----------|-------|------|",
    ]
    for t in sorted(traces, key=lambda x: (x.task_id, x.method)):
        gt  = "✓" if t.success_gt  else "✗"
        llm = "✓" if t.success_llm else "✗"
        lines.append(
            f"| {t.task_id} | {t.split} | {t.method} | {gt} | {llm} "
            f"| {t.total_steps} | {t.llm_calls} "
            f"| {t.tools_called}/{t.tools_required} "
            f"| {t.goal[:40]}... |"
        )

    lines += [
        "", "---", "", "## Methodology Notes", "",
        "- **Success metric**: ground truth = all APIs in `api_list` called at least once.",
        "  This is a proxy — it measures tool coverage, not answer correctness.",
        "- **LLM self-report** (`success_llm`) shown for comparison but not used as primary metric.",
        f"- **Sample size**: n={n} tasks. Small sample — CIs are wide. Treat as pilot results.",
        "- **Tools**: Mock implementations returning cached responses from `toolenv/response_examples/`.",
        "  Real API calls not made — avoids rate limits and costs.",
        "- **ReAct baseline**: full conversation history passed on each call; done=true blocked",
        "  until at least one tool called.",
        "- **Limitations**: Tasks with URL-style API names (e.g. `/tracking/:id`) require",
        "  sanitization that may lose semantic information. G2/G3 tasks with 7+ APIs",
        "  require planning horizons beyond what Haiku reliably achieves.",
        "", "---", "",
        f"*Generated by Dead Reckoning Agent eval harness — run `{run_id}`*",
    ]

    return "\n".join(lines)


# ================================================================== #
#  Main runner                                                         #
# ================================================================== #

def run_task(
    task: dict,
    provider: str,
    api_key: str,
    model: str,
    method: str,
    response_examples: dict,
    fix_threshold: float,
    max_steps: int,
    run_id: str,
) -> TaskTrace:
    goal     = task["query"]
    task_id  = task["_id"]
    split    = task["_split"]
    api_list = [a["api_name"] for a in task.get("api_list", [])]

    tools, called_log, name_map = build_tools_for_task(task, response_examples)
    n_required = len({api["api_name"] for api in task.get("api_list", [])})

    t0 = time.perf_counter()
    error = ""
    step_traces: list[StepTrace] = []

    # Build adapter
    if provider == "anthropic":
        if method == "dead_reckoning":
            adapter = TracingAnthropicDRAdapter(api_key=api_key, model=model, n_required=n_required)
        else:
            adapter = TracingAnthropicReActAdapter(api_key=api_key, model=model, goal=goal, n_required=n_required)
    else:  # openrouter
        if method == "dead_reckoning":
            adapter = TracingOpenRouterDRAdapter(api_key=api_key, model=model, n_required=n_required)
        else:
            adapter = TracingOpenRouterReActAdapter(api_key=api_key, model=model, goal=goal, n_required=n_required)

    try:
        if method == "dead_reckoning":
            agent = DeadReckoningAgent(
                adapter=adapter, goal=goal, tools=tools,
                fix_threshold=fix_threshold,
                max_steps_without_fix=5, max_total_steps=max_steps,
                verbose=False,
            )
        else:
            agent = DeadReckoningAgent(
                adapter=adapter, goal=goal, tools=tools,
                fix_threshold=0.0, hard_ceiling=0.01,
                max_steps_without_fix=1, checkpoint_interval=1,
                max_total_steps=max_steps, verbose=False,
            )

        for step in agent.run():
            step_traces.append(StepTrace(
                step_index=step.step_index,
                mode="llm_fix" if step.llm_call_made else "deterministic",
                action=step.action or "",
                result=str(step.result)[:200] if step.result else "",
                errored=False,  # StepResult has no errored field; errors handled internally
                confidence=round(step.confidence, 3),
                drift=round(step.drift, 3),
                llm_call_made=bool(step.llm_call_made),
            ))

        stats = agent.stats

    except Exception as e:
        error = str(e)
        stats = agent.stats if "agent" in dir() else None

    wall_time = time.perf_counter() - t0
    llm_done   = (stats.stop_reason == StopReason.TASK_COMPLETE) if stats else False
    gt_success = evaluate_success(task, called_log)

    return TaskTrace(
        task_id=task_id, split=split, goal=goal[:120],
        method=method, model=model, provider=provider,
        run_id=run_id,
        timestamp=datetime.datetime.now().isoformat(),
        api_list=api_list,
        toolbench_win=None,
        success_gt=gt_success,
        success_llm=llm_done,
        stop_reason=stats.stop_reason.value if (stats and stats.stop_reason) else "?",
        total_steps=stats.total_steps if stats else 0,
        llm_calls=stats.llm_calls if stats else 0,
        det_steps=(stats.total_steps - stats.llm_calls) if stats else 0,
        savings_pct=stats.savings_pct if stats else 0.0,
        tools_required=n_required,
        tools_called=len(set(called_log)),
        tools_called_list=list(set(called_log)),
        input_tokens=adapter.input_tokens,
        output_tokens=adapter.output_tokens,
        cost_usd=adapter.cost_usd,
        wall_time_s=round(wall_time, 2),
        steps=[asdict(s) for s in step_traces],
        llm_calls_log=[asdict(t) for t in adapter.llm_traces],
        error=error,
    )


def save_results(traces: list[TaskTrace], stats: dict, out_dir: Path, args):
    """Save all outputs in a structured directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(exist_ok=True)

    # 1. Full traces — one JSON per task×method
    for t in traces:
        fname = traces_dir / f"{t.task_id}_{t.method}.json"
        with open(fname, "w") as f:
            json.dump(asdict(t), f, indent=2, default=str)

    # 2. Per-task metrics CSV (flat, for pandas/Excel analysis)
    csv_path = out_dir / "metrics.csv"
    flat_fields = [
        "task_id", "split", "goal", "method", "model", "provider",
        "success_gt", "success_llm", "stop_reason",
        "total_steps", "llm_calls", "det_steps", "savings_pct",
        "tools_required", "tools_called",
        "input_tokens", "output_tokens", "cost_usd", "wall_time_s", "error",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_fields)
        w.writeheader()
        for t in traces:
            row = {k: getattr(t, k) for k in flat_fields}
            row["goal"] = row["goal"][:80]
            w.writerow(row)

    # 3. Summary JSON — all stats + CIs
    summary = {
        "run_id":    traces[0].run_id if traces else "",
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "model":         args.model,
            "provider":      args.provider,
            "split":         args.split,
            "fix_threshold": args.fix_threshold,
            "max_steps":     args.max_steps,
            "n_tasks":       len(set(t.task_id for t in traces)),
        },
        "stats": stats,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 4. Human-readable paper report
    report = generate_report(traces, stats, traces[0].run_id if traces else "", args)
    with open(out_dir / "report.md", "w") as f:
        f.write(report)

    # 5. LLM call log — all raw prompts and responses in one file (for qualitative analysis)
    llm_log = []
    for t in traces:
        for call in t.llm_calls_log:
            llm_log.append({
                "task_id": t.task_id, "method": t.method,
                "split": t.split, **call,
            })
    with open(out_dir / "llm_call_log.json", "w") as f:
        json.dump(llm_log, f, indent=2, default=str)

    print(f"\nSaved to {out_dir}/")
    print(f"  traces/          — {len(traces)} full task traces")
    print(f"  metrics.csv      — per-task numbers")
    print(f"  summary.json     — aggregate stats + 95% CIs")
    print(f"  report.md        — paper-ready report")
    print(f"  llm_call_log.json— all raw LLM calls")


def main():
    parser = argparse.ArgumentParser(description="Dead Reckoning full evaluation harness")
    parser.add_argument("--toolbench_dir",  required=True)
    parser.add_argument("--split",          default="all", choices=["G1", "G2", "G3", "all"])
    parser.add_argument("--provider",       default="anthropic", choices=["anthropic", "openrouter"])
    parser.add_argument("--model",          default="claude-haiku-4-5")
    parser.add_argument("--api_key",        default="")
    parser.add_argument("--fix_threshold",  type=float, default=0.35)
    parser.add_argument("--max_steps",      type=int,   default=20)
    parser.add_argument("--methods",        default="both", choices=["dr", "react", "both"])
    parser.add_argument("--output_dir",     default="results")
    parser.add_argument("--run_id",         default="",
                        help="Resume or name a run. Auto-generated if not set.")
    parser.add_argument("--delay",          type=float, default=0.3,
                        help="Seconds between calls (increase for free tier rate limits)")
    parser.add_argument("--n_tasks",        type=int,   default=0,
                        help="Limit tasks per split (0 = all)")
    args = parser.parse_args()

    # API key
    key = args.api_key
    if not key:
        env_var = "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENROUTER_API_KEY"
        key = os.environ.get(env_var, "")
    if not key:
        print(f"ERROR: set {env_var} or pass --api_key")
        sys.exit(1)

    if args.provider == "openrouter":
        try:
            import openai
        except ImportError:
            print("ERROR: pip3 install openai")
            sys.exit(1)

    # Run ID
    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check for partial results to resume
    existing = list((out_dir / "traces").glob("*.json")) if (out_dir / "traces").exists() else []
    done_keys = set()
    all_traces: list[TaskTrace] = []
    if existing:
        print(f"Resuming run {run_id} — found {len(existing)} existing traces")
        for f in existing:
            try:
                d = json.load(open(f))
                all_traces.append(TaskTrace(**{k: d[k] for k in TaskTrace.__dataclass_fields__}))
                done_keys.add(f"{d['task_id']}_{d['method']}")
            except Exception:
                pass

    toolbench_dir     = os.path.expanduser(args.toolbench_dir)
    response_examples = load_response_examples(toolbench_dir)
    splits            = ["G1", "G2", "G3"] if args.split == "all" else [args.split]
    methods           = (["dead_reckoning", "react"] if args.methods == "both"
                         else [{"dr": "dead_reckoning", "react": "react"}[args.methods]])

    print(f"\nRun ID  : {run_id}")
    print(f"Model   : {args.model}  ({args.provider})")
    print(f"Output  : {out_dir}/")
    print(f"Splits  : {splits}")
    print(f"Delay   : {args.delay}s between calls")
    print(f"Cached responses: {len(response_examples)}")
    if existing:
        print(f"Skipping: {len(done_keys)} already done")

    total_tasks = 0
    for split in splits:
        try:
            tasks = load_tasks(toolbench_dir, split)
        except FileNotFoundError as e:
            print(f"\nSKIP {split}: {e}")
            continue

        if args.n_tasks:
            tasks = tasks[:args.n_tasks]

        print(f"\n{'─'*60}")
        print(f"  {split} — {len(tasks)} tasks × {len(methods)} methods")

        for i, task in enumerate(tasks):
            goal = task["query"]
            print(f"\n  [{i+1}/{len(tasks)}] {goal[:58]}...")

            for method in methods:
                key_str = f"{task['_id']}_{method}"
                if key_str in done_keys:
                    print(f"    {method:<16}: SKIP (already done)")
                    continue

                try:
                    trace = run_task(
                        task=task, provider=args.provider,
                        api_key=key, model=args.model,
                        method=method,
                        response_examples=response_examples,
                        fix_threshold=args.fix_threshold,
                        max_steps=args.max_steps,
                        run_id=run_id,
                    )
                    all_traces.append(trace)
                    total_tasks += 1

                    gt  = "✓" if trace.success_gt  else "✗"
                    llm = "✓" if trace.success_llm else "✗"
                    print(f"    {method:<16}: GT={gt} LLM={llm}  "
                          f"steps={trace.total_steps}  llm={trace.llm_calls}  "
                          f"tools={trace.tools_called}/{trace.tools_required}  "
                          f"tok={trace.input_tokens+trace.output_tokens}  "
                          f"${trace.cost_usd:.4f}")

                    # Save after every task (crash-safe)
                    stats = compute_stats(all_traces)
                    save_results(all_traces, stats, out_dir, args)

                except Exception as e:
                    print(f"    {method}: ERROR — {e}")
                    import traceback; traceback.print_exc()

                time.sleep(args.delay)

    # Final summary
    if all_traces:
        stats = compute_stats(all_traces)
        save_results(all_traces, stats, out_dir, args)

        comp = stats.get("_comparison", {})
        print(f"\n{'='*60}")
        print(f"  FINAL RESULTS  (run {run_id})")
        print(f"{'='*60}")
        for method in ["dead_reckoning", "react"]:
            s = stats.get(method, {})
            if not s: continue
            gt = s["gt_success_pct"]
            lc = s["llm_calls_per_task"]
            print(f"  {method:<16}: GT={gt['mean']}% [{gt['ci95_lo']}-{gt['ci95_hi']}%]  "
                  f"LLM={lc['mean']} [{lc['ci95_lo']}-{lc['ci95_hi']}]/task")
        if comp:
            print(f"\n  LLM reduction  : {comp.get('llm_call_reduction_pct','?')}%")
            print(f"  Token reduction: {comp.get('token_reduction_pct','?')}%")
            print(f"  GT delta       : {comp.get('gt_success_delta_pp','?')} pp")
        print(f"\n  Results in: {out_dir}/")
        print(f"  Read the report: cat {out_dir}/report.md")


if __name__ == "__main__":
    main()
