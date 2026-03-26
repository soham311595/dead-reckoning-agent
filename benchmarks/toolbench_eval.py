"""
ToolBench Benchmark Harness
============================
Compares Dead Reckoning Agent vs a vanilla ReAct baseline on ToolBench tasks.
Uses OpenRouter so you can run on free models with no API cost.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SETUP (one-time)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Clone StableToolBench (virtual API server, no real API keys needed):

    git clone https://github.com/OpenBMB/ToolBench
    cd ToolBench && pip3 install -r requirements.txt

2. Download the data (Google Drive link in the ToolBench README):
    https://drive.google.com/drive/folders/1yBUQ732mPu-KclJnuQELEhtKakdXFc3J
    Unzip into ToolBench/data/

3. Start the virtual API server in a separate terminal:
    python toolbench/tooleval/server.py --tool_root_dir data/toolenv/tools/

4. Get a free OpenRouter key: https://openrouter.ai/keys

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    cd dead-reckoning-agent

    # Quick start — G1 split, 50 tasks, free Llama
    OPENROUTER_API_KEY=sk-or-... python3 benchmarks/toolbench_eval.py \
        --toolbench_dir ../ToolBench \
        --split G1 \
        --n_tasks 50 \
        --output results/G1.csv

    # Full paper run — all three splits
    OPENROUTER_API_KEY=sk-or-... python3 benchmarks/toolbench_eval.py \
        --toolbench_dir ../ToolBench \
        --split all \
        --n_tasks 100 \
        --output results/all.csv

Splits:
    G1 = I1 single-tool        best DR savings expected
    G2 = I2 intra-category     medium
    G3 = I3 multi-tool         hardest

Free models to try (best first):
    meta-llama/llama-3.3-70b-instruct:free   <- default
    deepseek/deepseek-r1:free
    google/gemma-3-27b-it:free
    mistralai/mistral-7b-instruct:free
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

from dead_reckoning import DeadReckoningAgent
from dead_reckoning.adapters import OpenRouterAdapter, _dispatch_action, _FIX_SYSTEM, _parse_fix_response
from dead_reckoning.core.agent import LLMAdapter, StopReason
from dead_reckoning.core.world_model import WorldModel


# ------------------------------------------------------------------ #
#  Data loading                                                        #
# ------------------------------------------------------------------ #

def load_tasks(toolbench_dir: str, split: str, n_tasks: int) -> list[dict]:
    data_dir   = Path(toolbench_dir) / "data"
    query_file = data_dir / "instruction" / f"{split}_query.json"
    id_file    = data_dir / "test_query_ids" / f"{split}_instruction_test_query_ids.json"

    if not query_file.exists():
        raise FileNotFoundError(
            f"Query file not found: {query_file}\n"
            "Download ToolBench data — see setup at top of file."
        )

    with open(query_file) as f:
        all_queries = json.load(f)

    if id_file.exists():
        with open(id_file) as f:
            test_ids = set(json.load(f))
        queries = [q for q in all_queries if q.get("query_id") in test_ids]
    else:
        queries = all_queries

    return queries[:n_tasks]


def load_tools_for_task(toolbench_dir: str, task: dict) -> dict[str, dict]:
    tools_dir = Path(toolbench_dir) / "data" / "toolenv" / "tools"
    result = {}
    for api_info in task.get("api_list", []):
        tool_name = api_info.get("tool_name", "")
        category  = api_info.get("category_name", "")
        tool_dir  = tools_dir / category / tool_name
        if tool_dir.exists():
            for fn_file in tool_dir.glob("*.json"):
                try:
                    with open(fn_file) as f:
                        fn_def = json.load(f)
                    fn_def["tool_name"] = tool_name
                    result[fn_file.stem] = fn_def
                except Exception:
                    pass
    return result


# ------------------------------------------------------------------ #
#  Virtual API server dispatcher                                       #
# ------------------------------------------------------------------ #

class VirtualAPIDispatcher:
    def __init__(self, server_url: str = "http://localhost:5000"):
        self.server_url = server_url.rstrip("/")
        self._call_log: list[dict] = []

    def make_tool_fn(self, tool_name: str, api_name: str) -> Callable:
        def call_tool(**kwargs) -> Any:
            payload = {"tool_name": tool_name, "api_name": api_name, "tool_input": kwargs}
            try:
                resp = requests.post(f"{self.server_url}/virtual", json=payload, timeout=10)
                self._call_log.append({"tool": f"{tool_name}.{api_name}", "status": resp.status_code})
                return resp.json().get("response", "") if resp.status_code == 200 else f"[API error {resp.status_code}]"
            except requests.exceptions.ConnectionError:
                return "[virtual server offline — run: python toolbench/tooleval/server.py]"
            except Exception as e:
                return f"[error: {e}]"
        call_tool.__name__ = api_name
        return call_tool

    def build_tools(self, tool_defs: dict[str, dict]) -> dict[str, Callable]:
        tools = {}
        for fn_name, fn_def in tool_defs.items():
            tool_name = fn_def.get("tool_name", fn_name.split(".")[0])
            api_name  = fn_def.get("name", fn_name)
            fn = self.make_tool_fn(tool_name, api_name)
            tools[api_name] = fn
            tools[f"{tool_name}.{api_name}"] = fn
        return tools

    def reset_log(self):
        self._call_log = []


# ------------------------------------------------------------------ #
#  Counting wrapper for Dead Reckoning adapter                        #
# ------------------------------------------------------------------ #

class CountingOpenRouterAdapter(OpenRouterAdapter):
    """OpenRouterAdapter that also tracks token usage from the response."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_tokens  = 0
        self.output_tokens = 0

    def get_fix(self, world: WorldModel, tools: dict) -> tuple[str, list[str], str, bool]:
        from openai import OpenAI
        tool_names = list(tools.keys()) if tools else ["none"]
        system = _FIX_SYSTEM.format(
            n_predictions=self.n_predictions,
            tool_names=", ".join(tool_names[:30]),
        )
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=self.extra_headers,
        )
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": world.summary()},
            ],
        )
        if hasattr(resp, "usage") and resp.usage:
            self.input_tokens  += resp.usage.prompt_tokens or 0
            self.output_tokens += resp.usage.completion_tokens or 0
        return _parse_fix_response(resp.choices[0].message.content or "")


# ------------------------------------------------------------------ #
#  ReAct baseline adapter (OpenRouter, no predictions)                #
# ------------------------------------------------------------------ #

_REACT_SYSTEM = """You are an agent solving a task using tools.

At each step respond ONLY with valid JSON — no markdown, no text outside the JSON:
{{
  "thought": "brief reasoning",
  "action": "tool_name(param='value')",
  "done": false
}}

Set "done": true and "action": "" when the task is fully complete.
Available tools: {tool_names}
Task: {goal}
"""

class OpenRouterReActAdapter(LLMAdapter):
    """Vanilla ReAct — every step calls the LLM, no predictions. Used as baseline."""

    def __init__(self, api_key: str, model: str, goal: str, tool_names: list[str]):
        self.api_key    = api_key
        self.model      = model
        self.goal       = goal
        self.tool_names = tool_names
        self.input_tokens  = 0
        self.output_tokens = 0

    def _client(self):
        from openai import OpenAI
        return OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "https://github.com/dead-reckoning-agent"},
        )

    def get_fix(self, world: WorldModel, tools: dict) -> tuple[str, list[str], str, bool]:
        system = _REACT_SYSTEM.format(
            tool_names=", ".join(self.tool_names[:25]),
            goal=self.goal,
        )
        resp = self._client().chat.completions.create(
            model=self.model,
            max_tokens=512,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": world.summary()},
            ],
        )
        if hasattr(resp, "usage") and resp.usage:
            self.input_tokens  += resp.usage.prompt_tokens or 0
            self.output_tokens += resp.usage.completion_tokens or 0

        text = re.sub(r"```(?:json)?\s*", "", resp.choices[0].message.content or "").strip().rstrip("`")
        try:
            data = json.loads(text)
            return data.get("thought", ""), [], data.get("action", ""), bool(data.get("done", False))
        except json.JSONDecodeError:
            m = re.search(r'"action"\s*:\s*"([^"]+)"', text)
            return "parse error", [], m.group(1) if m else "", False

    def execute_action(self, action: str, tools: dict, env: dict) -> tuple[Any, bool]:
        return _dispatch_action(action, tools, env)


# ------------------------------------------------------------------ #
#  Task result                                                         #
# ------------------------------------------------------------------ #

@dataclass
class TaskResult:
    task_id:       str
    split:         str
    instruction:   str
    method:        str
    model:         str
    success:       bool
    stop_reason:   str
    total_steps:   int
    llm_calls:     int
    llm_call_rate: float
    savings_pct:   float
    input_tokens:  int
    output_tokens: int
    cost_usd:      float
    duration_s:    float
    error:         str = ""


# ------------------------------------------------------------------ #
#  Cost table                                                          #
# ------------------------------------------------------------------ #

COST_PER_1K: dict[str, dict] = {
    "meta-llama/llama-3.3-70b-instruct:free": {"in": 0.0,      "out": 0.0},
    "deepseek/deepseek-r1:free":              {"in": 0.0,      "out": 0.0},
    "google/gemma-3-27b-it:free":             {"in": 0.0,      "out": 0.0},
    "mistralai/mistral-7b-instruct:free":     {"in": 0.0,      "out": 0.0},
    "meta-llama/llama-3.3-70b-instruct":      {"in": 0.00012,  "out": 0.00030},
    "deepseek/deepseek-chat":                 {"in": 0.00014,  "out": 0.00028},
    "claude-haiku-4-5":                       {"in": 0.00025,  "out": 0.00125},
    "gpt-4o-mini":                            {"in": 0.00015,  "out": 0.00060},
    "gpt-4o":                                 {"in": 0.005,    "out": 0.015},
}

def estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    r = COST_PER_1K.get(model, {"in": 0.001, "out": 0.002})
    return (in_tok * r["in"] + out_tok * r["out"]) / 1000


# ------------------------------------------------------------------ #
#  Run one task                                                        #
# ------------------------------------------------------------------ #

def run_task_dead_reckoning(task, tools, api_key, model, split, fix_threshold, max_steps) -> TaskResult:
    goal    = task["query"]
    task_id = str(task.get("query_id", task.get("id", "unknown")))

    adapter = CountingOpenRouterAdapter(api_key=api_key, model=model, n_predictions=4, max_tokens=512)
    agent   = DeadReckoningAgent(
        adapter=adapter, goal=goal, tools=tools,
        fix_threshold=fix_threshold, max_steps_without_fix=5,
        checkpoint_interval=12, max_total_steps=max_steps, verbose=False,
    )

    t0, error = time.perf_counter(), ""
    try:
        list(agent.run())
    except Exception as e:
        error = str(e)

    stats = agent.stats
    return TaskResult(
        task_id=task_id, split=split, instruction=goal[:120],
        method="dead_reckoning", model=model,
        success=(stats.stop_reason == StopReason.TASK_COMPLETE),
        stop_reason=stats.stop_reason.value if stats.stop_reason else "unknown",
        total_steps=stats.total_steps, llm_calls=stats.llm_calls,
        llm_call_rate=stats.llm_call_rate, savings_pct=stats.savings_pct,
        input_tokens=adapter.input_tokens, output_tokens=adapter.output_tokens,
        cost_usd=estimate_cost(model, adapter.input_tokens, adapter.output_tokens),
        duration_s=round(time.perf_counter() - t0, 2), error=error,
    )


def run_task_react(task, tools, api_key, model, split, max_steps) -> TaskResult:
    goal    = task["query"]
    task_id = str(task.get("query_id", task.get("id", "unknown")))

    react = OpenRouterReActAdapter(api_key=api_key, model=model, goal=goal, tool_names=list(tools.keys()))
    agent = DeadReckoningAgent(
        adapter=react, goal=goal, tools=tools,
        fix_threshold=0.0, hard_ceiling=0.01,
        max_steps_without_fix=1, checkpoint_interval=1,
        max_total_steps=max_steps, verbose=False,
    )

    t0, error = time.perf_counter(), ""
    try:
        list(agent.run())
    except Exception as e:
        error = str(e)

    stats = agent.stats
    return TaskResult(
        task_id=task_id, split=split, instruction=goal[:120],
        method="react", model=model,
        success=(stats.stop_reason == StopReason.TASK_COMPLETE),
        stop_reason=stats.stop_reason.value if stats.stop_reason else "unknown",
        total_steps=stats.total_steps, llm_calls=stats.llm_calls,
        llm_call_rate=1.0, savings_pct=0.0,
        input_tokens=react.input_tokens, output_tokens=react.output_tokens,
        cost_usd=estimate_cost(model, react.input_tokens, react.output_tokens),
        duration_s=round(time.perf_counter() - t0, 2), error=error,
    )


# ------------------------------------------------------------------ #
#  Summary                                                             #
# ------------------------------------------------------------------ #

def print_summary(results: list[TaskResult]):
    from collections import defaultdict
    by_method: dict[str, list[TaskResult]] = defaultdict(list)
    for r in results:
        by_method[r.method].append(r)

    print("\n" + "="*72)
    print("  RESULTS SUMMARY")
    print("="*72)
    print(f"\n{'Method':<20} {'Tasks':>6} {'Success%':>9} {'LLM/task':>9} {'Savings%':>9} {'$/task':>9} {'s/task':>8}")
    print("-"*72)
    for method in ["dead_reckoning", "react"]:
        rows = by_method.get(method, [])
        if not rows: continue
        n = len(rows)
        print(f"{method:<20} {n:>6} "
              f"{sum(r.success for r in rows)/n*100:>8.1f}% "
              f"{sum(r.llm_calls for r in rows)/n:>9.1f} "
              f"{sum(r.savings_pct for r in rows)/n:>8.1f}% "
              f"{sum(r.cost_usd for r in rows)/n:>9.4f} "
              f"{sum(r.duration_s for r in rows)/n:>7.1f}s")

    print()
    splits = sorted(set(r.split for r in results))
    if len(splits) > 1:
        print("Per-split breakdown:")
        for split in splits:
            for method in ["dead_reckoning", "react"]:
                rows = [r for r in by_method.get(method,[]) if r.split == split]
                if rows:
                    print(f"  {split}/{method:<20}: "
                          f"success={sum(r.success for r in rows)/len(rows)*100:.0f}%  "
                          f"llm/task={sum(r.llm_calls for r in rows)/len(rows):.1f}  n={len(rows)}")
        print()

    dr, base = by_method.get("dead_reckoning",[]), by_method.get("react",[])
    if dr and base:
        dr_llm   = sum(r.llm_calls for r in dr)   / len(dr)
        base_llm = sum(r.llm_calls for r in base) / len(base)
        delta_sr = sum(r.success for r in dr)/len(dr)*100 - sum(r.success for r in base)/len(base)*100
        print(f"  LLM call reduction : {(1-dr_llm/base_llm)*100:.0f}%")
        print(f"  Success rate delta : {delta_sr:+.1f}pp",
              "✓ paper-worthy" if abs(delta_sr) <= 3 else "— tune fix_threshold and retry")
    print()


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="ToolBench: Dead Reckoning vs ReAct via OpenRouter")
    parser.add_argument("--toolbench_dir",    required=True)
    parser.add_argument("--split",            default="G1", choices=["G1","G2","G3","all"])
    parser.add_argument("--n_tasks",          type=int,   default=50)
    parser.add_argument("--model",            default="meta-llama/llama-3.3-70b-instruct:free")
    parser.add_argument("--fix_threshold",    type=float, default=0.35)
    parser.add_argument("--max_steps",        type=int,   default=20)
    parser.add_argument("--output",           default="results/toolbench_results.csv")
    parser.add_argument("--methods",          default="both", choices=["dr","react","both"])
    parser.add_argument("--server_url",       default="http://localhost:5000")
    parser.add_argument("--api_key",          default=os.environ.get("OPENROUTER_API_KEY",""))
    parser.add_argument("--rate_limit_delay", type=float, default=0.5,
                        help="Seconds between tasks (free tier rate limiting)")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: set OPENROUTER_API_KEY or pass --api_key")
        print("Free key at: https://openrouter.ai/keys")
        sys.exit(1)

    try:
        import openai
    except ImportError:
        print("ERROR: pip3 install openai")
        sys.exit(1)

    dispatcher   = VirtualAPIDispatcher(server_url=args.server_url)
    splits       = ["G1","G2","G3"] if args.split == "all" else [args.split]
    all_results: list[TaskResult] = []

    print(f"\nModel  : {args.model}")
    print(f"Splits : {splits}  |  Tasks/split: {args.n_tasks}  |  Methods: {args.methods}")

    for split in splits:
        print(f"\n{'─'*62}\n  Split {split}")
        try:
            tasks = load_tasks(args.toolbench_dir, split, args.n_tasks)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}"); continue

        print(f"  Loaded {len(tasks)} tasks")

        for i, task in enumerate(tasks):
            goal = task.get("query", "")
            print(f"  [{i+1:3d}/{len(tasks)}] {goal[:58]}...")

            try:
                tool_defs = load_tools_for_task(args.toolbench_dir, task)
                tools     = dispatcher.build_tools(tool_defs)
            except Exception as e:
                print(f"    SKIP: {e}"); continue
            if not tools:
                print("    SKIP (no tools)"); continue

            if args.methods in ("dr", "both"):
                dispatcher.reset_log()
                try:
                    r = run_task_dead_reckoning(task, tools, args.api_key, args.model, split, args.fix_threshold, args.max_steps)
                    all_results.append(r)
                    print(f"    DR    {'✓' if r.success else '✗'}  steps={r.total_steps:>2}  llm={r.llm_calls:>2}  saved={r.savings_pct:.0f}%  tok={r.input_tokens+r.output_tokens}")
                except Exception as e:
                    print(f"    DR    ERROR: {e}"); traceback.print_exc()
                time.sleep(args.rate_limit_delay)

            if args.methods in ("react", "both"):
                dispatcher.reset_log()
                try:
                    r = run_task_react(task, tools, args.api_key, args.model, split, args.max_steps)
                    all_results.append(r)
                    print(f"    REACT {'✓' if r.success else '✗'}  steps={r.total_steps:>2}  llm={r.llm_calls:>2}  tok={r.input_tokens+r.output_tokens}")
                except Exception as e:
                    print(f"    REACT ERROR: {e}")
                time.sleep(args.rate_limit_delay)

    if not all_results:
        print("\nNo results. Check: data downloaded, virtual server running, API key valid.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in all_results)
    print(f"\nSaved → {output_path}")
    print_summary(all_results)


if __name__ == "__main__":
    main()
