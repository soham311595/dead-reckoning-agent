"""
ToolBench data_example Benchmark
==================================
Runs Dead Reckoning vs ReAct on the real ToolBench data_example tasks.

Uses:
  - data_example/instruction/G{1,2,3}_query.json  — real tasks + api_list
  - data_example/answer/G{1,2,3}_answer/*.json    — ground truth (win: true/false)
  - data_example/toolenv/response_examples/       — cached API responses as mock tools

Success metric: did the agent call ALL required tools from api_list?
(This is real ground truth, not Claude's self-report)

Run:
    cd dead-reckoning-agent
    ANTHROPIC_API_KEY=sk-ant-... python3 benchmarks/toolbench_data_example.py \
        --toolbench_dir ~/Desktop/AI work/ToolBench \
        --split all \
        --output results/toolbench_real.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from dead_reckoning import DeadReckoningAgent
from dead_reckoning.adapters import _parse_fix_response, _dispatch_action
from dead_reckoning.core.agent import LLMAdapter, StopReason
from dead_reckoning.core.world_model import WorldModel


# ------------------------------------------------------------------ #
#  Data loading                                                        #
# ------------------------------------------------------------------ #

def load_tasks(toolbench_dir: str, split: str) -> list[dict]:
    """Load tasks from data_example query files."""
    query_file = Path(toolbench_dir) / "data_example" / "instruction" / f"{split}_query.json"
    if not query_file.exists():
        raise FileNotFoundError(f"Not found: {query_file}")
    with open(query_file) as f:
        tasks = json.load(f)
    # Attach split label
    for i, t in enumerate(tasks):
        t["_split"] = split
        t["_id"] = f"{split}_{i:03d}"
    return tasks


def load_ground_truth(toolbench_dir: str, split: str) -> dict[str, bool]:
    """
    Load win/loss ground truth from answer files.
    Returns dict mapping answer filename stem → win bool.
    """
    answer_dir = Path(toolbench_dir) / "data_example" / "answer" / f"{split}_answer"
    truth = {}
    if answer_dir.exists():
        for f in answer_dir.glob("*.json"):
            try:
                d = json.load(open(f))
                truth[f.stem] = bool(d.get("win", False))
            except Exception:
                pass
    return truth


def load_response_examples(toolbench_dir: str) -> dict[str, Any]:
    """
    Load cached API responses from response_examples/.
    Returns dict: tool_name → response dict
    """
    resp_dir = Path(toolbench_dir) / "data_example" / "toolenv" / "response_examples"
    examples = {}
    if resp_dir.exists():
        for f in resp_dir.rglob("*.json"):
            try:
                d = json.load(open(f))
                tool_name = f.stem
                examples[tool_name] = d
            except Exception:
                pass
    return examples


# ------------------------------------------------------------------ #
#  Tool builder                                                        #
# ------------------------------------------------------------------ #

def sanitize_name(name: str) -> str:
    """
    Convert any API name into a valid Python identifier the LLM can call.

    Handles:
      /tracking/correo_argentino/result_task/:task_id  → tracking_result_task
      Get Tracking Data                                → get_tracking_data
      il                                               → il
      Checkhealth                                      → checkhealth
    """
    import re
    # strip leading slash, split on / and take the non-param segments
    parts = [p for p in name.split("/") if p and not p.startswith(":")]
    if parts:
        # join meaningful segments (skip very short filler like empty strings)
        name = "_".join(parts)
    # lowercase, replace spaces and hyphens with underscores
    name = name.lower().strip()
    name = re.sub(r"[\s\-]+", "_", name)
    # strip anything that isn't alphanumeric or underscore
    name = re.sub(r"[^a-z0-9_]", "", name)
    # collapse multiple underscores
    name = re.sub(r"_+", "_", name).strip("_")
    # must start with a letter
    if name and name[0].isdigit():
        name = "api_" + name
    return name or "api_call"


def build_tools_for_task(task: dict, response_examples: dict) -> tuple[dict[str, Callable], list[str], dict[str, str]]:
    """
    Build a tools dict for a task.

    Returns:
        tools       — callable dict keyed by sanitized name
        called_log  — list of raw api_names that were called
        name_map    — sanitized_name → original api_name (for success eval)
    """
    called: list[str] = []
    name_map: dict[str, str] = {}  # sanitized → original api_name

    def make_tool(tool_name: str, api_name: str, fn_name: str) -> Callable:
        # Try to find a cached response
        cached = None
        for key, val in response_examples.items():
            if tool_name.lower() in key.lower() or api_name.lower() in key.lower():
                cached = val
                break

        def tool_fn(**kwargs) -> Any:
            # Record the ORIGINAL api_name so evaluate_success can match it
            called.append(api_name)
            if cached:
                return cached
            return {
                "status": "success",
                "tool": tool_name,
                "api": api_name,
                "fn": fn_name,
                "result": f"Simulated result for {api_name}",
                "data": {}
            }
        tool_fn.__name__ = fn_name
        return tool_fn

    tools = {}
    for api_info in task.get("api_list", []):
        tool_name = api_info.get("tool_name", "unknown")
        api_name  = api_info.get("api_name", "call")
        fn_name   = sanitize_name(api_name)

        # Handle collisions by appending a counter
        base = fn_name
        counter = 2
        while fn_name in tools:
            fn_name = f"{base}_{counter}"
            counter += 1

        fn = make_tool(tool_name, api_name, fn_name)
        tools[fn_name] = fn
        name_map[fn_name] = api_name

    return tools, called, name_map


# ------------------------------------------------------------------ #
#  Success evaluator (real ground truth)                               #
# ------------------------------------------------------------------ #

def evaluate_success(task: dict, called_tools: list[str]) -> bool:
    """
    Check if all required APIs from api_list were called.
    called_tools contains original api_names (not sanitized).
    """
    required_originals = list({
        api["api_name"]
        for api in task.get("api_list", [])
    })
    called_set = set(called_tools)
    return all(r in called_set for r in required_originals)


# ------------------------------------------------------------------ #
#  Adapters                                                            #
# ------------------------------------------------------------------ #

_DR_SYSTEM = """You are the navigation module of an agent completing a task using specific API tools.

Output ONLY a JSON object, no prose, no markdown:
{{"reasoning": "1-2 sentences", "done": false, "next_action": "api_name(param='value')", "predicted_steps": ["api2()", "api3()"], "confidence": 0.9}}

Rules:
- next_action must use EXACTLY one of the available tool names listed below
- Use the exact function name as listed — not the API description, the exact name
- predicted_steps should list the next 2-3 tool calls in order
- Set done=true only after ALL required tools in the task have been called
- Never set done=true if steps_completed is 0

Available tools: {tool_names}"""


_REACT_SYSTEM = """You are an agent completing a task step by step using API tools.

Output ONLY a JSON object, no prose, no markdown:
{{"thought": "what to do next", "action": "api_name(param='value')", "done": false}}

Rules:
- action must use EXACTLY one of the available tool names listed below
- Use the exact function name as listed
- Set done=true only after ALL required tools have been called
- Never set done=true if no tools have been called yet
- Check completed_steps and do not repeat tools already called

Available tools: {tool_names}
Task: {goal}"""


class AnthropicDRAdapter(LLMAdapter):
    def __init__(self, api_key: str, model: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model  = model
        self.input_tokens  = 0
        self.output_tokens = 0

    def get_fix(self, world: WorldModel, tools: dict) -> tuple[str, list[str], str, bool]:
        tool_names = ", ".join(list(tools.keys())[:25])
        system = _DR_SYSTEM.format(tool_names=tool_names)
        resp = self.client.messages.create(
            model=self.model, max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": world.summary()}],
        )
        self.input_tokens  += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens
        return _parse_fix_response(resp.content[0].text)

    def execute_action(self, action: str, tools: dict, env: dict) -> tuple[Any, bool]:
        return _dispatch_action(action, tools, env)


class AnthropicReActAdapter(LLMAdapter):
    def __init__(self, api_key: str, model: str, goal: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model  = model
        self.goal   = goal
        self._history: list[dict] = []
        self._tools_called = 0
        self.input_tokens  = 0
        self.output_tokens = 0

    def get_fix(self, world: WorldModel, tools: dict) -> tuple[str, list[str], str, bool]:
        tool_names = ", ".join(list(tools.keys())[:25])
        system = _REACT_SYSTEM.format(tool_names=tool_names, goal=self.goal)

        completed = world.completed_steps
        if completed:
            last = completed[-1]
            user_msg = (
                f"Steps done: {len(completed)}\n"
                f"Last action: {last['action']}\n"
                f"Last result: {str(last['result'])[:150]}\n"
                f"What is the next action?"
            )
        else:
            user_msg = f"Goal: {self.goal}\nNo steps taken yet. What is the first action?"

        self._history.append({"role": "user", "content": user_msg})

        resp = self.client.messages.create(
            model=self.model, max_tokens=400,
            system=system,
            messages=self._history,
        )
        self.input_tokens  += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens

        text = resp.content[0].text.strip()
        self._history.append({"role": "assistant", "content": text})

        clean = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
        try:
            d = json.loads(clean)
        except Exception:
            m = re.search(r'"action"\s*:\s*"([^"]+)"', clean)
            return "parse error", [], m.group(1) if m else "", False

        action = d.get("action", "")
        done   = bool(d.get("done", False))

        # Block premature done
        if done and self._tools_called == 0:
            done = False
            if not action and tools:
                action = list(tools.keys())[0] + "()"

        return d.get("thought", ""), [], action, done

    def execute_action(self, action: str, tools: dict, env: dict) -> tuple[Any, bool]:
        result, errored = _dispatch_action(action, tools, env)
        if not errored and "no tool matched" not in str(result):
            self._tools_called += 1
        return result, errored


# ------------------------------------------------------------------ #
#  Result dataclass                                                    #
# ------------------------------------------------------------------ #

@dataclass
class TaskResult:
    task_id:          str
    split:            str
    goal:             str
    method:           str
    model:            str
    # Ground truth success (all required tools called)
    success_gt:       bool
    # LLM self-reported success
    success_llm:      bool
    stop_reason:      str
    total_steps:      int
    llm_calls:        int
    savings_pct:      float
    tools_required:   int
    tools_called:     int
    input_tokens:     int
    output_tokens:    int
    duration_s:       float
    error:            str = ""


# ------------------------------------------------------------------ #
#  Run one task                                                        #
# ------------------------------------------------------------------ #

def run_task(
    task: dict,
    api_key: str,
    model: str,
    method: str,
    response_examples: dict,
    fix_threshold: float,
    max_steps: int,
) -> TaskResult:
    goal    = task["query"]
    task_id = task["_id"]
    split   = task["_split"]

    tools, called_log, name_map = build_tools_for_task(task, response_examples)
    n_required = len(set(
        api["api_name"].lower().replace(" ", "_").replace("-", "_")
        for api in task.get("api_list", [])
    ))

    t0, error = time.perf_counter(), ""

    try:
        if method == "dead_reckoning":
            adapter = AnthropicDRAdapter(api_key=api_key, model=model)
            agent   = DeadReckoningAgent(
                adapter=adapter, goal=goal, tools=tools,
                fix_threshold=fix_threshold,
                max_steps_without_fix=5,
                max_total_steps=max_steps,
                verbose=False,
            )
        else:
            adapter = AnthropicReActAdapter(api_key=api_key, model=model, goal=goal)
            agent   = DeadReckoningAgent(
                adapter=adapter, goal=goal, tools=tools,
                fix_threshold=0.0, hard_ceiling=0.01,
                max_steps_without_fix=1, checkpoint_interval=1,
                max_total_steps=max_steps,
                verbose=False,
            )
        list(agent.run())
        stats = agent.stats

    except Exception as e:
        error = str(e)
        stats = agent.stats if "agent" in dir() else None

    duration    = time.perf_counter() - t0
    llm_done    = (stats.stop_reason == StopReason.TASK_COMPLETE) if stats else False
    gt_success  = evaluate_success(task, called_log)

    return TaskResult(
        task_id=task_id, split=split, goal=goal[:100],
        method=method, model=model,
        success_gt=gt_success,
        success_llm=llm_done,
        stop_reason=stats.stop_reason.value if (stats and stats.stop_reason) else "?",
        total_steps=stats.total_steps if stats else 0,
        llm_calls=stats.llm_calls if stats else 0,
        savings_pct=stats.savings_pct if stats else 0.0,
        tools_required=n_required,
        tools_called=len(set(called_log)),
        input_tokens=adapter.input_tokens,
        output_tokens=adapter.output_tokens,
        duration_s=round(duration, 2),
        error=error,
    )


# ------------------------------------------------------------------ #
#  Summary                                                             #
# ------------------------------------------------------------------ #

def print_summary(results: list[TaskResult]):
    by_m: dict[str, list[TaskResult]] = defaultdict(list)
    for r in results:
        by_m[r.method].append(r)

    print("\n" + "=" * 68)
    print("  TOOLBENCH data_example RESULTS")
    print("=" * 68)
    print(f"\n{'Method':<16} {'N':>3} {'GT%':>6} {'LLM%':>6} {'LLM/task':>9} {'Saved%':>7} {'Tok/task':>9}")
    print("-" * 60)

    for method in ["dead_reckoning", "react"]:
        rows = by_m.get(method, [])
        if not rows:
            continue
        n = len(rows)
        gt_sr  = sum(r.success_gt  for r in rows) / n * 100
        llm_sr = sum(r.success_llm for r in rows) / n * 100
        avg_llm = sum(r.llm_calls  for r in rows) / n
        avg_sav = sum(r.savings_pct for r in rows) / n
        avg_tok = sum(r.input_tokens + r.output_tokens for r in rows) / n
        print(f"{method:<16} {n:>3} {gt_sr:>5.0f}% {llm_sr:>5.0f}% "
              f"{avg_llm:>9.1f} {avg_sav:>6.0f}% {avg_tok:>9.0f}")

    print()
    print("  GT%  = ground truth (all required tools called)")
    print("  LLM% = Claude self-reported done=true")
    print()

    dr   = by_m.get("dead_reckoning", [])
    base = by_m.get("react", [])
    if dr and base:
        dr_llm   = sum(r.llm_calls for r in dr)   / len(dr)
        base_llm = sum(r.llm_calls for r in base) / len(base)
        gt_delta = sum(r.success_gt for r in dr)/len(dr)*100 - sum(r.success_gt for r in base)/len(base)*100
        reduc    = (1 - dr_llm / base_llm) * 100 if base_llm else 0
        print(f"  LLM call reduction  : {reduc:.0f}%")
        print(f"  GT success delta    : {gt_delta:+.0f}pp")
        tok_reduc = (1 - sum(r.input_tokens+r.output_tokens for r in dr)/len(dr) /
                     (sum(r.input_tokens+r.output_tokens for r in base)/len(base))) * 100
        print(f"  Token cost reduction: {tok_reduc:.0f}%")

    print("\n  Per-task breakdown:")
    print(f"  {'task_id':<10} {'split':>5}  {'method':<16}  {'GT':>3}  {'LLM':>3}  {'steps':>5}  {'llm':>4}  {'saved':>6}  goal")
    print("  " + "─" * 80)
    for r in sorted(results, key=lambda x: (x.task_id, x.method)):
        gt  = "✓" if r.success_gt  else "✗"
        llm = "✓" if r.success_llm else "✗"
        print(f"  {r.task_id:<10} {r.split:>5}  {r.method:<16}  {gt:>3}  {llm:>3}  "
              f"{r.total_steps:>5}  {r.llm_calls:>4}  {r.savings_pct:>5.0f}%  {r.goal[:35]}...")
    print()


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="ToolBench data_example benchmark with real ground truth"
    )
    parser.add_argument("--toolbench_dir", required=True,
                        help="Path to ToolBench repo (e.g. ~/Desktop/AI work/ToolBench)")
    parser.add_argument("--split",         default="all",
                        choices=["G1", "G2", "G3", "all"])
    parser.add_argument("--model",         default="claude-haiku-4-5")
    parser.add_argument("--fix_threshold", type=float, default=0.35)
    parser.add_argument("--max_steps",     type=int,   default=20)
    parser.add_argument("--methods",       default="both",
                        choices=["dr", "react", "both"])
    parser.add_argument("--output",        default="results/toolbench_example.csv")
    parser.add_argument("--api_key",
                        default=os.environ.get("ANTHROPIC_API_KEY", ""))
    parser.add_argument("--delay",         type=float, default=0.3,
                        help="Seconds between API calls")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: set ANTHROPIC_API_KEY or pass --api_key")
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("ERROR: pip3 install anthropic")
        sys.exit(1)

    toolbench_dir   = os.path.expanduser(args.toolbench_dir)
    splits          = ["G1", "G2", "G3"] if args.split == "all" else [args.split]
    response_examples = load_response_examples(toolbench_dir)
    print(f"Loaded {len(response_examples)} cached API response examples")

    all_results: list[TaskResult] = []

    for split in splits:
        try:
            tasks = load_tasks(toolbench_dir, split)
        except FileNotFoundError as e:
            print(f"SKIP {split}: {e}")
            continue

        print(f"\n{'─'*60}")
        print(f"  Split {split} — {len(tasks)} tasks")

        for i, task in enumerate(tasks):
            goal = task["query"]
            n_apis = len(task.get("api_list", []))
            print(f"\n  [{i+1}/{len(tasks)}] {goal[:60]}...")
            print(f"    Required APIs: {[a['api_name'] for a in task.get('api_list',[])]}")

            for method in (["dead_reckoning", "react"] if args.methods == "both"
                           else [{"dr": "dead_reckoning", "react": "react"}[args.methods]]):
                try:
                    r = run_task(
                        task=task, api_key=args.api_key, model=args.model,
                        method=method, response_examples=response_examples,
                        fix_threshold=args.fix_threshold, max_steps=args.max_steps,
                    )
                    all_results.append(r)
                    gt  = "✓" if r.success_gt  else "✗"
                    llm = "✓" if r.success_llm else "✗"
                    print(f"    {method:<16}: GT={gt} LLM={llm}  "
                          f"steps={r.total_steps}  llm={r.llm_calls}  "
                          f"saved={r.savings_pct:.0f}%  "
                          f"tools={r.tools_called}/{r.tools_required}")
                except Exception as e:
                    print(f"    {method}: ERROR — {e}")
                    import traceback; traceback.print_exc()

                time.sleep(args.delay)

    if not all_results:
        print("\nNo results.")
        return

    # Save CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
        w.writeheader()
        w.writerows(asdict(r) for r in all_results)
    print(f"\nSaved → {output_path}")

    print_summary(all_results)


if __name__ == "__main__":
    main()