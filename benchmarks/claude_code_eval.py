"""
ToolBench benchmark via Claude Code CLI adapter.

Compares Dead Reckoning vs ReAct using Claude Code as the LLM backend.
Both adapters embed full task context in every prompt — no conversation
memory across stateless CLI calls.

Setup:
    npm install -g @anthropic-ai/claude-code
    claude          # authenticate once

Run:
    cd dead-reckoning-agent
    python3 benchmarks/claude_code_eval.py \
        --toolbench_dir ~/Desktop/AI work/ToolBench \
        --split all \
        --output results/claude_code_results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dead_reckoning import DeadReckoningAgent
from dead_reckoning.core.agent import StopReason
from dead_reckoning.adapters_claude_code import ClaudeCodeAdapter, ClaudeCodeReActAdapter

sys.path.insert(0, str(Path(__file__).parent))
from toolbench_data_example import (
    load_tasks,
    load_response_examples,
    build_tools_for_task,
    evaluate_success,
)


# ------------------------------------------------------------------ #
#  Result                                                              #
# ------------------------------------------------------------------ #

@dataclass
class TaskResult:
    task_id:       str
    split:         str
    goal:          str
    method:        str
    model:         str
    success_gt:    bool
    success_llm:   bool
    stop_reason:   str
    total_steps:   int
    llm_calls:     int
    savings_pct:   float
    tools_required:int
    tools_called:  int
    input_tokens:  int
    output_tokens: int
    cost_usd:      float
    duration_s:    float
    error:         str = ""


# ------------------------------------------------------------------ #
#  Run one task                                                        #
# ------------------------------------------------------------------ #

def run_task(
    task: dict,
    model: str,
    method: str,
    response_examples: dict,
    fix_threshold: float,
    max_steps: int,
    timeout: int,
) -> TaskResult:
    goal     = task["query"]
    task_id  = task["_id"]
    split    = task["_split"]

    tools, called_log, name_map = build_tools_for_task(task, response_examples)
    n_required = len({api["api_name"] for api in task.get("api_list", [])})

    t0, error = time.perf_counter(), ""

    try:
        if method == "dead_reckoning":
            adapter = ClaudeCodeAdapter(model=model, n_predictions=4, timeout=timeout)
            agent   = DeadReckoningAgent(
                adapter=adapter, goal=goal, tools=tools,
                fix_threshold=fix_threshold,
                max_steps_without_fix=5,
                max_total_steps=max_steps,
                verbose=False,
            )
        else:
            adapter = ClaudeCodeReActAdapter(model=model, goal=goal, timeout=timeout)
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

    duration   = time.perf_counter() - t0
    llm_done   = (stats.stop_reason == StopReason.TASK_COMPLETE) if stats else False
    gt_success = evaluate_success(task, called_log)

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
        cost_usd=adapter.total_cost_usd,
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

    print("\n" + "=" * 70)
    print("  CLAUDE CODE ADAPTER RESULTS")
    print("=" * 70)
    print(f"\n{'Method':<16} {'N':>3} {'GT%':>6} {'LLM%':>6} "
          f"{'LLM/task':>9} {'Saved%':>7} {'Tok/task':>9} {'$/task':>8}")
    print("-" * 65)

    for method in ["dead_reckoning", "react"]:
        rows = by_m.get(method, [])
        if not rows:
            continue
        n = len(rows)
        print(f"{method:<16} {n:>3} "
              f"{sum(r.success_gt  for r in rows)/n*100:>5.0f}% "
              f"{sum(r.success_llm for r in rows)/n*100:>5.0f}% "
              f"{sum(r.llm_calls   for r in rows)/n:>9.1f} "
              f"{sum(r.savings_pct for r in rows)/n:>6.0f}% "
              f"{sum(r.input_tokens+r.output_tokens for r in rows)/n:>9.0f} "
              f"{sum(r.cost_usd    for r in rows)/n:>7.5f}")

    print()
    print("  GT%  = ground truth (all required tools called)")
    print("  LLM% = Claude self-reported done=true")

    dr   = by_m.get("dead_reckoning", [])
    base = by_m.get("react", [])
    if dr and base:
        dr_llm   = sum(r.llm_calls for r in dr)   / len(dr)
        base_llm = sum(r.llm_calls for r in base) / len(base)
        gt_delta = (sum(r.success_gt for r in dr)/len(dr) -
                    sum(r.success_gt for r in base)/len(base)) * 100
        tok_dr   = sum(r.input_tokens+r.output_tokens for r in dr)   / len(dr)
        tok_base = sum(r.input_tokens+r.output_tokens for r in base) / len(base)
        print(f"\n  LLM call reduction : {(1-dr_llm/base_llm)*100:.0f}%  ({dr_llm:.1f} vs {base_llm:.1f}/task)")
        if tok_base > 0:
            print(f"  Token reduction    : {(1-tok_dr/tok_base)*100:.0f}%  ({tok_dr:.0f} vs {tok_base:.0f}/task)")
        print(f"  GT success delta   : {gt_delta:+.0f}pp")

    print("\n  Per-task:")
    print(f"  {'id':<10} {'split':>5}  {'method':<16}  "
          f"{'GT':>3}  {'LLM':>3}  {'steps':>5}  {'llm':>4}  {'saved':>5}  goal")
    print("  " + "─" * 76)
    for r in sorted(results, key=lambda x: (x.task_id, x.method)):
        print(f"  {r.task_id:<10} {r.split:>5}  {r.method:<16}  "
              f"{'✓' if r.success_gt else '✗':>3}  "
              f"{'✓' if r.success_llm else '✗':>3}  "
              f"{r.total_steps:>5}  {r.llm_calls:>4}  "
              f"{r.savings_pct:>4.0f}%  {r.goal[:33]}...")
    print()


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="ToolBench benchmark via Claude Code CLI"
    )
    parser.add_argument("--toolbench_dir", required=True)
    parser.add_argument("--split",         default="all",
                        choices=["G1", "G2", "G3", "all"])
    parser.add_argument("--model",         default="claude-haiku-4-5")
    parser.add_argument("--fix_threshold", type=float, default=0.35)
    parser.add_argument("--max_steps",     type=int,   default=20)
    parser.add_argument("--methods",       default="both",
                        choices=["dr", "react", "both"])
    parser.add_argument("--output",        default="results/claude_code_results.csv")
    parser.add_argument("--timeout",       type=int,   default=60)
    parser.add_argument("--delay",         type=float, default=0.3)
    args = parser.parse_args()

    if shutil.which("claude") is None:
        print("ERROR: Claude Code CLI not found.")
        print("Install: npm install -g @anthropic-ai/claude-code")
        sys.exit(1)

    toolbench_dir     = os.path.expanduser(args.toolbench_dir)
    response_examples = load_response_examples(toolbench_dir)
    print(f"Loaded {len(response_examples)} cached API responses")
    print(f"Model: {args.model}  |  Backend: Claude Code CLI")

    splits      = ["G1", "G2", "G3"] if args.split == "all" else [args.split]
    all_results: list[TaskResult] = []

    for split in splits:
        try:
            tasks = load_tasks(toolbench_dir, split)
        except FileNotFoundError as e:
            print(f"SKIP {split}: {e}"); continue

        print(f"\n{'─'*60}\n  Split {split} — {len(tasks)} tasks")

        for i, task in enumerate(tasks):
            goal = task["query"]
            print(f"\n  [{i+1}/{len(tasks)}] {goal[:58]}...")
            print(f"    APIs: {[a['api_name'] for a in task.get('api_list',[])]}")

            methods = (
                ["dead_reckoning", "react"] if args.methods == "both"
                else [{"dr":"dead_reckoning","react":"react"}[args.methods]]
            )

            for method in methods:
                try:
                    r = run_task(
                        task=task, model=args.model, method=method,
                        response_examples=response_examples,
                        fix_threshold=args.fix_threshold,
                        max_steps=args.max_steps,
                        timeout=args.timeout,
                    )
                    all_results.append(r)
                    print(f"    {method:<16}: "
                          f"GT={'✓' if r.success_gt else '✗'} "
                          f"LLM={'✓' if r.success_llm else '✗'}  "
                          f"steps={r.total_steps}  llm={r.llm_calls}  "
                          f"saved={r.savings_pct:.0f}%  "
                          f"tools={r.tools_called}/{r.tools_required}  "
                          f"tok={r.input_tokens+r.output_tokens}")
                except Exception as e:
                    print(f"    {method}: ERROR — {e}")
                    import traceback; traceback.print_exc()
                time.sleep(args.delay)

    if not all_results:
        print("\nNo results.")
        return

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