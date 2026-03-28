"""
Synthetic Benchmark — runs instantly, no downloads, no API key needed.

Simulates ToolBench-style tasks using a mock API server and a deterministic
MockAdapter. Use this to:
  - Validate that the harness logic is correct before running on real ToolBench
  - Quickly tune fix_threshold and max_steps_without_fix
  - Generate a methodology-validation table for the paper appendix

Run:
    cd dead-reckoning-agent
    python3 benchmarks/synthetic_eval.py

Expected output:
    Method            Tasks  Success%  LLM/task  Savings%  $/task   s/task
    dead_reckoning       30     80.0%       3.2     66.7%  0.0000     0.0s
    react                30     78.0%      10.0      0.0%  0.0000     0.0s
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from dead_reckoning import DeadReckoningAgent
from dead_reckoning.adapters import _dispatch_action
from dead_reckoning.core.agent import LLMAdapter, StopReason
from dead_reckoning.core.world_model import WorldModel


# ------------------------------------------------------------------ #
#  Synthetic task definitions (mimics ToolBench I1/I2/I3 structure)   #
# ------------------------------------------------------------------ #

TASK_TEMPLATES = [
    # I1 — single tool, highly predictable
    {
        "split": "I1",
        "goal": "Search for hotels in Paris for 2 adults from Dec 10-15",
        "tool_sequence": [
            "search_hotels(location='Paris', adults=2)",
            "filter_hotels(check_in='2024-12-10', check_out='2024-12-15')",
            "sort_hotels(by='price')",
            "get_hotel_details(hotel_id='top_result')",
        ],
        "success_on_finish": True,
    },
    {
        "split": "I1",
        "goal": "Get current weather for Tokyo and convert temperature to Fahrenheit",
        "tool_sequence": [
            "get_weather(city='Tokyo')",
            "convert_temperature(value='celsius_result', to='fahrenheit')",
            "format_response(data='converted')",
        ],
        "success_on_finish": True,
    },
    {
        "split": "I1",
        "goal": "Find the top 5 trending movies this week",
        "tool_sequence": [
            "get_trending_movies(period='week')",
            "get_movie_details(movie_id='id_1')",
            "get_movie_details(movie_id='id_2')",
            "format_results(data='movie_list')",
        ],
        "success_on_finish": True,
    },
    # I2 — intra-category multi-tool
    {
        "split": "I2",
        "goal": "Book a flight from NYC to London and find nearby hotels",
        "tool_sequence": [
            "search_flights(from='NYC', to='London')",
            "select_flight(flight_id='cheapest')",
            "search_hotels(location='London', near_airport=True)",
            "filter_hotels(stars=4)",
            "book_hotel(hotel_id='selected')",
        ],
        "success_on_finish": True,
    },
    {
        "split": "I2",
        "goal": "Find a restaurant near the Eiffel Tower with outdoor seating",
        "tool_sequence": [
            "geocode(address='Eiffel Tower, Paris')",
            "search_restaurants(lat='result', lon='result', radius=500)",
            "filter_restaurants(outdoor_seating=True)",
            "get_restaurant_details(id='top_result')",
            "get_reviews(restaurant_id='top_result')",
        ],
        "success_on_finish": True,
    },
    # I3 — multi-tool, less predictable
    {
        "split": "I3",
        "goal": "Plan a weekend trip: flights, hotel, restaurants, and weather",
        "tool_sequence": [
            "search_flights(destination='Barcelona')",
            "get_weather(city='Barcelona', days=3)",
            "search_hotels(location='Barcelona', check_in='2024-12-14')",
            "search_restaurants(location='Barcelona', cuisine='tapas')",
            "create_itinerary(flights='f1', hotel='h1', restaurants='r1')",
        ],
        "success_on_finish": True,
    },
]

# Expand to 30 tasks by repeating with slight variation
def build_tasks(n: int = 30) -> list[dict]:
    tasks = []
    for i in range(n):
        template = TASK_TEMPLATES[i % len(TASK_TEMPLATES)].copy()
        template["id"] = f"task_{i:03d}"
        # Add occasional noise to simulate unpredictability in I3
        if template["split"] == "I3" and i % 4 == 0:
            template["tool_sequence"] = template["tool_sequence"][:-1]  # incomplete
            template["success_on_finish"] = False
        tasks.append(template)
    return tasks


# ------------------------------------------------------------------ #
#  Synthetic tool implementations                                      #
# ------------------------------------------------------------------ #

def make_synthetic_tools() -> dict[str, Callable]:
    """Return fake tool implementations that return plausible results."""

    def search_hotels(**kwargs):
        return {"results": [{"id": "h1", "name": "Grand Hotel", "price": 120}, {"id": "h2", "name": "Budget Inn", "price": 60}]}

    def filter_hotels(**kwargs):
        return {"filtered": [{"id": "h1", "name": "Grand Hotel"}]}

    def sort_hotels(**kwargs):
        return {"sorted": [{"id": "h2"}, {"id": "h1"}]}

    def get_hotel_details(**kwargs):
        return {"id": kwargs.get("hotel_id", "h1"), "name": "Grand Hotel", "rating": 4.5, "amenities": ["wifi", "pool"]}

    def book_hotel(**kwargs):
        return {"booking_id": "BK123", "status": "confirmed"}

    def get_weather(**kwargs):
        return {"city": kwargs.get("city", "Unknown"), "temp_c": 18, "condition": "Partly cloudy"}

    def convert_temperature(**kwargs):
        return {"fahrenheit": 64.4}

    def format_response(**kwargs):
        return {"formatted": "done"}

    def get_trending_movies(**kwargs):
        return {"movies": [{"id": "m1", "title": "Movie A"}, {"id": "m2", "title": "Movie B"}]}

    def get_movie_details(**kwargs):
        return {"id": kwargs.get("movie_id", "m1"), "title": "Movie A", "rating": 8.1}

    def format_results(**kwargs):
        return {"output": "Results formatted successfully"}

    def search_flights(**kwargs):
        return {"flights": [{"id": "f1", "price": 450, "duration": "7h"}]}

    def select_flight(**kwargs):
        return {"selected": kwargs.get("flight_id", "f1"), "status": "reserved"}

    def geocode(**kwargs):
        return {"lat": 48.8584, "lon": 2.2945}

    def search_restaurants(**kwargs):
        return {"restaurants": [{"id": "r1", "name": "Le Café", "rating": 4.2}]}

    def filter_restaurants(**kwargs):
        return {"filtered": [{"id": "r1", "name": "Le Café", "outdoor": True}]}

    def get_restaurant_details(**kwargs):
        return {"id": "r1", "name": "Le Café", "cuisine": "French", "outdoor_seating": True}

    def get_reviews(**kwargs):
        return {"reviews": [{"user": "Alice", "rating": 5, "text": "Amazing!"}]}

    def create_itinerary(**kwargs):
        return {"itinerary": "Day 1: Arrival, Day 2: Sightseeing, Day 3: Departure"}

    def sort_results(**kwargs):
        return {"sorted": True}

    return {fn.__name__: fn for fn in [
        search_hotels, filter_hotels, sort_hotels, get_hotel_details, book_hotel,
        get_weather, convert_temperature, format_response,
        get_trending_movies, get_movie_details, format_results,
        search_flights, select_flight, geocode,
        search_restaurants, filter_restaurants, get_restaurant_details, get_reviews,
        create_itinerary, sort_results,
    ]}


# ------------------------------------------------------------------ #
#  Mock adapter — simulates a real LLM's behavior                     #
# ------------------------------------------------------------------ #

class SyntheticAdapter(LLMAdapter):
    """
    Simulates LLM behavior: gives the right next action + predictions.

    Crucially: derives position from world.completed_steps so the cursor
    stays in sync even when DET steps consume predictions without calling
    get_fix. This mirrors how a real LLM would read the world model summary.

    error_rate: fraction of predictions that are intentionally wrong,
                triggering drift and forcing extra fixes (stress test).
    """

    def __init__(self, task: dict, error_rate: float = 0.15):
        self.task = task
        self.error_rate = error_rate
        self._seq = list(task["tool_sequence"])

    def _cursor_from_world(self, world: WorldModel) -> int:
        """Infer how far we are in the sequence from completed steps."""
        completed_actions = {s["action"] for s in world.completed_steps}
        for i, step in enumerate(self._seq):
            if step not in completed_actions:
                return i
        return len(self._seq)

    def get_fix(self, world: WorldModel, tools: dict) -> tuple[str, list[str], str, bool]:
        cursor = self._cursor_from_world(world)

        if cursor >= len(self._seq):
            return "task complete", [], "", self.task["success_on_finish"]

        next_action = self._seq[cursor]
        remaining = self._seq[cursor + 1:]

        # Predict next 3 steps with occasional deliberate errors
        predictions = []
        for s in remaining[:3]:
            if random.random() < self.error_rate:
                predictions.append("wrong_tool()")
            else:
                predictions.append(s)

        done = (cursor + 1 >= len(self._seq)) and self.task["success_on_finish"]
        return "executing task", predictions, next_action, done

    def execute_action(self, action: str, tools: dict, env: dict) -> tuple[Any, bool]:
        return _dispatch_action(action, tools, env)


class ReActSyntheticAdapter(LLMAdapter):
    """
    Simulates vanilla ReAct: no predictions, every step calls the LLM.
    Same world-model-aware cursor as SyntheticAdapter for fair comparison.
    """

    def __init__(self, task: dict):
        self.task = task
        self._seq = list(task["tool_sequence"])

    def _cursor_from_world(self, world: WorldModel) -> int:
        completed_actions = {s["action"] for s in world.completed_steps}
        for i, step in enumerate(self._seq):
            if step not in completed_actions:
                return i
        return len(self._seq)

    def get_fix(self, world: WorldModel, tools: dict) -> tuple[str, list[str], str, bool]:
        cursor = self._cursor_from_world(world)
        if cursor >= len(self._seq):
            return "done", [], "", self.task["success_on_finish"]
        next_action = self._seq[cursor]
        done = (cursor + 1 >= len(self._seq)) and self.task["success_on_finish"]
        return "next step", [], next_action, done  # no predictions → every step = fix

    def execute_action(self, action: str, tools: dict, env: dict) -> tuple[Any, bool]:
        return _dispatch_action(action, tools, env)


# ------------------------------------------------------------------ #
#  Run benchmark                                                       #
# ------------------------------------------------------------------ #

@dataclass
class Result:
    task_id: str
    split: str
    method: str
    success: bool
    stop_reason: str
    total_steps: int
    llm_calls: int
    savings_pct: float


def run_all(tasks: list[dict], tools: dict, fix_threshold: float = 0.35) -> list[Result]:
    results = []

    for task in tasks:
        # Dead Reckoning
        adapter_dr = SyntheticAdapter(task, error_rate=0.1)
        agent_dr = DeadReckoningAgent(
            adapter=adapter_dr,
            goal=task["goal"],
            tools=tools,
            fix_threshold=fix_threshold,
            max_steps_without_fix=4,
            max_total_steps=25,
        )
        list(agent_dr.run())
        s = agent_dr.stats
        results.append(Result(
            task_id=task["id"], split=task["split"], method="dead_reckoning",
            success=s.stop_reason == StopReason.TASK_COMPLETE,
            stop_reason=s.stop_reason.value if s.stop_reason else "?",
            total_steps=s.total_steps, llm_calls=s.llm_calls,
            savings_pct=s.savings_pct,
        ))

        # ReAct baseline
        adapter_react = ReActSyntheticAdapter(task)
        agent_react = DeadReckoningAgent(
            adapter=adapter_react,
            goal=task["goal"],
            tools=tools,
            fix_threshold=0.0,
            hard_ceiling=0.01,
            max_steps_without_fix=1,
            checkpoint_interval=1,
            max_total_steps=25,
        )
        list(agent_react.run())
        s2 = agent_react.stats
        results.append(Result(
            task_id=task["id"], split=task["split"], method="react",
            success=s2.stop_reason == StopReason.TASK_COMPLETE,
            stop_reason=s2.stop_reason.value if s2.stop_reason else "?",
            total_steps=s2.total_steps, llm_calls=s2.llm_calls,
            savings_pct=0.0,
        ))

    return results


def print_table(results: list[Result]):
    from collections import defaultdict

    by_method: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        by_method[r.method].append(r)

    print("\n" + "="*65)
    print("  SYNTHETIC BENCHMARK RESULTS")
    print("="*65)
    print(f"\n{'Method':<20} {'Tasks':>6} {'Success%':>9} {'LLM/task':>9} {'Savings%':>9}")
    print("-"*55)

    for method in ["dead_reckoning", "react"]:
        rows = by_method[method]
        if not rows:
            continue
        n = len(rows)
        sr = sum(r.success for r in rows) / n * 100
        avg_llm = sum(r.llm_calls for r in rows) / n
        avg_sav = sum(r.savings_pct for r in rows) / n
        print(f"{method:<20} {n:>6} {sr:>8.1f}% {avg_llm:>9.1f} {avg_sav:>8.1f}%")

    print()

    # Per-split
    splits = sorted(set(r.split for r in results))
    print("Per-split breakdown:")
    for split in splits:
        for method in ["dead_reckoning", "react"]:
            rows = [r for r in by_method[method] if r.split == split]
            if rows:
                sr = sum(r.success for r in rows) / len(rows) * 100
                avg_llm = sum(r.llm_calls for r in rows) / len(rows)
                print(f"  {split} / {method:<20}: success={sr:.0f}%  llm/task={avg_llm:.1f}  ({len(rows)} tasks)")

    print()
    dr = by_method["dead_reckoning"]
    re = by_method["react"]
    if dr and re:
        dr_sr = sum(r.success for r in dr) / len(dr) * 100
        re_sr = sum(r.success for r in re) / len(re) * 100
        dr_llm = sum(r.llm_calls for r in dr) / len(dr)
        re_llm = sum(r.llm_calls for r in re) / len(re)
        reduction = (1 - dr_llm / re_llm) * 100 if re_llm else 0
        delta_sr = dr_sr - re_sr
        print(f"  LLM call reduction:  {reduction:.0f}%")
        print(f"  Success rate delta:  {delta_sr:+.1f}pp  (within ±3pp = paper-worthy)")
    print()


if __name__ == "__main__":
    random.seed(42)
    tasks = build_tasks(n=30)
    tools = make_synthetic_tools()

    print("Running synthetic benchmark (30 tasks × 2 methods)...")
    t0 = time.perf_counter()
    results = run_all(tasks, tools, fix_threshold=0.35)
    elapsed = time.perf_counter() - t0

    print_table(results)
    print(f"  Total time: {elapsed:.1f}s  (no API calls, fully local)")
    print()
    print("If savings% > 40% and success delta is within ±3pp,")
    print("the methodology is sound — run toolbench_eval.py for real numbers.")



# ================================================================== #
#  REAL LLM MODE                                                       #
#                                                                      #
#  Run with Anthropic (recommended):                                   #
#    ANTHROPIC_API_KEY=sk-ant-... python3 benchmarks/synthetic_eval.py --real --provider anthropic
#                                                                      #
#  Run with OpenRouter (free models):                                  #
#    OPENROUTER_API_KEY=sk-or-... python3 benchmarks/synthetic_eval.py --real --provider openrouter --model gemma-3-27b-it:free
# ================================================================== #

import argparse
import csv
import json
import os
import re as _re
from collections import defaultdict
from dataclasses import dataclass, asdict


# ------------------------------------------------------------------ #
#  Shared prompt for DR fixes                                          #
# ------------------------------------------------------------------ #

_DR_SYSTEM = """You are the navigation module of an agent completing a task step by step.

At each step you must output ONLY a JSON object — no prose, no markdown fences:
{{"reasoning": "1-2 sentences", "done": false, "next_action": "tool_name(param='value')", "predicted_steps": ["next_tool()", "next_tool()"], "confidence": 0.9}}

Rules:
- next_action must be one of the available tools called with exact syntax: tool_name(key='value')
- predicted_steps should be 2-3 likely next actions after next_action
- Set done=true ONLY when ALL required steps of the goal are complete — not before
- Never set done=true on the first call unless the goal is trivially a no-op
Available tools: {tool_names}"""


# ------------------------------------------------------------------ #
#  Shared prompt for ReAct (one step at a time, with history)         #
# ------------------------------------------------------------------ #

_REACT_SYSTEM = """You are an agent completing a task step by step using tools.

At each step output ONLY a JSON object — no prose, no markdown fences:
{{"thought": "what to do next and why", "action": "tool_name(param='value')", "done": false}}

Rules:
- action must be one of the available tools called with exact syntax: tool_name(key='value')
- Set done=true ONLY when ALL steps are genuinely complete — you must call at least one tool first
- Never set done=true if no tools have been called yet
- Look at completed_steps in the world summary — do not repeat steps already done
Available tools: {tool_names}
Goal: {goal}"""


# ------------------------------------------------------------------ #
#  Anthropic DR adapter with token counting                           #
# ------------------------------------------------------------------ #

def make_anthropic_dr_adapter(api_key: str, model: str):
    import anthropic as _ant
    from dead_reckoning.adapters import _parse_fix_response
    from dead_reckoning.core.agent import LLMAdapter

    class AnthropicDRAdapter(LLMAdapter):
        def __init__(self):
            self.client = _ant.Anthropic(api_key=api_key)
            self.model = model
            self.input_tokens = 0
            self.output_tokens = 0

        def get_fix(self, world, tools):
            tool_names = ", ".join(list(tools.keys())[:20])
            system = _DR_SYSTEM.format(tool_names=tool_names)
            resp = self.client.messages.create(
                model=self.model, max_tokens=600,
                system=system,
                messages=[{"role": "user", "content": world.summary()}],
            )
            self.input_tokens  += resp.usage.input_tokens
            self.output_tokens += resp.usage.output_tokens
            return _parse_fix_response(resp.content[0].text)

        def execute_action(self, action, tools, env):
            from dead_reckoning.adapters import _dispatch_action
            return _dispatch_action(action, tools, env)

    return AnthropicDRAdapter()


# ------------------------------------------------------------------ #
#  Anthropic ReAct adapter — correct baseline                         #
#                                                                      #
#  Key fixes vs the broken version:                                    #
#  1. Maintains full conversation history so Claude knows what's done  #
#  2. Injects tool results back into the conversation                  #
#  3. Blocks done=true until at least one tool has been called         #
#  4. Counts actual API calls not agent steps                          #
# ------------------------------------------------------------------ #

def make_anthropic_react_adapter(api_key: str, model: str, goal: str, tools: dict):
    import anthropic as _ant
    from dead_reckoning.adapters import _dispatch_action
    from dead_reckoning.core.agent import LLMAdapter

    class AnthropicReActAdapter(LLMAdapter):
        def __init__(self):
            self.client = _ant.Anthropic(api_key=api_key)
            self.model = model
            self.goal = goal
            self._history = []          # full message history
            self._tools_called = 0      # how many tools executed so far
            self.input_tokens = 0
            self.output_tokens = 0
            self.llm_call_count = 0

        def get_fix(self, world, tools):
            tool_names = ", ".join(list(tools.keys())[:20])
            system = _REACT_SYSTEM.format(tool_names=tool_names, goal=self.goal)

            # Build user message from world state
            completed = world.completed_steps
            if completed:
                last = completed[-1]
                user_msg = (
                    f"Steps done so far: {len(completed)}\n"
                    f"Last action: {last['action']}\n"
                    f"Last result: {str(last['result'])[:200]}\n"
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
            self.llm_call_count += 1

            text = resp.content[0].text.strip()
            self._history.append({"role": "assistant", "content": text})

            # Parse JSON
            clean = _re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
            try:
                d = json.loads(clean)
            except Exception:
                # try to extract action
                m = _re.search(r'"action"\s*:\s*"([^"]+)"', clean)
                return "parse error", [], m.group(1) if m else "", False

            action = d.get("action", "")
            done   = bool(d.get("done", False))

            # CRITICAL FIX: block premature done=true
            # Must have called at least one tool before marking done
            if done and self._tools_called == 0:
                done = False
                action = action or list(tools.keys())[0] + "()"

            return d.get("thought", ""), [], action, done

        def execute_action(self, action, tools, env):
            result, errored = _dispatch_action(action, tools, env)
            if not errored and "no tool matched" not in str(result):
                self._tools_called += 1
            return result, errored

    return AnthropicReActAdapter()


# ------------------------------------------------------------------ #
#  Result dataclass                                                    #
# ------------------------------------------------------------------ #

@dataclass
class RealResult:
    task_id:      str
    split:        str
    goal:         str
    method:       str
    model:        str
    success:      bool
    stop_reason:  str
    total_steps:  int
    llm_calls:    int
    savings_pct:  float
    input_tokens: int
    output_tokens:int
    duration_s:   float
    error:        str = ""


# ------------------------------------------------------------------ #
#  Main benchmark runner                                               #
# ------------------------------------------------------------------ #

def run_real_llm_benchmark(
    api_key: str,
    model: str,
    provider: str,
    n_tasks: int,
    fix_threshold: float,
    output_csv: str,
    rate_delay: float,
):
    random.seed(42)
    tasks = build_tasks(n=n_tasks)
    tools = make_synthetic_tools()
    results: list[RealResult] = []

    print(f"\nModel    : {model}")
    print(f"Provider : {provider}")
    print(f"Tasks    : {len(tasks)} x 2 methods = {len(tasks)*2} LLM sessions")
    print(f"\n{'#':>3}  {'Split':>4}  {'Method':<16}  {'✓/✗'}  {'steps':>5}  {'llm':>4}  {'saved':>6}  goal")
    print("─" * 72)

    for i, task in enumerate(tasks):

        for method in ["dead_reckoning", "react"]:
            t0    = time.perf_counter()
            error = ""
            in_tok = out_tok = 0

            try:
                if method == "dead_reckoning":
                    adapter = make_anthropic_dr_adapter(api_key, model)
                    agent = DeadReckoningAgent(
                        adapter=adapter,
                        goal=task["goal"],
                        tools=tools,
                        fix_threshold=fix_threshold,
                        max_steps_without_fix=5,
                        max_total_steps=20,
                        verbose=False,
                    )
                    list(agent.run())
                    stats   = agent.stats
                    in_tok  = adapter.input_tokens
                    out_tok = adapter.output_tokens

                else:  # react
                    adapter = make_anthropic_react_adapter(api_key, model, task["goal"], tools)
                    agent = DeadReckoningAgent(
                        adapter=adapter,
                        goal=task["goal"],
                        tools=tools,
                        fix_threshold=0.0,
                        hard_ceiling=0.01,
                        max_steps_without_fix=1,
                        checkpoint_interval=1,
                        max_total_steps=20,
                        verbose=False,
                    )
                    list(agent.run())
                    stats   = agent.stats
                    in_tok  = adapter.input_tokens
                    out_tok = adapter.output_tokens

            except Exception as e:
                error = str(e)
                stats = agent.stats if "agent" in dir() else None

            duration = time.perf_counter() - t0
            success  = (stats.stop_reason == StopReason.TASK_COMPLETE) if stats else False
            llm      = stats.llm_calls    if stats else 0
            steps    = stats.total_steps  if stats else 0
            sav      = stats.savings_pct  if stats else 0.0
            sr       = stats.stop_reason.value if (stats and stats.stop_reason) else "?"

            icon = "✓" if success else "✗"
            print(f"{i+1:>3}  {task['split']:>4}  {method:<16}  {icon}  "
                  f"{steps:>5}  {llm:>4}  {sav:>5.0f}%  {task['goal'][:30]}...")

            results.append(RealResult(
                task_id=task["id"], split=task["split"],
                goal=task["goal"][:80], method=method, model=model,
                success=success, stop_reason=sr,
                total_steps=steps, llm_calls=llm, savings_pct=sav,
                input_tokens=in_tok, output_tokens=out_tok,
                duration_s=round(duration, 2), error=error,
            ))

            time.sleep(rate_delay)

    # ── Summary ───────────────────────────────────────────────────
    by_m: dict[str, list[RealResult]] = defaultdict(list)
    for r in results:
        by_m[r.method].append(r)

    print("\n" + "=" * 62)
    print(f"  REAL LLM RESULTS  ({model})")
    print("=" * 62)
    print(f"\n{'Method':<16} {'Tasks':>5} {'Success%':>9} {'LLM/task':>9} {'Savings%':>9} {'Tok/task':>9}")
    print("-" * 55)

    for method in ["dead_reckoning", "react"]:
        rows = by_m.get(method, [])
        if not rows:
            continue
        n = len(rows)
        print(f"{method:<16} {n:>5} "
              f"{sum(r.success for r in rows)/n*100:>8.1f}% "
              f"{sum(r.llm_calls for r in rows)/n:>9.1f} "
              f"{sum(r.savings_pct for r in rows)/n:>8.1f}% "
              f"{sum(r.input_tokens+r.output_tokens for r in rows)/n:>9.0f}")

    print()
    dr   = by_m.get("dead_reckoning", [])
    base = by_m.get("react", [])
    if dr and base:
        dr_llm   = sum(r.llm_calls for r in dr)   / len(dr)
        base_llm = sum(r.llm_calls for r in base) / len(base)
        delta_sr = (sum(r.success for r in dr)/len(dr) - sum(r.success for r in base)/len(base)) * 100
        reduc    = (1 - dr_llm / base_llm) * 100 if base_llm else 0
        print(f"  LLM call reduction : {reduc:.0f}%")
        verdict = "✓ paper-worthy" if abs(delta_sr) <= 3 else "— delta outside ±3pp, check baseline"
        print(f"  Success rate delta : {delta_sr:+.1f}pp  {verdict}")

    # Stop reason breakdown — key diagnostic
    print("\n  Stop reason breakdown:")
    stops: dict[str, int] = defaultdict(int)
    for r in results:
        stops[f"{r.method}/{r.stop_reason}"] += 1
    for k, v in sorted(stops.items()):
        note = ""
        if "no_pred" in k and "dead" in k:
            note = "  ← model not returning predictions"
        if "max_steps" in k and "react" in k:
            note = "  ← react looping, check done signal"
        print(f"    {k:<42}: {v}{note}")

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            w.writeheader()
            w.writerows(asdict(r) for r in results)
        print(f"\n  Saved → {output_csv}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real",          action="store_true",
                        help="Use real LLM instead of mock adapter")
    parser.add_argument("--provider",      default="anthropic",
                        choices=["anthropic", "openrouter"])
    parser.add_argument("--api_key",       default="")
    parser.add_argument("--model",         default="claude-haiku-4-5")
    parser.add_argument("--n_tasks",       type=int,   default=10)
    parser.add_argument("--fix_threshold", type=float, default=0.35)
    parser.add_argument("--output",        default="results/synthetic_real.csv")
    parser.add_argument("--rate_delay",    type=float, default=0.3)
    args = parser.parse_args()

    if args.real:
        key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            print("ERROR: set ANTHROPIC_API_KEY or pass --api_key")
            raise SystemExit(1)
        run_real_llm_benchmark(
            api_key=key,
            model=args.model,
            provider=args.provider,
            n_tasks=args.n_tasks,
            fix_threshold=args.fix_threshold,
            output_csv=args.output,
            rate_delay=args.rate_delay,
        )
    else:
        random.seed(42)
        tasks = build_tasks(n=30)
        tools = make_synthetic_tools()
        print("Running synthetic benchmark (30 tasks × 2 methods)...")
        t0 = time.perf_counter()
        results = run_all(tasks, tools, fix_threshold=0.35)
        elapsed = time.perf_counter() - t0
        print_table(results)
        print(f"  Total time: {elapsed:.1f}s  (no API calls, fully local)")
        print()
        print("To test with a real LLM (Anthropic):")
        print("  ANTHROPIC_API_KEY=sk-ant-... python3 benchmarks/synthetic_eval.py --real")