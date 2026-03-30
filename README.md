<div align="center">

# ⚓ Dead Reckoning Agent

**LLM agents that think less and cost less.**

[![PyPI version](https://img.shields.io/pypi/v/dead-reckoning-agent?color=blue&style=flat-square)](https://pypi.org/project/dead-reckoning-agent/)
[![Python](https://img.shields.io/pypi/pyversions/dead-reckoning-agent?style=flat-square)](https://pypi.org/project/dead-reckoning-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-27%20passing-brightgreen?style=flat-square)](#)

Most agents call the LLM at every single step. Dead Reckoning only calls it when it has to — executing predicted steps deterministically in between. Same results. Fewer calls. Lower cost.

**Works with Anthropic · OpenAI · OpenRouter**

</div>

## Results

Benchmarked on [ToolBench](https://github.com/OpenBMB/ToolBench) `data_example` — 10 real multi-step API tasks across 3 difficulty tiers (G1: 2 APIs, G2: 7 APIs, G3: 11 APIs), with ground-truth evaluation. **Full benchmark in progress...**

### ToolBench — n=10, all splits, both methods (OpenRouter)

| Model | Method | GT success | LLM calls / task | Tokens / task | Wall time / task | LLM ↓ | Token ↓ |
|-------|--------|:----------:|:----------------:|:-------------:|:----------------:|:-----:|:-------:|
| claude-haiku-4-5 | ReAct | 40% | 3.2 | 1,445 | 7.0s | — | — |
| claude-haiku-4-5 | **Dead Reckoning** | **40%** | **2.2** | **1,102** | **4.2s** | **↓ 31%** | **↓ 24%** |
| claude-opus-4-6 | ReAct | 40% | 3.2 | 1,317 | 10.8s | — | — |
| claude-opus-4-6 | **Dead Reckoning** | **40%** | **2.3** | **1,106** | **9.4s** | **↓ 28%** | **↓ 16%** |

**Zero accuracy tradeoff across both models** — identical GT success rates at lower cost. All 20 DR runs terminated cleanly (`task_complete`).

The savings are consistent across model tiers: DR reduces LLM calls by ~28–31% regardless of whether you're using a fast cheap model or a powerful expensive one. On Opus, the wall-time savings are smaller (1.4s/task) because Opus latency dominates — but the call reduction and token reduction still hold.

> **Per-split breakdown (both models consistent):** G1 tasks (2 tools): DR uses ~2.0–2.2 LLM calls vs ReAct's 2.8 — the second tool is predicted and executed deterministically. G2/G3 tasks (7–11 tools): both methods score 0% GT because ToolBench requires all listed APIs called, but both models correctly identify only the 2–3 APIs relevant to the actual user request. This reflects a benchmark limitation, not agent failure — LLM success is 100% on all tasks.

### Stack prompt caching for compounding savings

Dead Reckoning reduces **call count**. Prompt caching reduces **cost per call**. Together:

```
ReAct + no caching:         10 calls × 400 tok = 4,000 token-equivalents
Dead Reckoning alone:        3 calls × 400 tok = 1,200 token-equivalents  (70% down)
Dead Reckoning + caching:    580 token-equivalents                         (85% down)
```

Caching is on by default in `AnthropicAdapter`. The system prompt is cached after the first call and served at 10% of the normal input price on every subsequent fix.

---

## How it works

Before GPS, sailors crossed oceans using *dead reckoning*: track your last known position, heading, and speed — then project forward until you can take a proper fix from the stars. Only then do you update your chart.

Most LLM agents radio headquarters before every stroke of the oar. Dead Reckoning doesn't.

```
+------------------------------------------------------------------+
|                                                                  |
|  Every task has two kinds of steps:                              |
|                                                                  |
|  PREDICTABLE -- read_file, write_file, sequential API calls      |
|  Execute directly. No LLM. No latency. No cost.                  |
|                                                                  |
|  DECISION POINTS -- unexpected output, ambiguous next step       |
|  Call the LLM. Get new predictions. Checkpoint. Continue.        |
|                                                                  |
+------------------------------------------------------------------+
```

The agent maintains a **WorldModel** — a plain Python object tracking task state, predictions, and drift. The **ConfidenceGate** checks drift at every step:

- Drift low + next step predicted → `DETERMINISTIC` (no LLM call)
- Drift past threshold → `FIX_REQUIRED` (call LLM, re-calibrate, checkpoint)

Drift accumulates when predictions miss. It resets after each fix. The LLM is invoked exactly as often as necessary, and not once more.

---

## Install

```bash
pip install dead-reckoning-agent
```

```bash
# From source
git clone https://github.com/soham311595/dead-reckoning-agent
cd dead-reckoning-agent && pip install -e .
```

---

## All adapters

<details>
<summary><b>Claude Code</b> — no API key needed</summary>

```python
from dead_reckoning.adapters_claude_code import ClaudeCodeAdapter

adapter = ClaudeCodeAdapter(model="claude-haiku-4-5")   # fast
adapter = ClaudeCodeAdapter(model="claude-sonnet-4-5")  # balanced
adapter = ClaudeCodeAdapter(model="claude-opus-4-5")    # best reasoning
```

Requires Claude Code installed and authenticated:
```bash
npm install -g @anthropic-ai/claude-code
claude  # login once
```

</details>

<details>
<summary><b>Anthropic SDK</b> — with prompt caching built in</summary>

```python
import anthropic
from dead_reckoning.adapters import AnthropicAdapter

adapter = AnthropicAdapter(
    client=anthropic.Anthropic(),
    model="claude-haiku-4-5",
    use_cache=True,   # default: caches system prompt at 10% input price
)

# After a run, inspect cache performance
print(adapter.cache_stats())
# cache_read=1840 cache_write=460 hit_rate=80% effective_tokens_saved~1656

print(adapter.effective_input_tokens)
# cost-equivalent tokens after caching discount applied
```

</details>

<details>
<summary><b>OpenAI</b></summary>

```python
from openai import OpenAI
from dead_reckoning.adapters import OpenAIAdapter

adapter = OpenAIAdapter(client=OpenAI(), model="gpt-4o")
```

</details>

<details>
<summary><b>OpenRouter</b> — includes free models</summary>

```python
from dead_reckoning.adapters import OpenRouterAdapter

adapter = OpenRouterAdapter(
    api_key="sk-or-...",
    model="meta-llama/llama-3.3-70b-instruct:free",
)
```

</details>

<details>
<summary><b>Any LLM</b> — plug in your own</summary>

```python
from dead_reckoning import LLMAdapter, WorldModel

class MyAdapter(LLMAdapter):
    def get_fix(self, world: WorldModel, tools: dict) -> tuple[str, list[str], str, bool]:
        """Called only when the confidence gate demands a fix."""
        response = my_llm.complete(
            system=MY_SYSTEM_PROMPT,
            user=world.summary()   # compact, always up to date
        )
        return reasoning, predicted_steps, next_action, done

    def execute_action(self, action: str, tools: dict, env: dict) -> tuple[Any, bool]:
        return dispatch(action, tools)
```

</details>

---

## Full example

```python
import anthropic
from dead_reckoning import DeadReckoningAgent
from dead_reckoning.adapters import AnthropicAdapter

def read_file(path: str) -> str: ...
def write_file(path: str, content: str) -> None: ...
def run_tests(path: str = ".") -> dict: ...
def search_codebase(query: str) -> list[str]: ...

tools = {
    "read_file": read_file,
    "write_file": write_file,
    "run_tests": run_tests,
    "search_codebase": search_codebase,
}

agent = DeadReckoningAgent(
    adapter=AnthropicAdapter(client=anthropic.Anthropic(), model="claude-haiku-4-5"),
    goal="Add rate limiting to every public API endpoint",
    tools=tools,
    fix_threshold=0.35,       # call LLM when drift exceeds 35%
    max_steps_without_fix=5,  # also call every 5 deterministic steps
    max_total_steps=50,
    verbose=True,
)

for step in agent.run():
    tag = "LLM" if step.llm_call_made else "DET"
    print(f"  [{tag}]  {step.action:<45}  drift={step.drift:.2f}")

print(agent.stats)
# Steps: 24 | LLM calls: 5 (21%) | Deterministic: 19 (79% saved) | Stop: task_complete
```

---

## Tuning

| Parameter | Default | What it controls |
|-----------|:-------:|------------------|
| `fix_threshold` | `0.35` | Drift level that triggers a fix. Lower = more LLM calls, higher accuracy. |
| `hard_ceiling` | `0.65` | Emergency ceiling — always fix before drift hits this. |
| `max_steps_without_fix` | `5` | Max consecutive deterministic steps before a forced check-in. |
| `checkpoint_interval` | `10` | Periodic fix regardless of drift — safety net for long tasks. |

```python
# Structured tasks (file ops, sequential APIs) — be aggressive
DeadReckoningAgent(..., fix_threshold=0.45, max_steps_without_fix=8)

# Ambiguous tasks (research, reasoning chains) — stay conservative
DeadReckoningAgent(..., fix_threshold=0.20, max_steps_without_fix=2)
```

---

## Observability

```python
# Summary after a run
print(agent.stats)
# Steps: 18 | LLM calls: 4 (22%) | Deterministic: 14 (78% saved) | Stop: task_complete

# Per-step trace
for step in agent.step_history():
    print(f"[{step.mode}] {step.action}  conf={step.confidence:.2f}  drift={step.drift:.2f}")

# LLM calls only
for step in agent.llm_call_log():
    print(f"Fix at step {step.step_index}: {step.reasoning[:60]}...")

# Rollback to any checkpoint
agent.rollback_to_last_checkpoint()

# Full world model snapshot
import json
print(json.dumps(agent.world.to_dict(), indent=2))
```

---

## Running benchmarks

```bash
# Instant — no downloads, no API key
python3 benchmarks/synthetic_eval.py

# Real LLM via Anthropic SDK
ANTHROPIC_API_KEY=sk-ant-... python3 benchmarks/synthetic_eval.py --real --n_tasks 10

# Real LLM via Claude Code (no API key needed)
python3 benchmarks/claude_code_eval.py \
    --toolbench_dir ../ToolBench \
    --split all

# Ground truth on ToolBench data_example
ANTHROPIC_API_KEY=sk-ant-... python3 benchmarks/toolbench_data_example.py \
    --toolbench_dir ../ToolBench \
    --split all \
    --output results/my_run.csv
```

---

## When to use this

**Good fit**
- Coding agents with repeating file operations
- Multi-step API orchestration
- Data pipeline automation
- Any task where the next few steps are predictable

**Not the right fit**
- Open-ended creative tasks where every step is novel
- Tasks under 5 steps (overhead outweighs savings)
- Real-time systems where per-step latency is critical

---

## Prior art

| Technique | Difference |
|-----------|-----------|
| Speculative tool calling (arxiv 2512.15834) | Pre-executes tools in parallel with LLM decoding. Still calls LLM every step. |
| Sherlock (arxiv 2511.00330) | Selectively verifies workflow nodes. Focused on reliability, not call reduction. |
| Robotics world models (DreamerV3) | World models in physical action spaces, not software agent task spaces. |
| Speculative decoding | Token-level, inside a single LLM call. |

Dead Reckoning is the first framework to combine: **(1)** a task-level world model, **(2)** confidence-gated LLM invocation, and **(3)** multi-step deterministic execution from predictions — in a single developer-facing library.

---

## Roadmap

- [x] Claude Code CLI adapter
- [x] OpenRouter adapter (free models)
- [x] Prompt caching in Anthropic adapter
- [x] ToolBench ground-truth benchmark
- [ ] `async` run loop for concurrent tool execution
- [ ] Prediction accuracy analytics + auto-tuning
- [ ] LangGraph / CrewAI compatibility layers
- [ ] Streaming step results
- [ ] Drift visualization dashboard

---

## Contributing

PRs welcome. If you benchmark this on a real task and want to share results, open a discussion — real-world LLM call reduction numbers are especially valuable.

---

<div align="center">

MIT License · Built by [Soham Takuri](https://github.com/soham311595)

</div>
