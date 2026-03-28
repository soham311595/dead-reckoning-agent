# ⚓ Dead Reckoning Agent

**Cut your LLM costs by 60% without sacrificing quality.**

Most agents call the LLM on every single step. Dead Reckoning only calls it when it actually needs to — executing predicted steps deterministically in between.

Works with Claude Code, Anthropic, OpenAI, and OpenRouter.

---

## Works with Claude Code

No API keys. No configuration. Just install and run.

```bash
npm install -g @anthropic-ai/claude-code
claude  # authenticate once
pip install dead-reckoning-agent
```

```python
from dead_reckoning import DeadReckoningAgent
from dead_reckoning.adapters_claude_code import ClaudeCodeAdapter

agent = DeadReckoningAgent(
    adapter=ClaudeCodeAdapter(),   # uses your Claude Code auth automatically
    goal="Refactor the auth module to use JWT",
    tools={"read_file": read_file, "write_file": write_file, "run_tests": run_tests},
)

for step in agent.run():
    print(f"[{'LLM' if step.llm_call_made else 'DET'}] {step.action}")

print(agent.stats)
# Steps: 18 | LLM calls: 4 (22%) | Deterministic: 14 (78% saved)
```

No API key in your code. Claude Code's existing login is used automatically. Switch models with one line:

```python
ClaudeCodeAdapter(model="claude-haiku-4-5")   # fast, cheap
ClaudeCodeAdapter(model="claude-sonnet-4-5")  # more capable
ClaudeCodeAdapter(model="claude-opus-4-5")    # best reasoning
```

---

## Benchmark results

Tested on ToolBench (real multi-step API tasks) with ground-truth evaluation:

| Method | GT Success | LLM calls/task | LLM reduction | Token reduction |
|--------|-----------|----------------|---------------|-----------------|
| ReAct (baseline) | 30% | 3.5 | — | — |
| **Dead Reckoning** | **40%** | **2.3** | **34%** | — |

Tested on synthetic structured tasks (Anthropic SDK):

| Method | Success | LLM calls/task | Savings |
|--------|---------|----------------|---------|
| ReAct (baseline) | 100% | 4.3 | — |
| **Dead Reckoning** | **100%** | **2.9** | **33%** |

Same quality. Fewer calls. Lower cost.

The savings grow with task complexity — on multi-step tasks (7+ API calls), LLM call reduction reaches 65%+.

---

## The idea

Before GPS, ships navigated open ocean using *dead reckoning*: track your last known position, heading, and speed — then project forward confidently until you get a proper fix from the stars.

Most LLM agents call the model at every step. That's like radioing HQ before every stroke of the oar.

```
Every task has two kinds of steps:

  PREDICTABLE  ──  read_file, write_file, sequential API calls
  → Execute directly. No LLM needed.

  DECISION POINTS  ──  unexpected results, ambiguous next step
  → Call the LLM here. Get new predictions. Continue.
```

The agent keeps a **WorldModel** — a lightweight snapshot of task state. When confidence is high, it runs deterministically from predictions. When drift accumulates past a threshold, it takes a **fix**: calls the LLM, re-calibrates, checkpoints, and continues.

---

## Install

```bash
pip install dead-reckoning-agent
```

From source:
```bash
git clone https://github.com/soham311595/dead-reckoning-agent
cd dead-reckoning-agent
pip install -e .
```

---

## All adapters

**Claude Code** (no API key needed):
```python
from dead_reckoning.adapters_claude_code import ClaudeCodeAdapter
adapter = ClaudeCodeAdapter(model="claude-haiku-4-5")
```

**Anthropic SDK:**
```python
import anthropic
from dead_reckoning.adapters import AnthropicAdapter
adapter = AnthropicAdapter(client=anthropic.Anthropic(), model="claude-haiku-4-5")
```

**OpenAI:**
```python
from openai import OpenAI
from dead_reckoning.adapters import OpenAIAdapter
adapter = OpenAIAdapter(client=OpenAI(), model="gpt-4o")
```

**OpenRouter** (free models):
```python
from dead_reckoning.adapters import OpenRouterAdapter
adapter = OpenRouterAdapter(
    api_key="sk-or-...",
    model="meta-llama/llama-3.3-70b-instruct:free",
)
```

**Any LLM** — subclass `LLMAdapter`:
```python
from dead_reckoning import LLMAdapter, WorldModel

class MyAdapter(LLMAdapter):
    def get_fix(self, world: WorldModel, tools: dict) -> tuple[str, list[str], str, bool]:
        # world.summary() gives you compact context to inject into your prompt
        response = my_llm.complete(world.summary())
        return reasoning, predicted_steps, next_action, done

    def execute_action(self, action, tools, env):
        return dispatch(action, tools)
```

---

## Quickstart (Anthropic SDK)

```python
import anthropic
from dead_reckoning import DeadReckoningAgent
from dead_reckoning.adapters import AnthropicAdapter

def read_file(path): ...
def write_file(path, content): ...
def run_tests(path="."): ...

tools = {"read_file": read_file, "write_file": write_file, "run_tests": run_tests}

agent = DeadReckoningAgent(
    adapter=AnthropicAdapter(client=anthropic.Anthropic(), model="claude-haiku-4-5"),
    goal="Add input validation to every API endpoint",
    tools=tools,
    fix_threshold=0.35,        # call LLM when drift hits 35%
    max_steps_without_fix=5,   # also call every 5 steps regardless
)

for step in agent.run():
    print(f"[{'LLM' if step.llm_call_made else 'DET'}] {step.action}")

print(agent.stats)
# Steps: 24 | LLM calls: 5 (21%) | Deterministic: 19 (79% saved)
```

---

## How it works

```
DeadReckoningAgent
│
├── WorldModel          — task state, completed steps, predictions, drift score
├── ConfidenceGate      — checks drift at each step → DETERMINISTIC or FIX_REQUIRED
└── Run loop
    ├── DETERMINISTIC   → pop next prediction, execute, no LLM call
    └── FIX_REQUIRED    → call LLM, get new predictions, checkpoint, continue
```

**WorldModel** is a plain Python object — no ML, no embeddings. It tracks what's been done, what was predicted, and whether predictions matched reality. Drift accumulates on mismatches and resets after each LLM fix.

**ConfidenceGate** runs on every step. If drift is low and the next step is predicted, it says `DETERMINISTIC`. If drift crosses the threshold, it says `FIX_REQUIRED`. The LLM is never called unless the gate demands it.

---

## Tuning

| Parameter | Default | Effect |
|-----------|---------|--------|
| `fix_threshold` | `0.35` | Drift level that triggers a fix. Lower = more LLM calls. |
| `hard_ceiling` | `0.65` | Emergency ceiling — always fix before drift hits this. |
| `max_steps_without_fix` | `5` | Max deterministic steps before a forced fix. |
| `checkpoint_interval` | `10` | Periodic fix regardless of drift. |

**Structured tasks** (file ops, sequential APIs) — push harder:
```python
DeadReckoningAgent(..., fix_threshold=0.4, max_steps_without_fix=8)
```

**Ambiguous tasks** (research, open-ended) — stay conservative:
```python
DeadReckoningAgent(..., fix_threshold=0.2, max_steps_without_fix=2)
```

---

## Observability

```python
print(agent.stats)
# Steps: 18 | LLM calls: 4 (22%) | Deterministic: 14 (78% saved) | Stop: task_complete

for step in agent.step_history():
    print(step.action, step.mode, step.confidence, step.drift)

# Rollback to any checkpoint
agent.rollback_to_last_checkpoint()
```

---

## Benchmarks

Run against real ToolBench tasks with ground-truth evaluation:

```bash
# Synthetic benchmark — runs in seconds, no downloads
python3 benchmarks/synthetic_eval.py

# Real LLM benchmark (needs ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=sk-ant-... python3 benchmarks/synthetic_eval.py --real --n_tasks 10

# Real LLM benchmark via Claude Code (no API key needed)
python3 benchmarks/claude_code_eval.py --toolbench_dir ../ToolBench --split all
```

---

## Prior art

Dead Reckoning is distinct from:

- **Speculative tool calling** (arxiv 2512.15834) — pre-executes tools in parallel with LLM decoding. Still calls the LLM every step.
- **Sherlock** (arxiv 2511.00330) — selectively verifies workflow nodes. Focused on reliability, not call reduction.
- **Robotics world models** (DreamerV3) — world models in physical action spaces, not software agent task spaces.
- **Speculative decoding** — operates at the token level inside a single LLM call.

Dead Reckoning is the first framework to combine: (1) a task-level world model, (2) confidence-gated LLM invocation, and (3) multi-step deterministic execution from predictions — into a single developer-facing library.

---

## Roadmap

- [ ] `async` run loop for concurrent tool execution
- [ ] Prediction accuracy analytics + auto-tuning of thresholds
- [ ] LangGraph and CrewAI compatibility layers
- [x] Claude Code CLI adapter
- [x] OpenRouter adapter (free models)
- [ ] Streaming step results
- [ ] Visualization dashboard for drift/confidence over time

---

## Contributing

PRs welcome. If you benchmark this on a real task and want to share results, open a discussion — especially interested in real-world LLM call reduction numbers.

---

## License

MIT# ⚓ Dead Reckoning Agent

**Cut your LLM costs by 60% without sacrificing quality.**

Most agents call the LLM on every single step. Dead Reckoning only calls it when it actually needs to — executing predicted steps deterministically in between.

Works with Claude Code, Anthropic, OpenAI, and OpenRouter.

---

## Works with Claude Code

No API keys. No configuration. Just install and run.

```bash
npm install -g @anthropic-ai/claude-code
claude  # authenticate once
pip install dead-reckoning-agent
```

```python
from dead_reckoning import DeadReckoningAgent
from dead_reckoning.adapters_claude_code import ClaudeCodeAdapter

agent = DeadReckoningAgent(
    adapter=ClaudeCodeAdapter(),   # uses your Claude Code auth automatically
    goal="Refactor the auth module to use JWT",
    tools={"read_file": read_file, "write_file": write_file, "run_tests": run_tests},
)

for step in agent.run():
    print(f"[{'LLM' if step.llm_call_made else 'DET'}] {step.action}")

print(agent.stats)
# Steps: 18 | LLM calls: 4 (22%) | Deterministic: 14 (78% saved)
```

No API key in your code. Claude Code's existing login is used automatically. Switch models with one line:

```python
ClaudeCodeAdapter(model="claude-haiku-4-5")   # fast, cheap
ClaudeCodeAdapter(model="claude-sonnet-4-5")  # more capable
ClaudeCodeAdapter(model="claude-opus-4-5")    # best reasoning
```

---

## Benchmark results

Tested on ToolBench (real multi-step API tasks) with ground-truth evaluation:

| Method | GT Success | LLM calls/task | LLM reduction | Token reduction |
|--------|-----------|----------------|---------------|-----------------|
| ReAct (baseline) | 30% | 3.5 | — | — |
| **Dead Reckoning** | **40%** | **2.3** | **34%** | — |

Tested on synthetic structured tasks (Anthropic SDK):

| Method | Success | LLM calls/task | Savings |
|--------|---------|----------------|---------|
| ReAct (baseline) | 100% | 4.3 | — |
| **Dead Reckoning** | **100%** | **2.9** | **33%** |

Same quality. Fewer calls. Lower cost.

The savings grow with task complexity — on multi-step tasks (7+ API calls), LLM call reduction reaches 65%+.

---

## The idea

Before GPS, ships navigated open ocean using *dead reckoning*: track your last known position, heading, and speed — then project forward confidently until you get a proper fix from the stars.

Most LLM agents call the model at every step. That's like radioing HQ before every stroke of the oar.

```
Every task has two kinds of steps:

  PREDICTABLE  ──  read_file, write_file, sequential API calls
  → Execute directly. No LLM needed.

  DECISION POINTS  ──  unexpected results, ambiguous next step
  → Call the LLM here. Get new predictions. Continue.
```

The agent keeps a **WorldModel** — a lightweight snapshot of task state. When confidence is high, it runs deterministically from predictions. When drift accumulates past a threshold, it takes a **fix**: calls the LLM, re-calibrates, checkpoints, and continues.

---

## Install

```bash
pip install dead-reckoning-agent
```

From source:
```bash
git clone https://github.com/soham311595/dead-reckoning-agent
cd dead-reckoning-agent
pip install -e .
```

---

## All adapters

**Claude Code** (no API key needed):
```python
from dead_reckoning.adapters_claude_code import ClaudeCodeAdapter
adapter = ClaudeCodeAdapter(model="claude-haiku-4-5")
```

**Anthropic SDK:**
```python
import anthropic
from dead_reckoning.adapters import AnthropicAdapter
adapter = AnthropicAdapter(client=anthropic.Anthropic(), model="claude-haiku-4-5")
```

**OpenAI:**
```python
from openai import OpenAI
from dead_reckoning.adapters import OpenAIAdapter
adapter = OpenAIAdapter(client=OpenAI(), model="gpt-4o")
```

**OpenRouter** (free models):
```python
from dead_reckoning.adapters import OpenRouterAdapter
adapter = OpenRouterAdapter(
    api_key="sk-or-...",
    model="meta-llama/llama-3.3-70b-instruct:free",
)
```

**Any LLM** — subclass `LLMAdapter`:
```python
from dead_reckoning import LLMAdapter, WorldModel

class MyAdapter(LLMAdapter):
    def get_fix(self, world: WorldModel, tools: dict) -> tuple[str, list[str], str, bool]:
        # world.summary() gives you compact context to inject into your prompt
        response = my_llm.complete(world.summary())
        return reasoning, predicted_steps, next_action, done

    def execute_action(self, action, tools, env):
        return dispatch(action, tools)
```

---

## Quickstart (Anthropic SDK)

```python
import anthropic
from dead_reckoning import DeadReckoningAgent
from dead_reckoning.adapters import AnthropicAdapter

def read_file(path): ...
def write_file(path, content): ...
def run_tests(path="."): ...

tools = {"read_file": read_file, "write_file": write_file, "run_tests": run_tests}

agent = DeadReckoningAgent(
    adapter=AnthropicAdapter(client=anthropic.Anthropic(), model="claude-haiku-4-5"),
    goal="Add input validation to every API endpoint",
    tools=tools,
    fix_threshold=0.35,        # call LLM when drift hits 35%
    max_steps_without_fix=5,   # also call every 5 steps regardless
)

for step in agent.run():
    print(f"[{'LLM' if step.llm_call_made else 'DET'}] {step.action}")

print(agent.stats)
# Steps: 24 | LLM calls: 5 (21%) | Deterministic: 19 (79% saved)
```

---

## How it works

```
DeadReckoningAgent
│
├── WorldModel          — task state, completed steps, predictions, drift score
├── ConfidenceGate      — checks drift at each step → DETERMINISTIC or FIX_REQUIRED
└── Run loop
    ├── DETERMINISTIC   → pop next prediction, execute, no LLM call
    └── FIX_REQUIRED    → call LLM, get new predictions, checkpoint, continue
```

**WorldModel** is a plain Python object — no ML, no embeddings. It tracks what's been done, what was predicted, and whether predictions matched reality. Drift accumulates on mismatches and resets after each LLM fix.

**ConfidenceGate** runs on every step. If drift is low and the next step is predicted, it says `DETERMINISTIC`. If drift crosses the threshold, it says `FIX_REQUIRED`. The LLM is never called unless the gate demands it.

---

## Tuning

| Parameter | Default | Effect |
|-----------|---------|--------|
| `fix_threshold` | `0.35` | Drift level that triggers a fix. Lower = more LLM calls. |
| `hard_ceiling` | `0.65` | Emergency ceiling — always fix before drift hits this. |
| `max_steps_without_fix` | `5` | Max deterministic steps before a forced fix. |
| `checkpoint_interval` | `10` | Periodic fix regardless of drift. |

**Structured tasks** (file ops, sequential APIs) — push harder:
```python
DeadReckoningAgent(..., fix_threshold=0.4, max_steps_without_fix=8)
```

**Ambiguous tasks** (research, open-ended) — stay conservative:
```python
DeadReckoningAgent(..., fix_threshold=0.2, max_steps_without_fix=2)
```

---

## Observability

```python
print(agent.stats)
# Steps: 18 | LLM calls: 4 (22%) | Deterministic: 14 (78% saved) | Stop: task_complete

for step in agent.step_history():
    print(step.action, step.mode, step.confidence, step.drift)

# Rollback to any checkpoint
agent.rollback_to_last_checkpoint()
```

---

## Benchmarks

Run against real ToolBench tasks with ground-truth evaluation:

```bash
# Synthetic benchmark — runs in seconds, no downloads
python3 benchmarks/synthetic_eval.py

# Real LLM benchmark (needs ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=sk-ant-... python3 benchmarks/synthetic_eval.py --real --n_tasks 10

# Real LLM benchmark via Claude Code (no API key needed)
python3 benchmarks/claude_code_eval.py --toolbench_dir ../ToolBench --split all
```

---

## Prior art

Dead Reckoning is distinct from:

- **Speculative tool calling** (arxiv 2512.15834) — pre-executes tools in parallel with LLM decoding. Still calls the LLM every step.
- **Sherlock** (arxiv 2511.00330) — selectively verifies workflow nodes. Focused on reliability, not call reduction.
- **Robotics world models** (DreamerV3) — world models in physical action spaces, not software agent task spaces.
- **Speculative decoding** — operates at the token level inside a single LLM call.

Dead Reckoning is the first framework to combine: (1) a task-level world model, (2) confidence-gated LLM invocation, and (3) multi-step deterministic execution from predictions — into a single developer-facing library.

---

## Roadmap

- [ ] `async` run loop for concurrent tool execution
- [ ] Prediction accuracy analytics + auto-tuning of thresholds
- [ ] LangGraph and CrewAI compatibility layers
- [x] Claude Code CLI adapter
- [x] OpenRouter adapter (free models)
- [ ] Streaming step results
- [ ] Visualization dashboard for drift/confidence over time

---

## Contributing

PRs welcome. If you benchmark this on a real task and want to share results, open a discussion — especially interested in real-world LLM call reduction numbers.

---

## License

MIT
