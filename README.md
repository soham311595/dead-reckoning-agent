# ⚓ Dead Reckoning Agent

**An LLM agent that navigates tasks like a ship navigates open water.**

Before GPS, ships crossed oceans using *dead reckoning*: track your last known position, your heading, and your speed — then project forward confidently until you can take a proper fix from the stars.

Most LLM agents call the model at every single step. That's like radioing headquarters before every stroke of the oar.

Dead Reckoning Agent does it differently.

---

## The idea

```
Every agent task has two kinds of steps:

  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  PREDICTABLE  ──  read_file, write_file, sequential ops     │
  │  These don't need an LLM. Execute them directly.            │
  │                                                             │
  │  DECISION POINTS  ──  what to do given unexpected output    │
  │  These need the LLM. Invoke it here, and only here.         │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
```

The agent maintains a **WorldModel** — a lightweight snapshot of task state. When confidence is high, it executes steps *deterministically* from predictions. When drift accumulates past a threshold, it takes a **fix**: invokes the LLM, re-calibrates, takes a checkpoint, and continues.

The result: **60–80% fewer LLM calls** on structured tasks, with no loss of reasoning quality at decision points.

---

## Results

```
Task: Refactor auth module (18 steps)

  Step  Mode      Conf   Drift  LLM   Action
  ────  ────────  ─────  ─────  ───   ─────────────────────────────────
     1  ◆ FIX    0.80   0.20  YES   list_files(path='auth')          [cp_0000]
     2  ● DET    0.80   0.20        analyze_file(path='auth/login.py')
     3  ● DET    0.70   0.30        analyze_file(path='auth/logout.py')
     4  ◆ FIX    0.80   0.20  YES   read_file(path='auth/utils.py')  [cp_0001]
     5  ● DET    0.80   0.20        write_file(path='auth/utils.py')
     6  ● DET    0.70   0.30        analyze_file(path='auth/middleware.py')
     7  ◆ FIX    0.80   0.20  YES   read_file(path='auth/login.py')  [cp_0002]
    ...

  Total steps     : 18
  LLM calls made  : 6   (33% of steps)
  Deterministic   : 12  (67% LLM-free) ✓
  Checkpoints     : 6

  Naive agent: 18 LLM calls.
  Dead Reckoning: 6 calls. 67% reduction.
```

---

## Install

```bash
pip install dead-reckoning-agent
```

Or from source:
```bash
git clone https://github.com/your-org/dead-reckoning-agent
cd dead-reckoning-agent
pip install -e .
```

---

## Quickstart

```python
import anthropic
from dead_reckoning import DeadReckoningAgent
from dead_reckoning.adapters import AnthropicAdapter

# Your tools
def read_file(path): ...
def write_file(path, content): ...
def run_tests(path="."): ...

tools = {"read_file": read_file, "write_file": write_file, "run_tests": run_tests}

# Wire it up
client = anthropic.Anthropic()
adapter = AnthropicAdapter(client=client, model="claude-opus-4-5")

agent = DeadReckoningAgent(
    adapter=adapter,
    goal="Add input validation to every API endpoint",
    tools=tools,
    fix_threshold=0.35,     # invoke LLM when drift hits 35%
    max_steps_without_fix=5,  # also fix every 5 steps regardless
)

for step in agent.run():
    print(f"[{'LLM' if step.llm_call_made else 'DET'}] {step.action}")

print(agent.stats)
# Steps: 24 | LLM calls: 5 (21%) | Deterministic: 19 (79% saved)
```

Works with OpenAI too:
```python
from openai import OpenAI
from dead_reckoning.adapters import OpenAIAdapter

adapter = OpenAIAdapter(client=OpenAI(), model="gpt-4o")
```

---

## How it works

```
+--------------------------------------------------------------+
|                      DeadReckoningAgent                      |
|                                                              |
|  +----------------+  +----------------+  +----------------+  |
|  |   WorldModel   |  | ConfidenceGate |  | ExecutionMode  |  |
|  |                |  |                |  |                |  |
|  | - goal         |  | - drift >=     |  | DETERMINISTIC  |  |
|  | - history      |  |   threshold?   |  | - run from     |  |
|  | - env state    |  | - steps >=     |  |   predictions  |  |
|  | - predictions  |  |   max_fix?     |  |                |  |
|  | - drift        |  | - tool error?  |  | FIX_REQUIRED   |  |
|  |                |  |                |  | - call LLM     |  |
|  |                |  |                |  | - checkpoint   |  |
|  |                |  |                |  | - new preds    |  |
|  +----------------+  +----------------+  +----------------+  |
|          ^                                        |          |
|          |                                        |          |
|          +----------------------------------------+          |
|                                                              |
+--------------------------------------------------------------+
```

**WorldModel** tracks task state and accumulates drift as predictions miss or steps deviate from expectations. Think of it as the ship's log.

**ConfidenceGate** evaluates the world model at every step and returns one of three execution modes:
- `DETERMINISTIC` — run without LLM
- `FIX_REQUIRED` — invoke LLM immediately (drift threshold hit)
- `CHECKPOINT` — scheduled LLM invocation + snapshot

**Drift** accumulates when:
- A predicted step doesn't match what actually happened
- A tool returns an unexpected error
- Too many steps have passed since the last fix

**Drift decreases** when:
- Predictions are accurate
- A new LLM fix is taken

---

## Tuning

| Parameter | Default | Effect |
|-----------|---------|--------|
| `fix_threshold` | `0.35` | Drift level that triggers LLM re-invocation. Lower = more LLM calls, higher quality. |
| `hard_ceiling` | `0.65` | Hard stop — always invoke LLM before drift hits this. |
| `max_steps_without_fix` | `5` | Max consecutive deterministic steps before forced fix. |
| `checkpoint_interval` | `10` | Periodic LLM check-in regardless of drift. |

**For highly structured tasks** (file operations, sequential API calls):
```python
agent = DeadReckoningAgent(..., fix_threshold=0.4, max_steps_without_fix=8)
# More aggressive: fewer LLM calls, lean on predictions
```

**For ambiguous tasks** (research, open-ended reasoning):
```python
agent = DeadReckoningAgent(..., fix_threshold=0.2, max_steps_without_fix=2)
# Conservative: LLM re-invoked often, high confidence maintained
```

---

## Custom adapters

Plug in any LLM by subclassing `LLMAdapter`:

```python
from dead_reckoning import LLMAdapter, WorldModel

class MyAdapter(LLMAdapter):
    def get_fix(self, world: WorldModel, tools: dict) -> tuple[str, list[str], str]:
        """
        Returns: (reasoning, predicted_next_steps, immediate_action)
        Called only when confidence gate demands a fix.
        """
        response = my_llm.complete(
            system=FIX_SYSTEM_PROMPT,
            user=world.summary()  # compact context injection
        )
        return parse(response)

    def execute_action(self, action: str, tools: dict, env: dict) -> tuple[Any, bool]:
        """
        Returns: (result, errored)
        """
        return dispatch(action, tools)
```

---

## Rollback

Every LLM fix creates a checkpoint. You can roll back to any of them:

```python
# Something went wrong
agent.rollback_to_last_checkpoint()

# Or pick a specific checkpoint
step = agent.step_history()[5]
if step.checkpoint_id:
    agent.world.rollback(step.checkpoint_id)
```

---

## Observability

```python
# After a run
print(agent.stats)
# Steps: 24 | LLM calls: 5 (21%) | Deterministic: 19 (79% saved)

# Per-step log
for step in agent.step_history():
    print(step.action, step.mode, step.confidence, step.drift)

# Just the LLM invocations
for step in agent.llm_call_log():
    print(f"Fix at step {step.step_index}: confidence {step.confidence:.2f}")

# World model snapshot
import json
print(json.dumps(agent.world.to_dict(), indent=2))
```

---

## When to use this

**Great fit:**
- Coding agents (file read/write/analyze cycles)
- Data pipeline automation
- Multi-step API orchestration
- Any task with repeating patterns

**Not the right fit:**
- Open-ended creative tasks (every step needs the LLM)
- Tasks where the next step is always surprising
- Tasks under 5 steps (overhead not worth it)

---

## Prior art & novelty

This pattern is distinct from:

- **Speculative tool calling** (arxiv 2512.15834) — pre-executes tools in parallel with LLM decoding. Still calls the LLM at every step.
- **Sherlock** (arxiv 2511.00330) — selectively verifies workflow nodes. Focused on reliability, not LLM call reduction.
- **Robotics world models** (DreamerV3, etc.) — world models in physical action spaces, not software agent task spaces.
- **Speculative decoding** — operates at the token level within a single LLM call.

Dead Reckoning Agent is the first framework to combine: (1) a task-level world model, (2) confidence-gated LLM invocation, and (3) multi-step deterministic execution from predictions — into a single developer-facing library.

---

## Roadmap

- [ ] `async` run loop for concurrent tool execution
- [ ] Prediction accuracy analytics + auto-tuning of thresholds
- [ ] LangGraph and CrewAI compatibility layers
- [ ] Streaming step results
- [ ] Built-in benchmark suite (WebArena, ToolBench)
- [ ] Visualization dashboard for drift/confidence over time

---

## Contributing

Issues and PRs welcome. If you're benchmarking this against a real LLM on a structured task and want to share results, open a discussion.

---

## License

MIT
