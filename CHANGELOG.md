# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-03-25

### Added
- `WorldModel` — task-state tracker with drift accumulation and checkpoint/rollback
- `ConfidenceGate` — evaluates drift and returns DETERMINISTIC / FIX_REQUIRED / CHECKPOINT
- `DeadReckoningAgent` — main run loop with explicit `StopReason` on every exit path
- `AnthropicAdapter` and `OpenAIAdapter` — plug-and-play LLM adapters
- `_dispatch_action` — action string dispatcher supporting `tool(key='val')` syntax
- `RunStats` with `stop_reason`, `savings_pct`, and `llm_call_rate`
- Demo script with mock adapter showing 67% LLM call reduction
- 12 unit tests covering WorldModel and ConfidenceGate
