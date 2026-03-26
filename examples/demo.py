"""
Dead Reckoning Agent — self-contained demo.

Run with:  python examples/demo.py

This demo uses a MockAdapter that simulates LLM responses, so you can
see the architecture in action without an API key.

A real run would replace MockAdapter with AnthropicAdapter or OpenAIAdapter.
"""

from __future__ import annotations
import sys
import time

sys.path.insert(0, "..")

from dead_reckoning import (
    DeadReckoningAgent,
    LLMAdapter,
    WorldModel,
    ExecutionMode,
)
from typing import Any, Callable


# ------------------------------------------------------------------ #
#  Mock tools (simulate a codebase refactor task)                      #
# ------------------------------------------------------------------ #

FILES = {
    "auth/login.py": "def login(user, pass): return db.check(user, pass)",
    "auth/logout.py": "def logout(token): return db.delete(token)",
    "auth/register.py": "def register(user, pass): return db.insert(user, pass)",
    "auth/middleware.py": "def verify(request): token = request.headers.get('token')",
    "auth/utils.py": "def hash_password(p): return hashlib.md5(p)",
}

ANALYSIS_RESULTS = {
    "auth/login.py": {"issues": ["plaintext password comparison", "no rate limiting"]},
    "auth/logout.py": {"issues": ["token not invalidated server-side"]},
    "auth/register.py": {"issues": ["MD5 hash used", "no password validation"]},
    "auth/middleware.py": {"issues": ["token decoded without verification"]},
    "auth/utils.py": {"issues": ["MD5 deprecated", "no salt"]},
}

def list_files(path: str = "auth") -> list[str]:
    return [f for f in FILES if f.startswith(path)]

def read_file(path: str) -> str:
    return FILES.get(path, f"[not found: {path}]")

def analyze_file(path: str) -> dict:
    return ANALYSIS_RESULTS.get(path, {"issues": []})

def write_file(path: str, content: str = "") -> str:
    FILES[path] = content
    return f"✓ Written {path}"

def run_tests(path: str = ".") -> dict:
    time.sleep(0.05)  # simulate test run
    return {"passed": 12, "failed": 0, "path": path}


TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "analyze_file": analyze_file,
    "write_file": write_file,
    "run_tests": run_tests,
}


# ------------------------------------------------------------------ #
#  Mock LLM Adapter                                                    #
# ------------------------------------------------------------------ #

class MockAdapter(LLMAdapter):
    """
    Simulates an LLM that understands a code refactor task.
    Returns realistic fix responses to demonstrate the architecture.
    """

    _fix_sequence = [
        # Fix 0: initial
        {
            "reasoning": "Auth module needs security refactor. I'll analyze files in order, then rewrite each with JWT.",
            "next_action": "list_files(path='auth')",
            "done": False,
            "predicted_steps": [
                "analyze_file(path='auth/login.py')",
                "analyze_file(path='auth/logout.py')",
                "analyze_file(path='auth/register.py')",
            ],
        },
        # Fix 1: after drift from unexpected analyze result
        {
            "reasoning": "login.py and logout.py analyzed. register.py has MD5 issue — high priority. Will rewrite auth/utils.py first since it's the root cause.",
            "next_action": "read_file(path='auth/utils.py')",
            "done": False,
            "predicted_steps": [
                "write_file(path='auth/utils.py')",
                "analyze_file(path='auth/middleware.py')",
                "write_file(path='auth/middleware.py')",
            ],
        },
        # Fix 2: scheduled checkpoint
        {
            "reasoning": "Good progress. utils.py and middleware rewritten. Now update login and register with new JWT utils.",
            "next_action": "read_file(path='auth/login.py')",
            "done": False,
            "predicted_steps": [
                "write_file(path='auth/login.py')",
                "read_file(path='auth/register.py')",
                "write_file(path='auth/register.py')",
                "run_tests(path='auth')",
            ],
        },
    ]

    def __init__(self):
        self._call_count = 0

    def get_fix(self, world: WorldModel, tools: dict) -> tuple[str, list[str], str]:
        idx = min(self._call_count, len(self._fix_sequence) - 1)
        fix = self._fix_sequence[idx]
        self._call_count += 1
        return fix["reasoning"], fix["predicted_steps"], fix["next_action"], fix.get("done", False)

    def execute_action(self, action: str, tools: dict, env: dict) -> tuple[Any, bool]:
        from dead_reckoning.adapters import _dispatch_action
        return _dispatch_action(action, tools, env)


# ------------------------------------------------------------------ #
#  Run the demo                                                        #
# ------------------------------------------------------------------ #

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
GRAY   = "\033[90m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def fmt_mode(mode: ExecutionMode) -> str:
    if mode == ExecutionMode.DETERMINISTIC:
        return f"{GREEN}●{RESET} DET"
    return f"{YELLOW}◆{RESET} FIX"


def run_demo():
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  ⚓  Dead Reckoning Agent  —  Demo Run{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}\n")
    print(f"  Goal: {CYAN}Refactor the auth module to use JWT and bcrypt{RESET}\n")
    print(f"  {'Step':>4}  {'Mode':8}  {'Conf':5}  {'Drift':5}  {'LLM':4}  {'Action'}")
    print(f"  {'────':>4}  {'────────':8}  {'─────':5}  {'─────':5}  {'───':4}  {'──────────────────────────────────────────'}")

    agent = DeadReckoningAgent(
        adapter=MockAdapter(),
        goal="Refactor the auth module to use JWT and bcrypt",
        tools=TOOLS,
        fix_threshold=0.35,
        max_steps_without_fix=4,
        checkpoint_interval=6,
        max_total_steps=18,
        verbose=False,
    )

    for step in agent.run():
        llm_marker = f"{YELLOW}YES{RESET}" if step.llm_call_made else f"{GRAY}   {RESET}"
        action_display = step.action[:50] + "…" if len(step.action) > 50 else step.action
        cp_marker = f" {GRAY}[{step.checkpoint_id}]{RESET}" if step.checkpoint_id else ""

        print(
            f"  {step.step_index:>4}  {fmt_mode(step.mode):8}  "
            f"{step.confidence:.2f}   {step.drift:.2f}   {llm_marker}  "
            f"{action_display}{cp_marker}"
        )
        time.sleep(0.04)  # make it feel alive

    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  Results{RESET}")
    print(f"{'─'*60}")
    stats = agent.stats
    print(f"  Total steps        : {stats.total_steps}")
    print(f"  LLM calls made     : {YELLOW}{stats.llm_calls}{RESET}  ({stats.llm_call_rate:.0%} of steps)")
    print(f"  Deterministic steps: {GREEN}{stats.deterministic_steps}{RESET}  ({stats.savings_pct:.0f}% LLM-free)")
    print(f"  Checkpoints        : {stats.checkpoints}")
    print(f"\n  {BOLD}Equivalent naive agent would have made {stats.total_steps} LLM calls.{RESET}")
    print(f"  {GREEN}Dead Reckoning used {stats.llm_calls} — a {stats.savings_pct:.0f}% reduction.{RESET}")
    print(f"\n{'─'*60}\n")


if __name__ == "__main__":
    run_demo()
