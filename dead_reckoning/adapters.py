"""
Ready-to-use LLM adapters.

from dead_reckoning.adapters import AnthropicAdapter, OpenAIAdapter
"""

from __future__ import annotations
import json
import re
from typing import Any, Callable

from .core.agent import LLMAdapter
from .core.world_model import WorldModel


_FIX_SYSTEM = """You are the navigation module of a Dead Reckoning Agent.

Your job at each "fix" is to:
1. Understand where the agent is in the task (from the world model summary).
2. Decide the single BEST next action to take RIGHT NOW — or determine the task is done.
3. Predict the next {n_predictions} steps that will likely follow (only if structured enough).

Respond ONLY with valid JSON matching this schema:
{{
  "reasoning": "brief chain-of-thought (2-4 sentences)",
  "done": false,
  "next_action": "the exact action/tool call to execute now (empty string if done=true)",
  "predicted_steps": ["step2", "step3", ...],
  "confidence": 0.0-1.0
}}

Available tools: {tool_names}

Set "done": true ONLY when the original goal has been fully achieved.
Be decisive. Predictions should be specific enough to execute without ambiguity.
If the next steps are genuinely unpredictable, return an empty predicted_steps list.
"""


def _parse_fix_response(text: str) -> tuple[str, list[str], str, bool]:
    """Parse LLM fix response -> (reasoning, predicted_steps, next_action, done)."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        data = json.loads(text)
        return (
            data.get("reasoning", ""),
            data.get("predicted_steps", []),
            data.get("next_action", ""),
            bool(data.get("done", False)),
        )
    except json.JSONDecodeError:
        action_match = re.search(r'"next_action"\s*:\s*"([^"]+)"', text)
        action = action_match.group(1) if action_match else ""
        done_match = re.search(r'"done"\s*:\s*(true|false)', text)
        done = bool(done_match and done_match.group(1) == "true")
        return ("parse error — proceeding with partial data", [], action, done)


class AnthropicAdapter(LLMAdapter):
    """
    Adapter for Anthropic Claude models with optional prompt caching.

    Prompt caching marks the static system prompt with cache_control so
    Anthropic caches it server-side. Subsequent fix calls in the same task
    read the system prompt from cache at 10% of the normal input price —
    a 90% reduction on that portion.

    This stacks on top of Dead Reckoning's call reduction:
      - DR:      fewer LLM calls overall
      - Caching: cheaper input tokens on each call made

    Usage:
        import anthropic
        client = anthropic.Anthropic()

        # With caching (default, recommended)
        adapter = AnthropicAdapter(client=client, model="claude-haiku-4-5")

        # Without caching
        adapter = AnthropicAdapter(client=client, model="claude-haiku-4-5", use_cache=False)
    """

    def __init__(
        self,
        client,
        model: str = "claude-haiku-4-5",
        n_predictions: int = 4,
        max_tokens: int = 512,
        use_cache: bool = True,
    ):
        self.client = client
        self.model = model
        self.n_predictions = n_predictions
        self.max_tokens = max_tokens
        self.use_cache = use_cache

        # Token tracking
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0       # tokens served from cache (cheap)
        self.cache_creation_tokens = 0   # tokens written to cache (slightly more)

    def get_fix(
        self,
        world: WorldModel,
        tools: dict[str, Callable],
    ) -> tuple[str, list[str], str, bool]:
        tool_names = list(tools.keys()) if tools else ["none"]
        system_text = _FIX_SYSTEM.format(
            n_predictions=self.n_predictions,
            tool_names=", ".join(tool_names),
        )

        # Build system parameter — list format required for cache_control
        if self.use_cache:
            system_param = [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system_param = system_text

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_param,
            messages=[{"role": "user", "content": world.summary()}],
        )

        # Track token usage including cache metrics
        u = response.usage
        self.input_tokens            += getattr(u, "input_tokens", 0) or 0
        self.output_tokens           += getattr(u, "output_tokens", 0) or 0
        self.cache_read_tokens       += getattr(u, "cache_read_input_tokens", 0) or 0
        self.cache_creation_tokens   += getattr(u, "cache_creation_input_tokens", 0) or 0

        return _parse_fix_response(response.content[0].text)

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of cacheable calls that hit the cache (0.0 on first call)."""
        total = self.cache_read_tokens + self.cache_creation_tokens
        return self.cache_read_tokens / total if total > 0 else 0.0

    @property
    def effective_input_tokens(self) -> float:
        """
        Cost-equivalent input tokens after caching discount.
        Cache reads cost 10% of normal input price.
        Cache writes cost 125% of normal input price.
        Regular input tokens cost 100%.
        """
        return (
            self.input_tokens * 1.0
            + self.cache_creation_tokens * 1.25
            + self.cache_read_tokens * 0.10
        )

    def cache_stats(self) -> str:
        saved = self.cache_read_tokens * 0.90  # tokens saved vs full price
        return (
            f"cache_read={self.cache_read_tokens} "
            f"cache_write={self.cache_creation_tokens} "
            f"hit_rate={self.cache_hit_rate:.0%} "
            f"effective_tokens_saved≈{saved:.0f}"
        )

    def execute_action(
        self,
        action: str,
        tools: dict[str, Callable],
        env: dict[str, Any],
    ) -> tuple[Any, bool]:
        return _dispatch_action(action, tools, env)


class OpenAIAdapter(LLMAdapter):
    """
    Adapter for OpenAI models.

    Usage:
        from openai import OpenAI
        client = OpenAI()
        adapter = OpenAIAdapter(client=client, model="gpt-4o")
    """

    def __init__(
        self,
        client,  # openai.OpenAI instance
        model: str = "gpt-4o",
        n_predictions: int = 4,
        max_tokens: int = 512,
    ):
        self.client = client
        self.model = model
        self.n_predictions = n_predictions
        self.max_tokens = max_tokens

    def get_fix(
        self,
        world: WorldModel,
        tools: dict[str, Callable],
    ) -> tuple[str, list[str], str, bool]:
        tool_names = list(tools.keys()) if tools else ["none"]
        system = _FIX_SYSTEM.format(
            n_predictions=self.n_predictions,
            tool_names=", ".join(tool_names),
        )
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": world.summary()},
            ],
        )
        return _parse_fix_response(response.choices[0].message.content)

    def execute_action(
        self,
        action: str,
        tools: dict[str, Callable],
        env: dict[str, Any],
    ) -> tuple[Any, bool]:
        return _dispatch_action(action, tools, env)


# ------------------------------------------------------------------ #
#  Shared action dispatcher                                            #
# ------------------------------------------------------------------ #

def _dispatch_action(
    action: str,
    tools: dict[str, Callable],
    env: dict[str, Any],
) -> tuple[Any, bool]:
    """
    Route an action string to the appropriate tool.

    Supports three formats:
      "tool_name(key='val', key2=val2)"  → kwargs call (handles single/double quotes)
      "tool_name: arg"                   → positional call with one arg
      "tool_name"                        → no-arg call
    """
    action = action.strip()

    # Format 1: function-call syntax  tool(key='val', ...)
    fn_match = re.match(r"^(\w+)\((.*)\)$", action, re.DOTALL)
    if fn_match:
        fn_name, args_str = fn_match.group(1), fn_match.group(2).strip()
        if fn_name in tools:
            try:
                if not args_str:
                    return tools[fn_name](), False
                kwargs = _parse_kwargs(args_str)
                if kwargs is not None:
                    return tools[fn_name](**kwargs), False
                # fallback: single positional arg
                return tools[fn_name](args_str), False
            except Exception as e:
                return str(e), True

    # Format 2: "tool_name: arg"
    colon_match = re.match(r"^(\w+):\s*(.+)$", action)
    if colon_match:
        fn_name, arg = colon_match.group(1), colon_match.group(2).strip()
        if fn_name in tools:
            try:
                return tools[fn_name](arg), False
            except Exception as e:
                return str(e), True

    # Format 3: bare tool name
    if action in tools:
        try:
            return tools[action](), False
        except Exception as e:
            return str(e), True

    return f"[no tool matched for: {action!r}]", False


def _parse_kwargs(args_str: str) -> dict | None:
    """
    Parse a keyword-argument string like: path='auth/login.py', content='hello'

    Handles both single and double quoted string values, bare integers/floats/bools,
    and nested content with commas inside quoted strings.
    Returns None if parsing fails so caller can fall back.
    """
    if not args_str.strip():
        return {}

    result = {}
    # Tokenise: key=value pairs where value may be a quoted string or a literal
    pattern = re.compile(
        r"""(\w+)\s*=\s*(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)"|([^,\s]+))""",
        re.DOTALL,
    )
    found = pattern.findall(args_str)
    if not found:
        return None

    for key, sq_val, dq_val, bare_val in found:
        if sq_val != "" or (args_str.find(f"{key}='") != -1):
            result[key] = sq_val
        elif dq_val != "" or (args_str.find(f'{key}="') != -1):
            result[key] = dq_val
        else:
            # coerce bare values
            v = bare_val.strip()
            if v.lower() == "true":
                result[key] = True
            elif v.lower() == "false":
                result[key] = False
            else:
                try:
                    result[key] = int(v)
                except ValueError:
                    try:
                        result[key] = float(v)
                    except ValueError:
                        result[key] = v

    return result if result else None


class OpenRouterAdapter(LLMAdapter):
    """
    Adapter for OpenRouter — gives access to free and cheap models
    (Llama, Mistral, Gemma, DeepSeek, etc.) via an OpenAI-compatible API.

    Get a free API key at: https://openrouter.ai/keys

    Free models (as of 2026, check openrouter.ai/models?q=free for latest):
        "meta-llama/llama-3.3-70b-instruct:free"
        "mistralai/mistral-7b-instruct:free"
        "google/gemma-3-27b-it:free"
        "deepseek/deepseek-r1:free"
        "microsoft/phi-3-mini-128k-instruct:free"

    Usage:
        adapter = OpenRouterAdapter(
            api_key="sk-or-...",
            model="meta-llama/llama-3.3-70b-instruct:free",
        )
    """

    def __init__(
        self,
        api_key: str,
        model: str = "meta-llama/llama-3.3-70b-instruct:free",
        n_predictions: int = 4,
        max_tokens: int = 512,
        site_url: str = "https://github.com/your-org/dead-reckoning-agent",
        site_name: str = "Dead Reckoning Agent",
    ):
        self.api_key = api_key
        self.model = model
        self.n_predictions = n_predictions
        self.max_tokens = max_tokens
        # OpenRouter passes these through for rankings/analytics (optional)
        self.extra_headers = {
            "HTTP-Referer": site_url,
            "X-Title": site_name,
        }

    def _client(self):
        """Lazy-import openai so it's not a hard dependency."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required for OpenRouterAdapter: pip3 install openai"
            )
        return OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=self.extra_headers,
        )

    def get_fix(
        self,
        world: WorldModel,
        tools: dict[str, Callable],
    ) -> tuple[str, list[str], str, bool]:
        tool_names = list(tools.keys()) if tools else ["none"]
        system = _FIX_SYSTEM.format(
            n_predictions=self.n_predictions,
            tool_names=", ".join(tool_names[:30]),  # cap to avoid huge prompts
        )
        response = self._client().chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": world.summary()},
            ],
        )
        text = response.choices[0].message.content or ""
        return _parse_fix_response(text)

    def execute_action(
        self,
        action: str,
        tools: dict[str, Callable],
        env: dict[str, Any],
    ) -> tuple[Any, bool]:
        return _dispatch_action(action, tools, env)
