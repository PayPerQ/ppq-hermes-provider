from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class PPQProfile(ProviderProfile):
    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        if supports_reasoning:
            if reasoning_config is not None:
                extra_body["reasoning"] = dict(reasoning_config)
            else:
                extra_body["reasoning"] = {"enabled": True, "effort": "medium"}
        return extra_body, {}

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        import urllib.request
        import json

        url = f"{self.base_url}/models"
        req = urllib.request.Request(url)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            models = data.get("data", [])
            return [
                m["id"]
                for m in models
                if isinstance(m, dict) and m.get("type") == "chat"
            ]
        except Exception:
            return None


ppq = PPQProfile(
    name="ppq",
    aliases=("payperq", "ppq-inference"),
    display_name="PayPerQ",
    description="PayPerQ — OpenAI-compatible inference API",
    signup_url="https://payperq.ai",
    env_vars=("PPQ_API_KEY", "PPQ_BASE_URL"),
    base_url="https://api.ppq.ai/v1",
    auth_type="api_key",
    api_mode="chat_completions",
    default_aux_model="openai/gpt-4o-mini",
    default_headers={"User-Agent": "ppq-hermes-plugin/0.1.0"},
    fallback_models=(
        # OpenAI
        "gpt-4o",
        "gpt-4o-mini",
        "openai/gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "o1",
        "o1-mini",
        "o1-preview",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        # Anthropic
        "anthropic/claude-3.7-sonnet",
        "anthropic/claude-haiku-4.5",
        "anthropic/claude-3-opus:beta",
        "anthropic/claude-3-haiku:beta",
        # Google
        "google/gemini-flash-1.5",
        # Meta
        "meta-llama/llama-3.1-405b-instruct",
        "meta-llama/llama-3-70b-instruct",
        # Mistral
        "mistralai/mixtral-8x7b-instruct",
        # DeepSeek
        "deepseek/deepseek-r1",
    ),
)

register_provider(ppq)
