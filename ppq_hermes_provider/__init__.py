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
    signup_url="https://ppq.ai",
    env_vars=("PPQ_API_KEY", "PPQ_BASE_URL"),
    base_url="https://api.ppq.ai/v1",
    auth_type="api_key",
    api_mode="chat_completions",
    default_aux_model="openai/gpt-5.4-mini",
    default_headers={"User-Agent": "ppq-hermes-plugin/0.1.0"},
    fallback_models=(
        # Popular — premium
        "claude-sonnet-4.6",
        "claude-opus-4.7",
        "gpt-5.5",
        "gpt-5.3-chat",
        "gpt-5.3-codex",
        "gpt-5.5-pro",
        "grok-4.20",
        "sonar-reasoning",
        # Popular — budget
        "claude-haiku-4.5",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gemini-3-flash-preview",
        # Affordable open / third-party
        "deepseek/deepseek-r1",
        "deepseek/deepseek-chat-v3-0324",
        "meta-llama/llama-3.3-70b-instruct",
        "google/gemini-flash-1.5",
        "mistralai/mistral-small-3.1-24b-instruct",
    ),
)

register_provider(ppq)
