"""
Integration and unit tests for the PPQ Hermes provider plugin.

Set PPQ_API_KEY env var to run integration tests:
    PPQ_API_KEY=sk-... pytest tests/
    pytest tests/ -m unit   # unit tests only (no API key needed)
"""
import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Make the plugin importable without a Hermes install by stubbing the providers module.
_HERMES_AGENT = Path.home() / ".hermes" / "hermes-agent"
if _HERMES_AGENT.exists():
    sys.path.insert(0, str(_HERMES_AGENT))
else:
    # Minimal stubs so tests can import the plugin in CI without Hermes installed.
    from dataclasses import dataclass, field
    from typing import Any

    OMIT_TEMPERATURE = object()

    @dataclass
    class ProviderProfile:
        name: str = ""
        aliases: tuple = ()
        display_name: str = ""
        description: str = ""
        signup_url: str = ""
        env_vars: tuple = ()
        base_url: str = ""
        auth_type: str = "api_key"
        api_mode: str = "chat_completions"
        default_aux_model: str = ""
        default_headers: dict = field(default_factory=dict)
        fallback_models: tuple = ()
        fixed_temperature: Any = None
        default_max_tokens: int | None = None
        models_url: str = ""

        def build_api_kwargs_extras(self, *, reasoning_config=None, **_):
            return {}, {}

        def fetch_models(self, *, api_key=None, timeout=8.0):
            return None

    _registry = {}

    def register_provider(p):
        _registry[p.name] = p

    fake_providers = type(sys)("providers")
    fake_providers.register_provider = register_provider
    fake_providers.ProviderProfile = ProviderProfile
    fake_base = type(sys)("providers.base")
    fake_base.ProviderProfile = ProviderProfile
    fake_base.OMIT_TEMPERATURE = OMIT_TEMPERATURE
    sys.modules["providers"] = fake_providers
    sys.modules["providers.base"] = fake_base

import importlib.util

_PLUGIN_INIT = Path(__file__).parent.parent / "ppq_hermes_provider" / "__init__.py"

def _load_plugin():
    spec = importlib.util.spec_from_file_location("ppq_hermes_provider", _PLUGIN_INIT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_mod = _load_plugin()
ppq = _mod.ppq

PPQ_API_KEY = os.environ.get("PPQ_API_KEY", "")
BASE_URL = "https://api.ppq.ai/v1"

import pytest


# ── 1. Plugin loads correctly ────────────────────────────────────────────────

@pytest.mark.unit
def test_provider_name():
    assert ppq.name == "ppq"

@pytest.mark.unit
def test_provider_aliases():
    assert "payperq" in ppq.aliases

@pytest.mark.unit
def test_base_url():
    assert ppq.base_url == BASE_URL

@pytest.mark.unit
def test_auth_type():
    assert ppq.auth_type == "api_key"

@pytest.mark.unit
def test_api_mode():
    assert ppq.api_mode == "chat_completions"

@pytest.mark.unit
def test_class_is_ppq_profile():
    assert type(ppq).__name__ == "PPQProfile"


# ── 2. User-Agent header ─────────────────────────────────────────────────────

@pytest.mark.unit
def test_user_agent_header():
    assert "User-Agent" in ppq.default_headers
    assert ppq.default_headers["User-Agent"].startswith("ppq-hermes-plugin/")


# ── 3. Reasoning passthrough ─────────────────────────────────────────────────

@pytest.mark.unit
def test_reasoning_with_config():
    extra_body, top = ppq.build_api_kwargs_extras(
        supports_reasoning=True,
        reasoning_config={"effort": "high"},
    )
    assert extra_body == {"reasoning": {"effort": "high"}}
    assert top == {}

@pytest.mark.unit
def test_reasoning_default_when_no_config():
    extra_body, top = ppq.build_api_kwargs_extras(supports_reasoning=True)
    assert extra_body == {"reasoning": {"enabled": True, "effort": "medium"}}
    assert top == {}

@pytest.mark.unit
def test_reasoning_not_sent_when_unsupported():
    extra_body, top = ppq.build_api_kwargs_extras(supports_reasoning=False)
    assert extra_body == {}
    assert top == {}


# ── 4. fetch_models — unit (mocked HTTP) ────────────────────────────────────

@pytest.mark.unit
def test_fetch_models_filters_chat_only():
    mock_response_data = {
        "data": [
            {"id": "gpt-4o", "type": "chat"},
            {"id": "tts-model", "type": "tts"},
            {"id": "claude-3", "type": "chat"},
            {"id": "dall-e-3", "type": "image"},
            {"id": "deepseek-r1", "type": "chat"},
        ]
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_response_data).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        models = ppq.fetch_models(api_key="test-key")

    assert models == ["gpt-4o", "claude-3", "deepseek-r1"]
    assert "tts-model" not in models
    assert "dall-e-3" not in models

@pytest.mark.unit
def test_fetch_models_returns_none_on_error():
    with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
        result = ppq.fetch_models(api_key="test-key")
    assert result is None


# ── 5. Integration tests (require PPQ_API_KEY) ───────────────────────────────

@pytest.mark.integration
@pytest.mark.skipif(not PPQ_API_KEY, reason="PPQ_API_KEY not set")
def test_fetch_models_live():
    # Verify chat completions auth works before testing model list
    import urllib.request
    req = urllib.request.Request(
        f"{BASE_URL}/models",
        headers={"Authorization": f"Bearer {PPQ_API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200

    models = ppq.fetch_models(api_key=PPQ_API_KEY)
    assert models is not None
    assert len(models) > 24
    for m in models:
        assert isinstance(m, str) and m

@pytest.mark.integration
@pytest.mark.skipif(not PPQ_API_KEY, reason="PPQ_API_KEY not set")
def test_basic_chat():
    import urllib.request
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
        "max_tokens": 10,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {PPQ_API_KEY}",
            "Content-Type": "application/json",
            **ppq.default_headers,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    content = data["choices"][0]["message"]["content"]
    assert isinstance(content, str) and len(content) > 0

@pytest.mark.integration
@pytest.mark.skipif(not PPQ_API_KEY, reason="PPQ_API_KEY not set")
def test_reasoning_live():
    import urllib.request
    extra_body, _ = ppq.build_api_kwargs_extras(
        supports_reasoning=True,
        reasoning_config={"effort": "low"},
    )
    payload = {
        "model": "deepseek/deepseek-r1",
        "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        "max_tokens": 50,
        **extra_body,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {PPQ_API_KEY}",
            "Content-Type": "application/json",
            **ppq.default_headers,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    assert "choices" in data
    content = data["choices"][0]["message"]["content"]
    assert "4" in content
