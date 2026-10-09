"""Exercise the installed SDKs at the HTTP boundary without network/API charges."""

import json

import httpx2
import pytest
from anthropic import Anthropic, AuthenticationError
from openai import OpenAI

import models


def test_real_openai_sdk_request_and_text_parsing(monkeypatch):
    def handle(request):
        assert request.url.path == "/v1/responses"
        body = json.loads(request.content)
        assert body["model"] == models.DEFAULT_OPENAI_MODEL
        assert body["store"] is False
        assert "temperature" not in body
        return httpx2.Response(
            200,
            json={
                "id": "resp_offline",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": models.DEFAULT_OPENAI_MODEL,
                "output": [
                    {
                        "id": "msg_offline",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "code", "annotations": []}
                        ],
                    }
                ],
            },
        )

    with OpenAI(
        api_key="offline",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handle)),
    ) as client:
        monkeypatch.setattr(models, "OpenAI", lambda **kwargs: client)
        assert models.OpenAIModel("offline").generate("question") == "code"


def test_real_claude_sdk_request_and_text_parsing(monkeypatch):
    def handle(request):
        assert request.url.path == "/v1/messages"
        body = json.loads(request.content)
        assert body["model"] == models.DEFAULT_CLAUDE_MODEL
        assert "temperature" not in body
        return httpx2.Response(
            200,
            json={
                "id": "msg_offline",
                "type": "message",
                "role": "assistant",
                "model": models.DEFAULT_CLAUDE_MODEL,
                "content": [{"type": "text", "text": "code"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    with Anthropic(
        api_key="offline",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handle)),
    ) as client:
        monkeypatch.setattr(models, "Anthropic", lambda **kwargs: client)
        assert models.ClaudeModel("offline").generate("question") == "code"


def test_real_claude_http_errors_are_not_mistaken_for_content(monkeypatch):
    def handle(request):
        return httpx2.Response(
            401,
            json={
                "type": "error",
                "error": {"type": "authentication_error", "message": "Invalid API key"},
            },
        )

    with Anthropic(
        api_key="offline",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handle)),
    ) as client:
        monkeypatch.setattr(models, "Anthropic", lambda **kwargs: client)
        with pytest.raises(AuthenticationError):
            models.ClaudeModel("offline").generate("question")
