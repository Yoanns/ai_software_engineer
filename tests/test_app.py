import json
import re
from unittest.mock import MagicMock

import pytest

import app as app_module
from models import BaseAIModel, MemoryManager


def token(client):
    response = client.get("/")
    return re.search(r'name="csrf_token" value="([^"]+)"', response.text).group(1)


def configure(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    return client.post(
        "/configure-model", data={"csrf_token": token(client), "model_type": "OpenAI"}
    )


class FakeModel(BaseAIModel):
    def generate(self, prompt):
        if "Generate exactly 3 tests" in prompt:
            return json.dumps([{"inputs": [1], "expected": 1}] * 3)
        return "def solution_function(value):\n    return value"


def test_home_is_available_without_keys_or_local_package(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Save model" in response.text
    assert "Automatic execution is disabled" in response.text
    assert response.headers["Cache-Control"] == "no-store"
    assert "EventSource" not in response.text


def test_model_configuration_redirects_and_only_stores_serializable_settings(
    client, monkeypatch
):
    response = configure(client, monkeypatch)
    assert response.status_code == 303
    with client.session_transaction() as session:
        assert session["model"] == "OpenAI"
        assert set(session) <= {"model", "csrf_token", "_permanent"}
    page = client.get("/")
    assert "Selected: OpenAI" in page.text
    assert "Select and save a model to begin." not in page.text


def test_legacy_index_post_is_safe(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    response = client.post(
        "/", data={"csrf_token": token(client), "model_type": "OpenAI"}
    )
    assert response.status_code == 303


def test_configuration_error_is_visible(client):
    response = client.post(
        "/configure-model", data={"csrf_token": token(client), "model_type": "Claude"}
    )
    assert response.status_code == 400
    assert "ANTHROPIC_API_KEY" in response.text


@pytest.mark.parametrize("path", ["/", "/configure-model", "/solve"])
def test_csrf_is_required(client, path):
    response = client.post(path, data={"model_type": "OpenAI", "problem": "Add"})
    assert response.status_code == 400
    assert "form expired or is invalid" in response.text


def test_requires_model(client):
    response = client.post(
        "/solve", data={"csrf_token": token(client), "problem": "Add"}
    )
    assert response.status_code == 400
    assert "Select and save a model first" in response.text


@pytest.mark.parametrize("problem", ["", "  ", "x" * 20001])
def test_problem_validation_before_api_call(client, monkeypatch, problem):
    configure(client, monkeypatch)
    factory = MagicMock()
    monkeypatch.setattr(app_module, "create_model", factory)
    response = client.post(
        "/solve", data={"csrf_token": token(client), "problem": problem}
    )
    assert response.status_code == 400
    factory.assert_not_called()


def test_bad_language(client, monkeypatch):
    configure(client, monkeypatch)
    response = client.post(
        "/solve",
        data={"csrf_token": token(client), "problem": "Add", "language": "shell"},
    )
    assert response.status_code == 400


def test_request_size_limit(client):
    response = client.post("/solve", data={"problem": "x" * 70000})
    assert response.status_code == 413


def test_end_to_end_not_run_result_and_copy_script(client, monkeypatch):
    configure(client, monkeypatch)
    model = FakeModel()
    model.close = MagicMock()
    monkeypatch.setattr(app_module, "create_model", lambda provider: model)
    response = client.post(
        "/solve",
        data={"csrf_token": token(client), "problem": "Identity", "language": "python"},
    )
    assert response.status_code == 200
    assert "Tests were generated but not executed" in response.text
    assert "All three generated tests passed" not in response.text
    assert "navigator.clipboard.writeText" in response.text
    assert MemoryManager().count() == 1
    model.close.assert_called_once()


def test_generated_html_is_escaped(client, monkeypatch):
    configure(client, monkeypatch)
    model = FakeModel()
    model.generate = MagicMock(
        side_effect=[
            '<script>alert("unsafe")</script>',
            json.dumps([{"inputs": [], "expected": "<script>bad</script>"}] * 3),
        ]
    )
    monkeypatch.setattr(app_module, "create_model", lambda provider: model)
    response = client.post(
        "/solve", data={"csrf_token": token(client), "problem": "Identity"}
    )
    assert '<script>alert("unsafe")</script>' not in response.text
    assert "&lt;script&gt;" in response.text


def test_provider_error_has_non_success_status(client, monkeypatch):
    configure(client, monkeypatch)
    model = FakeModel()
    model.generate = MagicMock(side_effect=RuntimeError("sensitive detail"))
    monkeypatch.setattr(app_module, "create_model", lambda provider: model)
    response = client.post(
        "/solve", data={"csrf_token": token(client), "problem": "Identity"}
    )
    assert response.status_code == 502
    assert "Generation failed" in response.text
    assert "sensitive detail" not in response.text
