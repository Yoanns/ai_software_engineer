import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import models
from models import MemoryManager, SoftwareEngineerAgent

CASES = [
    {"inputs": [1, 2], "expected": 3},
    {"inputs": [-1, 1], "expected": 0},
    {"inputs": [0, 0], "expected": 0},
]
SOLUTION = "def solution_function(a, b):\n    return a + b"


class FakeModel(models.BaseAIModel):
    def __init__(self, outputs=None):
        self.outputs = iter(outputs or [SOLUTION, json.dumps(CASES)])
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return value


@pytest.mark.parametrize("value", ["```python\nx = 1\n```", "x = 1"])
def test_clean_fences(value):
    assert models.clean_response(value) == "x = 1"


def test_does_not_strip_internal_fences():
    value = 'def solution_function():\n    return "```"'
    assert models.clean_response(value) == value


def test_openai_responses_and_override(monkeypatch):
    factory = MagicMock()
    factory.return_value.responses.create.return_value = SimpleNamespace(
        status="completed", output_text="answer"
    )
    monkeypatch.setattr(models, "OpenAI", factory)
    monkeypatch.setenv("OPENAI_MODEL", "configured-model")
    model = models.OpenAIModel("test-key")
    assert model.generate("question") == "answer"
    factory.return_value.responses.create.assert_called_once_with(
        model="configured-model", input="question", max_output_tokens=8192, store=False
    )
    assert factory.call_args.kwargs["timeout"] == 120
    model.close()
    factory.return_value.close.assert_called_once()


@pytest.mark.parametrize("status,text", [("incomplete", "partial"), ("completed", "")])
def test_openai_rejects_incomplete_or_empty(monkeypatch, status, text):
    factory = MagicMock()
    factory.return_value.responses.create.return_value = SimpleNamespace(
        status=status, output_text=text
    )
    monkeypatch.setattr(models, "OpenAI", factory)
    with pytest.raises(ValueError):
        models.OpenAIModel("test-key").generate("question")


def test_claude_multiple_text_blocks_and_override(monkeypatch):
    factory = MagicMock()
    factory.return_value.messages.create.return_value = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking"),
            SimpleNamespace(type="text", text="first"),
            SimpleNamespace(type="text", text="second"),
        ],
    )
    monkeypatch.setattr(models, "Anthropic", factory)
    monkeypatch.setenv("CLAUDE_MODEL", "configured-claude")
    model = models.ClaudeModel("test-key")
    assert model.generate("question") == "first\nsecond"
    assert factory.return_value.messages.create.call_args.kwargs == {
        "model": "configured-claude",
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "question"}],
    }
    model.close()
    factory.return_value.close.assert_called_once()


@pytest.mark.parametrize("reason", ["max_tokens", "refusal", "tool_use"])
def test_claude_rejects_incomplete(monkeypatch, reason):
    factory = MagicMock()
    factory.return_value.messages.create.return_value = SimpleNamespace(
        stop_reason=reason, content=[]
    )
    monkeypatch.setattr(models, "Anthropic", factory)
    with pytest.raises(ValueError):
        models.ClaudeModel("test-key").generate("question")


@pytest.mark.parametrize(
    "provider", ["OpenAI", "Claude", "Deepseek", "Llama", "invalid"]
)
def test_missing_configuration(provider):
    with pytest.raises(ValueError):
        models.create_model(provider)


def test_legacy_claude_key(monkeypatch):
    factory = MagicMock()
    monkeypatch.setenv("CLAUDE_API_KEY", "legacy-key")
    monkeypatch.setattr(models, "ClaudeModel", factory)
    models.create_model("Claude")
    factory.assert_called_once_with("legacy-key")


def test_local_package_is_optional(monkeypatch, tmp_path):
    path = tmp_path / "example.gguf"
    path.touch()
    monkeypatch.setenv("LLAMA_MODEL_PATH", str(path))
    monkeypatch.setattr(models.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ValueError, match="requirements-local.txt"):
        models.create_model("Llama")


def test_local_adapter_uses_chat_completion(monkeypatch):
    import sys

    factory = MagicMock()
    factory.return_value.create_chat_completion.return_value = {
        "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}]
    }
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=factory))
    model = models.LocalGGUFModel("example.gguf")
    assert model.generate("question") == "answer"
    assert factory.call_args.kwargs["n_gpu_layers"] == 0
    assert factory.call_args.kwargs["n_ctx"] == 8192
    assert factory.return_value.create_chat_completion.call_args.kwargs["messages"] == [
        {"role": "user", "content": "question"}
    ]


def test_local_cache(monkeypatch, tmp_path):
    path = tmp_path / "example.gguf"
    path.touch()
    factory = MagicMock()
    monkeypatch.setenv("LLAMA_MODEL_PATH", str(path))
    monkeypatch.setattr(models.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(models, "LocalGGUFModel", factory)
    models._load_local_model.cache_clear()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            instances = list(pool.map(lambda _: models.create_model("Llama"), range(8)))
        assert all(value is instances[0] for value in instances)
        factory.assert_called_once()
    finally:
        models._load_local_model.cache_clear()


@pytest.mark.parametrize("value", ["invalid", "0", "-5"])
def test_invalid_numeric_config(monkeypatch, value):
    monkeypatch.setenv("API_TIMEOUT", value)
    with pytest.raises(ValueError, match="API_TIMEOUT"):
        models.env_int("API_TIMEOUT", 120)


def test_default_flow_does_not_execute(monkeypatch):
    runner = MagicMock(side_effect=AssertionError("Must not execute"))
    monkeypatch.setattr(models, "run_test_case", runner)
    model = FakeModel()
    result = SoftwareEngineerAgent(model).process_task("Add two numbers")
    assert result["solution"] == SOLUTION
    assert result["verification_result"]["status"] == "not_run"
    assert result["success"] is False
    assert len(model.prompts) == 2
    assert MemoryManager().count() == 1
    runner.assert_not_called()


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        "[]",
        "[1, 2, 3]",
        '[{"inputs": 1, "expected": 2}, {}, {}]',
        '[{"inputs": []}, {"inputs": []}, {"inputs": []}]',
    ],
)
def test_invalid_generated_tests(response):
    agent = SoftwareEngineerAgent(FakeModel([response]))
    result = agent.verify_solution("add", SOLUTION)
    assert result["status"] == "error"
    assert not result["success"]
    assert result["error"]


def test_fenced_json_is_supported():
    agent = SoftwareEngineerAgent(
        FakeModel(["```json\n" + json.dumps(CASES) + "\n```"])
    )
    result = agent.verify_solution("add", SOLUTION)
    assert result["status"] == "not_run"


def test_enabled_python_verification(monkeypatch):
    monkeypatch.setenv("ALLOW_CODE_EXECUTION", "true")
    result = SoftwareEngineerAgent(FakeModel()).process_task("Add two numbers")
    assert result["success"]
    assert result["verification_result"]["status"] == "passed"
    assert [test["received"] for test in result["verification_result"]["tests"]] == [
        3,
        0,
        0,
    ]


def test_failed_tests_not_claimed_success(monkeypatch):
    monkeypatch.setenv("ALLOW_CODE_EXECUTION", "true")
    model = FakeModel([SOLUTION.replace("a + b", "42"), json.dumps(CASES)])
    result = SoftwareEngineerAgent(model).process_task("Add two numbers")
    assert not result["success"]
    assert result["verification_result"]["status"] == "failed"


def test_generation_error_is_recorded_without_leaking_provider_details():
    model = FakeModel([RuntimeError("private-provider-detail")])
    result = SoftwareEngineerAgent(model).process_task("Add")
    assert result["error"]
    assert "private-provider-detail" not in json.dumps(result)
    saved = MemoryManager().get_memories()[0]
    assert saved["verification_result"]["error"] == result["error"]


def test_test_provider_error_keeps_solution():
    model = FakeModel([SOLUTION, RuntimeError("quota failure")])
    result = SoftwareEngineerAgent(model).process_task("Add")
    assert result["solution"] == SOLUTION
    assert result["verification_result"]["status"] == "error"
    assert "Test generation failed" in result["verification_result"]["error"]


def test_reflections_continue_after_ten_tasks_and_persist():
    for _ in range(12):
        result = SoftwareEngineerAgent(FakeModel()).process_task("Add")
    memories = MemoryManager().get_memories(20)
    assert len(memories) == 12
    assert result["reflections"]
    assert sum(bool(memory["reflections"]) for memory in memories) == 4
    assert memories[0]["reflections"] == result["reflections"]


def test_database_failure_does_not_lose_solution(monkeypatch):
    manager = MemoryManager()
    monkeypatch.setattr(
        manager, "store_memory", MagicMock(side_effect=sqlite3.OperationalError())
    )
    result = SoftwareEngineerAgent(FakeModel(), manager).process_task("Add")
    assert result["solution"] == SOLUTION
    assert result["warning"]


def test_memory_can_be_used_across_threads():
    manager = MemoryManager()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: SoftwareEngineerAgent(FakeModel(), manager).process_task(
                    "Add"
                ),
                range(12),
            )
        )
    assert all(result["solution"] == SOLUTION for result in results)
    assert manager.count() == 12
    assert sum(bool(memory["reflections"]) for memory in manager.get_memories(20)) == 4


def test_legacy_database_rows(tmp_path):
    manager = MemoryManager(tmp_path / "nested" / "memory.db")
    with sqlite3.connect(manager.db_path) as conn:
        conn.execute(
            "INSERT INTO memories (task, verification_result, reflections) VALUES (?, ?, ?)",
            ("old task", None, "invalid JSON"),
        )
    memory = manager.get_memories()[0]
    assert memory["verification_result"] == {}
    assert memory["reflections"] == []
