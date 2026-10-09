import pytest


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "ANTHROPIC_API_KEY",
        "CLAUDE_API_KEY",
        "CLAUDE_MODEL",
        "DEEPSEEK_MODEL_PATH",
        "LLAMA_MODEL_PATH",
        "ALLOW_CODE_EXECUTION",
        "MAX_OUTPUT_TOKENS",
        "API_TIMEOUT",
        "LOCAL_CONTEXT_SIZE",
        "LOCAL_MAX_TOKENS",
        "LOCAL_GPU_LAYERS",
        "LOCAL_CHAT_FORMAT",
        "THREADS",
        "TEST_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "history.db"))


@pytest.fixture
def client():
    from app import app

    app.config.update(
        TESTING=True, SECRET_KEY="test-only-secret", WTF_CSRF_ENABLED=True
    )
    return app.test_client()
