"""Model adapters and the small generate / verify / remember workflow."""

import datetime
import importlib.util
import json
import logging
import os
import re
import sqlite3
from abc import ABC, abstractmethod
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from threading import Lock

from anthropic import Anthropic
from openai import OpenAI

from verification import LANGUAGES, run_test_case

logger = logging.getLogger(__name__)
DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"
PROVIDERS = ("OpenAI", "Claude", "Deepseek", "Llama")


def env_int(name, default, minimum=1):
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def execution_enabled():
    return os.getenv("ALLOW_CODE_EXECUTION", "").lower() in {"1", "true", "yes"}


def clean_response(text):
    """Strip a single enclosing Markdown fence, never arbitrary code contents."""
    text = text.strip()
    match = re.fullmatch(r"```[^\n]*\n(.*?)\n?```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def require_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "The model returned no text. Try again or choose another model."
        )
    return text.strip()


class BaseAIModel(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return generated text or raise an actionable error."""

    def close(self):
        """Release per-request resources, if any."""


class OpenAIModel(BaseAIModel):
    def __init__(self, api_key):
        self.model = os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        self.max_tokens = env_int("MAX_OUTPUT_TOKENS", 8192)
        self.client = OpenAI(
            api_key=api_key, timeout=env_int("API_TIMEOUT", 120), max_retries=1
        )

    def generate(self, prompt):
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=self.max_tokens,
            store=False,
        )
        if response.status != "completed":
            raise ValueError(
                "OpenAI did not complete the response. Check MAX_OUTPUT_TOKENS "
                "or try another model."
            )
        return require_text(response.output_text)

    def close(self):
        self.client.close()


class ClaudeModel(BaseAIModel):
    def __init__(self, api_key):
        self.model = os.getenv("CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL
        self.max_tokens = env_int("MAX_OUTPUT_TOKENS", 8192)
        self.client = Anthropic(
            api_key=api_key, timeout=env_int("API_TIMEOUT", 120), max_retries=1
        )

    def generate(self, prompt):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason != "end_turn":
            raise ValueError(
                "Claude did not complete the response. Check MAX_OUTPUT_TOKENS "
                "or try another model."
            )
        return require_text(
            "\n".join(block.text for block in response.content if block.type == "text")
        )

    def close(self):
        self.client.close()


class LocalGGUFModel(BaseAIModel):
    def __init__(self, model_path):
        # Optional native dependency: cloud-only installs do not need a compiler.
        from llama_cpp import Llama

        options = {
            "model_path": model_path,
            "n_ctx": env_int("LOCAL_CONTEXT_SIZE", 8192),
            "n_threads": env_int("THREADS", 4),
            "n_gpu_layers": env_int("LOCAL_GPU_LAYERS", 0, minimum=-1),
            "verbose": False,
        }
        if chat_format := os.getenv("LOCAL_CHAT_FORMAT"):
            options["chat_format"] = chat_format
        self.llm = Llama(**options)
        self.lock = Lock()

    def generate(self, prompt):
        # llama.cpp contexts must not be used concurrently.
        with self.lock:
            response = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=env_int("LOCAL_MAX_TOKENS", 2048),
                temperature=0.2,
            )
        choice = response["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError(
                "Local response was incomplete. Adjust LOCAL_MAX_TOKENS and "
                "LOCAL_CONTEXT_SIZE or use a shorter problem."
            )
        return require_text(choice["message"]["content"])


# Keep the original import names for existing callers.
DeepseekLocalModel = LocalGGUFModel
LlamaModel = LocalGGUFModel
_local_load_lock = Lock()


@lru_cache(maxsize=2)
def _load_local_model(path):
    return LocalGGUFModel(path)


def validate_model_config(model_type):
    if model_type not in PROVIDERS:
        raise ValueError("Select a supported model provider.")
    if model_type == "OpenAI":
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise ValueError("Set OPENAI_API_KEY in .env, then restart the app.")
    elif model_type == "Claude":
        if not (
            os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY") or ""
        ).strip():
            raise ValueError("Set ANTHROPIC_API_KEY in .env, then restart the app.")
    else:
        name = "DEEPSEEK_MODEL_PATH" if model_type == "Deepseek" else "LLAMA_MODEL_PATH"
        path = os.getenv(name)
        if not path or not Path(path).expanduser().is_file():
            raise ValueError(
                f"Set {name} to an existing GGUF file, then restart the app."
            )
        if importlib.util.find_spec("llama_cpp") is None:
            raise ValueError(
                "Local models need: python -m pip install -r requirements-local.txt"
            )


def create_model(model_type):
    validate_model_config(model_type)
    if model_type == "OpenAI":
        return OpenAIModel(os.environ["OPENAI_API_KEY"])
    if model_type == "Claude":
        return ClaudeModel(
            os.getenv("ANTHROPIC_API_KEY") or os.environ["CLAUDE_API_KEY"]
        )
    name = "DEEPSEEK_MODEL_PATH" if model_type == "Deepseek" else "LLAMA_MODEL_PATH"
    path = str(Path(os.environ[name]).expanduser().resolve())
    # Also serialize cold loads: lru_cache alone permits duplicate concurrent loads.
    with _local_load_lock:
        return _load_local_model(path)


class MemoryManager:
    """Short-lived connections are safe across Flask's request threads."""

    def __init__(self, db_path=None):
        self.db_path = str(db_path or os.getenv("MEMORY_DB_PATH", "memory.db"))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS memories
                (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, task TEXT,
                solution TEXT, success BOOLEAN, verification_result TEXT,
                reflections TEXT)"""
            )

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def store_memory(self, memory, reflection_factory=None):
        with closing(self._connect()) as conn, conn:
            # Keep count, reflection, and insert in one transaction across workers.
            conn.execute("BEGIN IMMEDIATE")
            count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if reflection_factory is not None and (count + 1) % 3 == 0:
                previous = conn.execute(
                    "SELECT * FROM memories ORDER BY id DESC LIMIT 2"
                ).fetchall()
                memory["reflections"] = reflection_factory(
                    [memory, *self._decode_memories(previous)]
                )
            conn.execute(
                """INSERT INTO memories
                (timestamp, task, solution, success, verification_result, reflections)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    memory["timestamp"],
                    memory["task"],
                    memory["solution"],
                    memory["success"],
                    json.dumps(memory.get("verification_result", {})),
                    json.dumps(memory.get("reflections", [])),
                ),
            )

    def count(self):
        with closing(self._connect()) as conn:
            return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def get_memories(self, limit=10):
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return self._decode_memories(rows)

    @staticmethod
    def _decode_memories(rows):
        memories = []
        for row in rows:
            memory = dict(row)
            for key, default in (("verification_result", {}), ("reflections", [])):
                try:
                    value = json.loads(memory[key] or "null")
                    memory[key] = value if isinstance(value, type(default)) else default
                except (ValueError, TypeError):
                    memory[key] = default
            memories.append(memory)
        return memories


class SoftwareEngineerAgent:
    def __init__(self, model, memory_manager=None):
        self.model = model
        self.memory_manager = memory_manager or MemoryManager()

    def process_task(self, problem, language="python"):
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "task": problem,
            "language": language,
            "success": False,
            "solution": "",
            "verification_result": {"success": False, "status": "error", "tests": []},
            "reflections": [],
        }
        try:
            entry["solution"] = self.generate_solution(problem, language)
            entry["verification_result"] = self.verify_solution(
                problem, entry["solution"], language
            )
            entry["success"] = entry["verification_result"]["success"]
        except Exception:
            logger.exception("Solution generation failed")
            entry["error"] = (
                "Generation failed. Check the server log, API key, model access, "
                "quota, output/context limits, and connection, then try again."
            )
            entry["verification_result"]["error"] = entry["error"]

        try:
            self.memory_manager.store_memory(entry, self.reflect)
        except (OSError, sqlite3.Error):
            logger.exception("Could not save task history")
            entry["warning"] = (
                "The result is available, but local history could not be saved."
            )
        return entry

    def generate_solution(self, problem, language="python"):
        if language not in LANGUAGES:
            raise ValueError("Unsupported language.")
        prompt = (
            f"You are a software engineer. Solve the following problem in {language}.\n"
            f"{LANGUAGES[language]['contract']}\n"
            "Use only the standard library. Return only one source file, without "
            "Markdown fences or explanations. Do not run demonstrations or read stdin "
            "at module load. Return JSON-compatible values from the function.\n\n"
            f"Problem:\n{problem}"
        )
        return clean_response(require_text(self.model.generate(prompt)))

    def verify_solution(self, problem, solution, language="python"):
        result = {"success": False, "status": "error", "tests": []}
        try:
            if language not in LANGUAGES:
                raise ValueError("Unsupported language.")
            response = self.model.generate(
                f"Problem:\n{problem}\n\n{language} solution:\n{solution}\n\n"
                "Generate exactly 3 tests, including edge cases. Derive expected outputs "
                "from the problem, not by copying possible bugs in the solution. "
                'Return only a JSON array of objects: {"inputs": [positional arguments], '
                '"expected": expected JSON value}. No prose.'
            )
            cases = json.loads(clean_response(require_text(response)))
            if not isinstance(cases, list) or len(cases) != 3:
                raise ValueError("Expected exactly 3 test cases.")
            for case in cases:
                if (
                    not isinstance(case, dict)
                    or not isinstance(case.get("inputs"), list)
                    or "expected" not in case
                ):
                    raise ValueError(
                        "Each test needs an inputs array and an expected value."
                    )

            if not execution_enabled():
                result.update(
                    status="not_run",
                    tests=cases,
                    message="Tests were generated but not executed. Automatic execution is disabled.",
                )
                return result

            for number, case in enumerate(cases, 1):
                test = {
                    "test_case": number,
                    "input": case["inputs"],
                    "expected": case["expected"],
                    "received": None,
                    "success": False,
                }
                try:
                    received = run_test_case(
                        clean_response(solution),
                        case["inputs"],
                        language,
                        timeout=env_int("TEST_TIMEOUT", 5),
                    )
                    test.update(received=received, success=received == case["expected"])
                except (RuntimeError, OSError, ValueError) as exc:
                    test["error"] = str(exc)
                result["tests"].append(test)
            result["success"] = all(test["success"] for test in result["tests"])
            result["status"] = "passed" if result["success"] else "failed"
        except (ValueError, TypeError) as exc:
            result["error"] = str(exc)
        except Exception:
            logger.exception("Test generation failed")
            result["error"] = (
                "Test generation failed. Check the provider and server log."
            )
        return result

    def reflect(self, memories):
        verified = [
            memory
            for memory in memories
            if memory.get("verification_result", {}).get("status")
            in {"passed", "failed"}
        ]
        if not verified:
            return [
                "No executed tests in the last three tasks; correctness is not verified."
            ]
        rate = sum(bool(memory["success"]) for memory in verified) / len(verified)
        return [
            f"Tests passed for {rate:.0%} of {len(verified)} recently executed tasks.",
            "Model-generated tests are a starting point, not a guarantee of correctness.",
        ]
