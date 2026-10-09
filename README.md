# AI Software Engineer

A small Flask app for generating a single-file solution and three suggested tests
for a programming problem. Choose an API provider or a local GGUF model, describe
the problem, and inspect the result. No agent framework, background queue, or
front-end build step is required.

## What it does

- Generates Python, JavaScript (Node.js/CommonJS), or PHP functions.
- Generates three model-written test cases.
- Optionally executes tests in separate processes with a per-test timeout.
- Saves tasks, code, verification results, and a brief reflection every three tasks
  in a local SQLite database.
- Runs locally or with Docker Compose.

This is a single-user development tool, not an autonomous repository editor or a
production coding service. The memory is local history, not model training, and
previous tasks are not automatically sent back to the model. It does not validate
complete React, Vue, Next.js, Nuxt, or Laravel applications.

## Quick start

Use Python 3.12 or newer. Python 3.12 is recommended for the optional native GGUF
dependency and is the version used by the Docker image.

```bash
git clone https://github.com/Yoanns/ai_software_engineer.git
cd ai_software_engineer
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows Command Prompt, activate with `.venv\Scripts\activate.bat` and copy the
configuration with `copy .env.example .env`.

Edit `.env` and set **one** of `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. API access
and billing are separate from consumer chat subscriptions. Do not paste keys
into the web form or commit them.

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000), select a provider, click
**Save model**, choose the language, and submit a problem. For example:

> Given a list of integers, return the second-largest distinct value, or null
> (None in Python) when fewer than two distinct values exist.

Model selection checks configuration, not credentials with the provider. The
first solve makes the actual API requests. Normally each solve makes two calls:
one for code and one for tests; transient API errors may cause one retry per call.

## Models

Defaults were checked against provider documentation on **2026-10-09**.
Change model IDs in `.env` and restart the app; no source changes are necessary.

| Choice | Default / configuration | Notes |
| --- | --- | --- |
| OpenAI | `OPENAI_MODEL=gpt-6-luna` | Current efficient model, using the Responses API with `store=False`. See the [OpenAI model page](https://developers.openai.com/api/docs/models/gpt-6-luna). |
| Claude | `CLAUDE_MODEL=claude-sonnet-5-5` | Current Sonnet model, using the official Anthropic SDK. See the [Claude model page](https://platform.claude.com/docs/en/models/sonnet-5-5/overview). |
| DeepSeek / local GGUF | `DEEPSEEK_MODEL_PATH` | Your instruction-tuned GGUF file; no fixed model version or automatic download. |
| Llama / local GGUF | `LLAMA_MODEL_PATH` | Your instruction-tuned GGUF file; no fixed model version or automatic download. |

Use a model available to your API account. For a different OpenAI default, consult
the [current model catalog](https://developers.openai.com/api/docs/models);
for Claude, consult the [current model IDs](https://platform.claude.com/docs/en/models/overview).
Overrides must support the API used by their adapter. Incomplete or empty
responses are reported as errors rather than passed off as working code.

### Optional local models

Cloud-only installations do not import or build `llama-cpp-python`.
For local inference:

```bash
python -m pip install -r requirements-local.txt
```

Download a compatible instruction-tuned GGUF file yourself, put it in `models/`,
and set either `DEEPSEEK_MODEL_PATH` or `LLAMA_MODEL_PATH`. The two local choices
use the same adapter; choose a file that fits your RAM and is supported by the
installed llama.cpp version. This update does not replace an existing GGUF file.

The adapter uses chat completion and the GGUF's embedded chat template, with
`LOCAL_CHAT_FORMAT` available as an override. Installation may need a C/C++
compiler; CPU wheels and GPU-specific build instructions are documented in the
[llama-cpp-python installation guide](https://llama-cpp-python.readthedocs.io/en/latest/).
CPU-only is the default. Models load on the first solve, are cached within the
worker, and serialize inference so threads do not share a context concurrently.

## Configuration

Copy `.env.example` rather than creating a configuration from scratch.
Restart after changing it.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | Empty | OpenAI API key. |
| `OPENAI_MODEL` | `gpt-6-luna` | OpenAI Responses-compatible model. |
| `ANTHROPIC_API_KEY` | Empty | Anthropic key; legacy `CLAUDE_API_KEY` remains supported. |
| `CLAUDE_MODEL` | `claude-sonnet-5-5` | Claude Messages-compatible model. |
| `FLASK_SECRET_KEY` | Random on startup | Set a stable random secret for persistent sessions or multiple workers. Never use a public example value. |
| `SESSION_COOKIE_SECURE` | `false` | Set `true` behind HTTPS; leave false for local HTTP. |
| `API_TIMEOUT` | `120` | Timeout in seconds per API attempt; one retry is allowed. |
| `MAX_OUTPUT_TOKENS` | `8192` | Cloud output budget, including reasoning where applicable. Increase if responses are incomplete. |
| `MEMORY_DB_PATH` | `memory.db` | SQLite file location; `/app/data/memory.db` in Compose. |
| `LOG_LEVEL` | `INFO` | Logs go to stderr / Docker logs. |
| `DEEPSEEK_MODEL_PATH` | Empty | Local GGUF file path. |
| `LLAMA_MODEL_PATH` | Empty | Local GGUF file path. |
| `THREADS` | `4` | llama.cpp CPU thread count. |
| `LOCAL_CONTEXT_SIZE` | `8192` | Local context size; must fit the prompt plus output. |
| `LOCAL_MAX_TOKENS` | `2048` | Local output budget. |
| `LOCAL_GPU_LAYERS` | `0` | CPU-only by default; requires an appropriate build for GPU use. |
| `LOCAL_CHAT_FORMAT` | Empty | Override the GGUF chat template when needed. |
| `ALLOW_CODE_EXECUTION` | `false` | Explicit opt-in to running generated code. Read the warning below. |
| `TEST_TIMEOUT` | `5` | Seconds allowed per test process. |

Generate a session secret with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

## Verification and safety

**Generated code is untrusted. Automatic execution is disabled by default.**
The default flow still generates code and three test cases, but reports
**not executed**, never “all tests passed.”

Setting `ALLOW_CODE_EXECUTION=true` restores automatic tests. Only do this in a
disposable environment without sensitive files or credentials. Each test runs
in a temporary working directory with a timeout and a reduced environment; API
key variables are not inherited. The runner monitors output size and terminates
the process group on POSIX. Windows termination covers only the direct process.

**A subprocess is not a security sandbox.** Generated code can still access
files, the network, and other resources allowed to the OS user, and can spawn
processes. There are no hard memory, filesystem, or network isolation guarantees.
The ordinary app Docker container also contains provider credentials, so running
the app in Docker does not by itself make generated code safe.

The function contracts are:

- **Python:** `def solution_function(...): ...`; standard library only.
- **JavaScript:** `module.exports.solutionFunction = ...`; Node.js is required.
  Positional arguments are spread, and async functions are supported.
- **PHP:** `function solutionFunction(...) { ... }`; PHP CLI is required and
  positional arguments are unpacked.

Inputs and return values must be JSON-compatible. Python is included in the app
image; Node.js and PHP are not. Missing runtimes produce an explicit test error.
The tests come from the same model as the solution: passing them is not proof of
correctness. Review generated code and add independent tests.

There is no login or rate limiter. Keep the app on localhost or behind trusted
access controls; do not expose it directly to the Internet. Forms have CSRF
protection and request-size limits, but these are not a substitute for
authentication. Task text and generated code are stored unencrypted in SQLite.
Cloud providers receive the problem, code, and test-generation prompt;
`store=False` is not a promise of zero provider retention.

## Docker

Create `.env` first, then:

```bash
docker compose up --build
```

Open [http://127.0.0.1:5001](http://127.0.0.1:5001). The published port binds to
localhost only. The container runs as a non-root user with one threaded Gunicorn
worker; SQLite history is stored in the `ai-engineer-data` named volume.

For GGUF models, set `INSTALL_LOCAL_MODELS=true` in `.env`, set model paths to
`/app/models/your-file.gguf`, and rebuild. The `models/` directory is mounted
read-only. Native CPU builds can take time; GPU Docker setup is not included.

```bash
docker compose logs -f
docker compose down
```

`docker compose down` keeps history. `docker compose down -v` deletes the named
volume and its history.

### Upgrading an old installation

1. Back up `.env` and `memory.db` before updating.
2. Recreate the Python virtual environment, or rebuild the Docker image.
3. Compare `.env` against `.env.example`. Keep your keys; replace outdated model
   overrides. `CLAUDE_API_KEY`, `DEEPSEEK_MODEL_PATH`, `LLAMA_MODEL_PATH`, and
   `THREADS` remain supported.
4. The SQLite table layout is unchanged. Native installs reuse `memory.db`.
   Compose now uses a named volume instead of bind-mounting the whole repository;
   an existing root-level `memory.db` is **not imported automatically**.
5. To retain that history in Docker, start the new container and, before submitting
   tasks, copy your backed-up database:

   ```bash
   docker compose cp ./memory.db ai-engineer:/app/data/memory.db
   docker compose exec --user root ai-engineer chown appuser:appuser /app/data/memory.db
   docker compose restart ai-engineer
   ```

   Only do this for an initial migration, not while the app is processing tasks.

The broken progress stream has been removed in favor of a simple loading message.
Model selection now redirects back to an enabled form. Reflections are saved
with each qualifying task, and there is no long-lived cross-thread SQLite
connection or serialized model client in a browser session.

## Development and checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use fake provider responses and temporary databases; no paid API calls or
GGUF downloads are needed. Node.js/PHP execution tests skip if the corresponding
runtime is unavailable. CI tests Python 3.12 and 3.14 and builds/smoke-tests the
default Docker image. Live provider calls and local-model inference require your
own credentials/model files and are not covered by the offline test suite.

For a local Unix server without Docker:

```bash
gunicorn --bind 127.0.0.1:5000 --workers 1 --threads 4 --timeout 600 app:app
```

Use `python app.py` on Windows. Keep one worker for local GGUF inference to avoid
loading multiple copies into RAM. If API calls or local inference take longer
than expected, check output/context limits, server timeouts, account quota, and
the server logs. Do not enable Flask's debugger on a publicly reachable server.
