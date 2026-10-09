"""Opt-in subprocess tests. This is NOT a security sandbox."""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

LANGUAGES = {
    "python": {
        "label": "Python",
        "contract": "Expose a function named solution_function(*args).",
    },
    "javascript": {
        "label": "JavaScript (Node.js)",
        "contract": "Export a function named solutionFunction using module.exports (CommonJS).",
    },
    "php": {
        "label": "PHP",
        "contract": "Include the <?php opening tag and expose a function named solutionFunction.",
    },
}

PYTHON_RUNNER = """
import json
import runpy
from pathlib import Path
namespace = runpy.run_path("solution.py", run_name="solution_module")
function = namespace.get("solution_function")
if not callable(function):
    raise ValueError("Expected a function named solution_function")
inputs = json.loads(Path("inputs.json").read_text(encoding="utf-8"))
result = function(*inputs)
Path("result.json").write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
"""
JS_RUNNER = """
const fs = require('node:fs');
const solution = require('./solution.cjs');
const inputs = JSON.parse(fs.readFileSync('inputs.json', 'utf8'));
(async () => {
    if (typeof solution.solutionFunction !== 'function')
        throw new Error('Expected an exported solutionFunction');
    const result = await solution.solutionFunction(...inputs);
    fs.writeFileSync('result.json', JSON.stringify(result));
})().catch(error => { console.error(error.message); process.exitCode = 1; });
"""
PHP_RUNNER = """<?php
require 'solution.php';
$inputs = json_decode(file_get_contents('inputs.json'), true, 512, JSON_THROW_ON_ERROR);
$result = solutionFunction(...$inputs);
file_put_contents('result.json', json_encode($result, JSON_THROW_ON_ERROR));
"""
MAX_OUTPUT_BYTES = 1024 * 1024


def _stop_process(process):
    if os.name == "posix":
        # Include child processes, even if the immediate parent already exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    process.wait()


def run_test_case(solution, inputs, language="python", timeout=5):
    if language == "python":
        executable, extension, runner = sys.executable, "py", PYTHON_RUNNER
        flags = ["-I"]
    elif language == "javascript":
        executable, extension, runner = shutil.which("node"), "cjs", JS_RUNNER
        flags = []
    elif language == "php":
        executable, extension, runner = shutil.which("php"), "php", PHP_RUNNER
        flags = []
    else:
        raise ValueError("Unsupported language.")
    if not executable:
        raise RuntimeError(f"Install the {language} runtime to execute these tests.")

    with tempfile.TemporaryDirectory(prefix="ai-engineer-") as directory:
        root = Path(directory)
        (root / f"solution.{extension}").write_text(solution, encoding="utf-8")
        (root / f"runner.{extension}").write_text(runner, encoding="utf-8")
        (root / "inputs.json").write_text(json.dumps(inputs), encoding="utf-8")
        output = root / "output.log"
        result_file = root / "result.json"
        # Do not hand API keys and other environment secrets to generated code.
        # It still has filesystem/network access: a subprocess is NOT a sandbox.
        environment = {
            key: os.environ[key]
            for key in ("PATH", "SYSTEMROOT", "WINDIR")
            if key in os.environ
        }
        environment.update(
            HOME=directory, TMPDIR=directory, TMP=directory, TEMP=directory
        )
        with output.open("wb") as log:
            process = subprocess.Popen(
                [executable, *flags, f"runner.{extension}"],
                cwd=directory,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
            deadline = time.monotonic() + timeout
            try:
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"Test timed out after {timeout} seconds.")
                    if output.stat().st_size > MAX_OUTPUT_BYTES:
                        raise RuntimeError("Test produced too much output.")
                    time.sleep(0.02)
            finally:
                _stop_process(process)
        if output.stat().st_size > MAX_OUTPUT_BYTES:
            raise RuntimeError("Test produced too much output.")
        if process.returncode:
            # Bound diagnostics, even for very verbose failures.
            with output.open("rb") as log:
                message = log.read(4000).decode("utf-8", errors="replace")
            raise RuntimeError(
                message.strip() or f"Runtime exited with code {process.returncode}."
            )
        if not result_file.is_file():
            raise RuntimeError("The solution did not return a JSON-compatible result.")
        if result_file.stat().st_size > MAX_OUTPUT_BYTES:
            raise RuntimeError("Test result is too large.")
        return json.loads(result_file.read_text(encoding="utf-8"))
