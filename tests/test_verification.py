import os
import shutil
import time

import pytest

from verification import run_test_case


def test_explicit_python_entry_point_not_first_imported_callable():
    source = """
from math import sqrt
def helper(value):
    return value * 10
def solution_function(value):
    return helper(value) + int(sqrt(value))
"""
    assert run_test_case(source, [4]) == 42


def test_output_prints_do_not_break_result():
    source = "def solution_function():\n    print('debug')\n    return 'ok'"
    assert run_test_case(source, []) == "ok"


def test_provider_keys_not_inherited(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private")
    source = (
        "import os\ndef solution_function():\n    return os.getenv('OPENAI_API_KEY')"
    )
    assert run_test_case(source, []) is None


def test_infinite_loop_times_out():
    start = time.monotonic()
    source = "def solution_function():\n    while True: pass"
    with pytest.raises(RuntimeError, match="timed out"):
        run_test_case(source, [], timeout=0.2)
    assert time.monotonic() - start < 3


def test_top_level_infinite_loop_times_out():
    with pytest.raises(RuntimeError, match="timed out"):
        run_test_case("while True: pass", [], timeout=0.2)


def test_output_limit():
    source = "def solution_function():\n    print('x' * 1100000)\n    return 1"
    with pytest.raises(RuntimeError, match="too much output"):
        run_test_case(source, [])


def test_result_limit():
    source = "def solution_function():\n    return 'x' * 1100000"
    with pytest.raises(RuntimeError, match="too large"):
        run_test_case(source, [])


def test_syntax_error_is_diagnostic():
    with pytest.raises(RuntimeError, match="SyntaxError"):
        run_test_case("def broken(", [])


def test_missing_function_is_diagnostic():
    with pytest.raises(RuntimeError, match="solution_function"):
        run_test_case("def helper(): return 1", [])


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js not installed")
def test_javascript_positional_arguments_and_async():
    source = "module.exports.solutionFunction = async (a, b) => a + b;"
    assert run_test_case(source, [2, 3], "javascript") == 5


@pytest.mark.skipif(not shutil.which("php"), reason="PHP not installed")
def test_php_positional_arguments():
    source = "<?php function solutionFunction($a, $b) { return $a + $b; }"
    assert run_test_case(source, [2, 3], "php") == 5


def test_missing_runtime(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda executable: None)
    with pytest.raises(RuntimeError, match="Install the javascript runtime"):
        run_test_case("", [], "javascript")


def test_unsupported_language():
    with pytest.raises(ValueError, match="Unsupported"):
        run_test_case("", [], "shell")


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only process groups")
def test_child_process_is_stopped_after_parent_returns(tmp_path):
    marker = tmp_path / "should-not-exist"
    source = f"""
import subprocess
import sys
def solution_function():
    subprocess.Popen([sys.executable, '-c',
        "import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).touch()"])
    return True
"""
    assert run_test_case(source, []) is True
    time.sleep(1.1)
    assert not marker.exists()
