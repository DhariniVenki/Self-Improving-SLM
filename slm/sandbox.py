"""Run a generated program against its MBPP tests in a separate process.

Each program runs in its own Python subprocess, in a fresh temporary folder,
with memory and file-size limits. This is not a security boundary on its
own: we rely on the Colab VM being throwaway.

The time limit is CPU time, not wall-clock time, so a slow-but-correct
program isn't marked as a timeout just because other programs are running
in parallel (Colab has only 2 CPUs). A longer wall-clock limit catches
programs that hang without using CPU, such as sleep() or waiting on input.

Every program gets one of these statuses:
    pass          all tests passed
    no_code       nothing could be extracted from the completion
    syntax        the program does not parse
    runtime       an exception other than AssertionError (at import or in a test)
    wrong_answer  a test assert failed
    timeout       the program or a test ran past the time limit
"""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STATUSES = ("pass", "no_code", "syntax", "runtime", "wrong_answer", "timeout")

# Runs inside the child process. It sets resource limits, defines the
# solution, then runs each test separately and writes per-test results to
# result.json (a file, so the solution printing or closing stdout can't
# interfere). __name__ is not "__main__", so `if __name__ == "__main__":`
# example blocks in generated code are skipped.
_HARNESS = r'''
import json, sys

def _limit(memory_mb, cpu_s):
    try:
        import resource
    except ImportError:
        return
    for name, value in (("RLIMIT_AS", memory_mb * 1024 * 1024),
                        ("RLIMIT_CPU", cpu_s),
                        ("RLIMIT_FSIZE", 10 * 1024 * 1024)):
        try:
            resource.setrlimit(getattr(resource, name), (value, value))
        except (ValueError, OSError, AttributeError):
            pass  # not supported on this OS (e.g. RLIMIT_AS on macOS)

def _err(e):
    return (type(e).__name__ + ": " + str(e))[:300]

def _main():
    cfg = json.load(open("job.json"))
    _limit(cfg["memory_mb"], cfg["cpu_s"])
    out = {"stage": "compile", "tests": [], "error": None}

    def _write():
        with open("result.json", "w") as f:
            json.dump(out, f)

    # MBPP setup code builds test inputs from the solution's own classes
    # (e.g. Node), so it runs after the solution.
    src = cfg["code"] + "\n" + cfg["setup"]
    try:
        compiled = compile(src, "solution.py", "exec")
    except (SyntaxError, ValueError) as e:
        out["stage"], out["error"] = "syntax", _err(e)
        return _write()

    g = {"__name__": "solution", "__builtins__": __builtins__}
    out["stage"] = "define"
    _write()
    try:
        exec(compiled, g)
    except AssertionError as e:
        # An assert of the model's own (e.g. a copy of the test) failed.
        out["stage"], out["error"] = "wrong_answer", _err(e)
        return _write()
    except BaseException as e:
        out["stage"], out["error"] = "runtime", _err(e)
        return _write()

    out["stage"] = "tests"
    for t in cfg["tests"]:
        _write()  # so a timeout or hard crash still shows how far we got
        try:
            exec(t, g)
            out["tests"].append("pass")
        except AssertionError as e:
            out["tests"].append("wrong_answer")
            out["error"] = out["error"] or _err(e)
        except BaseException as e:
            out["tests"].append("runtime")
            out["error"] = out["error"] or _err(e)
    out["stage"] = "done"
    _write()

_main()
'''


def run_tests(
    code: str,
    tests: list[str],
    setup: str = "",
    timeout_s: float = 5.0,
    memory_mb: int = 1024,
) -> dict:
    """Run `code` followed by each test. Returns a result dict (see STATUSES)."""
    result = {"status": None, "passed": False, "tests": [], "error": None}
    if not code.strip():
        result["status"] = "no_code"
        return result

    with tempfile.TemporaryDirectory(prefix="slm_sandbox_") as tmp:
        tmp = Path(tmp)
        (tmp / "harness.py").write_text(_HARNESS)
        job = {
            "code": code,
            "tests": tests,
            "setup": setup or "",
            "memory_mb": memory_mb,
            "cpu_s": max(1, math.ceil(timeout_s)),
        }
        (tmp / "job.json").write_text(json.dumps(job))

        proc = subprocess.Popen(
            [sys.executable, "-I", "harness.py"],
            cwd=tmp,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,  # own process group, so we can kill children too
        )
        timed_out = False
        try:
            _, stderr = proc.communicate(timeout=3 * timeout_s + 2)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _, stderr = proc.communicate()
        # The CPU-time limit kills the child with SIGXCPU (or SIGKILL past the hard limit).
        if proc.returncode in (-signal.SIGXCPU, -signal.SIGKILL):
            timed_out = True

        out = None
        if (tmp / "result.json").exists():
            try:
                out = json.loads((tmp / "result.json").read_text())
            except json.JSONDecodeError:
                out = None

    if out is not None:
        result["tests"] = out["tests"]
        result["error"] = out["error"]

    if timed_out:
        result["status"] = "timeout"
    elif out is None or out["stage"] in ("compile", "define"):
        # The process died before it could report (memory limit, os._exit, segfault).
        result["status"] = "runtime"
        result["error"] = stderr.decode(errors="replace").strip().splitlines()[-1][:300] if stderr else "process died"
    elif out["stage"] == "syntax":
        result["status"] = "syntax"
    elif out["stage"] in ("runtime", "wrong_answer"):
        result["status"] = out["stage"]
    elif out["stage"] == "tests":
        # Died part-way through the tests without a timeout.
        result["status"] = "runtime"
        result["error"] = result["error"] or "process died during tests"
    elif all(t == "pass" for t in out["tests"]):
        result["status"] = "pass"
        result["passed"] = True
    else:
        result["status"] = next(t for t in out["tests"] if t != "pass")
    return result


def run_many(jobs: list[dict], timeout_s: float = 5.0, memory_mb: int = 1024, workers: int = 0) -> list[dict]:
    """Run many programs in parallel. Each job has keys code, tests, setup.

    workers=0 uses one worker per CPU.
    """
    workers = workers or os.cpu_count() or 2

    def one(job):
        return run_tests(job["code"], job["tests"], job.get("setup", ""), timeout_s, memory_mb)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, jobs))
