"""LiveCodeBench (task 25): competition problems released after the model.

MBPP may be in the model's training data. LiveCodeBench collects problems
from AtCoder and LeetCode with their release dates, so we can score only
problems released after Qwen2.5-Coder came out (September 2024). A gain
there can't come from memorized problems.

Two kinds of problems:
    stdin       (AtCoder) a program that reads input and prints the answer;
                the printed output is compared with the expected output
    functional  (LeetCode) a `class Solution` method given in starter code;
                it is called with the test's arguments and its return value compared

Grading follows LiveCodeBench: public tests (the examples in the problem
statement) first, then the hidden ones, stopping at the first failure. Output
is compared line by line ignoring surrounding whitespace, and numbers are
compared as decimals ("1.0" equals "1").

Each program runs in its own subprocess with a CPU-time limit per test (and a
wall-clock limit for programs that sleep), plus memory and file-size limits,
like slm/sandbox.py. The time and memory limits need Linux (Colab); on other
systems only the overall wall-clock limit applies.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .prompts import extract_code
from .sandbox import STATUSES

LCB_PROMPT_VERSION = 1

# LiveCodeBench's own prompt for chat models, so scores are comparable to published ones.
SYSTEM_PROMPT = ("You are an expert Python programmer. You will be given a question (problem specification) "
                 "and will generate a correct Python program that matches the specification and passes all tests.")
FORMAT_STARTER = ("### Format: You will use the following starter code to write the solution to the problem "
                  "and enclose your code within delimiters.\n```python\n{starter}\n```")
FORMAT_STDIN = ("### Format: Read the inputs from stdin solve the problem and write the answer to stdout "
                "(do not directly test on the sample inputs). Enclose your code within delimiters as follows. "
                "Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes "
                "output to STDOUT.\n```python\n# YOUR CODE HERE\n```")
USER_TEMPLATE = "### Question:\n{question}\n\n{format}\n\n### Answer: (use the provided format with backticks)\n\n"

# Prepended to every program, as LiveCodeBench does: LeetCode starter code uses
# typing names (List, Optional) without importing them.
HEADER = ("from typing import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\n"
          "from itertools import *\nfrom functools import *\n"
          "import math, string, re, sys, collections, heapq, bisect, itertools, functools\n")


# ---------------------------------------------------------------- data

class _NoGlobals(pickle.Unpickler):
    """The hidden tests are a pickled string. Refuse anything that isn't plain data,
    so a tampered file can't run code when it is unpickled."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def decode_tests(raw: str) -> list[dict]:
    """Public tests are JSON; hidden ones are base64(zlib(pickle(JSON string)))."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_NoGlobals(io.BytesIO(zlib.decompress(base64.b64decode(raw)))).load())


def download(cfg: dict) -> list[Path]:
    from huggingface_hub import hf_hub_download

    lcb = cfg["livecodebench"]
    return [Path(hf_hub_download(lcb["dataset"], f, repo_type="dataset")) for f in lcb["files"]]


def load_problems(paths: list[Path], since: str, until: str | None = None) -> list[dict]:
    """Problems with contest_date in [since, until), oldest first. Hidden tests stay
    encoded until they are run, since decoded they take several GB."""
    problems = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                date = r["contest_date"][:10]
                if date < since or (until and date >= until):
                    continue
                meta = json.loads(r["metadata"] or "{}")
                public = json.loads(r["public_test_cases"])
                problems[r["question_id"]] = {
                    "question_id": r["question_id"],
                    "title": r["question_title"],
                    "question": r["question_content"],
                    "platform": r["platform"],
                    "difficulty": r["difficulty"],
                    "contest_date": date,
                    "starter_code": r["starter_code"],
                    "func_name": meta.get("func_name"),
                    "kind": "functional" if r["starter_code"].strip() else "stdin",
                    "public_tests": public,
                    "private_raw": r["private_test_cases"],
                }
    return sorted(problems.values(), key=lambda p: (p["contest_date"], p["question_id"]))


def build_messages(problem: dict) -> list[dict]:
    if problem["kind"] == "functional":
        fmt = FORMAT_STARTER.format(starter=problem["starter_code"].rstrip())
    else:
        fmt = FORMAT_STDIN
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(question=problem["question"], format=fmt)}]


# ---------------------------------------------------------------- sandbox

# Runs inside the child process; see slm/sandbox.py for the general design.
_HARNESS = r'''
import io, json, os, signal, sys
from decimal import Decimal, InvalidOperation

class _TimeLimit(BaseException):
    pass  # BaseException, so `except Exception` in the solution can't swallow it

def _on_timer(signum, frame):
    raise _TimeLimit()

def _arm(cpu_s):
    if hasattr(signal, "setitimer"):
        signal.setitimer(signal.ITIMER_PROF, cpu_s)              # CPU time
        signal.setitimer(signal.ITIMER_REAL, 3 * cpu_s + 2 if cpu_s else 0)  # sleeping programs

def _limit(memory_mb):
    try:
        import resource
    except ImportError:
        return
    for name, value in (("RLIMIT_AS", memory_mb * 1024 * 1024), ("RLIMIT_FSIZE", 10 * 1024 * 1024)):
        try:
            resource.setrlimit(getattr(resource, name), (value, value))
        except (ValueError, OSError, AttributeError):
            pass

def _err(e):
    return (type(e).__name__ + ": " + str(e))[:300]

def _same_output(out, expected):
    a = [l.strip() for l in out.strip().splitlines()]
    b = [l.strip() for l in expected.strip().splitlines()]
    if a == b:
        return True
    ta, tb = out.split(), expected.split()
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x == y:
            continue
        try:
            if Decimal(x) != Decimal(y):
                return False
        except (InvalidOperation, ValueError):
            return False
    return True

def _plain(x):
    try:
        return json.loads(json.dumps(x))  # tuples -> lists, as in the expected JSON
    except (TypeError, ValueError):
        return x

def _run_stdin(compiled, inp):
    with open("in.txt", "w", encoding="utf-8") as f:
        f.write(inp)
    fd = os.open("in.txt", os.O_RDONLY)
    if fd != 0:         # 0 is free again if the last test's open(0) closed it
        os.dup2(fd, 0)  # so input(), sys.stdin and open(0) all read the test input
        os.close(fd)
    sys.stdin = io.TextIOWrapper(io.FileIO(0, "r", closefd=False), encoding="utf-8")
    buf = io.BytesIO()
    sys.stdout = io.TextIOWrapper(buf, encoding="utf-8", write_through=True)
    try:
        try:
            exec(compiled, {"__name__": "__main__", "__builtins__": __builtins__})
        except SystemExit:
            pass
        sys.stdout.flush()
        return buf.getvalue().decode("utf-8", errors="replace")
    finally:
        sys.stdout = sys.__stdout__

def _main():
    cfg = json.load(open("job.json", encoding="utf-8"))
    _limit(cfg["memory_mb"])
    if hasattr(signal, "SIGPROF"):
        signal.signal(signal.SIGPROF, _on_timer)
        signal.signal(signal.SIGALRM, _on_timer)
    out = {"stage": "compile", "tests": [], "error": None}

    def _write():
        with open("result.json", "w") as f:
            json.dump(out, f)

    try:
        compiled = compile(cfg["header"] + cfg["code"], "solution.py", "exec")
    except (SyntaxError, ValueError) as e:
        out["stage"], out["error"] = "syntax", _err(e)
        return _write()

    method = None
    if cfg["kind"] == "functional":
        out["stage"] = "define"
        _write()
        g = {"__name__": "solution", "__builtins__": __builtins__}
        try:
            _arm(cfg["cpu_s"])
            exec(compiled, g)
            _arm(0)
            method = getattr(g["Solution"](), cfg["func_name"])
        except _TimeLimit:
            out["stage"], out["error"] = "timeout", "time limit while defining the solution"
            return _write()
        except BaseException as e:
            _arm(0)
            out["stage"], out["error"] = "runtime", _err(e)
            return _write()

    out["stage"] = "tests"
    for t in cfg["tests"]:
        _write()  # so a hard crash still shows how far we got
        status = "pass"
        try:
            _arm(cfg["cpu_s"])
            if method is not None:
                args = [json.loads(line) for line in t["input"].split("\n")]
                ok = _plain(method(*args)) == json.loads(t["output"])
            else:
                ok = _same_output(_run_stdin(compiled, t["input"]), t["output"])
            _arm(0)
            if not ok:
                status = "wrong_answer"
                out["error"] = "wrong output"
        except _TimeLimit:
            status = "timeout"
            out["error"] = "time limit"
        except BaseException as e:
            _arm(0)
            status = "runtime"
            out["error"] = _err(e)
        out["tests"].append(status)
        if status != "pass":
            break  # LiveCodeBench stops at the first failing test
    out["stage"] = "done"
    _write()

_main()
'''


def run_problem(code: str, kind: str, tests: list[dict], func_name: str | None = None,
                timeout_s: float = 6.0, memory_mb: int = 1024) -> dict:
    """Run one program against a problem's tests. Status is one of STATUSES."""
    result = {"status": None, "passed": False, "tests": [], "error": None}
    if not code.strip():
        result["status"] = "no_code"
        return result

    cpu_s = max(1, math.ceil(timeout_s))
    with tempfile.TemporaryDirectory(prefix="slm_lcb_") as tmp:
        tmp = Path(tmp)
        (tmp / "harness.py").write_text(_HARNESS, encoding="utf-8")
        job = {"code": code, "kind": kind, "func_name": func_name, "header": HEADER,
               "tests": [{"input": t["input"], "output": t["output"]} for t in tests],
               "cpu_s": cpu_s, "memory_mb": memory_mb}
        (tmp / "job.json").write_text(json.dumps(job), encoding="utf-8")

        popen_kwargs = {"start_new_session": True} if os.name == "posix" else {}
        proc = subprocess.Popen([sys.executable, "-I", "harness.py"], cwd=tmp, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **popen_kwargs)
        # Backstop only: each test has its own limits inside the child.
        wall = (3 * cpu_s + 2) * (len(tests) + 1) + 10
        timed_out = False
        try:
            _, stderr = proc.communicate(timeout=wall)
        except subprocess.TimeoutExpired:
            timed_out = True
            if hasattr(os, "killpg"):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                proc.kill()
            _, stderr = proc.communicate()

        out = None
        if (tmp / "result.json").exists():
            try:
                out = json.loads((tmp / "result.json").read_text())
            except json.JSONDecodeError:
                out = None

    if out is not None:
        result["tests"], result["error"] = out["tests"], out["error"]
    if timed_out:
        result["status"] = "timeout"
    elif out is None or out["stage"] in ("compile", "define"):
        result["status"] = "runtime"
        result["error"] = stderr.decode(errors="replace").strip().splitlines()[-1][:300] if stderr else "process died"
    elif out["stage"] in ("syntax", "runtime", "timeout"):
        result["status"] = out["stage"]
    elif out["stage"] == "tests":
        # Died part-way (e.g. memory limit) without reporting.
        result["status"] = "runtime"
        result["error"] = result["error"] or "process died during tests"
    elif out["tests"] and all(t == "pass" for t in out["tests"]) and len(out["tests"]) == len(tests):
        result["status"], result["passed"] = "pass", True
    else:
        result["status"] = next((t for t in out["tests"] if t != "pass"), "runtime")
    return result


# ---------------------------------------------------------------- evaluation

def score(problems: list[dict], completions: list[list[str]], cfg: dict) -> list[dict]:
    lcb, sb = cfg["livecodebench"], cfg["sandbox"]

    def one(item):
        p, k, comp = item
        tests = p["public_tests"] + decode_tests(p["private_raw"])
        code = extract_code(comp)
        r = run_problem(code, p["kind"], tests, p["func_name"], lcb["timeout_s"], sb["memory_mb"])
        n_public = len(p["public_tests"])
        return {
            "question_id": p["question_id"], "sample_idx": k,
            "platform": p["platform"], "difficulty": p["difficulty"], "contest_date": p["contest_date"],
            "status": r["status"], "passed": r["passed"],
            # Passed the examples shown in the problem statement.
            "passed_shown": len(r["tests"]) >= n_public and all(t == "pass" for t in r["tests"][:n_public]),
            "n_tests": len(tests), "tests_passed": sum(t == "pass" for t in r["tests"]),
            "error": r["error"], "code": code, "completion": comp,
        }

    items = [(p, k, c) for p, comps in zip(problems, completions) for k, c in enumerate(comps)]
    with ThreadPoolExecutor(max_workers=sb["workers"] or os.cpu_count() or 2) as pool:
        return list(pool.map(one, items))


def summarize(records: list[dict]) -> dict:
    def rate(rs):
        return {"n": len(rs), "pass@1": sum(r["passed"] for r in rs) / len(rs) if rs else 0.0}

    statuses = Counter(r["status"] for r in records)
    groups = {}
    for key in ("difficulty", "platform"):
        groups[f"by_{key}"] = {v: rate([r for r in records if r[key] == v])
                               for v in sorted({r[key] for r in records})}
    return {
        "n_problems": len({r["question_id"] for r in records}),
        "n_samples": len(records),
        "pass@1": rate(records)["pass@1"],
        **groups,
        "status_counts": {s: statuses.get(s, 0) for s in STATUSES},
        "shown_pass_hidden_fail": sum(r["passed_shown"] and not r["passed"] for r in records),
        "first_date": min(r["contest_date"] for r in records),
        "last_date": max(r["contest_date"] for r in records),
    }
