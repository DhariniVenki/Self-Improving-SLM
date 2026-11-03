"""Build fine-tuning sets from sampled solutions (Runs B, C) or the human
solutions (Run D).

Every training example is the same prompt the model sees at evaluation
(build_messages) plus a response holding the code in a single ```python
block, the format the prompt asks for. Loss is computed on the response only.

    B  samples that pass all 3 tests, deduplicated, at most max_per_problem
    C  samples pass or fail (only empty ones dropped), same dedup and cap
    D  the MBPP reference solution, one per problem
"""
from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

from .prompts import build_messages


def normalize(code: str) -> str:
    """Key for deduplication: the same program up to comments and formatting.

    ast.unparse drops comments and normalizes whitespace and quotes. Code that
    doesn't parse (only possible in Run C) is compared as stripped text.
    """
    try:
        return ast.unparse(ast.parse(code))
    except (SyntaxError, ValueError):
        return "\n".join(line.rstrip() for line in code.strip().splitlines())


def format_response(code: str) -> str:
    return f"```python\n{code.strip()}\n```"


def per_problem_caps(greedy_records: list[dict], max_per_problem: int, solved_cap: int) -> dict[int, int]:
    """Cap per problem: solved_cap where greedy already passes, else max_per_problem.

    Solutions to problems the model already solves teach it little, so these
    caps shift the training set toward problems it gets wrong (0 drops the
    solved problems, 1 keeps one example of each).
    """
    return {r["task_id"]: solved_cap if r["passed"] else max_per_problem for r in greedy_records}


def fix_candidates(cfg: dict) -> list[dict]:
    """The self-training variants to compare on heldout, in tie-break order."""
    fixes = cfg["fixes"]
    return [{"run": f"{name}_lr{lr:g}", "data_run": name, "solved_cap": cap, "learning_rate": lr}
            for name, cap in fixes["datasets"].items() for lr in fixes["learning_rates"]]


def select_samples(records: list[dict], only_passing: bool, max_per_problem: int,
                   caps: dict[int, int] | None = None) -> list[dict]:
    """Distinct samples per problem, in sample order, at most max_per_problem each
    (or caps[task_id] for the problems listed in caps)."""
    by_task: dict[int, list[dict]] = {}
    for r in sorted(records, key=lambda r: (r["task_id"], r["sample_idx"])):
        by_task.setdefault(r["task_id"], []).append(r)

    chosen = []
    for task_id, rs in by_task.items():
        cap = (caps or {}).get(task_id, max_per_problem)
        seen = set()
        for r in rs:
            if not r["code"].strip() or (only_passing and not r["passed"]):
                continue
            key = normalize(r["code"])
            if key in seen:
                continue
            seen.add(key)
            if len(seen) <= cap:
                chosen.append(r)
    return chosen


def distinct_counts(records: list[dict], only_passing: bool) -> dict[int, int]:
    """Distinct (normalized) programs per problem, before the cap (task 14)."""
    keys: dict[int, set] = {}
    for r in records:
        keys.setdefault(r["task_id"], set())
        if r["code"].strip() and (r["passed"] or not only_passing):
            keys[r["task_id"]].add(normalize(r["code"]))
    return {t: len(k) for t, k in keys.items()}


def examples_from_samples(records: list[dict], problems: dict[int, dict], only_passing: bool,
                          max_per_problem: int, n_shown_tests: int,
                          caps: dict[int, int] | None = None) -> list[dict]:
    return [{
        "task_id": r["task_id"],
        "sample_idx": r["sample_idx"],
        "status": r["status"],
        "messages": build_messages(problems[r["task_id"]], n_shown_tests),
        "response": format_response(r["code"]),
    } for r in select_samples(records, only_passing, max_per_problem, caps)]


def examples_from_references(problems: list[dict], n_shown_tests: int, exclude: set[int] = frozenset()) -> list[dict]:
    return [{
        "task_id": p["task_id"],
        "sample_idx": 0,
        "status": "reference",
        "messages": build_messages(p, n_shown_tests),
        "response": format_response(p["code"]),
    } for p in problems if p["task_id"] not in exclude]


def stats(examples: list[dict], n_problems: int, distinct: dict[int, int] | None = None) -> dict:
    per_task = Counter(e["task_id"] for e in examples)
    out = {
        "n_examples": len(examples),
        "n_problems": n_problems,
        "problems_with_examples": len(per_task),
        "problems_without_examples": n_problems - len(per_task),
        "examples_per_problem": dict(sorted(Counter(per_task.values()).items())),
        "status_counts": dict(Counter(e["status"] for e in examples)),
    }
    wrong = sum(e["status"] not in ("pass", "reference") for e in examples)
    out["share_not_passing"] = wrong / len(examples) if examples else 0.0
    if distinct is not None:
        out["distinct_before_cap_per_problem"] = dict(sorted(Counter(distinct.values()).items()))
        out["distinct_before_cap_total"] = sum(distinct.values())
    return out


def save_examples(examples: list[dict], info: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train.jsonl", "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    with open(out_dir / "stats.json", "w") as f:
        json.dump(info, f, indent=2)


def load_examples(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]
