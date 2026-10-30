"""Generate solutions for a set of problems, run them, and summarize.

Saves one line per sample (records.jsonl) so later analysis, such as
McNemar's test or failure-type breakdowns, can work from the raw results.
"""
from __future__ import annotations

import json
from collections import Counter
from math import comb
from pathlib import Path

from .prompts import build_messages, extract_code
from .sandbox import STATUSES, run_many


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimate from n samples with c correct (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def score(problems: list[dict], completions: list[list[str]], cfg: dict) -> list[dict]:
    """Run every completion against its problem's tests. One record per sample."""
    n_shown = cfg["prompt"]["n_shown_tests"]
    sb = cfg["sandbox"]
    jobs, meta = [], []
    for p, comps in zip(problems, completions):
        for k, comp in enumerate(comps):
            jobs.append({"code": extract_code(comp), "tests": p["test_list"], "setup": p["test_setup_code"]})
            meta.append((p["task_id"], k, comp))
    results = run_many(jobs, timeout_s=sb["timeout_s"], memory_mb=sb["memory_mb"], workers=sb["workers"])

    records = []
    for (task_id, k, comp), job, r in zip(meta, jobs, results):
        shown = r["tests"][:n_shown]
        records.append({
            "task_id": task_id,
            "sample_idx": k,
            "status": r["status"],
            "passed": r["passed"],
            # Passed the test(s) the model saw in its prompt.
            "passed_shown": len(shown) == n_shown and all(t == "pass" for t in shown),
            "tests": r["tests"],
            "error": r["error"],
            "code": job["code"],
            "completion": comp,
        })
    return records


def summarize(records: list[dict]) -> dict:
    by_task: dict[int, list[dict]] = {}
    for r in records:
        by_task.setdefault(r["task_id"], []).append(r)
    n = len(next(iter(by_task.values())))
    correct = [sum(r["passed"] for r in rs) for rs in by_task.values()]

    statuses = Counter(r["status"] for r in records)
    summary = {
        "n_problems": len(by_task),
        "samples_per_problem": n,
        "n_samples": len(records),
        "pass@1": sum(pass_at_k(n, c, 1) for c in correct) / len(correct),
        "problems_solved_at_least_once": sum(c > 0 for c in correct),
        "status_counts": {s: statuses.get(s, 0) for s in STATUSES},
        "status_share": {s: statuses.get(s, 0) / len(records) for s in STATUSES},
        # Passed the test shown in the prompt but failed a hidden one.
        "shown_pass_hidden_fail": sum(r["passed_shown"] and not r["passed"] for r in records),
    }
    if n > 1:
        summary[f"pass@{n}"] = sum(pass_at_k(n, c, n) for c in correct) / len(correct)
    return summary


def evaluate(generator, problems: list[dict], cfg: dict, n: int = 1, temperature: float = 0.0, top_p: float = 1.0):
    messages = [build_messages(p, cfg["prompt"]["n_shown_tests"]) for p in problems]
    completions = generator.generate(messages, n=n, temperature=temperature, top_p=top_p)
    records = score(problems, completions, cfg)
    return records, summarize(records)


def save(records: list[dict], summary: dict, out_dir: str | Path, meta: dict | None = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "summary.json", "w") as f:
        json.dump({**(meta or {}), **summary}, f, indent=2)


def print_summary(name: str, summary: dict) -> None:
    print(f"\n== {name} ==")
    print(f"problems: {summary['n_problems']}   samples: {summary['n_samples']}")
    print(f"pass@1: {summary['pass@1']:.4f}")
    for key in summary:
        if key.startswith("pass@") and key != "pass@1":
            print(f"{key}: {summary[key]:.4f}")
    print("status counts: " + ", ".join(f"{s}={c}" for s, c in summary["status_counts"].items()))
    print(f"passed shown test but failed a hidden one: {summary['shown_pass_hidden_fail']}")
