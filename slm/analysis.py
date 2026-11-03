"""Compare runs on the same problems (tasks 26-29).

Everything works from the per-problem greedy results (records.jsonl), so
runs are compared problem by problem rather than only by their averages:

    McNemar's test     exact two-sided test on the problems where two runs disagree
    bootstrap CIs      resample problems; one set of resamples for every run, so
                       differences and ratios keep the pairing between runs
    gain recovered     (B - A) / (D - A): how much of fine-tuning on human solutions
                       self-training gets back
    failure types      share of problems per status, before and after training

Only the standard library is used, so it runs anywhere the results are.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from math import comb
from pathlib import Path

from .sandbox import STATUSES


def load_run(results_dir: str | Path, run: str, split: str, mode: str = "greedy") -> tuple[dict, list[dict]] | None:
    """(summary, records) for one run, or None if it hasn't been run yet."""
    d = Path(results_dir) / run / split / mode
    if not (d / "summary.json").exists() or not (d / "records.jsonl").exists():
        return None
    summary = json.loads((d / "summary.json").read_text())
    with open(d / "records.jsonl") as f:
        records = [json.loads(line) for line in f]
    return summary, records


def passed_by_task(records: list[dict]) -> dict[int, bool]:
    """Greedy pass/fail per problem (the first sample of each problem)."""
    out = {}
    for r in sorted(records, key=lambda r: (r["task_id"], r["sample_idx"])):
        out.setdefault(r["task_id"], bool(r["passed"]))
    return out


def status_shares(records: list[dict]) -> dict[str, float]:
    """Share of problems with each status (greedy: one sample per problem)."""
    status = {}
    for r in sorted(records, key=lambda r: (r["task_id"], r["sample_idx"])):
        status.setdefault(r["task_id"], r["status"])
    n = len(status)
    return {s: sum(v == s for v in status.values()) / n for s in STATUSES}


def sampling_stats(summary: dict, records: list[dict]) -> dict:
    """Round 0 statistics for one split sampled k times (tasks 12-13)."""
    n = summary["n_problems"]
    k = summary["samples_per_problem"]
    correct = Counter()
    for r in records:
        correct[r["task_id"]] += r["passed"]
    shown_pass = sum(r["passed_shown"] for r in records)
    return {
        "n_problems": n,
        "samples_per_problem": k,
        "pass@1": summary["pass@1"],
        f"pass@{k}": summary[f"pass@{k}"],
        "never_solved": n - summary["problems_solved_at_least_once"],
        "always_solved": sum(c == k for c in correct.values()),
        # How many of the k samples solve each problem: {0: problems, 1: ..., k: ...}.
        "solve_count_histogram": {c: sum(v == c for v in correct.values()) for c in range(k + 1)},
        "samples_passing_shown_test": shown_pass,
        "shown_pass_hidden_fail": summary["shown_pass_hidden_fail"],
        # Of the samples that pass the test in the prompt, the share that fail a hidden one.
        "hidden_fail_share_of_shown_pass": summary["shown_pass_hidden_fail"] / shown_pass if shown_pass else 0.0,
        "status_share": summary["status_share"],
    }


def mcnemar(a: dict[int, bool], b: dict[int, bool]) -> dict:
    """Exact McNemar test (two-sided binomial on the discordant problems)."""
    tasks = sorted(a.keys() & b.keys())
    only_a = sum(a[t] and not b[t] for t in tasks)
    only_b = sum(b[t] and not a[t] for t in tasks)
    n = only_a + only_b
    if n == 0:
        p = 1.0
    else:
        tail = sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n
        p = min(1.0, 2 * tail)
    return {
        "n_problems": len(tasks),
        "both_pass": sum(a[t] and b[t] for t in tasks),
        "neither_pass": sum(not a[t] and not b[t] for t in tasks),
        "only_first": only_a,
        "only_second": only_b,
        "p_value": p,
    }


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def gain_recovered(base: float, run: float, reference: float) -> float | None:
    """(run - base) / (reference - base); None when the reference doesn't beat the base."""
    if reference <= base:
        return None
    return (run - base) / (reference - base)


def percentile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def ci(xs: list[float], level: float = 0.95) -> list[float]:
    return [percentile(xs, (1 - level) / 2), percentile(xs, (1 + level) / 2)]


def group_difference(first: list[dict[int, bool]], second: list[dict[int, bool]],
                     n_boot: int = 10000, seed: int = 0) -> dict:
    """Mean pass@1 of repeated runs of second minus first, with a bootstrap CI.

    Each problem's pass rate is averaged over the repeats, and problems are
    resampled, so the interval covers the choice of problems. With 3 repeats
    the run-to-run noise is averaged down but not resampled.
    """
    tasks = sorted(set.intersection(*(set(p) for p in first + second)))
    rate = lambda runs, t: sum(p[t] for p in runs) / len(runs)
    diffs = [rate(second, t) - rate(first, t) for t in tasks]
    n = len(tasks)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.choices(range(n), k=n)
        boots.append(sum(diffs[i] for i in idx) / n)
    return {"n_problems": n, "difference": sum(diffs) / n, "ci95": ci(boots), "n_boot": n_boot}


def bootstrap(passed: dict[str, dict[int, bool]], pairs: list[tuple[str, str]],
              recover: list[tuple[str, str, str]], n_boot: int = 10000, seed: int = 0) -> dict:
    """Percentile bootstrap over problems, shared by all runs.

    passed: run -> {task_id: passed}, all on the same problems.
    pairs: (first, second) for the difference second - first.
    recover: (base, run, reference) for gain_recovered.
    Resamples where the reference doesn't beat the base are left out of that
    ratio's interval; how many were used is reported.
    """
    tasks = sorted(set.intersection(*(set(p) for p in passed.values())))
    vec = {run: [int(p[t]) for t in tasks] for run, p in passed.items()}
    n = len(tasks)
    rng = random.Random(seed)
    means = {run: [] for run in vec}
    for _ in range(n_boot):
        idx = rng.choices(range(n), k=n)
        for run, v in vec.items():
            means[run].append(sum(v[i] for i in idx) / n)

    out = {"n_boot": n_boot, "n_problems": n, "pass@1": {}, "difference": {}, "gain_recovered": {}}
    for run, m in means.items():
        out["pass@1"][run] = {"estimate": sum(vec[run]) / n, "ci95": ci(m)}
    for a, b in pairs:
        diffs = [y - x for x, y in zip(means[a], means[b])]
        out["difference"][f"{b} - {a}"] = {"estimate": (sum(vec[b]) - sum(vec[a])) / n, "ci95": ci(diffs)}
    for base, run, ref in recover:
        est = gain_recovered(*(sum(vec[r]) / n for r in (base, run, ref)))
        ratios = [g for g in map(gain_recovered, means[base], means[run], means[ref]) if g is not None]
        out["gain_recovered"][f"{run} vs {ref}"] = {
            "estimate": est,
            "ci95": ci(ratios) if ratios else None,
            "resamples_used": len(ratios),
        }
    return out
