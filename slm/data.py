"""MBPP loading and the fixed train / heldout / test splits.

The splits are written once to data/splits.json and committed, so everyone
in the group trains and evaluates on exactly the same problem ids.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from .config import resolve

SPLIT_NAMES = ("train", "heldout", "test")

# Standard-library modules that a few MBPP tests use without importing
# (e.g. task 596 calls sys.getsizeof). The reference solutions import them,
# so a correct solution that doesn't would otherwise fail the test.
_TEST_MODULES = ("sys", "math", "re", "heapq", "collections", "itertools", "cmath", "string", "operator")


def _add_missing_imports(tests: list[str], setup: str) -> str:
    text = "\n".join(tests) + "\n" + setup
    missing = [m for m in _TEST_MODULES
               if re.search(rf"\b{m}\.\w", text) and not re.search(rf"^\s*(import|from)\s+{m}\b", text, re.M)]
    return "\n".join([f"import {m}" for m in missing] + ([setup] if setup else []))


def load_mbpp(cfg: dict) -> dict[int, dict]:
    """Return every MBPP problem keyed by task_id, with the HF split it came from."""
    from datasets import load_dataset

    ds = load_dataset(cfg["data"]["dataset"], cfg["data"]["config"])
    problems = {}
    for hf_split in ds:
        for ex in ds[hf_split]:
            tests = list(ex["test_list"])
            problems[ex["task_id"]] = {
                "task_id": ex["task_id"],
                "text": ex["text"],
                "code": ex["code"],
                "test_list": tests,
                "test_setup_code": _add_missing_imports(tests, ex.get("test_setup_code") or ""),
                "hf_split": hf_split,
            }
    return problems


def load_mbpp_sanitized(cfg: dict) -> dict[int, dict]:
    """The hand-checked 'sanitized' MBPP subset (427 problems), in the same format.

    Its descriptions and tests were cleaned up by the MBPP authors; it is
    close to what EvalPlus and the Qwen2.5-Coder report evaluate on. Used
    only for the comparison with published scores, never for training.
    """
    from datasets import load_dataset

    ds = load_dataset(cfg["data"]["dataset"], "sanitized")
    problems = {}
    for hf_split in ds:
        for ex in ds[hf_split]:
            tests = list(ex["test_list"])
            problems[ex["task_id"]] = {
                "task_id": ex["task_id"],
                "text": ex["prompt"],
                "code": ex["code"],
                "test_list": tests,
                "test_setup_code": _add_missing_imports(tests, "\n".join(ex["test_imports"])),
                "hf_split": hf_split,
            }
    return problems


def split_hash(ids: list[int]) -> str:
    """Short fingerprint of a split's problem ids, stored with each result so
    results computed on an older version of a split are recognised as stale."""
    return hashlib.sha1(",".join(map(str, sorted(ids))).encode()).hexdigest()[:12]


def make_splits(problems: dict[int, dict], heldout_size: int, seed: int) -> dict[str, list[int]]:
    """train+validation -> train / heldout; HF test stays the test set.

    The 10 'prompt' problems are left out of everything.
    """
    pool = sorted(t for t, p in problems.items() if p["hf_split"] in ("train", "validation"))
    heldout = sorted(random.Random(seed).sample(pool, heldout_size))
    heldout_set = set(heldout)
    return {
        "train": [t for t in pool if t not in heldout_set],
        "heldout": heldout,
        "test": sorted(t for t, p in problems.items() if p["hf_split"] == "test"),
    }


def save_splits(splits: dict[str, list[int]], path: str | Path) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=1)


def load_splits(path: str | Path) -> dict[str, list[int]]:
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/make_splits.py first.")
    with open(path) as f:
        return json.load(f)


def get_split(cfg: dict, name: str, problems: dict[int, dict] | None = None) -> list[dict]:
    """The problems in one of our splits, in task_id order."""
    if name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {name!r}; expected one of {SPLIT_NAMES}")
    if problems is None:
        problems = load_mbpp(cfg)
    ids = load_splits(cfg["data"]["splits_file"])[name]
    return [problems[t] for t in ids]
