"""Task 3: split MBPP into train / heldout / test and save the ids.

    python scripts/make_splits.py                     # writes data/splits.json
    python scripts/make_splits.py --check-references  # also runs the human solutions

Run once and commit data/splits.json so the whole group uses the same split.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slm.config import load_config, resolve  # noqa: E402
from slm.data import load_mbpp, make_splits, save_splits  # noqa: E402
from slm.sandbox import run_many  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check-references", action="store_true",
                    help="Run every human-written solution against its tests (needed later for Run D).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    problems = load_mbpp(cfg)
    path = resolve(cfg["data"]["splits_file"])

    if path.exists() and not args.overwrite:
        print(f"{path} already exists; leaving it unchanged (use --overwrite to regenerate).")
        splits = json.loads(path.read_text())
    else:
        splits = make_splits(problems, cfg["data"]["heldout_size"], cfg["seed"])
        save_splits(splits, path)
        print(f"Wrote {path}")
    for name, ids in splits.items():
        print(f"  {name:8s} {len(ids):4d} problems")

    if args.check_references:
        ids = [t for name in ("train", "heldout", "test") for t in splits[name]]
        jobs = [{"code": problems[t]["code"], "tests": problems[t]["test_list"],
                 "setup": problems[t]["test_setup_code"]} for t in ids]
        sb = cfg["sandbox"]
        results = run_many(jobs, timeout_s=sb["timeout_s"], memory_mb=sb["memory_mb"], workers=sb["workers"])
        failing = {t: r for t, r in zip(ids, results) if not r["passed"]}
        print(f"\nReference solutions passing their own tests: {len(ids) - len(failing)}/{len(ids)}")
        print("Failures by status:", dict(Counter(r["status"] for r in failing.values())))
        out = resolve("data/reference_failures.json")
        out.write_text(json.dumps({str(t): {"status": r["status"], "error": r["error"]}
                                   for t, r in sorted(failing.items())}, indent=1))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
