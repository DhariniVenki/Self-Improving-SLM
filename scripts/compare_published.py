"""Why is our MBPP pass@1 lower than the published score for this model?

The Qwen2.5-Coder technical report (arXiv:2409.12186, Table 16) gives
52.4% on MBPP for Qwen2.5-Coder-0.5B-Instruct, using EvalPlus, which
evaluates a hand-cleaned subset of MBPP. Our protocol uses the 500 original
MBPP test problems. This script changes one thing at a time, with greedy
decoding, so the report can say where the gap comes from:

    1. ours           original descriptions and tests, 1 test shown, 500 problems
    2. ours, subset   same results, restricted to the 257 problems in sanitized MBPP
    3. sanitized      cleaned descriptions and tests for those 257, 1 test shown
    4. sanitized, 3   same, with all 3 tests shown in the prompt

    python scripts/compare_published.py --backend vllm

This is a diagnostic only. Every run (A, B, C, D) is still scored with
setting 1, so changing the protocol for one run never makes runs
incomparable.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slm.config import load_config  # noqa: E402
from slm.data import get_split, load_mbpp, load_mbpp_sanitized  # noqa: E402
from slm.evaluate import evaluate, save, summarize  # noqa: E402
from slm.generate import Generator  # noqa: E402

PUBLISHED = 0.524  # Qwen2.5-Coder-0.5B-Instruct, MBPP (EvalPlus), arXiv:2409.12186 Table 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--backend", default=None, choices=["auto", "vllm", "hf"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Only the first N problems (for a quick test).")
    args = ap.parse_args()

    cfg = load_config(args.config, args.output_dir)
    g = cfg["generation"]
    model = args.model or cfg["model"]
    out_dir = Path(cfg["output_dir"]) / "A_compare"

    test = get_split(cfg, "test", load_mbpp(cfg))[: args.limit]
    sanitized = load_mbpp_sanitized(cfg)
    subset_ids = {p["task_id"] for p in test if p["task_id"] in sanitized}
    sanitized_test = [sanitized[p["task_id"]] for p in test if p["task_id"] in subset_ids]

    show3 = copy.deepcopy(cfg)
    show3["prompt"]["n_shown_tests"] = 3

    gen = Generator(model, args.backend or g["backend"], g["max_new_tokens"], g["batch_size"], cfg["seed"])
    rows = []

    records, s = evaluate(gen, test, cfg)
    save(records, s, out_dir / "1_ours")
    rows.append(("1. ours (original MBPP, 1 test shown)", s))

    sub = [r for r in records if r["task_id"] in subset_ids]
    s = summarize(sub)
    save(sub, s, out_dir / "2_ours_subset")
    rows.append(("2. ours, sanitized-subset problems only", s))

    records, s = evaluate(gen, sanitized_test, cfg)
    save(records, s, out_dir / "3_sanitized")
    rows.append(("3. sanitized descriptions and tests", s))

    records, s = evaluate(gen, sanitized_test, show3)
    save(records, s, out_dir / "4_sanitized_show3")
    rows.append(("4. sanitized, all 3 tests shown", s))

    table = [{"setting": name, "n_problems": s["n_problems"], "pass@1": s["pass@1"],
              "wrong_answer": s["status_counts"]["wrong_answer"], "runtime": s["status_counts"]["runtime"]}
             for name, s in rows]
    with open(out_dir / "comparison.json", "w") as f:
        json.dump({"model": model, "published_mbpp_evalplus": PUBLISHED, "rows": table}, f, indent=2)

    print(f"\n{'setting':45s} {'problems':>8s} {'pass@1':>8s}")
    for t in table:
        print(f"{t['setting']:45s} {t['n_problems']:8d} {t['pass@1']:8.3f}")
    print(f"{'published (Qwen2.5-Coder report, EvalPlus)':45s} {'~378':>8s} {PUBLISHED:8.3f}")
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
