"""Build a fine-tuning set (tasks 14, 17, 20, 21).

    # Run D: the human-written solutions for the training problems
    python scripts/build_dataset.py --run D --source reference

    # Run B / C round 1, from the base model's 8 samples per training problem
    python scripts/build_dataset.py --run B_round1 --source A/train/sample8 --filter pass
    python scripts/build_dataset.py --run C_round1 --source A/train/sample8 --filter none

--source is a records folder under output_dir (made by scripts/evaluate.py
--splits train --pass-at-k 8). Writes <output_dir>/<run>/data/{train.jsonl,stats.json}.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slm.config import load_config, resolve  # noqa: E402
from slm.data import get_split, load_mbpp  # noqa: E402
from slm.selftrain import (distinct_counts, examples_from_references, examples_from_samples,  # noqa: E402
                           save_examples, stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", required=True, help="Run name, e.g. D or B_round1.")
    ap.add_argument("--source", required=True, help="'reference', or a records folder such as A/train/sample8.")
    ap.add_argument("--filter", choices=["pass", "none"], default="pass", help="Keep passing samples only (B) or all (C).")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.output_dir)
    out_root = Path(cfg["output_dir"])
    n_shown = cfg["prompt"]["n_shown_tests"]
    problems = load_mbpp(cfg)
    train = get_split(cfg, "train", problems)

    if args.source == "reference":
        # Reference solutions that fail their own tests (none are in train, but check anyway).
        broken = {int(t) for t in json.loads(resolve("data/reference_failures.json").read_text())}
        examples = examples_from_references(train, n_shown, exclude=broken)
        info = {"source": "reference", **stats(examples, len(train))}
    else:
        src = out_root / args.source
        with open(src / "records.jsonl") as f:
            records = [json.loads(line) for line in f]
        train_ids = {p["task_id"] for p in train}
        if {r["task_id"] for r in records} - train_ids:
            sys.exit(f"{src} has problems outside the train split; only train samples may be trained on.")
        only_passing = args.filter == "pass"
        cap = cfg["selftrain"]["max_per_problem"]
        examples = examples_from_samples(records, problems, only_passing, cap, n_shown)
        info = {"source": args.source, "filter": args.filter, "max_per_problem": cap,
                **stats(examples, len(train), distinct_counts(records, only_passing))}

    out = out_root / args.run / "data"
    save_examples(examples, info, out)
    print(json.dumps(info, indent=2))
    print(f"saved {len(examples)} examples to {out}")


if __name__ == "__main__":
    main()
