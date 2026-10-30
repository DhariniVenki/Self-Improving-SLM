"""Greedy evaluation of a model on our splits (task 8). Run A is this script
on the base model:

    python scripts/evaluate.py --run A --splits test heldout

Later runs pass their fine-tuned model:

    python scripts/evaluate.py --run B_round1 --model <path> --splits test heldout

Add --pass-at-k 8 to also sample 8 solutions per problem and report pass@8.
Results go to <output_dir>/<run>/<split>/<greedy|sampleK>/{records.jsonl,summary.json}.
Existing results are skipped unless --overwrite, so a Colab restart can
just rerun the same command.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slm.config import load_config  # noqa: E402
from slm.data import get_split, load_mbpp  # noqa: E402
from slm.evaluate import evaluate, print_summary, save  # noqa: E402
from slm.generate import Generator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", default="A", help="Name for this run's results folder.")
    ap.add_argument("--backend", default=None, choices=["auto", "vllm", "hf"], help="Override generation.backend.")
    ap.add_argument("--model", default=None, help="Model or checkpoint path (default: base model from config).")
    ap.add_argument("--splits", nargs="+", default=["test", "heldout"])
    ap.add_argument("--output-dir", default=None, help="Override output_dir, e.g. a Google Drive folder.")
    ap.add_argument("--pass-at-k", type=int, default=0, help="Also sample k solutions per problem for pass@k.")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N problems (for a quick test).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.output_dir)
    model = args.model or cfg["model"]
    g = cfg["generation"]
    run_dir = Path(cfg["output_dir"]) / args.run

    todo = []
    for split in args.splits:
        todo.append((split, "greedy", 1, g["eval"]["temperature"], 1.0))
        if args.pass_at_k > 1:
            todo.append((split, f"sample{args.pass_at_k}", args.pass_at_k, g["sample"]["temperature"], g["sample"]["top_p"]))
    todo = [t for t in todo if args.overwrite or not (run_dir / t[0] / t[1] / "summary.json").exists()]
    if not todo:
        print(f"All results already in {run_dir}; use --overwrite to redo them.")
        return

    problems = load_mbpp(cfg)
    gen = Generator(model, args.backend or g["backend"], g["max_new_tokens"], g["batch_size"], cfg["seed"])
    for split, mode, n, temperature, top_p in todo:
        items = get_split(cfg, split, problems)[: args.limit]
        t0 = time.time()
        records, summary = evaluate(gen, items, cfg, n=n, temperature=temperature, top_p=top_p)
        meta = {"run": args.run, "model": model, "split": split, "mode": mode, "temperature": temperature,
                "top_p": top_p, "backend": gen.backend, "seconds": round(time.time() - t0, 1)}
        out = run_dir / split / mode
        save(records, summary, out, meta)
        print_summary(f"run {args.run} / {split} / {mode}  ({meta['seconds']}s)", summary)
        print(f"saved to {out}")


if __name__ == "__main__":
    main()
