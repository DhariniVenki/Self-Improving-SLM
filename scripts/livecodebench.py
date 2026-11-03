"""Score a model on LiveCodeBench problems released after it (task 25).

    # Run A (base model)
    python scripts/livecodebench.py --run A

    # A fine-tuned model
    python scripts/livecodebench.py --run B_round1 --model /content/models/B_round1/merged

Greedy decoding, one solution per problem. Downloads about 700 MB from
Hugging Face the first time. Results go to
<output_dir>/<run>/livecodebench/greedy/{records.jsonl,summary.json}; existing
results are skipped unless --overwrite (or the problem set or prompt changed).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slm.config import load_config  # noqa: E402
from slm.evaluate import save  # noqa: E402
from slm.generate import Generator  # noqa: E402
from slm.livecodebench import LCB_PROMPT_VERSION, build_messages, download, load_problems, score, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", default="A")
    ap.add_argument("--model", default=None, help="Model path (default: base model from config).")
    ap.add_argument("--backend", default=None, choices=["auto", "vllm", "hf"])
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--since", default=None, help="Only problems released on or after this date (default: config).")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N problems (for a quick test).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.output_dir)
    lcb = cfg["livecodebench"]
    since = args.since or lcb["since"]
    model = args.model or cfg["model"]
    out = Path(cfg["output_dir"]) / args.run / "livecodebench" / "greedy"

    problems = load_problems(download(cfg), since)[: args.limit]
    ids = [p["question_id"] for p in problems]
    print(f"{len(problems)} problems released {problems[0]['contest_date']} to {problems[-1]['contest_date']}")

    if (out / "summary.json").exists() and not args.overwrite:
        saved = json.loads((out / "summary.json").read_text())
        if saved.get("question_ids") == ids and saved.get("prompt_version") == LCB_PROMPT_VERSION:
            print(f"Already scored in {out}; use --overwrite to redo it.")
            return
        print("Saved results used a different problem set or prompt; redoing them.")

    g = cfg["generation"]
    gen = Generator(model, args.backend or g["backend"], lcb["max_new_tokens"], g["batch_size"], cfg["seed"],
                    max_model_len=lcb["max_model_len"])
    t0 = time.time()
    completions = gen.generate([build_messages(p) for p in problems], n=1, temperature=0.0)
    t1 = time.time()
    records = score(problems, completions, cfg)
    summary = summarize(records)
    meta = {"run": args.run, "model": model, "benchmark": "livecodebench", "since": since,
            "files": lcb["files"], "prompt_version": LCB_PROMPT_VERSION, "backend": gen.backend,
            "generate_seconds": round(t1 - t0, 1), "score_seconds": round(time.time() - t1, 1),
            "question_ids": ids}
    save(records, summary, out, meta)

    print(f"\n== {args.run} / LiveCodeBench ({summary['n_problems']} problems, "
          f"{summary['first_date']} to {summary['last_date']}) ==")
    print(f"pass@1: {summary['pass@1']:.4f}")
    for key in ("by_difficulty", "by_platform"):
        print(f"{key}: " + ", ".join(f"{k} {v['pass@1']:.3f} (n={v['n']})" for k, v in summary[key].items()))
    print("status counts: " + ", ".join(f"{s}={c}" for s, c in summary["status_counts"].items()))
    print(f"passed the examples but failed a hidden test: {summary['shown_pass_hidden_fail']}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
