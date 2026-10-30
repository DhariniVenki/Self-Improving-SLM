"""Task 2: load the model and check that it writes runnable code.

    python scripts/check_model.py

Uses MBPP's 10 'prompt' problems, which are not in any of our splits.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slm.config import load_config  # noqa: E402
from slm.data import load_mbpp  # noqa: E402
from slm.evaluate import print_summary, score, summarize  # noqa: E402
from slm.generate import Generator  # noqa: E402
from slm.prompts import build_messages  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--backend", default=None, choices=["auto", "vllm", "hf"], help="Override generation.backend.")
    ap.add_argument("--model", default=None)
    ap.add_argument("--show", type=int, default=3, help="How many completions to print in full.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    g = cfg["generation"]
    gen = Generator(args.model or cfg["model"], args.backend or g["backend"], g["max_new_tokens"], g["batch_size"], cfg["seed"])

    problems = [p for p in load_mbpp(cfg).values() if p["hf_split"] == "prompt"]
    messages = [build_messages(p, cfg["prompt"]["n_shown_tests"]) for p in problems]
    print("Example prompt as the model sees it:\n" + gen.chat_prompts(messages[:1])[0])

    completions = gen.generate(messages, n=1, temperature=0.0)
    records = score(problems, completions, cfg)
    for p, r in list(zip(problems, records))[: args.show]:
        print(f"\n--- task {p['task_id']}: {p['text']}")
        print(r["completion"])
        print(f"--> status: {r['status']}  tests: {r['tests']}  error: {r['error']}")
    print_summary("prompt split (10 problems)", summarize(records))


if __name__ == "__main__":
    main()
