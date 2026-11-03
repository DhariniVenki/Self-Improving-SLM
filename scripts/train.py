"""LoRA fine-tuning from the base model (tasks 16-19).

    python scripts/train.py --run D --model-dir /content/models

Trains on <output_dir>/<data-run>/data/train.jsonl (made by
scripts/build_dataset.py; --data-run defaults to --run), saves the adapter
and merged model to <model-dir>/<run>/, and writes the settings, training
time and peak memory to <output_dir>/<run>/train_info.json. Then evaluate:

    python scripts/evaluate.py --run D --model /content/models/D/merged --splits test heldout

For tuning (task 18), --learning-rate and --epochs override the config:

    python scripts/train.py --run D_lr1e-4_ep2 --data-run D --learning-rate 1e-4 --epochs 2
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slm.config import load_config  # noqa: E402
from slm.selftrain import load_examples  # noqa: E402
from slm.train import save_info, train_lora  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", required=True, help="Name for this model and its results folder.")
    ap.add_argument("--data-run", default=None, help="Run whose data/train.jsonl to train on (default: --run).")
    ap.add_argument("--output-dir", default=None, help="Override output_dir, e.g. a Google Drive folder.")
    ap.add_argument("--model-dir", default=None, help="Where models are saved (default: model_dir in config).")
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None, help="Override the config seed, for repeated runs.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.output_dir)
    if args.seed is not None:
        cfg["seed"] = args.seed
    out_root = Path(cfg["output_dir"])
    info_path = out_root / args.run / "train_info.json"
    model_dir = Path(args.model_dir or cfg["model_dir"]) / args.run
    if (model_dir / "merged").exists() and info_path.exists() and not args.overwrite:
        print(f"{model_dir / 'merged'} already exists; use --overwrite to retrain.")
        return

    data_path = out_root / (args.data_run or args.run) / "data" / "train.jsonl"
    examples = load_examples(data_path)
    print(f"training on {len(examples)} examples from {data_path}")
    info = train_lora(examples, cfg, model_dir, args.learning_rate, args.epochs)
    info["data"] = str(data_path)
    save_info(info, info_path)
    print(json.dumps({k: v for k, v in info.items() if k != "loss_history"}, indent=2))
    print(f"saved model to {model_dir}, info to {info_path}")


if __name__ == "__main__":
    main()
