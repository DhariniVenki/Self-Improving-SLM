"""LoRA fine-tuning (task 16).

Every run starts from the base model in the config, never from an earlier
round's model (ReST-EM). The loss covers only the response: prompt tokens
get label -100. The prompt and response are tokenized separately and
joined, so the model is trained on exactly the token sequence it sees when
it generates after the prompt.

After training the LoRA weights are merged into the base model and saved
as a normal model folder, which scripts/evaluate.py --model can load with
vLLM like any other model.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path


def tokenize_example(tokenizer, example: dict, max_length: int) -> dict:
    prompt = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    # eos is <|im_end|> for Qwen2.5 instruct models: training on it teaches the model to stop.
    response_ids = tokenizer(example["response"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + response_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]
    return {"input_ids": input_ids, "labels": labels}


class Collator:
    """Right-pads a batch; padded positions are masked out of attention and loss."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        width = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_token_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}


def train_lora(examples: list[dict], cfg: dict, out_dir: str | Path, learning_rate: float | None = None,
               epochs: float | None = None) -> dict:
    """Fine-tune the base model on examples; save the adapter and merged model in out_dir.

    Returns training info: settings, time, peak GPU memory and losses.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    from .generate import bf16_supported

    t = cfg["train"]
    learning_rate = learning_rate or t["learning_rate"]
    epochs = epochs or t["epochs"]
    out_dir = Path(out_dir)
    base = cfg["model"]

    tokenizer = AutoTokenizer.from_pretrained(base)
    data = [tokenize_example(tokenizer, e, t["max_length"]) for e in examples]
    n_truncated = sum(len(d["input_ids"]) == t["max_length"] for d in data)

    # Weights stay in float32; bf16 (A100) or fp16 (T4) is used only for the forward pass.
    use_bf16 = bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=t["lora_r"], lora_alpha=t["lora_alpha"],
        lora_dropout=t["lora_dropout"], target_modules=t["target_modules"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    steps = math.ceil(len(data) / (t["batch_size"] * t["grad_accum"])) * epochs
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        lr_scheduler_type="cosine",
        warmup_steps=int(t["warmup_ratio"] * steps),
        weight_decay=0.0,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        bf16=use_bf16,
        fp16=use_fp16,
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        remove_unused_columns=False,
        label_names=["labels"],
    )
    trainer = Trainer(model=model, args=args, train_dataset=data,
                      data_collator=Collator(tokenizer.pad_token_id or tokenizer.eos_token_id))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    seconds = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None

    model.save_pretrained(out_dir / "adapter")
    merged = model.merge_and_unload()
    merged.to(torch.bfloat16 if use_bf16 else torch.float16).save_pretrained(out_dir / "merged")
    tokenizer.save_pretrained(out_dir / "merged")

    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    return {
        "base_model": base,
        "n_examples": len(examples),
        "n_truncated": n_truncated,
        "learning_rate": learning_rate,
        "epochs": epochs,
        **{k: t[k] for k in ("lora_r", "lora_alpha", "lora_dropout", "target_modules",
                             "batch_size", "grad_accum", "max_length", "warmup_ratio")},
        "trainable_params": trainable,
        "trainable_share": trainable / total,
        "precision": "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32",
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "train_seconds": round(seconds, 1),
        "peak_gpu_memory_gb": round(peak_gb, 2) if peak_gb is not None else None,
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "loss_history": losses,
        "merged_model": str(out_dir / "merged"),
    }


def save_info(info: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2))
