"""Numbers and figures for the report from Round 0 and Run D's tuning
(tasks 12-14, 18-21), all from results already in the repo:

    python scripts/round0_report.py

Writes to <output_dir>/analysis/round0/:

    round0.json, round0.md   sampling on train (and test, for comparison),
                             the training sets for B, C and D, training cost
    tuning.json, tuning.md   heldout pass@1 for every learning rate and epoch
                             count, with McNemar's test against the base model
    tuning.png               the same, against the base model and its 95% CI
    solve_counts.png         how many of the 8 samples solve each problem

Every tuning setting is compared with the base model on the same 150
heldout problems; those p-values are Holm-corrected over all settings.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slm.analysis import bootstrap, holm, load_run, mcnemar, passed_by_task, sampling_stats  # noqa: E402
from slm.config import load_config  # noqa: E402
from slm.sandbox import STATUSES  # noqa: E402


def tuning_runs(root: Path) -> list[tuple[float, int, str]]:
    """(learning rate, epochs, run name) for every D_tune/lr<lr>_ep<epochs> folder."""
    out = []
    for d in sorted((root / "D_tune").glob("lr*_ep*")):
        m = re.fullmatch(r"lr([0-9.e+-]+)_ep(\d+)", d.name)
        if m:
            out.append((float(m.group(1)), int(m.group(2)), f"D_tune/{d.name}"))
    return sorted(out)


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def write_round0(root: Path, out: Path) -> dict:
    report = {"sampling": {}, "training_sets": {}, "training_cost": {}}
    for split in ("train", "test"):
        got = load_run(root, "A", split, "sample8")
        if got is not None:
            report["sampling"][split] = sampling_stats(*got)
    for run in ("B_round1", "C_round1", "D"):
        path = root / run / "data" / "stats.json"
        if path.exists():
            report["training_sets"][run] = json.loads(path.read_text())
    for run in ("D", "B_round1", "C_round1"):
        path = root / run / "train_info.json"
        if path.exists():
            t = json.loads(path.read_text())
            report["training_cost"][run] = {k: t.get(k) for k in (
                "n_examples", "learning_rate", "epochs", "train_seconds", "peak_gpu_memory_gb", "gpu",
                "trainable_params", "trainable_share", "first_loss", "final_loss")}
    (out / "round0.json").write_text(json.dumps(report, indent=2) + "\n")

    md = ["## Round 0: 8 samples per problem from the base model (temperature 0.8)", "",
          "| | train | test |", "|---|---|---|"]
    s = report["sampling"]
    rows = [
        ("Problems", lambda x: str(x["n_problems"])),
        ("pass@1 (from the 8 samples)", lambda x: pct(x["pass@1"])),
        ("pass@8", lambda x: pct(x["pass@8"])),
        ("Never solved (task 12)", lambda x: f"{x['never_solved']} ({pct(x['never_solved'] / x['n_problems'])}%)"),
        ("Solved by all 8 samples", lambda x: f"{x['always_solved']} ({pct(x['always_solved'] / x['n_problems'])}%)"),
        ("Samples passing the shown test", lambda x: str(x["samples_passing_shown_test"])),
        ("...of which fail a hidden test (task 13)",
         lambda x: f"{x['shown_pass_hidden_fail']} ({pct(x['hidden_fail_share_of_shown_pass'])}%)"),
    ] + [(f"Samples: {st.replace('_', ' ')}", (lambda st: lambda x: pct(x["status_share"][st]))(st))
         for st in STATUSES]
    for name, f in rows:
        md.append(f"| {name} | " + " | ".join(f(s[sp]) if sp in s else "" for sp in ("train", "test")) + " |")

    ts = report["training_sets"]
    if ts:
        md += ["", "## Training sets (tasks 14, 20, 21)", "",
               "| Run | Examples | Problems covered | Distinct before cap | Share not passing |", "|---|---|---|---|---|"]
        for run, t in ts.items():
            md.append(f"| {run} | {t['n_examples']} | {t['problems_with_examples']}/{t['n_problems']} | "
                      f"{t.get('distinct_before_cap_total', '')} | {pct(t['share_not_passing'])}% |")
    tc = report["training_cost"]
    if tc:
        md += ["", "## Training cost (task 19)", "",
               "| Run | Examples | LR | Epochs | Seconds | Peak GPU GB | GPU | Loss first → final |",
               "|---|---|---|---|---|---|---|---|"]
        for run, t in tc.items():
            md.append(f"| {run} | {t['n_examples']} | {t['learning_rate']:g} | {t['epochs']:g} | {t['train_seconds']} | "
                      f"{t['peak_gpu_memory_gb']} | {t['gpu']} | {t['first_loss']:.3f} → {t['final_loss']:.3f} |")
    (out / "round0.md").write_text("\n".join(md) + "\n")
    return report


def write_tuning(root: Path, out: Path, seed: int) -> dict | None:
    base = load_run(root, "A", "heldout")
    settings = tuning_runs(root)
    if base is None or not settings:
        return None
    base_passed = passed_by_task(base[1])
    rows, passed = [], {"A": base_passed}
    for lr, ep, run in settings:
        got = load_run(root, run, "heldout")
        if got is None:
            continue
        passed[run] = passed_by_task(got[1])
        info_path = root / run / "train_info.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        rows.append({"run": run, "learning_rate": lr, "epochs": ep, "pass@1": got[0]["pass@1"],
                     "train_loss": info.get("train_loss"), **mcnemar(base_passed, passed[run])})
    for row, p in zip(rows, holm([r["p_value"] for r in rows])):
        row["p_value_holm"] = p

    # The final Run D used one of these settings; its heldout score is a repeat of that setting.
    repeat = None
    d = load_run(root, "D", "heldout")
    d_info = root / "D" / "train_info.json"
    if d is not None and d_info.exists():
        t = json.loads(d_info.read_text())
        same = [r["pass@1"] for r in rows if r["learning_rate"] == t["learning_rate"] and r["epochs"] == t["epochs"]]
        repeat = {"learning_rate": t["learning_rate"], "epochs": t["epochs"], "pass@1": d[0]["pass@1"],
                  "tuning_pass@1": same[0] if same else None}

    boot = bootstrap({"A": base_passed}, [], [], seed=seed)["pass@1"]["A"]
    tuning = {"base_pass@1": boot["estimate"], "base_ci95": boot["ci95"], "settings": rows,
              "below_base": sum(r["pass@1"] < boot["estimate"] for r in rows),
              "best": max(rows, key=lambda r: r["pass@1"])["run"], "repeat_of_chosen_setting": repeat}
    (out / "tuning.json").write_text(json.dumps(tuning, indent=2) + "\n")

    md = [f"Run D tuning on heldout ({rows[0]['n_problems']} problems). Base model: "
          f"{pct(boot['estimate'])}% [{pct(boot['ci95'][0])}, {pct(boot['ci95'][1])}]. "
          f"{tuning['below_base']} of {len(rows)} settings score below the base model. "
          f"McNemar p against the base model, Holm-corrected over all settings.", "",
          "| LR | Epochs | heldout pass@1 | vs base | only base / only tuned | p (Holm) | train loss |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        loss = "" if r["train_loss"] is None else f"{r['train_loss']:.3f}"
        md.append(f"| {r['learning_rate']:g} | {r['epochs']} | {pct(r['pass@1'])} | "
                  f"{100 * (r['pass@1'] - boot['estimate']):+.1f} | {r['only_first']} / {r['only_second']} | "
                  f"{r['p_value_holm']:.3f} | {loss} |")
    if repeat and repeat["tuning_pass@1"] is not None:
        md += ["", f"Run D used the setting LR {repeat['learning_rate']:g}, {repeat['epochs']:g} epochs, which scored "
                   f"{pct(repeat['tuning_pass@1'])}% in tuning. Trained and evaluated again, it scored "
                   f"{pct(repeat['pass@1'])}%: repeating one setting moves heldout pass@1 by "
                   f"{100 * abs(repeat['pass@1'] - repeat['tuning_pass@1']):.1f} points."]
    (out / "tuning.md").write_text("\n".join(md) + "\n")
    return tuning


def write_figures(report: dict, tuning: dict | None, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if tuning:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        lo, hi = (100 * x for x in tuning["base_ci95"])
        ax.axhspan(lo, hi, color="#C44E52", alpha=0.08, lw=0)
        ax.axhline(100 * tuning["base_pass@1"], ls="--", color="#C44E52",
                   label=f"base model ({pct(tuning['base_pass@1'])}, 95% CI shaded)")
        lrs = sorted({r["learning_rate"] for r in tuning["settings"]})
        for lr in lrs:
            pts = sorted((r["epochs"], 100 * r["pass@1"]) for r in tuning["settings"] if r["learning_rate"] == lr)
            ax.plot(*zip(*pts), marker="o", label=f"LR {lr:g}")
        rep = tuning["repeat_of_chosen_setting"]
        if rep:
            ax.plot(rep["epochs"], 100 * rep["pass@1"], marker="o", mfc="none", ms=9, color="black", ls="none",
                    label="Run D (same setting, retrained)")
        ax.set_xticks(sorted({r["epochs"] for r in tuning["settings"]}))
        ax.set_xlabel("epochs")
        ax.set_ylabel("heldout pass@1 (%)")
        ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower left")
        ax.set_ylim(min(30, lo - 2), max(60, hi + 2))
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "tuning.png", dpi=200)
        plt.close(fig)

    s = report["sampling"]
    if s:
        fig, ax = plt.subplots(figsize=(5, 3))
        width = 0.8 / len(s)
        for i, (split, st) in enumerate(s.items()):
            hist = st["solve_count_histogram"]
            xs = [int(c) + (i - (len(s) - 1) / 2) * width for c in hist]
            ax.bar(xs, [100 * v / st["n_problems"] for v in hist.values()], width=width, label=split)
        ax.set_xticks(range(len(next(iter(s.values()))["solve_count_histogram"])))
        ax.set_xlabel("samples (of 8) that pass all tests")
        ax.set_ylabel("share of problems (%)")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "solve_counts.png", dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--output-dir", default=None, help="Where the run results are (default: output_dir).")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.output_dir)
    root = Path(cfg["output_dir"])
    out = root / "analysis" / "round0"
    out.mkdir(parents=True, exist_ok=True)

    report = write_round0(root, out)
    tuning = write_tuning(root, out, cfg["seed"])
    if not args.no_figures:
        try:
            write_figures(report, tuning, out)
        except ImportError:
            print("matplotlib is not installed; skipped the figures (pip install matplotlib).")
    print((out / "round0.md").read_text())
    if tuning:
        print((out / "tuning.md").read_text())
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
