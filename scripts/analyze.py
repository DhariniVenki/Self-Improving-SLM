"""Analysis for the report (tasks 26-30): significance tests, confidence
intervals, failure types, and the figures.

    python scripts/analyze.py
    python scripts/analyze.py --runs A B_round1 C_round1 D --split test

Reads the greedy results of each run from output_dir (default results/) and
skips runs that haven't been run yet, so it can be rerun as runs finish.
Writes to <output_dir>/analysis/<split>/:

    analysis.json        every number below
    results_table.md     main results table
    results_table.tex    the same table for the ICML template
    pass_at_1.png        pass@1 per run with 95% CIs, against the base model's pass@8
    failure_types.png    share of problems per failure type, per run

McNemar's test compares runs problem by problem: the --compare pairs plus
every run against the base, with Holm's correction over all of them. Greedy decoding on vLLM
flips about 2% of problems between identical runs, so treat smaller
differences as noise whatever the p-value.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slm.analysis import bootstrap, holm, load_run, mcnemar, passed_by_task, status_shares  # noqa: E402
from slm.config import load_config  # noqa: E402
from slm.sandbox import STATUSES  # noqa: E402

TRAINED_ON = {"A": "nothing (base model)", "B": "own passing solutions",
              "C": "own solutions, unfiltered", "D": "human solutions"}
FAILURES = [s for s in STATUSES if s != "pass"]


def label(run: str) -> str:
    """B_round1 -> 'B (round 1)'."""
    name, _, rnd = run.partition("_round")
    return f"{name} (round {rnd})" if rnd else name


def parse_pair(text: str) -> tuple[str, str]:
    a, sep, b = text.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected FIRST:SECOND, got {text!r}")
    return a, b


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def table_rows(a: dict) -> list[dict]:
    base = a["base"]
    rows = []
    for run in a["runs"]:
        boot = a["bootstrap"]["pass@1"][run]
        test = a["mcnemar"].get(f"{base}:{run}")
        diff = a["bootstrap"]["difference"].get(f"{run} - {base}")
        rows.append({
            "run": label(run),
            "trained_on": TRAINED_ON.get(run.split("_")[0], ""),
            "pass1": f"{pct(boot['estimate'])} [{pct(boot['ci95'][0])}, {pct(boot['ci95'][1])}]",
            "delta": "" if diff is None else
                     f"{100 * diff['estimate']:+.1f} [{100 * diff['ci95'][0]:+.1f}, {100 * diff['ci95'][1]:+.1f}]",
            "p": "" if test is None else fmt_p(test["p_value_holm"]),
            "pass8": pct(a["pass@k"].get(run)) if run in a["pass@k"] else "",
            "wrong": pct(a["status_shares"][run]["wrong_answer"]),
            "runtime": pct(a["status_shares"][run]["runtime"]),
            "other": pct(sum(a["status_shares"][run][s] for s in ("syntax", "timeout", "no_code"))),
        })
    return rows


def write_tables(a: dict, out: Path) -> None:
    rows = table_rows(a)
    head = ["Run", "Trained on", "pass@1 [95% CI]", f"vs {a['base']}", "McNemar p (Holm)",
            "pass@8", "Wrong answer", "Runtime error", "Other"]
    keys = ["run", "trained_on", "pass1", "delta", "p", "pass8", "wrong", "runtime", "other"]
    md = [f"MBPP {a['split']} ({a['n_problems']} problems), greedy decoding. Percentages; "
          f"failure columns are shares of all problems. Differences under about 2 points "
          f"are within run-to-run noise.", "",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(r[k] for k in keys) + " |" for r in rows]
    (out / "results_table.md").write_text("\n".join(md) + "\n")

    tex = [r"\begin{table}[t]", r"\caption{Greedy pass@1 on MBPP " + a["split"] + f" ({a['n_problems']} problems), "
           r"with 95\% bootstrap intervals. $\Delta$ is the change from run " + a["base"] +
           r"; $p$ is McNemar's exact test, Holm-corrected. Failure columns are shares of all problems.}",
           r"\label{tab:results}", r"\centering\small", r"\begin{tabular}{llccccc}", r"\toprule",
           r"Run & Trained on & pass@1 & $\Delta$ & $p$ & Wrong & Runtime \\", r"\midrule"]
    for r in rows:
        cells = [r["run"], r["trained_on"], r["pass1"], r["delta"], r["p"], r["wrong"], r["runtime"]]
        tex.append(" & ".join(c.replace("%", r"\%").replace("<", "$<$") for c in cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "results_table.tex").write_text("\n".join(tex) + "\n")


def write_figures(a: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = a["runs"]
    names = [label(r) for r in runs]

    # pass@1 per run with 95% CIs; the base model's pass@k as the bar to reach (task 29).
    fig, ax = plt.subplots(figsize=(5, 3.2))
    est = [100 * a["bootstrap"]["pass@1"][r]["estimate"] for r in runs]
    lo = [e - 100 * a["bootstrap"]["pass@1"][r]["ci95"][0] for e, r in zip(est, runs)]
    hi = [100 * a["bootstrap"]["pass@1"][r]["ci95"][1] - e for e, r in zip(est, runs)]
    ax.bar(names, est, yerr=[lo, hi], capsize=4, color="#4C72B0")
    for i, e in enumerate(est):
        ax.text(i, 1.5, f"{e:.1f}", ha="center", color="white", fontsize=9)
    base_k = {k: v for k, v in a["pass@k"].items() if k == a["base"]}
    if base_k:
        ax.axhline(100 * base_k[a["base"]], ls="--", color="#C44E52",
                   label=f"{a['base']} pass@{a['k']} ({100 * base_k[a['base']]:.1f})")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_ylabel(f"pass@1 on MBPP {a['split']} (%)")
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "pass_at_1.png", dpi=200)
    plt.close(fig)

    # Failure types per run (task 28): stacked shares of all problems.
    colors = {"pass": "#55A868", "wrong_answer": "#C44E52", "runtime": "#DD8452",
              "syntax": "#8172B3", "timeout": "#937860", "no_code": "#8C8C8C"}
    fig, ax = plt.subplots(figsize=(5.5, 0.55 * len(runs) + 1.2))
    left = [0.0] * len(runs)
    for s in STATUSES:
        vals = [100 * a["status_shares"][r][s] for r in runs]
        ax.barh(names, vals, left=left, color=colors[s], label=s.replace("_", " "))
        left = [l + v for l, v in zip(left, vals)]
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"share of MBPP {a['split']} problems (%)")
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "failure_types.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--output-dir", default=None, help="Where the run results are (default: output_dir).")
    ap.add_argument("--runs", nargs="+", default=["A", "B_round1", "C_round1", "D"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--base", default="A", help="Baseline run.")
    ap.add_argument("--reference", default="D", help="Run whose gain the others are measured against.")
    ap.add_argument("--compare", nargs="+", type=parse_pair,
                    default=[("A", "B_round1"), ("A", "D"), ("B_round1", "C_round1")],
                    help="McNemar comparisons as FIRST:SECOND.")
    ap.add_argument("--k", type=int, default=8, help="pass@k from each run's sample<k> results, if any.")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.output_dir)
    root = Path(cfg["output_dir"])

    loaded, missing = {}, []
    for run in args.runs:
        got = load_run(root, run, args.split)
        if got is None:
            missing.append(run)
        else:
            loaded[run] = got
    if missing:
        print(f"Not run yet, skipped: {', '.join(missing)}")
    if args.base not in loaded:
        sys.exit(f"No greedy {args.split} results for the base run {args.base} in {root}.")
    runs = list(loaded)

    passed = {r: passed_by_task(recs) for r, (_, recs) in loaded.items()}
    task_sets = {r: frozenset(p) for r, p in passed.items()}
    if len(set(task_sets.values())) > 1:
        sizes = ", ".join(f"{r}={len(t)}" for r, t in task_sets.items())
        sys.exit(f"Runs were scored on different problems ({sizes}); rerun them on the same split.")

    # The planned comparisons, plus every run against the base (for the table).
    compare = [(x, y) for x, y in args.compare if x in loaded and y in loaded]
    compare += [(args.base, r) for r in runs if r != args.base and (args.base, r) not in compare]
    tests = [mcnemar(passed[x], passed[y]) for x, y in compare]
    for t, p in zip(tests, holm([t["p_value"] for t in tests])):
        t["p_value_holm"] = p

    recover = [(args.base, r, args.reference) for r in runs
               if args.reference in loaded and r not in (args.base, args.reference)]

    pass_k = {}
    for run in runs:
        got = load_run(root, run, args.split, f"sample{args.k}")
        if got is not None and f"pass@{args.k}" in got[0]:
            pass_k[run] = got[0][f"pass@{args.k}"]

    analysis = {
        "split": args.split,
        "n_problems": len(next(iter(task_sets.values()))),
        "base": args.base,
        "reference": args.reference,
        "runs": runs,
        "missing_runs": missing,
        "k": args.k,
        "pass@k": pass_k,
        "status_shares": {r: status_shares(recs) for r, (_, recs) in loaded.items()},
        "mcnemar": {f"{x}:{y}": t for (x, y), t in zip(compare, tests)},
        "bootstrap": bootstrap(passed, compare, recover, args.n_boot, cfg["seed"]),
    }
    out = root / "analysis" / args.split
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    write_tables(analysis, out)
    if not args.no_figures:
        try:
            write_figures(analysis, out)
        except ImportError:
            print("matplotlib is not installed; skipped the figures (pip install matplotlib).")

    print((out / "results_table.md").read_text())
    for key, t in analysis["mcnemar"].items():
        x, y = key.split(":")
        print(f"McNemar {x} vs {y}: only {x} {t['only_first']}, only {y} {t['only_second']}, "
              f"p = {fmt_p(t['p_value'])} (Holm {fmt_p(t['p_value_holm'])})")
    for key, g in analysis["bootstrap"]["gain_recovered"].items():
        interval = "" if g["ci95"] is None else f" [{pct(g['ci95'][0])}, {pct(g['ci95'][1])}]"
        print(f"Gain recovered, {key}: {pct(g['estimate'])}%{interval}")
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
