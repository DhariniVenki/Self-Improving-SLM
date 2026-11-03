"""Pick and report the self-training fixes (notebooks/self_training_fixes_colab.ipynb).

    # After the candidates are scored on heldout: choose one, without looking at test
    python scripts/fixes_report.py select

    # After the repeated runs: compare the chosen fix, Run B and the base model on test
    python scripts/fixes_report.py report

Writes to <output_dir>/fixes/: selection.json, report.json, report.md. The
chosen fix is the candidate with the best heldout pass@1; ties go to the
earlier one in configs/default.yaml. Scores on heldout are biased upward by
this choice, so test is the split to quote.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slm.analysis import group_difference, holm, load_run, mcnemar, passed_by_task  # noqa: E402
from slm.config import load_config  # noqa: E402
from slm.selftrain import fix_candidates  # noqa: E402


def repeats(base_run: str, seeds: list[int], name: str = "seed") -> list[str]:
    return [base_run] + [f"{base_run}_{name}{s}" for s in seeds]


def select(cfg: dict, out: Path) -> None:
    root = cfg["output_dir"]
    rows = []
    for c in fix_candidates(cfg):
        got = load_run(root, c["run"], "heldout")
        if got is None:
            sys.exit(f"{c['run']} has no heldout results yet.")
        rows.append({**c, "heldout_pass@1": got[0]["pass@1"]})
    best = max(rows, key=lambda r: r["heldout_pass@1"])  # max keeps the first of a tie
    context = {r: load_run(root, r, "heldout")[0]["pass@1"] for r in ("A", "B_round1")}
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps(
        {"chosen": best, "candidates": rows, "context": context}, indent=2))
    for r in rows:
        mark = "  <- chosen" if r is best else ""
        print(f"{r['run']:28s} {100 * r['heldout_pass@1']:5.1f}{mark}")
    print(f"for context: A {100 * context['A']:.1f}, B_round1 {100 * context['B_round1']:.1f}")


def load_group(root: str, runs: list[str], split: str) -> tuple[list[float], list[dict[int, bool]]]:
    scores, passed = [], []
    for run in runs:
        got = load_run(root, run, split)
        if got is None:
            sys.exit(f"{run} has no {split} results yet.")
        scores.append(got[0]["pass@1"])
        passed.append(passed_by_task(got[1]))
    return scores, passed


def report(cfg: dict, out: Path) -> None:
    root = cfg["output_dir"]
    chosen = json.loads((out / "selection.json").read_text())["chosen"]["run"]
    seeds = cfg["fixes"]["repeat_seeds"]
    groups = {"A": repeats("A", seeds, "repeat"), "B_round1": repeats("B_round1", seeds), chosen: repeats(chosen, seeds)}

    result = {"chosen": chosen, "groups": groups, "splits": {}}
    lines = [f"Chosen fix: {chosen} (best heldout score of {len(fix_candidates(cfg))} candidates). "
             "Each row averages 3 repeated runs. Test is the split to quote: "
             "heldout was used to choose the fix.", ""]
    for split in ("test", "heldout"):
        data = {g: load_group(root, runs, split) for g, runs in groups.items()}
        lines += [f"### {split}", "", "| Group | Repeats | Mean pass@1 |", "|---|---|---|"]
        for g, (scores, _) in data.items():
            each = ", ".join(f"{100 * s:.1f}" for s in scores)
            lines.append(f"| {g} | {each} | {100 * sum(scores) / len(scores):.1f} |")
        lines.append("")

        pairs = [("A", chosen), ("B_round1", chosen), ("A", "B_round1")]
        tests = [mcnemar(data[a][1][i], data[b][1][i]) for a, b in pairs for i in range(len(seeds) + 1)]
        adjusted = holm([t["p_value"] for t in tests])
        for t, p in zip(tests, adjusted):
            t["p_value_holm"] = p
        rows = []
        per_pair = len(seeds) + 1
        for j, (a, b) in enumerate(pairs):
            diff = group_difference(data[a][1], data[b][1])
            pair_tests = tests[j * per_pair:(j + 1) * per_pair]
            rows.append({"first": a, "second": b, **diff, "mcnemar_by_repeat": pair_tests})
            ps = ", ".join(f"{t['p_value_holm']:.3f}" for t in pair_tests)
            lines.append(f"- {b} minus {a}: {100 * diff['difference']:+.1f} points "
                         f"[{100 * diff['ci95'][0]:+.1f}, {100 * diff['ci95'][1]:+.1f}]; "
                         f"McNemar p per repeat (Holm): {ps}")
        lines.append("")
        result["splits"][split] = {"scores": {g: d[0] for g, d in data.items()}, "comparisons": rows}

    (out / "report.json").write_text(json.dumps(result, indent=2))
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "report"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.output_dir)
    out = Path(cfg["output_dir"]) / "fixes"
    select(cfg, out) if args.step == "select" else report(cfg, out)


if __name__ == "__main__":
    main()
