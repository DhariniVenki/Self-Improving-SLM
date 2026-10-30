# Self-Improving-SLM

STAT 453 project, Group 17. Can Qwen2.5-Coder-0.5B-Instruct get better at
Python by training only on its own solutions that pass unit tests?

| Run | Trained on | Purpose |
|---|---|---|
| A | nothing | baseline |
| B | its own solutions that **pass** the tests | the main idea |
| C | its own solutions, pass or fail | does the test filter matter? |
| D | human-written MBPP solutions | standard fine-tuning, for comparison |

All runs are scored the same way: greedy decoding on the 500 MBPP test problems.

## Layout

```
configs/default.yaml   every setting (model, splits, sampling, sandbox limits)
data/splits.json       fixed problem ids: train 314, heldout 150, test 500
slm/data.py            MBPP loading and splits
slm/prompts.py         prompt template and code extraction (fixed after Run A)
slm/sandbox.py         runs a program against its tests, labels the failure type
slm/generate.py        generation with vLLM (GPU) or Hugging Face
slm/evaluate.py        generate, run, summarize, save per-problem results
slm/selftrain.py       training sets: filter, deduplicate, cap per problem
slm/train.py           LoRA fine-tuning from the base model, merged for vLLM
scripts/               command-line entry points (below)
results/               run outputs, committed for later analysis
notebooks/             Colab notebooks: Run A; Round 0 and Run D
tests/                 unit tests for the sandbox and extraction
```

## Quick start (Colab)

Open `notebooks/run_a_colab.ipynb` in Colab with a GPU runtime. It mounts
Drive, installs dependencies, checks the model and runs Run A. Results are
written to Drive while running (so a disconnect loses nothing), and the last
cell copies them into `results/` and pushes them to GitHub. That cell needs
a `GITHUB_TOKEN` secret in Colab's Secrets panel.

## Commands

```bash
pip install -r requirements.txt

# Unit tests (no GPU or model needed)
python -m unittest discover tests

# Task 3: make the splits (already committed in data/splits.json)
python scripts/make_splits.py --check-references

# Task 2: check the model loads and writes code (10 MBPP 'prompt' problems)
python scripts/check_model.py

# Task 9: Run A on the test and heldout splits, plus pass@8 (finding 5)
python scripts/evaluate.py --run A --splits test heldout --pass-at-k 8 \
    --output-dir /content/drive/MyDrive/self-improving-slm

# Tasks 10-13: 8 samples per training problem from the base model
python scripts/evaluate.py --run A --splits train --pass-at-k 8

# Task 14 and 20-21: training sets for Runs B and C round 1, and Run D
python scripts/build_dataset.py --run B_round1 --source A/train/sample8 --filter pass
python scripts/build_dataset.py --run C_round1 --source A/train/sample8 --filter none
python scripts/build_dataset.py --run D --source reference

# Tasks 16-19: LoRA fine-tuning, then evaluate the merged model
python scripts/train.py --run D --model-dir /content/models
python scripts/evaluate.py --run D --model /content/models/D/merged --splits test heldout

# Diagnostic: where the gap to the published MBPP score (52.4%) comes from
python scripts/compare_published.py
```

Results land in `results/<run>/<split>/<greedy|sample8>/` and are committed
to the repo: `summary.json` (pass@1, failure counts) and `records.jsonl`
(one line per sample with code, status and per-test results, for McNemar's
test later). `--output-dir` writes somewhere else instead, such as Drive.

## Protocol decisions

- **Training data format:** the same prompt as evaluation, and the code in a
  single ```` ```python ```` block as the response. Loss is on the response
  (and the end-of-turn token) only. Every run trains from the base model.
- **Deduplication:** two solutions are duplicates if they are the same
  program apart from comments and formatting (`ast.unparse`). The first 4
  distinct ones (in sample order) are kept per problem. Run C keeps failing
  solutions too, but not empty ones.
- **Hyperparameters** are tuned once, with Run D on heldout, and reused for
  B and C so the runs differ only in their training data.

- **Prompt shows 1 test; the sandbox checks all 3.** `passed_shown` in each
  record says whether the shown test passed, and `shown_pass_hidden_fail`
  in the summary counts solutions that pass the shown test but fail a
  hidden one.
- **Extraction:** the first ```` ```python ```` block, else the text after an
  unclosed fence, else the whole completion. Top-level `assert` lines are
  removed, because the model often copies the prompt's test under its
  function and only our tests should decide pass/fail.
- **Failure types:** `pass`, `no_code`, `syntax`, `runtime`, `wrong_answer`,
  `timeout`.
- **Sandbox:** one subprocess per program in a temporary folder, a 5-second
  CPU-time limit (wall-clock backstop 17 s), 1 GB memory and 10 MB file
  limits. `if __name__ == "__main__":` blocks don't run. MBPP's
  `test_setup_code` runs after the solution, since it uses the solution's
  classes. The isolation boundary is the throwaway Colab VM.
- **Splits:** train+validation (464) → 314 train / 150 heldout (seed 0).
  The heldout set was 60 at first, but that gives a ±12.6-point 95%
  confidence interval, too wide to choose hyperparameters with; 150 gives
  about ±8. The 500 test problems are only for evaluation. The 10 `prompt`
  problems are only for smoke tests. Each result stores a fingerprint of
  its split, so results from an older split are redone, not reused.
- **Missing imports in tests:** task 596's tests use `sys` without
  importing it; any standard-library module a test uses without importing
  is added to its setup code. No hidden test calls a helper that the shown
  test doesn't reveal.
- **Reference solutions:** 961/964 pass their own tests. The failures are
  listed in `data/reference_failures.json`; drop them from Run D's data.
- **Setup code is shown in the prompt** (prompt version 2). Three problems
  (367 in test, 596 and 927 in train) build test inputs in setup code,
  e.g. a tree made of `Node` objects; without seeing it the model can't
  know the interface. Results store `prompt_version` and older ones are redone.
- **Greedy decoding on vLLM is not perfectly repeatable.** Two identical
  greedy runs on the test set gave the same completion for 382/500
  problems and flipped pass/fail on 9 (1.8%). Treat differences under about
  2 points between runs as noise, use McNemar's test on per-problem
  results, and for final numbers average a few repeated evaluations.
- **Published scores:** the Qwen2.5-Coder report gives 52.4% on MBPP for
  this model, measured with EvalPlus on cleaned-up problems. Our protocol
  is stricter (original problems and tests, one test shown).
  `scripts/compare_published.py` measures how much each difference
  matters; results are in `results/A_compare/`.
