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
data/splits.json       fixed problem ids: train 404, heldout 60, test 500
slm/data.py            MBPP loading and splits
slm/prompts.py         prompt template and code extraction (fixed after Run A)
slm/sandbox.py         runs a program against its tests, labels the failure type
slm/generate.py        generation with vLLM (GPU) or Hugging Face
slm/evaluate.py        generate, run, summarize, save per-problem results
scripts/               command-line entry points (below)
notebooks/             Colab notebook for Run A
tests/                 unit tests for the sandbox and extraction
```

## Quick start (Colab)

Open `notebooks/run_a_colab.ipynb` in Colab with a GPU runtime. It mounts
Drive, installs dependencies, checks the model and runs Run A, saving
results to Drive.

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
```

Results land in `<output_dir>/<run>/<split>/<greedy|sample8>/`:
`summary.json` (pass@1, failure counts) and `records.jsonl` (one line per
sample with code, status and per-test results, for McNemar's test later).

## Protocol decisions

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
- **Splits:** train+validation (464) → 404 train / 60 heldout (seed 0).
  The 500 test problems are only for evaluation. The 10 `prompt` problems
  are only for smoke tests.
- **Reference solutions:** 961/964 pass their own tests. The failures are
  listed in `data/reference_failures.json`; drop them from Run D's data.
