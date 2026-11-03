import unittest

from slm.selftrain import (distinct_counts, examples_from_references, fix_candidates, format_response, normalize,
                           per_problem_caps, select_samples, stats)


def rec(task_id, idx, code, passed=True, status=None):
    return {"task_id": task_id, "sample_idx": idx, "code": code, "passed": passed,
            "status": status or ("pass" if passed else "wrong_answer")}


MAX = "def f(xs):\n    return max(xs)"
MAX_COMMENTED = "def f(xs):\n    # biggest\n    return max( xs )"
SORTED = "def f(xs):\n    return sorted(xs)[-1]"
FIRST = "def f(xs):\n    return xs[0]"


class SelfTrainTest(unittest.TestCase):
    def test_normalize_ignores_comments_and_spacing(self):
        self.assertEqual(normalize(MAX), normalize(MAX_COMMENTED))
        self.assertNotEqual(normalize(MAX), normalize(SORTED))

    def test_normalize_unparseable_code(self):
        self.assertEqual(normalize("def f(:\n  x  \n"), "def f(:\n  x")

    def test_filter_and_dedup(self):
        records = [rec(1, 0, MAX), rec(1, 1, MAX_COMMENTED), rec(1, 2, FIRST, passed=False), rec(1, 3, SORTED)]
        chosen = select_samples(records, only_passing=True, max_per_problem=4)
        self.assertEqual([r["sample_idx"] for r in chosen], [0, 3])
        unfiltered = select_samples(records, only_passing=False, max_per_problem=4)
        self.assertEqual([r["sample_idx"] for r in unfiltered], [0, 2, 3])

    def test_cap_keeps_first_samples(self):
        codes = [f"def f():\n    return {i}" for i in range(6)]
        records = [rec(1, i, c) for i, c in enumerate(codes)]
        chosen = select_samples(records, only_passing=True, max_per_problem=4)
        self.assertEqual([r["sample_idx"] for r in chosen], [0, 1, 2, 3])

    def test_caps_favor_problems_greedy_fails(self):
        greedy = [rec(1, 0, MAX), rec(2, 0, FIRST, passed=False)]
        self.assertEqual(per_problem_caps(greedy, 4, 1), {1: 1, 2: 4})
        records = [rec(t, i, f"def f():\n    return {i}") for t in (1, 2) for i in range(3)]
        for solved_cap, expected in [(0, [2, 2, 2]), (1, [1, 2, 2, 2])]:
            chosen = select_samples(records, True, 4, per_problem_caps(greedy, 4, solved_cap))
            self.assertEqual([r["task_id"] for r in chosen], expected)

    def test_fix_candidates(self):
        cfg = {"fixes": {"datasets": {"X": 4, "Y": 0}, "learning_rates": [5e-4, 1e-4]}}
        cands = fix_candidates(cfg)
        self.assertEqual([c["run"] for c in cands], ["X_lr0.0005", "X_lr0.0001", "Y_lr0.0005", "Y_lr0.0001"])
        self.assertEqual(cands[2]["solved_cap"], 0)

    def test_empty_code_never_kept(self):
        records = [rec(1, 0, "", passed=False, status="no_code")]
        self.assertEqual(select_samples(records, only_passing=False, max_per_problem=4), [])

    def test_distinct_counts_include_unsolved_problems(self):
        records = [rec(1, 0, MAX), rec(1, 1, MAX_COMMENTED), rec(2, 0, FIRST, passed=False)]
        self.assertEqual(distinct_counts(records, only_passing=True), {1: 1, 2: 0})

    def test_reference_examples(self):
        problems = [{"task_id": 7, "text": "Add.", "code": "def add(a, b):\r\n  return a + b\r\n",
                     "test_list": ["assert add(1, 2) == 3"], "test_setup_code": ""},
                    {"task_id": 8, "text": "x", "code": "", "test_list": ["assert 1"], "test_setup_code": ""}]
        ex = examples_from_references(problems, n_shown_tests=1, exclude={8})
        self.assertEqual(len(ex), 1)
        self.assertTrue(ex[0]["response"].startswith("```python\ndef add"))
        self.assertTrue(ex[0]["response"].endswith("\n```"))
        self.assertIn("assert add(1, 2) == 3", ex[0]["messages"][1]["content"])

    def test_stats(self):
        ex = [{"task_id": 1, "status": "pass"}, {"task_id": 1, "status": "wrong_answer"}]
        s = stats(ex, n_problems=3)
        self.assertEqual(s["problems_without_examples"], 2)
        self.assertEqual(s["examples_per_problem"], {2: 1})
        self.assertEqual(s["share_not_passing"], 0.5)

    def test_format_response(self):
        self.assertEqual(format_response("\nx = 1\n\n"), "```python\nx = 1\n```")


if __name__ == "__main__":
    unittest.main()
