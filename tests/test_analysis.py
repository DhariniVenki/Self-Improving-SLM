import unittest

from slm.analysis import (bootstrap, ci, gain_recovered, holm, mcnemar, passed_by_task, sampling_stats,
                          status_shares)


def rec(task_id, passed, idx=0, status=None):
    return {"task_id": task_id, "sample_idx": idx, "passed": passed,
            "status": status or ("pass" if passed else "wrong_answer")}


class AnalysisTest(unittest.TestCase):
    def test_mcnemar_counts_and_exact_p(self):
        a = {1: True, 2: True, 3: False, 4: False, 5: True}
        b = {1: True, 2: False, 3: True, 4: True, 5: True}
        t = mcnemar(a, b)
        self.assertEqual((t["both_pass"], t["neither_pass"], t["only_first"], t["only_second"]), (2, 0, 1, 2))
        # Binomial(3, 1/2): P(X <= 1) = 4/8, two-sided 1.0.
        self.assertAlmostEqual(t["p_value"], 1.0)

    def test_mcnemar_one_sided_discordance(self):
        a = {i: False for i in range(10)}
        b = {i: True for i in range(10)}
        # Every discordant problem favours b: 2 * 0.5 ** 10.
        self.assertAlmostEqual(mcnemar(a, b)["p_value"], 2 / 1024)

    def test_mcnemar_no_discordance(self):
        self.assertEqual(mcnemar({1: True}, {1: True})["p_value"], 1.0)

    def test_holm(self):
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(holm([0.5, 0.9]), [1.0, 1.0])

    def test_gain_recovered(self):
        self.assertAlmostEqual(gain_recovered(0.3, 0.35, 0.4), 0.5)
        self.assertIsNone(gain_recovered(0.3, 0.35, 0.3))

    def test_greedy_takes_first_sample(self):
        records = [rec(2, False, idx=1), rec(2, True, idx=0), rec(1, False, status="runtime")]
        self.assertEqual(passed_by_task(records), {1: False, 2: True})
        shares = status_shares(records)
        self.assertEqual((shares["pass"], shares["runtime"], shares["wrong_answer"]), (0.5, 0.5, 0.0))

    def test_ci(self):
        lo, hi = ci(list(range(101)))
        self.assertAlmostEqual(lo, 2.5)
        self.assertAlmostEqual(hi, 97.5)

    def test_bootstrap_identical_runs(self):
        a = {i: i % 3 == 0 for i in range(60)}
        out = bootstrap({"A": a, "B": dict(a), "D": {i: True for i in range(60)}},
                        pairs=[("A", "B")], recover=[("A", "B", "D")], n_boot=200)
        self.assertEqual(out["difference"]["B - A"]["ci95"], [0.0, 0.0])
        self.assertAlmostEqual(out["pass@1"]["A"]["estimate"], 20 / 60)
        lo, hi = out["pass@1"]["A"]["ci95"]
        self.assertLess(lo, 20 / 60)
        self.assertGreater(hi, 20 / 60)
        self.assertEqual(out["gain_recovered"]["B vs D"]["estimate"], 0.0)

    def test_bootstrap_is_seeded(self):
        a = {i: i % 2 == 0 for i in range(30)}
        run = lambda: bootstrap({"A": a}, [], [], n_boot=50, seed=1)["pass@1"]["A"]["ci95"]
        self.assertEqual(run(), run())

    def test_sampling_stats(self):
        # Problem 1: both samples pass. Problem 2: one passes the shown test only. Problem 3: none.
        records = [dict(rec(1, True, i), passed_shown=True) for i in range(2)]
        records += [dict(rec(2, False, 0), passed_shown=True), dict(rec(2, False, 1), passed_shown=False)]
        records += [dict(rec(3, False, i), passed_shown=False) for i in range(2)]
        summary = {"n_problems": 3, "samples_per_problem": 2, "pass@1": 1 / 3, "pass@2": 1 / 3,
                   "problems_solved_at_least_once": 1, "shown_pass_hidden_fail": 1, "status_share": {}}
        s = sampling_stats(summary, records)
        self.assertEqual((s["never_solved"], s["always_solved"]), (2, 1))
        self.assertEqual(s["solve_count_histogram"], {0: 2, 1: 0, 2: 1})
        self.assertAlmostEqual(s["hidden_fail_share_of_shown_pass"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
