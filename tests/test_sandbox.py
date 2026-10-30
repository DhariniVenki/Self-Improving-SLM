import unittest

from slm.sandbox import run_tests

ADD = "def add(a, b):\n    return a + b\n"
TESTS = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"]


class SandboxTest(unittest.TestCase):
    def test_pass(self):
        r = run_tests(ADD, TESTS)
        self.assertEqual(r["status"], "pass")
        self.assertTrue(r["passed"])
        self.assertEqual(r["tests"], ["pass", "pass", "pass"])

    def test_wrong_answer_records_each_test(self):
        r = run_tests("def add(a, b):\n    return 3\n", TESTS)
        self.assertEqual(r["status"], "wrong_answer")
        self.assertEqual(r["tests"], ["pass", "wrong_answer", "wrong_answer"])

    def test_syntax(self):
        r = run_tests("def add(a, b)\n    return a + b\n", TESTS)
        self.assertEqual(r["status"], "syntax")

    def test_runtime_at_definition(self):
        r = run_tests("import not_a_real_module\n" + ADD, TESTS)
        self.assertEqual(r["status"], "runtime")
        self.assertIn("ModuleNotFoundError", r["error"])

    def test_runtime_in_test(self):
        r = run_tests("def add(a, b):\n    return a + undefined_name\n", TESTS)
        self.assertEqual(r["status"], "runtime")
        self.assertEqual(r["tests"], ["runtime"] * 3)

    def test_timeout(self):
        r = run_tests("def add(a, b):\n    while True:\n        pass\n", TESTS, timeout_s=1)
        self.assertEqual(r["status"], "timeout")

    def test_no_code(self):
        self.assertEqual(run_tests("  \n", TESTS)["status"], "no_code")

    def test_main_block_and_input_are_skipped(self):
        code = ADD + "if __name__ == '__main__':\n    print(add(int(input()), 1))\n"
        self.assertEqual(run_tests(code, TESTS)["status"], "pass")

    def test_own_failing_assert_is_wrong_answer(self):
        r = run_tests("def add(a, b):\n    return 3\nassert add(0, 0) == 0\n", TESTS)
        self.assertEqual(r["status"], "wrong_answer")

    def test_sys_exit_is_runtime(self):
        r = run_tests("import sys\nsys.exit(0)\n" + ADD, TESTS)
        self.assertEqual(r["status"], "runtime")

    def test_setup_code_can_use_solution_classes(self):
        code = "class Node:\n    def __init__(self, v):\n        self.v = v\n\ndef val(n):\n    return n.v\n"
        r = run_tests(code, ["assert val(root) == 1"], setup="root = Node(1)")
        self.assertEqual(r["status"], "pass")

    def test_sleep_hits_wall_clock_backstop(self):
        r = run_tests("import time\ndef add(a, b):\n    time.sleep(60)\n", TESTS, timeout_s=1)
        self.assertEqual(r["status"], "timeout")


if __name__ == "__main__":
    unittest.main()
