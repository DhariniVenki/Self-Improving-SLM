import base64
import json
import os
import pickle
import unittest
import zlib

from slm.livecodebench import build_messages, decode_tests, run_problem

STDIN_TESTS = [{"input": "1 2\n", "output": "3\n"}, {"input": "10 -4\n", "output": "6"}]
FUNC_TESTS = [{"input": "[1, 2, 3]\n2", "output": "[3, 4, 5]"}, {"input": "[]\n5", "output": "[]"}]
FUNC_CODE = ("class Solution:\n"
             "    def addAll(self, nums: List[int], k: int) -> List[int]:\n"
             "        return [x + k for x in nums]\n")


class StdinTest(unittest.TestCase):
    def test_pass_with_input(self):
        r = run_problem("a, b = map(int, input().split())\nprint(a + b)\n", "stdin", STDIN_TESTS)
        self.assertEqual(r["status"], "pass", r)
        self.assertEqual(r["tests"], ["pass", "pass"])

    def test_pass_with_sys_stdin_and_main_guard(self):
        code = ("import sys\n"
                "def main():\n"
                "    a, b = map(int, sys.stdin.read().split())\n"
                "    sys.stdout.write(str(a + b) + '\\n')\n"
                "if __name__ == '__main__':\n"
                "    main()\n")
        self.assertEqual(run_problem(code, "stdin", STDIN_TESTS)["status"], "pass")

    def test_pass_with_open0_and_buffer(self):
        code = "a, b = map(int, open(0).read().split())\nimport sys\nsys.stdout.buffer.write(str(a+b).encode())\n"
        self.assertEqual(run_problem(code, "stdin", STDIN_TESTS)["status"], "pass")

    def test_sys_exit_is_not_an_error(self):
        code = "import sys\na, b = map(int, input().split())\nprint(a + b)\nsys.exit(0)\n"
        self.assertEqual(run_problem(code, "stdin", STDIN_TESTS)["status"], "pass")

    def test_whitespace_and_number_format(self):
        tests = [{"input": "", "output": "1 2\n0.5\n"}]
        self.assertEqual(run_problem("print('1  2 ')\nprint('0.50')\n", "stdin", tests)["status"], "pass")

    def test_wrong_answer_stops_at_first_failure(self):
        r = run_problem("print(3)\n", "stdin", STDIN_TESTS)
        self.assertEqual(r["status"], "wrong_answer")
        self.assertEqual(r["tests"], ["pass", "wrong_answer"])

    def test_runtime(self):
        r = run_problem("print(1 // 0)\n", "stdin", STDIN_TESTS)
        self.assertEqual(r["status"], "runtime")
        self.assertIn("ZeroDivisionError", r["error"])

    def test_syntax(self):
        self.assertEqual(run_problem("print(\n", "stdin", STDIN_TESTS)["status"], "syntax")

    def test_no_code(self):
        self.assertEqual(run_problem("  \n", "stdin", STDIN_TESTS)["status"], "no_code")

    @unittest.skipUnless(os.name == "posix", "per-test time limits need Linux")
    def test_timeout_even_if_caught(self):
        code = "try:\n    while True: pass\nexcept Exception:\n    print(3)\n"
        r = run_problem(code, "stdin", STDIN_TESTS, timeout_s=1)
        self.assertEqual(r["status"], "timeout")


class FunctionalTest(unittest.TestCase):
    def test_pass_uses_typing_without_import(self):
        r = run_problem(FUNC_CODE, "functional", FUNC_TESTS, "addAll")
        self.assertEqual(r["status"], "pass", r)

    def test_tuple_result_matches_list(self):
        code = "class Solution:\n    def addAll(self, nums, k):\n        return tuple(x + k for x in nums)\n"
        self.assertEqual(run_problem(code, "functional", FUNC_TESTS, "addAll")["status"], "pass")

    def test_wrong_answer(self):
        code = "class Solution:\n    def addAll(self, nums, k):\n        return nums\n"
        self.assertEqual(run_problem(code, "functional", FUNC_TESTS, "addAll")["status"], "wrong_answer")

    def test_missing_method(self):
        code = "class Solution:\n    def other(self):\n        pass\n"
        r = run_problem(code, "functional", FUNC_TESTS, "addAll")
        self.assertEqual(r["status"], "runtime")
        self.assertIn("AttributeError", r["error"])


class DataTest(unittest.TestCase):
    def test_decode_plain_and_compressed(self):
        tests = [{"input": "1", "output": "2", "testtype": "stdin"}]
        self.assertEqual(decode_tests(json.dumps(tests)), tests)
        raw = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()
        self.assertEqual(decode_tests(raw), tests)

    def test_decode_refuses_objects(self):
        raw = base64.b64encode(zlib.compress(pickle.dumps(os.getcwd))).decode()
        with self.assertRaises(pickle.UnpicklingError):
            decode_tests(raw)

    def test_prompt_shows_starter_code_or_stdin_format(self):
        func = build_messages({"kind": "functional", "question": "Q", "starter_code": "class Solution:\n    pass"})
        self.assertIn("class Solution", func[1]["content"])
        stdin = build_messages({"kind": "stdin", "question": "Q", "starter_code": ""})
        self.assertIn("stdin", stdin[1]["content"])


if __name__ == "__main__":
    unittest.main()
