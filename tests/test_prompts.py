import unittest

from slm.prompts import build_messages, extract_code

PROBLEM = {
    "text": "Write a function to add two numbers.",
    "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"],
}


class PromptTest(unittest.TestCase):
    def test_only_shown_tests_in_prompt(self):
        user = build_messages(PROBLEM, n_shown_tests=1)[1]["content"]
        self.assertIn("assert add(1, 2) == 3", user)
        self.assertNotIn("assert add(0, 0) == 0", user)

    def test_extract_fenced(self):
        out = "Here you go:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
        self.assertEqual(extract_code(out), "def add(a, b):\n    return a + b")

    def test_extract_first_block_only(self):
        out = "```python\ndef add(a, b):\n    return a + b\n```\n```python\nprint(add(1, 2))\n```"
        self.assertEqual(extract_code(out), "def add(a, b):\n    return a + b")

    def test_extract_unclosed_fence(self):
        out = "```python\ndef add(a, b):\n    return a +"
        self.assertEqual(extract_code(out), "def add(a, b):\n    return a +")

    def test_copied_test_asserts_are_removed(self):
        out = ("```python\ndef add(a, b):\n    assert isinstance(a, int)\n    return a + b\n\n"
               "# Test\nassert add(1, 2) == 3\nassert (\n    add(0, 0) == 0\n)\n```")
        code = extract_code(out)
        self.assertNotIn("add(1, 2)", code)
        self.assertNotIn("add(0, 0)", code)
        self.assertIn("assert isinstance(a, int)", code)  # asserts inside the function stay

    def test_unparseable_code_left_alone(self):
        self.assertEqual(extract_code("def add(a, b)\nassert add(1, 2) == 3"), "def add(a, b)\nassert add(1, 2) == 3")

    def test_extract_bare_code(self):
        self.assertEqual(extract_code("def add(a, b):\n    return a + b\n"), "def add(a, b):\n    return a + b")


if __name__ == "__main__":
    unittest.main()
