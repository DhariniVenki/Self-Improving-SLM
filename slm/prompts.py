"""The fixed prompt template and code extraction rule.

Do not change these after Run A: every run must be evaluated with the same
prompt and extraction, or the comparisons between runs stop being fair.
"""
from __future__ import annotations

import ast
import re

SYSTEM_PROMPT = "You are an expert Python programmer."

USER_TEMPLATE = """{text}
Your code should pass this test:

{tests}

Write the complete function in a single ```python code block. Do not include test cases or example usage."""

_FENCED = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.DOTALL)
_OPEN_FENCE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*)", re.DOTALL)


def build_messages(problem: dict, n_shown_tests: int = 1) -> list[dict]:
    """Chat messages for one MBPP problem, showing the first n_shown_tests asserts."""
    tests = "\n".join(problem["test_list"][:n_shown_tests])
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=problem["text"], tests=tests)},
    ]


def extract_code(completion: str) -> str:
    """Pull the program out of a model completion.

    1. The first ```python ... ``` block.
    2. Otherwise, everything after an opening fence that was never closed
       (the model ran out of tokens).
    3. Otherwise, the whole completion, which is often bare code.

    Then drop top-level asserts (see strip_top_level_asserts).
    """
    m = _FENCED.search(completion) or _OPEN_FENCE.search(completion)
    code = m.group(1) if m else completion
    return strip_top_level_asserts(code.strip("\n"))


def strip_top_level_asserts(code: str) -> str:
    """Remove module-level `assert` statements.

    The model often copies the test from its prompt below the function
    despite being told not to. Only our tests should decide pass/fail, and a
    copied assert would otherwise fail while the program loads and show up
    as a runtime error instead of a wrong answer. Code that doesn't parse
    is returned unchanged (it will be scored as a syntax error).
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code
    drop = set()
    for node in tree.body:
        if isinstance(node, ast.Assert):
            drop.update(range(node.lineno, node.end_lineno + 1))
    if not drop:
        return code
    lines = code.split("\n")
    return "\n".join(l for i, l in enumerate(lines, start=1) if i not in drop).rstrip("\n")
