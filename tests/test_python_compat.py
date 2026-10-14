"""Every source file must parse on the oldest Python the project supports.

pyproject.toml declares >=3.10, but a developer on 3.12 or newer cannot see a
3.12-only construct: their own interpreter accepts it. CI's 3.10 job does, which
is how a nested f-string with backslashes in tasks/roles.py broke every CI run
for three weeks while the app kept working locally and on the 3.12 container.

- test_every_file_parses: on 3.10 and 3.11 an unsupported construct is a
  SyntaxError, so parsing every file is the whole check there.
- test_no_py312_only_fstring_fields: 3.12+ (PEP 701) parses those forms happily,
  so read the AST and flag the two the project actually risks: a backslash inside
  a replacement field, and a replacement field that reuses its f-string's own
  quote character. Skipped on older Pythons, whose parser already rejects them
  and whose AST positions inside f-strings are not reliable.
"""

import ast
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIRS = ("research_assistant", "tests", "scripts")


def _source_files():
    files = [os.path.join(ROOT, f) for f in os.listdir(ROOT) if f.endswith(".py")]
    for top in SOURCE_DIRS:
        for folder, _dirs, names in os.walk(os.path.join(ROOT, top)):
            files += [os.path.join(folder, n) for n in names if n.endswith(".py")]
    return sorted(files)


class TestPythonCompatibility(unittest.TestCase):
    def test_every_file_parses(self):
        for path in _source_files():
            with self.subTest(file=os.path.relpath(path, ROOT)):
                with open(path, encoding="utf-8") as fh:
                    ast.parse(fh.read(), filename=path)

    @unittest.skipIf(sys.version_info < (3, 12), "older parsers reject these forms themselves")
    def test_no_py312_only_fstring_fields(self):
        offenders = []
        for path in _source_files():
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
            for node in ast.walk(ast.parse(src)):
                if not isinstance(node, ast.JoinedStr):
                    continue
                opener = re.match(r"[A-Za-z]*(\"\"\"|'''|\"|')", ast.get_source_segment(src, node) or "")
                quote = opener.group(1) if opener else ""
                for part in node.values:
                    if not isinstance(part, ast.FormattedValue):
                        continue
                    field = ast.get_source_segment(src, part.value) or ""
                    if "\\" in field or (len(quote) == 1 and quote in field):
                        offenders.append(f"{os.path.relpath(path, ROOT)}:{node.lineno}  {{{field[:40]}}}")
        self.assertEqual(offenders, [], "f-string replacement fields that only Python 3.12+ accepts")


if __name__ == "__main__":
    unittest.main()
