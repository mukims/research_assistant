"""Static checks on the Streamlit front-end: app.py and research_assistant/ui/.

Streamlit tab bodies are long and mostly reached only by clicking, so a name that
is used but never defined can sit unnoticed until someone hits that branch (app.py
called `logger.warning` and `logger.error` in two error paths with no `logger`
anywhere). Moving code between modules is exactly how such a name goes missing,
so this reads every file's symbol tables and fails on any global that nothing
defines, imports or builtins provide. Stdlib only.
"""

import builtins
import symtable
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI_FILES = [ROOT / "app.py", *sorted((ROOT / "research_assistant" / "ui").glob("*.py"))]
MODULE_ATTRS = {"__file__", "__name__", "__doc__", "__package__", "__spec__", "__builtins__", "__loader__"}


def undefined_globals(source, filename="<src>"):
    """Names a module reads as globals that it neither binds, imports nor gets from builtins."""
    top = symtable.symtable(source, filename, "exec")
    bound = {
        s.get_name()
        for s in top.get_symbols()
        if s.is_assigned() or s.is_imported() or s.is_namespace()
    }
    known = bound | set(dir(builtins)) | MODULE_ATTRS
    missing = {s.get_name() for s in top.get_symbols() if s.is_referenced() and s.get_name() not in known}
    stack = list(top.get_children())
    while stack:
        table = stack.pop()
        missing |= {s.get_name() for s in table.get_symbols() if s.is_global() and s.is_referenced() and s.get_name() not in known}
        stack.extend(table.get_children())
    return missing


class TestNoUndefinedNames(unittest.TestCase):
    def test_the_checker_catches_what_it_is_for(self):
        self.assertEqual(undefined_globals("def f():\n    logger.warning('x')\n"), {"logger"})
        self.assertEqual(undefined_globals("with ctx:\n    n = 1\ndef f():\n    return n\n"), {"ctx"})
        self.assertEqual(undefined_globals("import os\nx = 1\ndef f(a):\n    return [os.sep, x, a, len(a)]\n"), set())
        self.assertEqual(undefined_globals("def f():\n    y = 1\n    def g():\n        return y\n    return g\n"), set())

    def test_app_and_every_ui_module_define_what_they_use(self):
        for path in UI_FILES:
            with self.subTest(file=path.relative_to(ROOT).as_posix()):
                self.assertEqual(sorted(undefined_globals(path.read_text(encoding="utf-8"), str(path))), [])


if __name__ == "__main__":
    unittest.main()
