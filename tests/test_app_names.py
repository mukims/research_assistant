"""Every bare name app.py calls must be bound somewhere in app.py.

app.py imports lazily inside Streamlit branches ("from … import ingest_pdfs"
next to the call that needs it), so a deleted import is not an ImportError at
startup and CI's import sweep cannot see it — it is a NameError the first time
a user takes that branch. This is a targeted static check for that failure:
collect every bare-name call, subtract everything the module binds anywhere
(imports at any depth, defs, assignments, loop/with targets, comprehension
variables, parameters) and builtins, and require the remainder to be empty.
"""

import ast
import builtins
import os
import unittest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def _bound_names(tree):
    bound = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
            if not isinstance(node, ast.ClassDef):
                args = node.args
                for a in args.args + args.posonlyargs + args.kwonlyargs:
                    bound.add(a.arg)
                for a in (args.vararg, args.kwarg):
                    if a is not None:
                        bound.add(a.arg)
        elif isinstance(node, ast.Lambda):
            for a in node.args.args:
                bound.add(a.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return bound


class TestBareCallsAreBound(unittest.TestCase):
    def test_every_called_name_is_bound(self):
        with open(APP, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), APP)
        bound = _bound_names(tree)
        unbound = sorted({
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id not in bound
        })
        self.assertEqual(unbound, [], f"app.py calls names it never binds: {unbound}")
