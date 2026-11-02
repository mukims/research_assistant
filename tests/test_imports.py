"""Every module in the package imports on the light CI install.

CI used to check a hand-written list of modules, which missed most of the package. This
walks the whole tree, so a new module with a broken (or heavy, module-scope) import fails
here instead of in production. The three root entry points are imported by CI separately.
"""

import importlib
import pkgutil
import unittest

import research_assistant


class TestImports(unittest.TestCase):
    def test_every_module_imports(self):
        failures = []
        for info in pkgutil.walk_packages(research_assistant.__path__, "research_assistant."):
            try:
                importlib.import_module(info.name)
            except BaseException as exc:  # noqa: BLE001 -- SystemExit from a script counts as a failure
                failures.append((info.name, repr(exc)))
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
