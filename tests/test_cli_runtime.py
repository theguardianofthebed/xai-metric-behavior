"""The file-producing CLI must work outside its parent notebook's UI stack."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CliRuntimeTests(unittest.TestCase):
    def child(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        # Deliberately unavailable even if this test interpreter has IPython.
        # This exercises import-time backend selection rather than GUI display.
        env["MPLBACKEND"] = "missing_notebook_display_backend"
        return subprocess.run(
            [sys.executable, *args], cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
        )

    def test_cli_starts_with_an_unavailable_inherited_display_backend(self):
        result = self.child("-m", "xai_metric_behavior", "--help")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("run-real", result.stdout)
        self.assertIn("--help", result.stdout)

    def test_importing_cli_module_does_not_reconfigure_notebook_display(self):
        result = self.child(
            "-c",
            "import os, sys; import xai_metric_behavior.__main__; "
            "assert os.environ['MPLBACKEND'] == 'missing_notebook_display_backend'; "
            "assert 'matplotlib' not in sys.modules",
        )
        self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
