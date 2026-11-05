"""Exercise the EXP-01 notebook runner with real child processes, not metric runs."""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def notebook_runner():
    """Compile the notebook's actual imports/helper without executing an experiment."""
    notebook = json.loads((ROOT / "notebooks" / "EXP-01.ipynb").read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and "def run_logged(" in "".join(cell["source"]))
    tree = ast.parse(source)
    helper = ast.Module(body=[node for node in tree.body
                             if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))],
                        type_ignores=[])
    namespace = {}
    exec(compile(helper, "notebooks/EXP-01.ipynb:run-cell", "exec"), namespace)
    return namespace["run_logged"]


class NotebookRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_logged = staticmethod(notebook_runner())

    def test_success_streams_both_outputs_and_keeps_backend_change_in_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "results" / "exp01-smoke-unique"
            acknowledgement = root / "output-was-visible"

            class VisibleOutput(io.StringIO):
                def write(self, value):
                    # The child cannot finish until its output is visible here.
                    if value == "stream-ready\n":
                        acknowledgement.touch()
                    return super().write(value)

            visible = VisibleOutput()
            script = """
import os, pathlib, sys, time
output_dir, acknowledgement = map(pathlib.Path, sys.argv[1:])
assert not output_dir.exists(), 'runner pre-created the experiment destination'
assert os.environ['MPLBACKEND'] == 'Agg'
assert os.environ['NOTEBOOK_RUNNER_SENTINEL'] == 'preserved'
print('stream-ready', flush=True)
deadline = time.monotonic() + 5
while not acknowledgement.exists():
    assert time.monotonic() < deadline, 'output was not streamed before process exit'
    time.sleep(0.01)
print('stderr was visible', file=sys.stderr, flush=True)
output_dir.mkdir()
(output_dir / 'manifest.json').write_text('{"status": "test-success"}')
"""
            original_backend = "module://matplotlib_inline.backend_inline"
            with patch.dict(os.environ, {"MPLBACKEND": original_backend,
                                         "NOTEBOOK_RUNNER_SENTINEL": "preserved"}):
                with redirect_stdout(visible):
                    log_path = self.run_logged(
                        [sys.executable, "-u", "-c", script, str(output_dir), str(acknowledgement)],
                        cwd=root, output_dir=output_dir,
                    )
                self.assertEqual(os.environ["MPLBACKEND"], original_backend)
            self.assertEqual(log_path, output_dir.with_name(output_dir.name + ".log"))
            self.assertTrue(acknowledgement.exists())
            self.assertEqual(json.loads((output_dir / "manifest.json").read_text()),
                             {"status": "test-success"})
            for message in ("stream-ready", "stderr was visible"):
                self.assertIn(message, visible.getvalue())
                self.assertIn(message, log_path.read_text())

    def test_failure_exposes_traceback_exit_status_and_retains_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "results" / "exp01-smoke-failure"
            visible = io.StringIO()
            script = """
import pathlib, sys, traceback
output_dir = pathlib.Path(sys.argv[1])
assert not output_dir.exists()
output_dir.mkdir()
(output_dir / 'manifest.json').write_text('{"status": "test-failure"}')
(output_dir / 'failures.csv').write_text('reason\\ndiagnostic sentinel\\n')
for i in range(70):
    print(f'progress {i}', flush=True)
try:
    raise ValueError('original child failure')
except ValueError:
    traceback.print_exc()
    sys.exit(7)
"""
            with redirect_stdout(visible), self.assertRaises(RuntimeError) as raised:
                self.run_logged([sys.executable, "-u", "-c", script, str(output_dir)],
                                cwd=root, output_dir=output_dir)
            log_path = output_dir.with_name(output_dir.name + ".log")
            error = str(raised.exception)
            for detail in ("status 7", str(log_path), "Traceback (most recent call last)",
                           "ValueError: original child failure", str(output_dir / "manifest.json"),
                           str(output_dir / "failures.csv")):
                self.assertIn(detail, error)
            self.assertNotIn("progress 0\n", error)
            self.assertIn("progress 0\n", log_path.read_text())
            self.assertIn("ValueError: original child failure", log_path.read_text())
            self.assertIn("ValueError: original child failure", visible.getvalue())
            self.assertNotIn("Saved:", visible.getvalue())
            self.assertEqual((output_dir / "failures.csv").read_text(),
                             "reason\ndiagnostic sentinel\n")

    def test_launch_error_keeps_log_and_does_not_create_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "results" / "exp01-smoke-launch-error"
            with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError) as raised:
                self.run_logged([str(root / "missing-python")], cwd=root, output_dir=output_dir)
            log_path = output_dir.with_name(output_dir.name + ".log")
            self.assertIn(str(log_path), str(raised.exception))
            self.assertIn("Could not start the experiment", log_path.read_text())
            self.assertFalse(output_dir.exists())

    def test_existing_log_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "exp01-smoke-collision"
            log_path = output_dir.with_name(output_dir.name + ".log")
            log_path.write_text("earlier evidence\n")
            with redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                self.run_logged([sys.executable, "-u", "-c", "raise AssertionError('must not run')"],
                                cwd=root, output_dir=output_dir)
            self.assertEqual(log_path.read_text(), "earlier evidence\n")
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
