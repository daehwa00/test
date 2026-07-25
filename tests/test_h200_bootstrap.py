import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "h200_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("h200_bootstrap", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class H200BootstrapTest(unittest.TestCase):
    def test_fingerprint_is_deterministic_and_pin_sensitive(self):
        self.assertEqual(MODULE.environment_fingerprint(), MODULE.environment_fingerprint(dict(reversed(list(MODULE.QLAB_PINS.items())))))
        changed = dict(MODULE.QLAB_PINS, brax="0.12.4")
        self.assertNotEqual(MODULE.environment_fingerprint(), MODULE.environment_fingerprint(changed))
        fingerprint = MODULE.environment_fingerprint()
        with patch.object(MODULE, "VIRTUALENV_VERSION", "different"):
            self.assertNotEqual(fingerprint, MODULE.environment_fingerprint())

    def test_runtime_path_is_versioned_by_fingerprint(self):
        fingerprint = MODULE.environment_fingerprint()
        path = MODULE.runtime_path(Path("/cache"), fingerprint)
        self.assertEqual(path, Path("/cache/venvs") / f"time-h200-qlab-{fingerprint}")
        self.assertEqual(MODULE.venv_interpreter(path), path / "bin/python")

    def test_install_commands_mutate_only_venv_and_pin_torch_triton(self):
        commands = MODULE.installation_commands(Path("/isolated/bin/python"), Path("/cache/pip/wheels"))
        arguments = [argument for command in commands for argument in command]
        self.assertEqual(commands[0][0], "/isolated/bin/python")
        self.assertIn("torch==2.6.0+cu124", commands[0])
        self.assertIn("triton==3.2.0", commands[0])
        self.assertIn(MODULE.PYTORCH_CU124_INDEX, commands[0])
        self.assertIn("causal-conv1d==1.5.0.post8", commands[2])
        self.assertIn("mamba-ssm==2.2.4", commands[2])
        self.assertIn("nvidia-cudnn-cu12==9.8.0.87", commands[3])
        self.assertIn("--no-deps", commands[3])
        self.assertNotIn(sys.executable, arguments)
    def test_virtualenv_bootstrap_does_not_depend_on_ensurepip(self):
        cache_root = Path("/cache")
        command = MODULE.bootstrap_tool_command(cache_root, cache_root / "pip/wheels")
        create = MODULE.create_venv_command(Path("/cache/venvs/runtime"))
        self.assertEqual(command[:4], (sys.executable, "-m", "pip", "install"))
        self.assertIn("--target", command)
        self.assertIn(f"virtualenv=={MODULE.VIRTUALENV_VERSION}", command)
        self.assertEqual(create[:3], (sys.executable, "-m", "virtualenv"))
        self.assertNotIn("venv", create)

    def test_partial_environment_without_pip_is_removed_before_recreation(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory)
            tool = MODULE.bootstrap_tool_path(cache_root) / "virtualenv"
            tool.mkdir(parents=True)
            (tool / "__init__.py").write_text("")
            interpreter = MODULE.venv_interpreter(
                MODULE.runtime_path(cache_root, MODULE.environment_fingerprint())
            )
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text("")
            failed_probe = {
                "command": [],
                "returncode": 1,
                "stdout_tail": [],
                "stderr_tail": [],
            }
            failed_create = {
                "command": [],
                "returncode": 1,
                "stdout_tail": [],
                "stderr_tail": [],
            }
            with patch.object(
                MODULE,
                "run_bounded",
                side_effect=[failed_probe, failed_create],
            ):
                _, _, audit, reason = MODULE.ensure_runtime(cache_root)
            self.assertEqual(reason, "create-venv-failed")
            self.assertEqual(audit[0]["stage"], "partial-venv")
            self.assertEqual(audit[1]["stage"], "create-venv")
            self.assertFalse(interpreter.exists())

    def test_stale_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            venv = Path(directory)
            (venv / MODULE.READY_MARKER).write_text(json.dumps({"fingerprint": "wrong", "pins": MODULE.QLAB_PINS}))
            self.assertIsNone(MODULE.read_ready(venv, MODULE.environment_fingerprint()))

    def test_ready_marker_is_atomic_and_contains_exact_pin_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            venv = Path(directory)
            versions = dict(MODULE.QLAB_PINS)
            with patch.object(MODULE.os, "replace", wraps=MODULE.os.replace) as replace:
                MODULE.write_ready_atomic(venv, "fingerprint", versions)
            self.assertEqual(replace.call_count, 1)
            marker = json.loads((venv / MODULE.READY_MARKER).read_text())
            self.assertEqual(marker["fingerprint"], "fingerprint")
            self.assertEqual(marker["pins"], MODULE.QLAB_PINS)
            self.assertEqual(marker["versions"], versions)

    def test_smoke_dispatch_uses_venv_interpreter(self):
        venv = Path("/cache/venvs/time-h200-qlab-test")
        smoke = {"command": [], "returncode": 0, "stdout_tail": [], "stderr_tail": []}
        with patch.object(MODULE, "ensure_runtime", return_value=(venv, "test", [], None)), \
             patch.object(MODULE, "run_bounded", return_value=smoke) as run, \
             patch.object(sys, "argv", ["h200_bootstrap.py"]):
            self.assertEqual(MODULE.main(), 0)
        command = run.call_args.args[0]
        self.assertEqual(command[0], str(MODULE.venv_interpreter(venv)))
        self.assertIn("--scientific", command)


if __name__ == "__main__":
    unittest.main()
