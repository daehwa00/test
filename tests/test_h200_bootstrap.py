import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "h200_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("h200_bootstrap", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class H200BootstrapTest(unittest.TestCase):
    def test_bootstrap_never_replaces_base_torch_or_triton(self):
        arguments = [argument for command in MODULE.BOOTSTRAP_COMMANDS for argument in command]
        self.assertFalse(any(argument.startswith("torch") for argument in arguments))
        self.assertFalse(any(argument.startswith("triton") for argument in arguments))

    def test_native_extensions_disable_build_isolation_and_dependencies(self):
        for command in MODULE.BOOTSTRAP_COMMANDS[1:]:
            self.assertIn("--no-build-isolation", command)
            self.assertIn("--no-deps", command)

    def test_cuda13_jax_is_requested(self):
        self.assertIn("jax[cuda13]", MODULE.BOOTSTRAP_COMMANDS[0])


if __name__ == "__main__":
    unittest.main()
