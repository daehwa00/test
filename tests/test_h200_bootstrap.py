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

    def test_causal_extension_disables_build_isolation_and_dependencies(self):
        command = MODULE.BOOTSTRAP_COMMANDS[1]
        self.assertIn("--no-build-isolation", command)
        self.assertIn("--no-deps", command)

    def test_cuda12_jax_runtime_is_pinned_for_driver_compatibility(self):
        self.assertIn("jax[cuda12]==0.6.0", MODULE.BOOTSTRAP_COMMANDS[0])
        self.assertNotIn("jax[cuda13]", MODULE.BOOTSTRAP_COMMANDS[0])

    def test_h200_compatible_causal_conv_release_is_used(self):
        self.assertIn("causal-conv1d==1.6.2.post1", MODULE.BOOTSTRAP_COMMANDS[1])

    def test_h200_mamba_release_installs_required_runtime_dependencies(self):
        command = MODULE.BOOTSTRAP_COMMANDS[2]
        self.assertIn("mamba-ssm==2.3.2.post1", command)
        self.assertIn("--no-build-isolation", command)
        self.assertNotIn("--no-deps", command)


if __name__ == "__main__":
    unittest.main()
