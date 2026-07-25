import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "h200_smoke.py"
SPEC = importlib.util.spec_from_file_location("h200_smoke", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class H200SmokeTest(unittest.TestCase):
    def test_expected_versions_pin_scientific_reference(self):
        self.assertEqual(MODULE.EXPECTED_QLAB_VERSIONS["torch"], "2.6.0+cu124")
        self.assertEqual(MODULE.EXPECTED_QLAB_VERSIONS["triton"], "3.2.0")
        self.assertEqual(MODULE.EXPECTED_QLAB_VERSIONS["causal-conv1d"], "1.5.0.post8")
        self.assertEqual(MODULE.EXPECTED_QLAB_VERSIONS["einops"], "0.8.1")
        self.assertEqual(MODULE.EXPECTED_QLAB_VERSIONS["ninja"], "1.11.1.4")
        self.assertEqual(MODULE.EXPECTED_QLAB_VERSIONS["mamba-ssm"], "2.2.4")
        self.assertEqual(
            MODULE.EXPECTED_QLAB_VERSIONS["nvidia-cudnn-cu12"],
            "9.8.0.87",
        )

    def test_stage_records_success(self):
        receipt = MODULE.Receipt()
        receipt.stage("ok", lambda: {"value": 7})
        self.assertEqual(receipt.stages, [{"name": "ok", "status": "passed", "value": 7}])

    def test_stage_records_bounded_failure(self):
        receipt = MODULE.Receipt()
        receipt.stage("bad", lambda: (_ for _ in ()).throw(RuntimeError("x" * 2_000)))
        stage = receipt.stages[0]
        self.assertEqual(stage["status"], "failed")
        self.assertEqual(stage["error_type"], "RuntimeError")
        self.assertLessEqual(len(stage["error"]), 1_000)
        self.assertLessEqual(len(stage["traceback_tail"]), 8)

    def test_scientific_mode_fails_closed_before_runtime_stages_on_pin_mismatch(self):
        versions = dict(MODULE.EXPECTED_QLAB_VERSIONS, torch="2.11.0+cu130")
        with patch.object(MODULE, "installed_versions", return_value=versions), \
             patch.object(MODULE, "collect_runtime") as runtime, \
             patch.object(sys, "argv", ["h200_smoke.py", "--cache-fingerprint", "abc", "--venv", "/cache/venv"]), \
             patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(MODULE.main(), 1)
        runtime.assert_not_called()
        receipt = json.loads(stdout.getvalue().split("=", 1)[1])
        self.assertFalse(receipt["passed"])
        self.assertFalse(receipt["scientific_compatible_with_qlab"])
        self.assertEqual(receipt["stages"][0]["name"], "exact-qlab-pins")
        self.assertEqual(receipt["cache"]["fingerprint"], "abc")
        self.assertEqual(receipt["cache"]["interpreter"], sys.executable)

    def test_exact_pins_gate_all_scientific_stages_and_receipt_is_scientific(self):
        with patch.object(MODULE, "installed_versions", return_value=dict(MODULE.EXPECTED_QLAB_VERSIONS)), \
             patch.object(MODULE, "collect_runtime", return_value=None), \
             patch.object(MODULE, "check_jax_cuda", return_value={}), \
             patch.object(MODULE, "check_backbones", return_value={}), \
             patch.object(MODULE, "check_brax_torch_bridge", return_value={}), \
             patch.object(MODULE, "check_one_update", return_value={}), \
             patch.object(sys, "argv", ["h200_smoke.py", "--scientific"]), \
             patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(MODULE.main(), 0)
        receipt = json.loads(stdout.getvalue().split("=", 1)[1])
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["kind"], "time-h200-qlab-scientific-smoke")
        self.assertEqual([stage["name"] for stage in receipt["stages"]], [
            "cuda-runtime", "jax-cuda", "time-and-vanilla-backbones", "brax-torch-dlpack", "corrected-v2-one-update"
        ])


if __name__ == "__main__":
    unittest.main()
