import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "h200_smoke.py"
SPEC = importlib.util.spec_from_file_location("h200_smoke", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class H200SmokeTest(unittest.TestCase):
    def test_expected_versions_pin_scientific_reference(self):
        self.assertEqual(MODULE.EXPECTED_QLAB_VERSIONS["torch"], "2.6.0+cu124")
        self.assertEqual(MODULE.EXPECTED_QLAB_VERSIONS["brax"], "0.12.3")
        self.assertEqual(MODULE.EXPECTED_QLAB_VERSIONS["mamba-ssm"], "2.2.4")

    def test_stage_records_success(self):
        receipt = MODULE.Receipt()
        receipt.stage("ok", lambda: {"value": 7})
        self.assertEqual(receipt.stages, [{"name": "ok", "status": "passed", "value": 7}])

    def test_stage_records_bounded_failure(self):
        receipt = MODULE.Receipt()

        def fail():
            raise RuntimeError("x" * 2_000)

        receipt.stage("bad", fail)
        stage = receipt.stages[0]
        self.assertEqual(stage["status"], "failed")
        self.assertEqual(stage["error_type"], "RuntimeError")
        self.assertLessEqual(len(stage["error"]), 1_000)
        self.assertLessEqual(len(stage["traceback_tail"]), 8)


if __name__ == "__main__":
    unittest.main()
