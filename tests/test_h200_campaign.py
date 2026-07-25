import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "h200_campaign.py"
SPEC = importlib.util.spec_from_file_location("h200_campaign", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class H200CampaignTest(unittest.TestCase):
    def test_campaign_is_exactly_the_canonical_primary_eighty_runs(self):
        manifest = MODULE.canonical_primary_manifest()
        self.assertEqual(manifest["run_count"], 80)
        self.assertEqual(set(manifest["tasks"]), set(MODULE.TASKS))
        self.assertEqual(
            {variant["name"] for variant in manifest["variants"]},
            {"time", "vanilla-mamba2"},
        )
        self.assertNotIn("ef-only", {variant["name"] for variant in manifest["variants"]})
        self.assertNotIn("long-horizon", json.dumps(manifest))
        self.assertEqual(len(manifest["runs"]), 80)

    def test_execution_requires_explicit_authorization(self):
        args = argparse.Namespace(
            output_dir=Path("results/test"), reset_dt=5.0,
            authorize_full_execution=False, resume=False,
        )
        with patch.object(MODULE, "execute_matrix") as execute:
            with self.assertRaisesRegex(RuntimeError, "authorize-full-execution"):
                MODULE.execute_campaign(args)
        execute.assert_not_called()

    def test_campaign_uses_canonical_runner_with_cuda_and_wandb_disabled(self):
        args = argparse.Namespace(
            output_dir=Path("results/test"), reset_dt=5.0,
            authorize_full_execution=True, resume=True,
        )
        with patch.object(MODULE, "execute_matrix", return_value=0) as execute:
            self.assertEqual(MODULE.execute_campaign(args), 0)
        runner_args = execute.call_args.args[0]
        self.assertTrue(runner_args.execute)
        self.assertTrue(runner_args.authorize_full_execution)
        self.assertEqual(runner_args.device, "cuda")
        self.assertFalse(runner_args.use_wandb)
        self.assertTrue(runner_args.resume)
    def test_interrupt_escapes_per_run_exception_handling(self):
        self.assertTrue(issubclass(MODULE.CampaignInterrupted, BaseException))
        self.assertFalse(issubclass(MODULE.CampaignInterrupted, Exception))
        with self.assertRaises(MODULE.CampaignInterrupted):
            MODULE._interrupt(MODULE.signal.SIGTERM, None)


    def test_compact_receipt_strips_histories_and_run_ids_for_eighty_runs(self):
        manifest = MODULE.canonical_primary_manifest()
        summary = {"n_seeds": 5, "mean": 1.0, "sample_std": 0.1, "sem": 0.01,
                   "ci95_low": 0.9, "ci95_high": 1.1}
        drift = {str(seed): {"event": "delta_drift_summary", "layers": [{
            "layer_index": 0, "delta_mean_change": 0.01,
            "timescale_median_change": 0.02,
            "effective_projection_norm_change": 0.03,
            "raw_history": [{"run_id": f"secret-{seed}"}],
        }]} for seed in range(5)}
        conditions = [
            {"task": task, "variant": variant, "final_return": summary, "auc": summary,
             "delta_drift_by_seed": drift}
            for task in MODULE.TASKS for variant in ("time", "vanilla-mamba2")
        ]
        aggregate_result = {
            "conditions": conditions,
            "paired_differences": [
                {"task": task, "contrast": "time-minus-vanilla-mamba2",
                 "final_return": summary, "auc": summary}
                for task in MODULE.TASKS
            ],
            "completeness": {"completed_runs": 80, "missing_run_ids": [],
                               "invalid_run_ids": [], "invalid_pairs": [], "complete": True},
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(MODULE, "source_commit", return_value="abc"):
            receipt = MODULE.compact_receipt(
                manifest, aggregate_result, Path(directory), None, None
            )
        encoded = json.dumps(receipt)
        self.assertLess(len(encoded), 65_000)
        self.assertNotIn("run_id", encoded)
        self.assertNotIn("raw_history", encoded)
        self.assertEqual(receipt["canonical_run_count"], 80)
        self.assertEqual(len(receipt["conditions"]), 16)

    def test_final_receipt_is_printed_when_execution_fails(self):
        with patch.object(MODULE, "execute_campaign", side_effect=RuntimeError("boom")), \
             patch.object(MODULE, "aggregate", return_value={"conditions": [], "paired_differences": [],
                 "completeness": {"completed_runs": 0, "missing_run_ids": ["x"],
                                  "invalid_run_ids": [], "invalid_pairs": [], "complete": False}}), \
             patch.object(MODULE, "source_commit", return_value=None), \
             patch.object(MODULE, "install_signal_handlers"), \
             patch.object(sys, "argv", ["h200_campaign.py", "--authorize-full-execution"]), \
             patch("builtins.print") as printed:
            self.assertEqual(MODULE.main(), 1)
        self.assertTrue(printed.call_args.args[0].startswith("H200_CAMPAIGN_RESULT="))

    def test_final_receipt_is_printed_when_aggregation_fails(self):
        with patch.object(MODULE, "execute_campaign", return_value=0), \
             patch.object(MODULE, "aggregate", side_effect=ValueError("bad artifacts")), \
             patch.object(MODULE, "source_commit", return_value=None), \
             patch.object(MODULE, "install_signal_handlers"), \
             patch.object(sys, "argv", ["h200_campaign.py", "--authorize-full-execution"]), \
             patch("builtins.print") as printed:
            self.assertEqual(MODULE.main(), 1)
        receipt = json.loads(printed.call_args.args[0].split("=", 1)[1])
        self.assertIn("ValueError: bad artifacts", receipt["errors"]["aggregation"])

if __name__ == "__main__":
    unittest.main()
