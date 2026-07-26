import argparse
import importlib.util
import json
import run_brax_matrix
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
            output_dir=Path("results/test"),
            reset_dt=5.0,
            batch="batch-1",
            authorize_full_execution=False,
            resume=False,
        )
        with patch.object(MODULE, "execute_run_configs") as execute:
            with self.assertRaisesRegex(RuntimeError, "authorize-full-execution"):
                MODULE.execute_campaign(args)
        execute.assert_not_called()

    def test_campaign_uses_canonical_runner_with_cuda_and_wandb_disabled(self):
        args = argparse.Namespace(
            output_dir=Path("results/test"),
            reset_dt=5.0,
            batch="batch-2",
            authorize_full_execution=True,
            resume=True,
        )
        with patch.object(MODULE, "write_matrix_manifest") as write_manifest, \
             patch.object(MODULE, "execute_run_configs", return_value=0) as execute:
            self.assertEqual(MODULE.execute_campaign(args), 0)
        self.assertEqual(write_manifest.call_args.args[1]["run_count"], 80)
        selected = tuple(execute.call_args.args[1])
        self.assertEqual(len(selected), 10)
        self.assertEqual({run.task for run in selected}, {"hopper"})
        self.assertEqual(execute.call_args.kwargs["device"], "cuda")
        self.assertFalse(execute.call_args.kwargs["use_wandb"])
        self.assertTrue(execute.call_args.kwargs["resume"])
    def test_interrupt_escapes_per_run_exception_handling(self):
        self.assertTrue(issubclass(MODULE.CampaignInterrupted, BaseException))
        self.assertFalse(issubclass(MODULE.CampaignInterrupted, Exception))
        with self.assertRaises(MODULE.CampaignInterrupted):
            MODULE._interrupt(MODULE.signal.SIGTERM, None)

    def test_explicit_batches_are_disjoint_and_cover_canonical_eighty(self):
        run_ids = {
            batch: {run.run_id for run in MODULE.batch_runs(batch)}
            for batch in MODULE.BATCH_TASKS
        }
        self.assertTrue(all(len(ids) == 10 for ids in run_ids.values()))
        for left, left_ids in run_ids.items():
            for right, right_ids in run_ids.items():
                if left != right:
                    self.assertFalse(left_ids & right_ids)
        self.assertEqual(len(set().union(*run_ids.values())), 80)


    def test_batch_receipt_is_recoverable_and_strips_raw_artifacts(self):
        manifest = MODULE.canonical_primary_manifest()
        runs = MODULE.batch_runs("batch-1")
        statuses = {
            run.run_id: {"state": "completed", "attempt_id": "attempt-0001"}
            for run in runs
        }

        def evidence(_output_dir, run, _status):
            return {
                "task": run.task,
                "variant": run.variant.name,
                "seed": run.seed,
                "final_return": 1.0,
                "auc": 2.0,
                "delta_drift": [{
                    "layer_index": 0,
                    "delta_mean_change": 0.01,
                    "timescale_median_change": 0.02,
                    "effective_projection_norm_change": 0.03,
                }],
                "shared_initialization_digest": "a" * 64,
                "provenance_valid": True,
            }

        with patch.object(MODULE, "run_evidence", side_effect=evidence), \
             patch.object(MODULE, "source_commit", return_value="abc"):
            receipt = MODULE.batch_receipt(
                manifest,
                "batch-1",
                Path("results/test"),
                statuses,
                None,
            )
        encoded = json.dumps(receipt)
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["status"]["completed"], 10)
        self.assertEqual(len(receipt["runs"]), 10)
        self.assertLess(len(encoded), 25_000)
        self.assertNotIn("run_id", encoded)
        self.assertNotIn("history", encoded)

    def test_final_receipt_is_printed_when_execution_fails(self):
        with patch.object(
            MODULE,
            "execute_campaign",
            side_effect=RuntimeError("boom"),
        ), patch.object(
            MODULE,
            "install_signal_handlers",
        ), patch.object(
            sys,
            "argv",
            [
                "h200_campaign.py",
                "--batch",
                "batch-1",
                "--authorize-full-execution",
            ],
        ), patch("builtins.print") as printed:
            self.assertEqual(MODULE.main(), 1)
        receipt = json.loads(printed.call_args.args[0].split("=", 1)[1])
        self.assertIn("RuntimeError: boom", receipt["errors"]["execution"])

    def test_public_runner_callback_receives_each_terminal_run(self):
        runs = [type("Run", (), {"run_id": f"run-{index}"})() for index in range(2)]
        statuses = {
            "run-0": {"state": "completed"},
            "run-1": {"state": "failed"},
        }
        observed = []
        with patch.object(run_brax_matrix, "_run_one", side_effect=[False, True]), \
             patch.object(
                 run_brax_matrix,
                 "read_status",
                 side_effect=lambda _output, run_id: statuses[run_id],
             ):
            failed = run_brax_matrix.execute_run_configs(
                Path("results/test"),
                runs,
                device="cuda",
                use_wandb=False,
                resume=False,
                on_terminal=lambda run, status: observed.append(
                    (run.run_id, status["state"])
                ),
            )
        self.assertEqual(failed, 1)
        self.assertEqual(
            observed,
            [("run-0", "completed"), ("run-1", "failed")],
        )

if __name__ == "__main__":
    unittest.main()
