import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggregate_results import aggregate
from artifacts import initialize_run, write_history, write_matrix_manifest, write_status
from metrics import evaluation_events_match_plan, summarize_history
from run_brax_matrix import matrix_manifest


class MetricsAndAggregateTest(unittest.TestCase):
    def test_final_return_and_trapezoidal_auc(self):
        metrics = summarize_history(
            {
                "events": [
                    {
                        "event": "evaluation",
                        "total_steps": 0,
                        "evaluation_mean_return": 2,
                    },
                    {
                        "event": "evaluation",
                        "total_steps": 10,
                        "evaluation_mean_return": 4,
                    },
                ]
            }
        )
        self.assertEqual(metrics, {"final_return": 4.0, "auc": 3.0})

    def test_training_diagnostics_do_not_contaminate_evaluation_metrics(self):
        metrics = summarize_history(
            {
                "events": [
                    {
                        "event": "evaluation",
                        "total_steps": 0,
                        "evaluation_mean_return": 2.0,
                    },
                    {
                        "total_steps": 10,
                        "episode_return": 999.0,
                        "training_unroll_reward": 999.0,
                    },
                    {
                        "event": "evaluation",
                        "total_steps": 10,
                        "evaluation_mean_return": 4.0,
                    },
                ]
            }
        )
        self.assertEqual(metrics, {"final_return": 4.0, "auc": 3.0})

    def test_evaluation_events_must_match_episode_plan(self):
        plan = {
            0: {
                "evaluation_kind": "intermediate",
                "evaluation_episodes": 2,
            },
            10: {
                "evaluation_kind": "final",
                "evaluation_episodes": 3,
            },
        }
        events = [
            {
                "event": "evaluation",
                "total_steps": 0,
                "evaluation_kind": "intermediate",
                "evaluation_episodes": 2,
                "episode_return": [1.0, 3.0],
                "evaluation_mean_return": 2.0,
            },
            {
                "event": "evaluation",
                "total_steps": 10,
                "evaluation_kind": "final",
                "evaluation_episodes": 3,
                "episode_return": [2.0, 4.0, 6.0],
                "evaluation_mean_return": 4.0,
            },
        ]
        self.assertTrue(evaluation_events_match_plan(events, plan))
        events[-1]["episode_return"] = [2.0]
        self.assertFalse(evaluation_events_match_plan(events, plan))

    def test_aggregation_keeps_training_seed_as_independent_unit(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed, value in ((0, 1.0), (1, 3.0)):
                run_id = f"ant__time__seed-{seed}"
                initialize_run(root, {"protocol": "time-brax-corrected-v2", "run_id": run_id, "task": "ant", "variant": "time", "seed": seed})
                attempt_id = "attempt-0001"
                write_history(
                    root,
                    run_id,
                    [
                        {
                            "event": "evaluation",
                            "attempt_id": attempt_id,
                            "total_steps": 1,
                            "evaluation_mean_return": value,
                        }
                    ],
                )
                write_status(root, run_id, "running", attempt_id=attempt_id)
                write_status(root, run_id, "completed", attempt_id=attempt_id)
            condition = aggregate(root)["conditions"][0]
            self.assertEqual(condition["seeds"], [0, 1])
            self.assertEqual(condition["final_return"]["mean"], 2.0)
            self.assertEqual(condition["final_return"]["n_seeds"], 2)

    def test_paired_difference_matches_training_seed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for variant, values in (
                ("time", {0: 4.0, 1: 8.0}),
                ("vanilla-mamba2", {0: 1.0, 1: 2.0}),
            ):
                for seed, value in values.items():
                    run_id = f"ant__{variant}__seed-{seed}"
                    initialize_run(
                        root,
                        {
                            "protocol": "time-brax-corrected-v2",
                            "run_id": run_id,
                            "task": "ant",
                            "variant": variant,
                            "seed": seed,
                        },
                    )
                    attempt_id = "attempt-0001"
                    write_history(
                        root,
                        run_id,
                        [
                            {
                                "event": "shared_initialization_verified",
                                "attempt_id": attempt_id,
                                "backbone_provenance": {"upstream": {}},
                                "shared_initialization_digest": f"{seed:x}" * 64,
                            },
                            {
                                "event": "evaluation",
                                "attempt_id": attempt_id,
                                "total_steps": 1,
                                "evaluation_mean_return": value,
                            },
                        ],
                    )
                    write_status(root, run_id, "running", attempt_id=attempt_id)
                    write_status(root, run_id, "completed", attempt_id=attempt_id)
            paired = aggregate(root)["paired_differences"][0]
            self.assertEqual(paired["seeds"], [0, 1])
            self.assertEqual(paired["final_return"]["mean"], 4.5)
            self.assertIsNotNone(paired["final_return"]["ci95_low"])

    def test_manifest_completion_rejects_invalid_scientific_artifacts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "protocol": "time-brax-corrected-v2",
                "run_id": "ant__time__seed-0",
                "task": "ant",
                "variant": "time",
                "seed": 0,
            }
            write_matrix_manifest(
                root,
                {
                    "protocol": "time-brax-corrected-v2",
                    "runs": [config],
                },
            )
            initialize_run(root, config)
            write_history(
                root,
                config["run_id"],
                [
                    {
                        "event": "evaluation",
                        "total_steps": 0,
                        "evaluation_mean_return": 1.0,
                    }
                ],
            )
            write_status(root, config["run_id"], "running")
            write_status(root, config["run_id"], "completed")
            with self.assertRaisesRegex(ValueError, "valid reset_dt"):
                aggregate(root)

    def test_aggregation_rejects_truncated_canonical_manifest(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = matrix_manifest(5.0)
            manifest["runs"] = []
            manifest["run_count"] = 0
            write_matrix_manifest(root, manifest)
            with self.assertRaisesRegex(ValueError, "canonical 80-run matrix"):
                aggregate(root)


if __name__ == "__main__":
    unittest.main()
