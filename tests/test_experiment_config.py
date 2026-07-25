import importlib.util
import sys
import types
from pathlib import Path
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_config import (
    LONG_HORIZON_PROFILE,
    LONG_HORIZON_PROTOCOL,
    LONG_HORIZON_TRAINER_OVERRIDES,
    LONG_HORIZON_VARIANTS,
    PROTOCOL,
    VARIANTS,
    build_long_horizon_matrix,
    build_matrix,
    expected_evaluation_plan,
    expected_delta_telemetry_steps,
)
from run_brax_matrix import matrix_manifest
from run_long_horizon_matrix import matrix_manifest as long_horizon_matrix_manifest

class ExperimentConfigTest(unittest.TestCase):
    def test_matrix_has_exactly_one_key_per_task_variant_seed(self):
        runs = list(build_matrix())
        self.assertEqual(len(runs), 80)
        self.assertEqual(len({run.run_id for run in runs}), 80)
        self.assertEqual(runs[0].to_dict()["protocol"], PROTOCOL)

    def test_variant_controls_are_independent(self):
        controls = {
            (variant.name, variant.ef_enabled, variant.mr_enabled)
            for variant in VARIANTS
        }
        self.assertEqual(
            controls,
            {
                ("time", True, True),
                ("ef-only", True, False),
                ("mr-only", False, True),
                ("vanilla-mamba2", False, False),
            },
        )
    def test_manifest_is_deterministic_and_lists_all_run_keys(self):
        first = matrix_manifest(5.0)
        self.assertEqual(first, matrix_manifest(5.0))
        self.assertEqual(first["run_count"], 80)
        self.assertEqual(len(first["runs"]), 80)
    def test_non_default_reset_is_applied_only_to_ef_variants(self):
        runs = list(build_matrix(reset_dt=3.0, variants=VARIANTS, seeds=(0,)))
        by_variant = {
            run.variant.name: run
            for run in runs
            if run.task == "halfcheetah"
        }
        self.assertEqual(by_variant["time"].trainer_config("cuda")["reset_dt"], 3.0)
        self.assertEqual(
            by_variant["vanilla-mamba2"].trainer_config("cuda")["reset_dt"], 5.0
        )
        self.assertEqual(by_variant["mr-only"].trainer_config("cuda")["reset_dt"], 5.0)
        recipe = by_variant["time"].to_dict()["recipe"]
        self.assertEqual(recipe["epsilon"], 0.2)
        self.assertEqual(recipe["num_minibatches"], 4)
        self.assertEqual(recipe["num_update_epochs"], 4)

    def test_evaluation_plan_uses_six_points_and_larger_final_sample(self):
        config = next(
            run.to_dict() for run in build_matrix(seeds=(0,)) if run.task == "ant"
        )
        plan = expected_evaluation_plan(config)
        self.assertEqual(len(plan), 6)
        ordered = [plan[step] for step in sorted(plan)]
        self.assertTrue(
            all(item["evaluation_episodes"] == 16 for item in ordered[:-1])
        )
        self.assertTrue(
            all(item["evaluation_kind"] == "intermediate" for item in ordered[:-1])
        )
        self.assertEqual(
            ordered[-1],
            {"evaluation_kind": "final", "evaluation_episodes": 64},
        )

    def test_delta_telemetry_uses_six_fixed_probe_points(self):
        config = next(
            run.to_dict() for run in build_matrix(seeds=(0,)) if run.task == "ant"
        )
        steps = expected_delta_telemetry_steps(config)
        self.assertEqual(len(steps), 6)
        self.assertEqual(steps, set(expected_evaluation_plan(config)))
        self.assertTrue(config["recipe"]["delta_telemetry_enabled"])
        self.assertEqual(config["recipe"]["delta_telemetry_probe_sequences"], 8)

    def test_long_horizon_matrix_is_exactly_twenty_deterministic_treatment_runs(self):
        first = list(build_long_horizon_matrix())
        second = list(build_long_horizon_matrix())
        self.assertEqual(len(first), 20)
        self.assertEqual([run.run_id for run in first], [run.run_id for run in second])
        self.assertEqual(len({run.run_id for run in first}), 20)
        self.assertEqual(
            {run.variant.name for run in first}, {"time", "vanilla-mamba2"}
        )
        self.assertEqual(
            {run.variant for run in first}, set(LONG_HORIZON_VARIANTS)
        )
        self.assertEqual({run.seed for run in first}, set(range(5)))

    def test_long_horizon_runs_bind_recovered_profile_masks_and_controls(self):
        runs = {run.task: run for run in build_long_horizon_matrix() if run.seed == 0}
        expected_masks = {"ant": [0, 1, 2, 3, 4], "halfcheetah": [0, 1, 2]}
        self.assertEqual(set(runs), set(expected_masks))
        for task, run in runs.items():
            config = run.trainer_config("cuda")
            manifest_run = run.to_dict()
            self.assertEqual(manifest_run["protocol"], LONG_HORIZON_PROTOCOL)
            self.assertEqual(manifest_run["profile"], LONG_HORIZON_PROFILE)
            self.assertEqual(manifest_run["episode_length"], 5_000)
            self.assertEqual(manifest_run["keep_dims_override"], expected_masks[task])
            self.assertIn("ep-5000", run.run_id)
            self.assertIn(run.mask_name, run.run_id)
            for key, value in LONG_HORIZON_TRAINER_OVERRIDES.items():
                self.assertEqual(config[key], value)
            self.assertEqual(config["keep_dims_override"], expected_masks[task])
            self.assertEqual(len(expected_evaluation_plan(manifest_run)), 6)
            self.assertEqual(len(expected_delta_telemetry_steps(manifest_run)), 6)

    def test_standard_matrix_recipe_and_identity_controls_remain_default(self):
        run = next(
            run
            for run in build_matrix(seeds=(0,))
            if run.task == "ant" and run.variant.name == "time"
        )
        config = run.trainer_config("cuda")
        self.assertEqual(run.to_dict()["protocol"], PROTOCOL)
        self.assertNotIn("episode_length", config)
        self.assertNotIn("keep_dims_override", config)
        self.assertEqual(config["num_envs"], 64)
        self.assertEqual(config["unroll_length"], 128)
        self.assertEqual(config["num_update_epochs"], 4)
        self.assertEqual(config["num_minibatches"], 4)
        self.assertEqual(run.run_id, next(
            candidate.run_id
            for candidate in build_matrix(seeds=(0,))
            if candidate.task == "ant" and candidate.variant.name == "time"
        ))

    def test_long_horizon_dry_run_manifest_lists_only_bound_runs(self):
        manifest = long_horizon_matrix_manifest(5.0)
        self.assertEqual(manifest["run_count"], 20)
        self.assertEqual(manifest["protocol"], LONG_HORIZON_PROTOCOL)
        self.assertEqual(manifest["profile"], LONG_HORIZON_PROFILE)
        self.assertEqual(manifest["episode_length"], 5_000)
        self.assertEqual(
            manifest["keep_dims"],
            {"ant": [0, 1, 2, 3, 4], "halfcheetah": [0, 1, 2]},
        )
        self.assertEqual(
            {run["variant"] for run in manifest["runs"]},
            {"time", "vanilla-mamba2"},
        )
        self.assertTrue(
            all(
                run["episode_length"] == 5_000
                and run["profile"] == LONG_HORIZON_PROFILE
                and run["keep_dims_override"]
                for run in manifest["runs"]
            )
        )

    def test_env_control_validators_reject_invalid_masks_and_horizons_portably(self):
        gym = types.ModuleType("gym")
        brax = types.ModuleType("brax")
        envs = types.ModuleType("brax.envs")
        wrappers = types.ModuleType("brax.envs.wrappers")
        gym_wrapper = types.ModuleType("brax.envs.wrappers.gym")
        torch_wrapper = types.ModuleType("brax.envs.wrappers.torch")
        gym_wrapper.VectorGymWrapper = object
        brax.envs = envs
        wrappers.gym = gym_wrapper
        wrappers.torch = torch_wrapper
        module_names = {
            "gym": gym,
            "brax": brax,
            "brax.envs": envs,
            "brax.envs.wrappers": wrappers,
            "brax.envs.wrappers.gym": gym_wrapper,
            "brax.envs.wrappers.torch": torch_wrapper,
        }
        with mock.patch.dict(sys.modules, module_names):
            spec = importlib.util.spec_from_file_location(
                "portable_env_utils",
                Path(__file__).resolve().parents[1] / "env_utils.py",
            )
            env_utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(env_utils)

        for value in (0, -1, True, 1.5):
            with self.assertRaises((TypeError, ValueError)):
                env_utils._validate_episode_length(value)
        for value in ([], [0, 0], [-1], [True], [0, "1"]):
            with self.assertRaises((TypeError, ValueError)):
                env_utils._validate_keep_dims(value)

if __name__ == "__main__":
    unittest.main()
