import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


try:
    import torch
    from TiME.backbones import build_backbone
except (ImportError, OSError):
    torch = None
    build_backbone = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA Mamba stack required")
class BackboneContractTest(unittest.TestCase):
    def test_all_four_states_use_expected_implementations(self):
        variants = ((True, True), (True, False), (False, True), (False, False))
        models = [
            build_backbone(
                d_model=256,
                d_state=64,
                d_conv=4,
                expand=2,
                ef_enabled=ef_enabled,
                mr_enabled=mr_enabled,
            ).cuda()
            for ef_enabled, mr_enabled in variants
        ]
        self.assertTrue(all(type(model).__module__ == "mamba2" for model in models[:3]))
        self.assertTrue(type(models[-1]).__module__.startswith("mamba_ssm."))
    def test_delta_telemetry_supports_time_and_upstream_vanilla(self):
        from TiME import TiMEAgent
        from TiME.delta_telemetry import collect_policy_delta_telemetry

        probe = torch.linspace(-1.0, 1.0, 2 * 8 * 9, device="cuda").reshape(
            2, 8, 9
        )
        for ef_enabled, mr_enabled in ((True, True), (False, False)):
            with self.subTest(ef_enabled=ef_enabled, mr_enabled=mr_enabled):
                agent = TiMEAgent(
                    obs_size=9,
                    action_size=6,
                    entropy_cost=0.0,
                    discounting=0.98,
                    reward_scaling=1.0,
                    lambda_=0.95,
                    epsilon=0.2,
                    device="cuda",
                    ef_enabled=ef_enabled,
                    mr_enabled=mr_enabled,
                ).cuda()
                agent.train(True)
                before = {
                    name: value.detach().clone()
                    for name, value in agent.state_dict().items()
                }
                cpu_rng = torch.get_rng_state().clone()
                cuda_rng = torch.cuda.get_rng_state().clone()
                event = collect_policy_delta_telemetry(agent, probe, 0)

                self.assertEqual(len(event["layers"]), 2)
                self.assertEqual(event["probe"]["batch_size"], 2)
                self.assertTrue(agent.training)
                self.assertTrue(
                    all(
                        torch.equal(before[name], value)
                        for name, value in agent.state_dict().items()
                    )
                )
                self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
                self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
                for layer in event["layers"]:
                    generator = layer["generator"]
                    if mr_enabled:
                        self.assertIsNotNone(generator["gamma"])
                        self.assertIsNotNone(generator["sigma_estimate"])
                    else:
                        self.assertIsNone(generator["gamma"])
                        self.assertIsNone(generator["sigma_estimate"])

    def test_delta_telemetry_preserves_the_next_loss_exactly(self):
        from TiME import TiMEAgent
        from TiME.delta_telemetry import collect_policy_delta_telemetry

        batch_size = 2
        sequence_length = 8
        observation_size = 9
        action_size = 6
        observation = torch.linspace(
            -1.0,
            1.0,
            batch_size * (sequence_length + 1) * observation_size,
            device="cuda",
        ).reshape(batch_size, sequence_length + 1, observation_size)
        batch = {
            "observation": observation,
            "done": torch.zeros(
                batch_size, sequence_length, dtype=torch.bool, device="cuda"
            ),
            "truncation": torch.zeros(
                batch_size, sequence_length, dtype=torch.bool, device="cuda"
            ),
            "reward": torch.linspace(
                -0.5,
                0.5,
                batch_size * sequence_length,
                device="cuda",
            ).reshape(batch_size, sequence_length),
            "logits": torch.zeros(
                batch_size,
                sequence_length,
                action_size * 2,
                device="cuda",
            ),
            "action": torch.zeros(
                batch_size,
                sequence_length,
                action_size,
                device="cuda",
            ),
        }

        for ef_enabled, mr_enabled in ((True, True), (False, False)):
            with self.subTest(ef_enabled=ef_enabled, mr_enabled=mr_enabled):
                torch.manual_seed(17)
                torch.cuda.manual_seed_all(17)
                agent_with_telemetry = TiMEAgent(
                    obs_size=observation_size,
                    action_size=action_size,
                    entropy_cost=0.01,
                    discounting=0.98,
                    reward_scaling=1.0,
                    lambda_=0.95,
                    epsilon=0.2,
                    device="cuda",
                    ef_enabled=ef_enabled,
                    mr_enabled=mr_enabled,
                ).cuda()
                agent_without_telemetry = TiMEAgent(
                    obs_size=observation_size,
                    action_size=action_size,
                    entropy_cost=0.01,
                    discounting=0.98,
                    reward_scaling=1.0,
                    lambda_=0.95,
                    epsilon=0.2,
                    device="cuda",
                    ef_enabled=ef_enabled,
                    mr_enabled=mr_enabled,
                ).cuda()
                agent_without_telemetry.load_state_dict(
                    agent_with_telemetry.state_dict()
                )

                cpu_rng = torch.get_rng_state().clone()
                cuda_rng = torch.cuda.get_rng_state().clone()
                collect_policy_delta_telemetry(
                    agent_with_telemetry,
                    observation[:, :-1],
                    total_steps=0,
                )
                self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
                self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))

                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng)
                losses_with_telemetry = agent_with_telemetry.loss(batch)

                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng)
                losses_without_telemetry = agent_without_telemetry.loss(batch)

                for actual, expected in zip(
                    losses_with_telemetry, losses_without_telemetry
                ):
                    self.assertTrue(torch.equal(actual, expected))
    def test_reset_aware_forward_and_backward_are_finite(self):
        torch.manual_seed(0)
        inputs = torch.randn(2, 8, 256, device="cuda", requires_grad=True)
        resets = torch.zeros(2, 8, dtype=torch.bool, device="cuda")
        resets[0, 1] = True
        resets[1, 2] = True
        model = build_backbone(
            d_model=256,
            d_state=64,
            d_conv=4,
            expand=2,
            ef_enabled=True,
            mr_enabled=True,
        ).cuda()
        output = model(inputs, resets=resets)
        self.assertEqual(output.shape, inputs.shape)
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_unfused_local_backbone_works_without_causal_extension(self):
        import mamba2

        previous = mamba2.causal_conv1d_fn
        mamba2.causal_conv1d_fn = None
        try:
            model = mamba2.Mamba2(
                d_model=256,
                d_state=64,
                d_conv=4,
                expand=2,
                ef_enabled=True,
                mr_enabled=True,
                use_mem_eff_path=False,
            ).cuda()
            inputs = torch.randn(2, 8, 256, device="cuda", requires_grad=True)
            resets = torch.zeros(2, 8, dtype=torch.bool, device="cuda")
            resets[:, 4] = True
            output = model(inputs, resets=resets)
            self.assertEqual(output.shape, inputs.shape)
            self.assertTrue(torch.isfinite(output).all())
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(inputs.grad).all())
        finally:
            mamba2.causal_conv1d_fn = previous
    def test_brax_torch_bridge_runs_on_cuda(self):
        from env_utils import create_env

        env = create_env("swimmer", "cuda", seed=7, num_envs=2)
        observation = env.reset()
        action = torch.zeros(env.action_space.shape, device="cuda")
        next_observation, reward, done, _ = env.step(action)
        self.assertEqual(next_observation.shape, observation.shape)
        self.assertTrue(torch.isfinite(next_observation).all())
        self.assertEqual(reward.shape, done.shape)
    def test_long_horizon_masks_run_through_brax_cuda_bridge(self):
        from env_utils import create_env

        for env_name, keep_dims in (
            ("ant", [0, 1, 2, 3, 4]),
            ("halfcheetah", [0, 1, 2]),
        ):
            with self.subTest(env_name=env_name):
                env = create_env(
                    env_name,
                    "cuda",
                    seed=11,
                    num_envs=2,
                    episode_length=5_000,
                    keep_dims_override=keep_dims,
                )
                observation = env.reset()
                self.assertEqual(observation.shape, (2, len(keep_dims)))
                action = torch.zeros(env.action_space.shape, device="cuda")
                next_observation, reward, done, info = env.step(action)
                self.assertEqual(next_observation.shape, observation.shape)
                self.assertTrue(torch.isfinite(next_observation).all())
                self.assertEqual(reward.shape, done.shape)
                self.assertIn("truncation", info)
    def test_matched_agent_rebuilds_memory_rebalancer_buffers(self):
        from trainer import _build_matched_agent

        kwargs = {
            "obs_size": 4,
            "action_size": 2,
            "entropy_cost": 0.0,
            "discounting": 0.99,
            "reward_scaling": 1.0,
            "lambda_": 0.95,
            "epsilon": 0.2,
            "device": "cuda",
        }
        _, event = _build_matched_agent(
            kwargs,
            {"ef_enabled": True, "mr_enabled": True, "reset_dt": 5.0},
            "swimmer",
            0,
        )
        self.assertEqual(event["rebalanced_buffer_count"], 4)
        self.assertEqual(event["backbone_provenance"]["upstream"]["version"], "2.2.4")
    def test_attempt_completion_requires_provenance_and_two_evaluations(self):
        from artifacts import (
            append_history_event,
            initialize_run,
            read_json,
            run_history_path,
            write_json,
        )
        from run_brax_matrix import _validate_attempt_completion
        from TiME.provenance import backbone_provenance

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = "swimmer__time__seed-0"
            attempt_id = "attempt-0001"
            initialize_run(
                root,
                {
                    "run_id": run_id,
                    "mr_enabled": True,
                    "device": "cuda",
                    "recipe": {
                        "num_envs": 1,
                        "unroll_length": 1,
                        "num_timesteps": 10,
                        "evaluation_points": 2,
                        "evaluation_intermediate_episodes": 2,
                        "evaluation_final_episodes": 3,
                    },
                },
            )
            for event in (
                {
                    "event": "attempt_started",
                    "device": "cuda",
                },
                {
                    "event": "shared_initialization_verified",
                    "backbone_provenance": backbone_provenance(),
                    "shared_initialization_digest": "a" * 64,
                    "shared_tensor_count": 56,
                    "rebalanced_buffer_count": 4,
                },
                {
                    "event": "evaluation",
                    "total_steps": 0,
                    "evaluation_kind": "intermediate",
                    "evaluation_episodes": 2,
                    "episode_return": [1.0, 1.0],
                    "evaluation_mean_return": 1.0,
                },
                {
                    "event": "evaluation",
                    "total_steps": 10,
                    "evaluation_kind": "final",
                    "evaluation_episodes": 3,
                    "episode_return": [2.0, 2.0, 2.0],
                    "evaluation_mean_return": 2.0,
                },
            ):
                append_history_event(
                    root, run_id, {**event, "attempt_id": attempt_id}
                )
            _validate_attempt_completion(root, run_id, attempt_id)
            path = run_history_path(root, run_id)
            history = read_json(path)
            history["events"] = history["events"][1:]
            write_json(path, history)
            with self.assertRaises(RuntimeError):
                _validate_attempt_completion(root, run_id, attempt_id)
    def test_strict_aggregation_validates_current_attempt_provenance(self):
        from aggregate_results import aggregate
        from artifacts import (
            append_history_event,
            initialize_run,
            write_matrix_manifest,
            write_status,
        )
        from TiME.provenance import backbone_provenance
        from experiment_config import (
            expected_delta_telemetry_steps,
            expected_evaluation_plan,
        )
        from run_brax_matrix import matrix_manifest

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = matrix_manifest(5.0)
            config = manifest["runs"][0]
            run_id = config["run_id"]
            attempt_id = "attempt-0001"
            write_matrix_manifest(root, manifest)
            initialize_run(root, config)
            append_history_event(
                root,
                run_id,
                {
                    "event": "attempt_started",
                    "attempt_id": attempt_id,
                    "device": "cuda",
                },
            )
            append_history_event(
                root,
                run_id,
                {
                    "event": "shared_initialization_verified",
                    "attempt_id": attempt_id,
                    "backbone_provenance": backbone_provenance(),
                    "shared_initialization_digest": "a" * 64,
                    "shared_tensor_count": 56,
                    "rebalanced_buffer_count": 4,
                },
            )
            for step, evaluation_spec in sorted(
                expected_evaluation_plan(config).items()
            ):
                episode_returns = [
                    float(step)
                    for _ in range(evaluation_spec["evaluation_episodes"])
                ]
                append_history_event(
                    root,
                    run_id,
                    {
                        "event": "evaluation",
                        "attempt_id": attempt_id,
                        "total_steps": step,
                        **evaluation_spec,
                        "episode_return": episode_returns,
                        "evaluation_mean_return": float(step),
                    },
                )
            telemetry_summary = {
                "count": 4,
                "mean": 1.0,
                "std": 0.1,
                "min": 0.5,
                "max": 1.5,
                "p05": 0.6,
                "p50": 1.0,
                "p95": 1.4,
            }
            for step in sorted(expected_delta_telemetry_steps(config)):
                append_history_event(
                    root,
                    run_id,
                    {
                        "event": "delta_telemetry",
                        "attempt_id": attempt_id,
                        "step": step,
                        "total_steps": step,
                        "probe": {
                            "batch_size": 8,
                            "sequence_length": 128,
                            "observation_size": 13,
                            "dtype": "torch.float32",
                            "source": "first_training_rollout_after_normalization",
                        },
                        "reset_policy": "all_false_fixed_probe",
                        "layers": [
                            {
                                "layer_index": 0,
                                "delta": telemetry_summary,
                                "timescale_steps": telemetry_summary,
                                "generator": {
                                    "nheads": 8,
                                    "dt_bias_mean": -3.0,
                                    "raw_projection_spectral_norm": 1.0,
                                    "effective_projection_spectral_norm": 1.0,
                                    "gamma": 1.0,
                                    "sigma_estimate": 1.0,
                                },
                            }
                        ],
                    },
                )
            write_status(root, run_id, "running", attempt_id=attempt_id)
            write_status(root, run_id, "completed", attempt_id=attempt_id)
            completeness = aggregate(root)["completeness"]
            self.assertFalse(completeness["complete"])
            self.assertEqual(completeness["completed_runs"], 1)
            self.assertEqual(len(completeness["missing_run_ids"]), 79)
            self.assertEqual(completeness["invalid_run_ids"], [])
    def test_completed_reuse_is_bound_to_backbone_provenance(self):
        from artifacts import append_history_event, initialize_run, write_status
        from run_brax_matrix import _validate_completed_provenance
        from TiME.provenance import backbone_provenance

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = "swimmer__time__seed-0"
            attempt_id = "attempt-0001"
            initialize_run(
                root,
                {
                    "run_id": run_id,
                    "mr_enabled": True,
                    "device": "cuda",
                    "recipe": {
                        "num_envs": 1,
                        "unroll_length": 1,
                        "num_timesteps": 10,
                        "evaluation_points": 2,
                        "evaluation_intermediate_episodes": 2,
                        "evaluation_final_episodes": 3,
                    },
                },
            )
            for event in (
                {
                    "event": "attempt_started",
                    "device": "cuda",
                },
                {
                    "event": "shared_initialization_verified",
                    "backbone_provenance": backbone_provenance(),
                    "shared_initialization_digest": "a" * 64,
                    "shared_tensor_count": 56,
                    "rebalanced_buffer_count": 4,
                },
                {
                    "event": "evaluation",
                    "total_steps": 0,
                    "evaluation_kind": "intermediate",
                    "evaluation_episodes": 2,
                    "episode_return": [1.0, 1.0],
                    "evaluation_mean_return": 1.0,
                },
                {
                    "event": "evaluation",
                    "total_steps": 10,
                    "evaluation_kind": "final",
                    "evaluation_episodes": 3,
                    "episode_return": [2.0, 2.0, 2.0],
                    "evaluation_mean_return": 2.0,
                },
            ):
                append_history_event(
                    root, run_id, {**event, "attempt_id": attempt_id}
                )
            write_status(root, run_id, "running", attempt_id=attempt_id)
            write_status(root, run_id, "completed", attempt_id=attempt_id)
            _validate_completed_provenance(root, run_id)

            from artifacts import read_json, run_history_path, write_json

            path = run_history_path(root, run_id)
            history = read_json(path)
            initialization = next(
                event
                for event in history["events"]
                if event.get("event") == "shared_initialization_verified"
            )
            initialization["backbone_provenance"]["upstream"]["source_sha256"] = "0" * 64
            write_json(path, history)
            with self.assertRaises(RuntimeError):
                _validate_completed_provenance(root, run_id)


if __name__ == "__main__":
    unittest.main()
