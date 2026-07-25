import json
import pathlib
import sys
import unittest

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - exercised on torch-free installations
    torch = None
    nn = None

_Module = nn.Module if nn is not None else object

if torch is not None:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from TiME.delta_telemetry import collect_policy_delta_telemetry, summarize_delta_drift


@unittest.skipIf(torch is None, "torch is unavailable")
class DeltaDriftSummaryTests(unittest.TestCase):
    @staticmethod
    def _event(step, mean, std, median, norm):
        summary = {"count": 4, "mean": mean, "std": std, "min": 1.0, "max": 4.0, "p05": 1.0, "p50": median, "p95": 4.0}
        return {
            "event": "delta_telemetry",
            "step": step,
            "total_steps": step,
            "probe": {
                "batch_size": 2,
                "sequence_length": 4,
                "observation_size": 3,
                "dtype": "torch.float32",
                "source": "first_training_rollout_after_normalization",
            },
            "reset_policy": "all_false_fixed_probe",
            "layers": [{"layer_index": 0, "delta": summary, "timescale_steps": summary, "generator": {"nheads": 2, "dt_bias_mean": -1.0, "raw_projection_spectral_norm": norm, "effective_projection_spectral_norm": norm, "gamma": None, "sigma_estimate": None}}],
        }

    def test_summary_reports_exact_changes_and_slope(self):
        result = summarize_delta_drift([self._event(100, 2.0, 0.5, 8.0, 3.0), self._event(600, 3.5, 0.75, 5.0, 4.0)])
        layer = result["layers"][0]
        self.assertEqual(layer["points"], 2)
        self.assertEqual(layer["delta_mean_change"], 1.5)
        self.assertEqual(layer["delta_mean_slope_per_million_steps"], 3000.0)
        self.assertEqual(layer["timescale_median_change"], -3.0)
        self.assertEqual(layer["effective_projection_norm_change"], 1.0)

    def test_summary_rejects_duplicate_steps_and_inconsistent_layers(self):
        with self.assertRaises(ValueError):
            summarize_delta_drift([self._event(1, 1.0, 1.0, 1.0, 1.0), self._event(1, 2.0, 1.0, 1.0, 1.0)])
        altered = self._event(2, 2.0, 1.0, 1.0, 1.0)
        altered["layers"][0]["layer_index"] = 1
        with self.assertRaises(ValueError):
            summarize_delta_drift([self._event(1, 1.0, 1.0, 1.0, 1.0), altered])
        malformed = self._event(1, float("nan"), 1.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            summarize_delta_drift([malformed])


@unittest.skipIf(torch is None, "torch is unavailable")
class DeltaTelemetryTests(unittest.TestCase):
    class FakeMamba(_Module):
        def __init__(self):
            super().__init__()
            self.nheads = 2
            self.in_proj = nn.Linear(3, 6, bias=False)
            self.dt_bias = nn.Parameter(torch.tensor([-1.0, -0.5]))
            self.A_log = nn.Parameter(torch.tensor([0.0, 0.5]))

        def forward(self, value):
            return self.in_proj(value)

    class Wrapper(_Module):
        def __init__(self, mamba):
            super().__init__()
            self.mamba = mamba

    class FakeAgent(_Module):
        def __init__(self, fail=False):
            super().__init__()
            self.policy_encoder = nn.Module()
            self.policy_encoder.mambas = nn.ModuleList([DeltaTelemetryTests.Wrapper(DeltaTelemetryTests.FakeMamba())])
            self.fail = fail

        def policy(self, observation, rollout, resets):
            assert rollout is False
            assert resets.dtype == torch.bool and not resets.any()
            output = self.policy_encoder.mambas[0].mamba(observation)
            if self.fail:
                raise RuntimeError("forced policy failure")
            return output

    def test_collection_is_deterministic_and_observational(self):
        torch.manual_seed(7)
        agent = self.FakeAgent()
        agent.train(True)
        agent.policy_encoder.mambas[0].mamba.train(False)
        before = {name: value.detach().clone() for name, value in agent.state_dict().items()}
        modes = {id(module): module.training for module in agent.modules()}
        rng = torch.get_rng_state().clone()
        probe = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / 10
        first = collect_policy_delta_telemetry(agent, probe, 11)
        second = collect_policy_delta_telemetry(agent, probe, 11)
        self.assertEqual(first, second)
        self.assertEqual(first["reset_policy"], "all_false_fixed_probe")
        self.assertEqual(first["probe"]["batch_size"], 2)
        self.assertEqual(first["probe"]["sequence_length"], 4)
        self.assertEqual(first["probe"]["observation_size"], 3)
        json.dumps(first, allow_nan=False)
        layer = first["layers"][0]
        self.assertEqual(layer["delta"]["count"], 16)
        self.assertIsNone(layer["generator"]["gamma"])
        self.assertIsNone(layer["generator"]["sigma_estimate"])
        self.assertTrue(all(isinstance(value, (int, float, type(None), str, list, dict)) for value in layer["generator"].values()))
        self.assertTrue(all(torch.equal(before[name], value) for name, value in agent.state_dict().items()))
        self.assertEqual(modes, {id(module): module.training for module in agent.modules()})
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(agent.policy_encoder.mambas[0].mamba.in_proj._forward_hooks)

    def test_hooks_and_training_mode_are_restored_after_forward_error(self):
        agent = self.FakeAgent(fail=True)
        agent.train(True)
        modes = {id(module): module.training for module in agent.modules()}
        with self.assertRaisesRegex(RuntimeError, "forced policy failure"):
            collect_policy_delta_telemetry(agent, torch.ones(1, 2, 3), 1)
        self.assertEqual(modes, {id(module): module.training for module in agent.modules()})
        self.assertFalse(agent.policy_encoder.mambas[0].mamba.in_proj._forward_hooks)
