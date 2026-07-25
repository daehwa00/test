"""Bounded Aerodrone H200 compatibility probe for the TiME Brax stack.

This is intentionally not a scientific campaign. It prints one compact JSON receipt,
exercises both matched backbones, the Brax/Torch bridge, and one tiny corrected-v2
training update, then exits non-zero if any runtime stage fails.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import io
import json
import platform
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EXPECTED_QLAB_VERSIONS = {
    "brax": "0.12.3",
    "causal-conv1d": "1.5.0.post8",
    "jax": "0.6.0",
    "jaxlib": "0.6.0",
    "mamba-ssm": "2.2.4",
    "torch": "2.6.0+cu124",
    "triton": "3.2.0",
}


@dataclass
class Receipt:
    schema_version: int = 1
    kind: str = "time-h200-compatibility-smoke"
    runtime: dict = field(default_factory=dict)
    versions: dict = field(default_factory=dict)
    version_match: dict = field(default_factory=dict)
    stages: list[dict] = field(default_factory=list)
    scientific_compatible_with_qlab: bool = False
    passed: bool = False

    def stage(self, name: str, operation: Callable[[], dict]) -> None:
        try:
            evidence = operation()
        except Exception as exc:  # The receipt must survive dependency/kernel failures.
            self.stages.append({
                "name": name,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
                "traceback_tail": traceback.format_exc().splitlines()[-8:],
            })
        else:
            self.stages.append({"name": name, "status": "passed", **evidence})


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_runtime(receipt: Receipt) -> None:
    import torch

    receipt.runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")


def check_jax_cuda() -> dict:
    import jax

    devices = [str(device) for device in jax.devices()]
    if not any(getattr(device, "platform", None) == "gpu" for device in jax.devices()):
        raise RuntimeError(f"JAX has no GPU device: {devices}")
    return {"devices": devices}


def check_backbones() -> dict:
    import torch

    from TiME.backbones import build_backbone

    implementations = {}
    for name, ef_enabled, mr_enabled in (
        ("time", True, True),
        ("vanilla-mamba2", False, False),
    ):
        model = build_backbone(
            d_model=256,
            d_state=64,
            d_conv=4,
            expand=2,
            ef_enabled=ef_enabled,
            mr_enabled=mr_enabled,
        ).cuda()
        inputs = torch.randn(2, 8, 256, device="cuda", requires_grad=True)
        resets = torch.zeros(2, 8, dtype=torch.bool, device="cuda")
        resets[:, 4] = True
        output = model(inputs, resets=resets) if ef_enabled else model(inputs)
        loss = output.square().mean()
        loss.backward()
        if output.shape != inputs.shape or not torch.isfinite(output).all():
            raise RuntimeError(f"{name} produced an invalid output")
        if inputs.grad is None or not torch.isfinite(inputs.grad).all():
            raise RuntimeError(f"{name} produced an invalid gradient")
        implementations[name] = type(model).__module__
        del model, inputs, output, loss
    torch.cuda.synchronize()
    return {"implementations": implementations}


def check_brax_torch_bridge() -> dict:
    import torch

    from env_utils import create_env

    env = create_env(
        "ant",
        "cuda",
        seed=7,
        num_envs=2,
        episode_length=5_000,
        keep_dims_override=[0, 1, 2, 3, 4],
    )
    observation = env.reset()
    next_observation, reward, done, info = env.step(
        torch.zeros(env.action_space.shape, device="cuda")
    )
    if observation.shape != (2, 5) or next_observation.shape != observation.shape:
        raise RuntimeError("unexpected masked Ant observation shape")
    if not torch.isfinite(next_observation).all() or not torch.isfinite(reward).all():
        raise RuntimeError("Brax/Torch bridge produced non-finite values")
    return {
        "observation_shape": list(observation.shape),
        "reward_shape": list(reward.shape),
        "done_shape": list(done.shape),
        "has_truncation": "truncation" in info,
    }


def check_one_update() -> dict:
    from env_utils import ENV_HYPERPARAMS
    from trainer import train

    events: list[dict] = []
    config = {
        **ENV_HYPERPARAMS["ant"],
        "protocol": "time-brax-corrected-v2-h200-smoke",
        "device": "cuda",
        "seed": 0,
        "env_name": "ant",
        "use_wandb": False,
        "ef_enabled": True,
        "mr_enabled": True,
        "reset_dt": 5.0,
        "epsilon": 0.2,
        "episode_length": 5_000,
        "keep_dims_override": [0, 1, 2, 3, 4],
        "num_envs": 8,
        "unroll_length": 32,
        "num_timesteps": 256,
        "num_minibatches": 2,
        "num_update_epochs": 1,
        "evaluation_points": 2,
        "evaluation_intermediate_episodes": 2,
        "evaluation_final_episodes": 2,
        "delta_telemetry_enabled": True,
        "delta_telemetry_points": 2,
        "delta_telemetry_probe_sequences": 2,
        "history_recorder": events.append,
    }
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        train(config=config, env_name="ant")
    event_counts: dict[str, int] = {}
    for event in events:
        name = str(event.get("event", "unknown"))
        event_counts[name] = event_counts.get(name, 0) + 1
    required = {"shared_initialization_verified", "evaluation", "delta_telemetry"}
    missing = sorted(required - event_counts.keys())
    if missing:
        raise RuntimeError(f"training receipt is missing events: {missing}")
    return {
        "num_timesteps": config["num_timesteps"],
        "episode_length": config["episode_length"],
        "event_counts": event_counts,
        "captured_stdout_tail": captured.getvalue().splitlines()[-8:],
    }


def main() -> int:
    receipt = Receipt()
    receipt.versions = {
        name: installed_version(name) for name in EXPECTED_QLAB_VERSIONS
    }
    receipt.version_match = {
        name: receipt.versions[name] == expected
        for name, expected in EXPECTED_QLAB_VERSIONS.items()
    }
    receipt.stage("cuda-runtime", lambda: (collect_runtime(receipt) or receipt.runtime))
    if receipt.stages[-1]["status"] == "passed":
        receipt.stage("jax-cuda", check_jax_cuda)
        receipt.stage("time-and-vanilla-backbones", check_backbones)
        receipt.stage("brax-torch-dlpack", check_brax_torch_bridge)
        receipt.stage("corrected-v2-one-update", check_one_update)
    receipt.scientific_compatible_with_qlab = all(receipt.version_match.values())
    receipt.passed = bool(receipt.stages) and all(
        stage["status"] == "passed" for stage in receipt.stages
    )
    print("H200_SMOKE_RESULT=" + json.dumps(receipt.__dict__, sort_keys=True))
    return 0 if receipt.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
