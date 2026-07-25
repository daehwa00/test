"""Immutable corrected-v2 Brax experiment definitions."""

from __future__ import annotations
import hashlib
import json
from types import MappingProxyType
import math

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

PROTOCOL = "time-brax-corrected-v2"
EXECUTION_DEVICE = "cuda"
TRAINING_SEEDS = tuple(range(5))
TASKS = (
    "halfcheetah",
    "hopper",
    "ant",
    "walker2d",
    "swimmer",
    "reacher",
    "pusher",
    "humanoidstandup",
)

# These are deliberately literal copies of the submitted Brax recipes.  Keeping
# them here means a matrix manifest remains self-contained if runtime defaults change.
TASK_RECIPES: Mapping[str, Mapping[str, Any]] = {
    "halfcheetah": {"discounting": 0.98, "lambda_": 0.92, "num_envs": 64, "unroll_length": 256, "entropy_cost": 0.0005558684366667077, "learning_rate": 0.00136509431273925, "reward_scaling": 0.32044334240011263, "num_timesteps": 3_000_000},
    "hopper": {"discounting": 0.99, "lambda_": 0.95, "num_envs": 64, "unroll_length": 128, "entropy_cost": 0.00001430586420329152, "learning_rate": 0.00010018555363337202, "reward_scaling": 0.13706167636580946, "num_timesteps": 1_000_000},
    "ant": {"discounting": 0.98, "lambda_": 0.93, "num_envs": 64, "unroll_length": 128, "entropy_cost": 0.00003698473904078169, "learning_rate": 0.0004037540234187324, "reward_scaling": 0.38578535484188625, "num_timesteps": 3_000_000},
    "walker2d": {"discounting": 0.995, "lambda_": 0.98, "num_envs": 32, "unroll_length": 128, "entropy_cost": 0.00003066781225197179, "learning_rate": 0.00021918420087285032, "reward_scaling": 0.2281923341486558, "num_timesteps": 1_000_000},
    "swimmer": {"discounting": 0.98, "lambda_": 0.99, "num_envs": 32, "unroll_length": 64, "entropy_cost": 0.00003430319237955782, "learning_rate": 0.0012626482323111997, "reward_scaling": 1.0, "num_timesteps": 1_000_000},
    "reacher": {"discounting": 0.995, "lambda_": 0.98, "num_envs": 32, "unroll_length": 64, "entropy_cost": 0.00023869519228979816, "learning_rate": 0.0005078717564188115, "reward_scaling": 0.12877286712431638, "num_timesteps": 1_000_000},
    "pusher": {"discounting": 0.99, "lambda_": 0.95, "num_envs": 32, "unroll_length": 128, "entropy_cost": 0.0009387007322881968, "learning_rate": 0.00037492908011401826, "reward_scaling": 0.26359850133423257, "num_timesteps": 1_000_000},
    "humanoidstandup": {"discounting": 0.99, "lambda_": 0.8, "num_envs": 32, "unroll_length": 256, "entropy_cost": 0.00001250166915452103, "learning_rate": 0.0006849469830690655, "reward_scaling": 0.17345286822592065, "num_timesteps": 1_000_000},
}
TRAINER_DEFAULTS: Mapping[str, Any] = {
    "epsilon": 0.2,
    "num_minibatches": 4,
    "num_update_epochs": 4,
    "evaluation_points": 6,
    "evaluation_intermediate_episodes": 16,
    "evaluation_final_episodes": 64,
    "delta_telemetry_enabled": True,
    "delta_telemetry_points": 6,
    "delta_telemetry_probe_sequences": 8,
}


@dataclass(frozen=True)
class Variant:
    name: str
    ef_enabled: bool
    mr_enabled: bool

VARIANTS = (
    Variant("time", True, True),
    Variant("ef-only", True, False),
    Variant("mr-only", False, True),
    Variant("vanilla-mamba2", False, False),
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}
PRIMARY_VARIANTS = (VARIANT_BY_NAME["time"], VARIANT_BY_NAME["vanilla-mamba2"])

LONG_HORIZON_PROTOCOL = "time-brax-corrected-v2-long-horizon"
LONG_HORIZON_PROFILE = "much_stronger"
LONG_HORIZON_TASKS = ("ant", "halfcheetah")
LONG_HORIZON_KEEP_DIMS: Mapping[str, tuple[int, ...]] = MappingProxyType({
    "ant": (0, 1, 2, 3, 4),
    "halfcheetah": (0, 1, 2),
})
LONG_HORIZON_MASK_NAME = "much_stronger"
LONG_HORIZON_TRAINER_OVERRIDES: Mapping[str, Any] = MappingProxyType({
    "episode_length": 5_000,
    "num_timesteps": 3_000_000,
    "num_envs": 256,
    "unroll_length": 256,
    "num_update_epochs": 2,
    "num_minibatches": 8,
})
LONG_HORIZON_VARIANTS = PRIMARY_VARIANTS

@dataclass(frozen=True)
class RunConfig:
    task: str
    variant: Variant
    seed: int
    reset_dt: float = 5.0
    @property
    def effective_reset_dt(self) -> float:
        return self.reset_dt if self.variant.ef_enabled else 5.0


    @property
    def identity_payload(self) -> Dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "task": self.task,
            "variant": self.variant.name,
            "seed": self.seed,
            "ef_enabled": self.variant.ef_enabled,
            "mr_enabled": self.variant.mr_enabled,
            "reset_dt": self.effective_reset_dt,
            "device": EXECUTION_DEVICE,
            "recipe": {**TASK_RECIPES[self.task], **TRAINER_DEFAULTS},
        }

    @property
    def run_id(self) -> str:
        canonical = json.dumps(
            self.identity_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()[:16]
        return f"{self.task}__{self.variant.name}__seed-{self.seed}__{digest}"

    def to_dict(self) -> Dict[str, Any]:
        return {**self.identity_payload, "run_id": self.run_id}

    def trainer_config(self, device: str, use_wandb: bool = False) -> Dict[str, Any]:
        if device != EXECUTION_DEVICE:
            raise ValueError(f"corrected-v2 runs require device={EXECUTION_DEVICE!r}")
        return {
            **TASK_RECIPES[self.task],
            **TRAINER_DEFAULTS,
            "device": device,
            "seed": self.seed,
            "env_name": self.task,
            "use_wandb": use_wandb,
            "ef_enabled": self.variant.ef_enabled,
            "mr_enabled": self.variant.mr_enabled,
            "reset_dt": self.effective_reset_dt,
        }
@dataclass(frozen=True)
class LongHorizonRunConfig:
    task: str
    variant: Variant
    seed: int
    reset_dt: float = 5.0

    def __post_init__(self) -> None:
        if self.task not in LONG_HORIZON_TASKS:
            raise ValueError(f"unsupported long-horizon task: {self.task!r}")
        if self.variant not in LONG_HORIZON_VARIANTS:
            raise ValueError(
                "long-horizon runs support only TiME and vanilla-Mamba2 variants"
            )
        if self.seed not in TRAINING_SEEDS:
            raise ValueError("long-horizon seeds must be in 0..4")
        if not math.isfinite(self.reset_dt) or self.reset_dt <= 0:
            raise ValueError("reset_dt must be finite and positive")

    @property
    def effective_reset_dt(self) -> float:
        return self.reset_dt if self.variant.ef_enabled else 5.0

    @property
    def keep_dims_override(self) -> tuple[int, ...]:
        return LONG_HORIZON_KEEP_DIMS[self.task]

    @property
    def mask_name(self) -> str:
        return LONG_HORIZON_MASK_NAME

    @property
    def recipe(self) -> Dict[str, Any]:
        return {
            **TASK_RECIPES[self.task],
            **TRAINER_DEFAULTS,
            **LONG_HORIZON_TRAINER_OVERRIDES,
            "keep_dims_override": list(self.keep_dims_override),
        }

    @property
    def identity_payload(self) -> Dict[str, Any]:
        return {
            "protocol": LONG_HORIZON_PROTOCOL,
            "profile": LONG_HORIZON_PROFILE,
            "task": self.task,
            "episode_length": LONG_HORIZON_TRAINER_OVERRIDES["episode_length"],
            "mask_name": self.mask_name,
            "keep_dims_override": list(self.keep_dims_override),
            "variant": self.variant.name,
            "seed": self.seed,
            "ef_enabled": self.variant.ef_enabled,
            "mr_enabled": self.variant.mr_enabled,
            "reset_dt": self.effective_reset_dt,
            "device": EXECUTION_DEVICE,
            "recipe": self.recipe,
        }

    @property
    def run_id(self) -> str:
        canonical = json.dumps(
            self.identity_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()[:16]
        dims = "-".join(map(str, self.keep_dims_override))
        return (
            f"{self.task}__ep-{LONG_HORIZON_TRAINER_OVERRIDES['episode_length']}"
            f"__{self.mask_name}-dims-{dims}__{self.variant.name}"
            f"__seed-{self.seed}__{digest}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {**self.identity_payload, "run_id": self.run_id}

    def trainer_config(self, device: str, use_wandb: bool = False) -> Dict[str, Any]:
        if device != EXECUTION_DEVICE:
            raise ValueError(
                f"long-horizon runs require device={EXECUTION_DEVICE!r}"
            )
        return {
            **self.recipe,
            "device": device,
            "seed": self.seed,
            "env_name": self.task,
            "use_wandb": use_wandb,
            "protocol": LONG_HORIZON_PROTOCOL,
            "profile": LONG_HORIZON_PROFILE,
            "mask_name": self.mask_name,
            "ef_enabled": self.variant.ef_enabled,
            "mr_enabled": self.variant.mr_enabled,
            "reset_dt": self.effective_reset_dt,
        }


def build_long_horizon_matrix(
    reset_dt: float = 5.0,
) -> Iterable[LongHorizonRunConfig]:
    if not math.isfinite(reset_dt) or reset_dt <= 0:
        raise ValueError("reset_dt must be finite and positive")
    for task in LONG_HORIZON_TASKS:
        for variant in LONG_HORIZON_VARIANTS:
            for seed in TRAINING_SEEDS:
                yield LongHorizonRunConfig(task, variant, seed, reset_dt)


def expected_evaluation_steps(config: Mapping[str, Any]) -> set[int]:
    recipe = config["recipe"]
    num_steps = int(recipe["num_envs"]) * int(recipe["unroll_length"])
    num_epochs = int(recipe["num_timesteps"]) // num_steps
    evaluation_points = int(recipe["evaluation_points"])
    if evaluation_points < 2:
        raise ValueError("evaluation_points must be at least 2")
    update_indices = {
        (index * num_epochs) // (evaluation_points - 1)
        for index in range(evaluation_points)
    }
    return {index * num_steps for index in update_indices}


def expected_evaluation_plan(config: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    recipe = config["recipe"]
    steps = sorted(expected_evaluation_steps(config))
    final_step = steps[-1]
    return {
        step: {
            "evaluation_kind": "final" if step == final_step else "intermediate",
            "evaluation_episodes": int(
                recipe[
                    "evaluation_final_episodes"
                    if step == final_step
                    else "evaluation_intermediate_episodes"
                ]
            ),
        }
        for step in steps
    }


def expected_delta_telemetry_steps(config: Mapping[str, Any]) -> set[int]:
    recipe = config["recipe"]
    if not recipe.get("delta_telemetry_enabled", False):
        return set()
    num_steps = int(recipe["num_envs"]) * int(recipe["unroll_length"])
    num_epochs = int(recipe["num_timesteps"]) // num_steps
    telemetry_points = int(recipe["delta_telemetry_points"])
    if telemetry_points < 2:
        raise ValueError("delta_telemetry_points must be at least 2")
    update_indices = {
        (index * num_epochs) // (telemetry_points - 1)
        for index in range(telemetry_points)
    }
    return {index * num_steps for index in update_indices}

def build_matrix(
    reset_dt: float = 5.0,
    variants: Iterable[Variant] = PRIMARY_VARIANTS,
    seeds: Iterable[int] = TRAINING_SEEDS,
) -> Iterable[RunConfig]:
    if not math.isfinite(reset_dt) or reset_dt <= 0:
        raise ValueError("reset_dt must be finite and positive")
    selected_variants = tuple(variants)
    selected_seeds = tuple(seeds)
    for task in TASKS:
        for variant in selected_variants:
            for seed in selected_seeds:
                yield RunConfig(task, variant, seed, reset_dt)
