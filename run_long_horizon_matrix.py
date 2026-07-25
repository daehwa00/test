"""Safe launcher for the corrected-v2 long-horizon TiME-vs-Mamba2 matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from artifacts import write_matrix_manifest
from experiment_config import (
    EXECUTION_DEVICE,
    LONG_HORIZON_PROFILE,
    LONG_HORIZON_KEEP_DIMS,
    LONG_HORIZON_PROTOCOL,
    LONG_HORIZON_TASKS,
    LONG_HORIZON_VARIANTS,
    LONG_HORIZON_TRAINER_OVERRIDES,
    TRAINING_SEEDS,
    build_long_horizon_matrix,
)
from run_brax_matrix import _run_one


def matrix_manifest(reset_dt: float) -> dict[str, Any]:
    runs = [run.to_dict() for run in build_long_horizon_matrix(reset_dt)]
    return {
        "protocol": LONG_HORIZON_PROTOCOL,
        "profile": LONG_HORIZON_PROFILE,
        "tasks": list(LONG_HORIZON_TASKS),
        "variants": [
            {
                "name": variant.name,
                "ef_enabled": variant.ef_enabled,
                "mr_enabled": variant.mr_enabled,
            }
            for variant in LONG_HORIZON_VARIANTS
        ],
        "training_seeds": list(TRAINING_SEEDS),
        "episode_length": LONG_HORIZON_TRAINER_OVERRIDES["episode_length"],
        "keep_dims": {
            task: list(dims) for task, dims in LONG_HORIZON_KEEP_DIMS.items()
        },
        "trainer_overrides": dict(LONG_HORIZON_TRAINER_OVERRIDES),
        "reset_dt": reset_dt,
        "run_count": len(runs),
        "runs": runs,
    }


def execute_matrix(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    manifest = matrix_manifest(args.reset_dt)
    write_matrix_manifest(output_dir, manifest)
    if not args.execute:
        print(
            f"Dry run: wrote {manifest['run_count']} deterministic long-horizon "
            f"run keys to {output_dir}"
        )
        return 0
    if not args.authorize_long_horizon_execution:
        raise RuntimeError(
            "Long-horizon execution requires --authorize-long-horizon-execution"
        )

    failed = 0
    for run in build_long_horizon_matrix(args.reset_dt):
        failed += _run_one(output_dir, run, args.device, args.use_wandb, args.resume)
    if failed:
        raise RuntimeError(
            f"{failed} long-horizon runs failed; inspect per-run status.json artifacts"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="results/time-brax-corrected-v2-long-horizon"
    )
    parser.add_argument("--reset-dt", type=float, default=5.0)
    parser.add_argument("--device", choices=(EXECUTION_DEVICE,), default=EXECUTION_DEVICE)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument(
        "--execute", action="store_true", help="Run training instead of the default dry-run"
    )
    parser.add_argument(
        "--authorize-long-horizon-execution",
        action="store_true",
        help="Required acknowledgement before the 20 long-horizon training runs start",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Restart incomplete or failed existing runs"
    )
    return parser


def main() -> int:
    return execute_matrix(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
