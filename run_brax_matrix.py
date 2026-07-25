"""Safe launcher for the corrected-v2 TiME-vs-Mamba2 Brax matrix."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

from artifacts import (
    BufferedHistoryRecorder,
    append_history_event,
    initialize_run,
    read_json,
    run_history_path,
    run_config_path,
    run_status_history_path,
    read_status,
    write_matrix_manifest,
    write_status,
)
from experiment_config import (
    EXECUTION_DEVICE,
    PRIMARY_VARIANTS,
    PROTOCOL,
    TASKS,
    TRAINING_SEEDS,
    VARIANTS,
    build_matrix,
    expected_evaluation_plan,
    expected_delta_telemetry_steps,
)
from metrics import evaluation_events_match_plan, summarize_history


def matrix_manifest(reset_dt: float) -> dict[str, Any]:
    runs = [run.to_dict() for run in build_matrix(reset_dt)]
    return {
        "protocol": PROTOCOL,
        "tasks": list(TASKS),
        "variants": [
            {
                "name": variant.name,
                "ef_enabled": variant.ef_enabled,
                "mr_enabled": variant.mr_enabled,
            }
            for variant in PRIMARY_VARIANTS
        ],
        "available_ablation_variants": [variant.name for variant in VARIANTS],
        "training_seeds": list(TRAINING_SEEDS),
        "reset_dt": reset_dt,
        "run_count": len(runs),
        "runs": runs,
    }


def _provenance_projection(value: dict[str, Any]) -> dict[str, Any]:
    return {
        side: {
            key: item
            for key, item in fields.items()
            if not key.endswith("_path")
        }
        for side, fields in value.items()
    }


def _validate_completed_provenance(output_dir: Path, run_id: str) -> None:
    status = read_status(output_dir, run_id)
    attempt_id = status.get("attempt_id") if status else None
    if status is None or status.get("state") != "completed" or not attempt_id:
        raise RuntimeError(f"completed run {run_id} lacks a bound attempt")
    _validate_attempt_completion(output_dir, run_id, str(attempt_id))


def _validate_attempt_completion(
    output_dir: Path, run_id: str, attempt_id: str
) -> None:
    from TiME.provenance import backbone_provenance

    history = read_json(run_history_path(output_dir, run_id))
    events = [
        event
        for event in history.get("events", [])
        if event.get("attempt_id") == attempt_id
    ]
    initialization = [
        event
        for event in events
        if event.get("event") == "shared_initialization_verified"
        and event.get("backbone_provenance")
    ]
    if len(initialization) != 1:
        raise RuntimeError(f"attempt {attempt_id} lacks unique provenance evidence")
    recorded = _provenance_projection(initialization[0]["backbone_provenance"])
    current = _provenance_projection(dict(backbone_provenance()))
    if recorded != current:
        raise RuntimeError(f"attempt {attempt_id} provenance mismatch")
    initialization_event = initialization[0]
    digest = initialization_event.get("shared_initialization_digest")
    config = read_json(run_config_path(output_dir, run_id))
    attempt_starts = [
        event for event in events if event.get("event") == "attempt_started"
    ]
    if (
        len(attempt_starts) != 1
        or attempt_starts[0].get("device") != config.get("device")
    ):
        raise RuntimeError(f"attempt {attempt_id} lacks matching device evidence")
    expected_rebalanced = 4 if config.get("mr_enabled") else 0
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or initialization_event.get("shared_tensor_count", 0) <= 0
        or initialization_event.get("rebalanced_buffer_count")
        != expected_rebalanced
    ):
        raise RuntimeError(f"attempt {attempt_id} has invalid initialization evidence")
    evaluations = [
        event for event in events if event.get("event") == "evaluation"
    ]
    evaluation_steps = [
        event.get("total_steps", event.get("step")) for event in evaluations
    ]
    metrics = summarize_history(events)
    delta_telemetry = [
        event for event in events if event.get("event") == "delta_telemetry"
    ]
    telemetry_steps = [
        event.get("total_steps", event.get("step")) for event in delta_telemetry
    ]
    expected_telemetry_steps = sorted(expected_delta_telemetry_steps(config))
    try:
        if delta_telemetry:
            from TiME.delta_telemetry import summarize_delta_drift

            summarize_delta_drift(delta_telemetry)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"attempt {attempt_id} has invalid delta telemetry evidence"
        ) from error
    if (
        evaluation_steps != sorted(evaluation_steps)
        or not evaluation_events_match_plan(
            evaluations, expected_evaluation_plan(config)
        )
        or metrics["final_return"] is None
        or metrics["auc"] is None
        or telemetry_steps != expected_telemetry_steps
    ):
        raise RuntimeError(
            f"attempt {attempt_id} lacks complete evaluation or delta telemetry evidence"
        )


def _next_attempt_id(output_dir: Path, run_id: str) -> str:
    path = run_status_history_path(output_dir, run_id)
    history = read_json(path) if path.exists() else {"events": []}
    count = sum(event.get("state") == "running" for event in history["events"])
    return f"attempt-{count + 1:04d}"


def _run_one(output_dir: Path, run, device: str, use_wandb: bool, resume: bool) -> bool:
    prior = read_status(output_dir, run.run_id)
    if prior:
        initialize_run(output_dir, run.to_dict())
        if prior["state"] == "completed":
            _validate_completed_provenance(output_dir, run.run_id)
            return False
        if not resume:
            raise RuntimeError(
                f"{run.run_id} already has state {prior['state']}; "
                "use --resume to restart incomplete runs"
            )
        if prior["state"] == "running":
            write_status(
                output_dir,
                run.run_id,
                "failed",
                attempt_id=prior.get("attempt_id"),
                error_type="InterruptedRunRecovered",
                error="previous process ended without a terminal status",
            )
    else:
        initialize_run(output_dir, run.to_dict())
    attempt_id = _next_attempt_id(output_dir, run.run_id)
    append_history_event(
        output_dir,
        run.run_id,
        {
            "event": "attempt_started",
            "attempt_id": attempt_id,
            "device": device,
            "use_wandb": use_wandb,
        },
    )
    write_status(output_dir, run.run_id, "running", attempt_id=attempt_id)
    trainer_config = run.trainer_config(device=device, use_wandb=use_wandb)
    history_recorder = BufferedHistoryRecorder(output_dir, run.run_id)
    trainer_config["history_recorder"] = lambda event: history_recorder(
        {**event, "attempt_id": attempt_id}
    )
    try:
        # Import only after the explicit execution gate; dry-runs need no Brax stack.
        from trainer import train

        try:
            train(config=trainer_config, env_name=run.task)
        finally:
            history_recorder.flush()
        _validate_attempt_completion(output_dir, run.run_id, attempt_id)
    except Exception as error:
        write_status(
            output_dir,
            run.run_id,
            "failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
            attempt_id=attempt_id,
        )
        return True
    write_status(
        output_dir, run.run_id, "completed", attempt_id=attempt_id
    )
    return False


def execute_matrix(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    manifest = matrix_manifest(args.reset_dt)
    write_matrix_manifest(output_dir, manifest)
    if not args.execute:
        print(f"Dry run: wrote {manifest['run_count']} deterministic run keys to {output_dir}")
        return 0
    if not args.authorize_full_execution:
        raise RuntimeError("Full execution requires --authorize-full-execution")

    failed = 0
    for run in build_matrix(args.reset_dt):
        failed += _run_one(output_dir, run, args.device, args.use_wandb, args.resume)
    if failed:
        raise RuntimeError(f"{failed} corrected-v2 runs failed; inspect per-run status.json artifacts")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/time-brax-corrected-v2")
    parser.add_argument("--reset-dt", type=float, default=5.0)
    parser.add_argument("--device", choices=(EXECUTION_DEVICE,), default=EXECUTION_DEVICE)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Run training instead of the default dry-run")
    parser.add_argument(
        "--authorize-full-execution",
        action="store_true",
        help="Required acknowledgement before all 80 training runs can start",
    )
    parser.add_argument("--resume", action="store_true", help="Restart incomplete or failed existing runs")
    return parser


def main() -> int:
    return execute_matrix(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
