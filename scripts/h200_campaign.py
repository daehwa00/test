"""Execute recoverable 10-run task batches of the canonical H200 primary matrix."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from artifacts import read_json, run_history_path, write_matrix_manifest
from experiment_config import EXECUTION_DEVICE, PRIMARY_VARIANTS, TASKS, TRAINING_SEEDS, build_matrix
from metrics import summarize_history
from run_brax_matrix import execute_run_configs, matrix_manifest


class CampaignInterrupted(BaseException):
    """Stop the matrix without being swallowed by per-run Exception handling."""


def _interrupt(signum: int, _frame: Any) -> None:
    raise CampaignInterrupted(f"received signal {signum}")


def install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGINT, _interrupt)


CAMPAIGN_NAME = "standard-primary"  # Compatibility name for the canonical campaign.
DEFAULT_OUTPUT_DIR = Path("results/h200-standard-primary")
BATCH_TASKS = {
    "batch-1": ("halfcheetah",),
    "batch-2": ("hopper",),
    "batch-3": ("ant",),
    "batch-4": ("walker2d",),
    "batch-5": ("swimmer",),
    "batch-6": ("reacher",),
    "batch-7": ("pusher",),
    "batch-8": ("humanoidstandup",),
}


def canonical_primary_manifest(reset_dt: float = 5.0) -> dict[str, Any]:
    manifest = matrix_manifest(reset_dt)
    expected_count = len(TASKS) * len(PRIMARY_VARIANTS) * len(TRAINING_SEEDS)
    if (manifest["run_count"] != expected_count or
            {variant["name"] for variant in manifest["variants"]} != {"time", "vanilla-mamba2"}):
        raise RuntimeError("canonical corrected-v2 primary matrix is not the expected 80 runs")
    return manifest


def batch_runs(batch: str, reset_dt: float = 5.0) -> tuple[Any, ...]:
    """Return the explicit canonical RunConfigs authorized for one H200 request."""
    if batch not in BATCH_TASKS:
        raise ValueError(f"unknown H200 batch: {batch}")
    runs = tuple(run for run in build_matrix(reset_dt) if run.task in BATCH_TASKS[batch])
    if len(runs) != 10:
        raise RuntimeError(f"{batch} must contain exactly 10 canonical primary runs")
    return runs


def campaign_command(interpreter: Path, output_dir: Path, batch: str = "batch-1") -> tuple[str, ...]:
    return (str(interpreter), str(Path(__file__).resolve()), "--output-dir", str(output_dir),
            "--batch", batch, "--authorize-full-execution")


def source_commit() -> str | None:
    try:
        result = subprocess.run(("git", "rev-parse", "HEAD"), cwd=PROJECT_ROOT, check=False,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def _compact_delta_drift(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    telemetry = [event for event in events if event.get("event") == "delta_telemetry"]
    if not telemetry:
        return []
    from TiME.delta_telemetry import summarize_delta_drift
    summary = summarize_delta_drift(telemetry)
    fields = ("delta_mean_change", "timescale_median_change", "effective_projection_norm_change")
    return [{"layer_index": layer.get("layer_index"),
             **{field: layer.get(field) for field in fields}}
            for layer in summary.get("layers", []) if isinstance(layer, dict)]


def run_evidence(output_dir: Path, run: Any, status: dict[str, Any] | None) -> dict[str, Any] | None:
    """Read a completed run's compact, per-seed recovery evidence."""
    if not status or status.get("state") != "completed":
        return None
    try:
        attempt_id = status["attempt_id"]
        events = [event for event in read_json(run_history_path(output_dir, run.run_id)).get("events", [])
                  if event.get("attempt_id") == attempt_id]
        initialization = [event for event in events
                          if event.get("event") == "shared_initialization_verified"]
        digest = initialization[0].get("shared_initialization_digest") if len(initialization) == 1 else None
        valid = (isinstance(digest, str) and len(digest) == 64 and
                 all(character in "0123456789abcdef" for character in digest) and
                 bool(initialization[0].get("backbone_provenance")) if len(initialization) == 1 else False)
        metrics = summarize_history(events)
        return {"task": run.task, "variant": run.variant.name, "seed": run.seed,
                "final_return": metrics["final_return"], "auc": metrics["auc"],
                "delta_drift": _compact_delta_drift(events),
                "shared_initialization_digest": digest, "provenance_valid": valid}
    except (KeyError, OSError, TypeError, ValueError):
        return None


def batch_receipt(manifest: dict[str, Any], batch: str, output_dir: Path,
                  statuses: dict[str, dict[str, Any] | None], execution_error: str | None) -> dict[str, Any]:
    runs = batch_runs(batch, manifest["reset_dt"])
    evidence = [value for run in runs
                if (value := run_evidence(output_dir, run, statuses.get(run.run_id))) is not None]
    failed = sum(bool(status and status.get("state") == "failed") for status in statuses.values())
    completed = len(evidence)
    missing = len(runs) - completed - failed
    passed = execution_error is None and completed == len(runs) and failed == 0 and missing == 0 and all(
        item["final_return"] is not None and item["auc"] is not None and item["provenance_valid"] for item in evidence)
    return {"passed": passed, "campaign": CAMPAIGN_NAME, "batch": batch,
            "source_commit": source_commit(), "canonical_run_count": manifest["run_count"],
            "status": {"expected": len(runs), "completed": completed, "failed": failed, "missing": missing},
            "runs": evidence, "errors": {"execution": execution_error}}


def execute_campaign(args: argparse.Namespace, on_terminal: Any = None) -> int:
    manifest = canonical_primary_manifest(args.reset_dt)
    if not args.authorize_full_execution:
        raise RuntimeError("Full execution requires --authorize-full-execution")
    write_matrix_manifest(args.output_dir, manifest)  # Full manifest is required by aggregators.
    return execute_run_configs(args.output_dir, batch_runs(args.batch, args.reset_dt), device=EXECUTION_DEVICE,
                               use_wandb=False, resume=args.resume, on_terminal=on_terminal)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch", choices=tuple(BATCH_TASKS), required=True)
    parser.set_defaults(reset_dt=5.0)
    parser.add_argument("--authorize-full-execution", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = canonical_primary_manifest(args.reset_dt)
    install_signal_handlers()
    statuses: dict[str, dict[str, Any] | None] = {}

    def checkpoint(run: Any, status: dict[str, Any] | None) -> None:
        statuses[run.run_id] = status
        receipt = batch_receipt(manifest, args.batch, args.output_dir, statuses, None)
        print("H200_BATCH_CHECKPOINT=" + json.dumps(receipt, sort_keys=True, separators=(",", ":")))

    execution_error = None
    try:
        failed = execute_campaign(args, checkpoint)
        if failed:
            execution_error = f"RuntimeError: {failed} batch runs failed"
    except BaseException as error:
        execution_error = f"{type(error).__name__}: {error}"
    receipt = batch_receipt(manifest, args.batch, args.output_dir, statuses, execution_error)
    print("H200_CAMPAIGN_RESULT=" + json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
