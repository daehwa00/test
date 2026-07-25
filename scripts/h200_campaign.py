"""Execute and compactly report the canonical corrected-v2 H200 primary matrix."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aggregate_results import aggregate
from artifacts import read_json, run_status_path
from experiment_config import EXECUTION_DEVICE, PRIMARY_VARIANTS, TASKS, TRAINING_SEEDS
from run_brax_matrix import execute_matrix, matrix_manifest

class CampaignInterrupted(BaseException):
    """Stop the matrix without being swallowed by per-run Exception handling."""


def _interrupt(signum: int, _frame: Any) -> None:
    raise CampaignInterrupted(f"received signal {signum}")


def install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGINT, _interrupt)


CAMPAIGN_NAME = "standard-primary"
DEFAULT_OUTPUT_DIR = Path("results/h200-standard-primary")


def canonical_primary_manifest(reset_dt: float = 5.0) -> dict[str, Any]:
    """Return the sole H200 campaign definition, rejecting matrix drift early."""
    manifest = matrix_manifest(reset_dt)
    expected_count = len(TASKS) * len(PRIMARY_VARIANTS) * len(TRAINING_SEEDS)
    if (
        manifest["run_count"] != expected_count
        or {variant["name"] for variant in manifest["variants"]}
        != {"time", "vanilla-mamba2"}
    ):
        raise RuntimeError("canonical corrected-v2 primary matrix is not the expected 80 runs")
    return manifest


def campaign_command(interpreter: Path, output_dir: Path) -> tuple[str, ...]:
    return (
        str(interpreter),
        str(Path(__file__).resolve()),
        "--output-dir",
        str(output_dir),
        "--authorize-full-execution",
    )


def source_commit() -> str | None:
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=PROJECT_ROOT, check=False,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _compact_delta_drift(delta_by_seed: dict[str, Any]) -> list[dict[str, Any]]:
    """Collapse per-seed, per-layer drift into receipt-safe condition summaries."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for drift in delta_by_seed.values():
        if not isinstance(drift, dict):
            continue
        for layer in drift.get("layers", []):
            if isinstance(layer, dict) and isinstance(layer.get("layer_index"), int):
                grouped[layer["layer_index"]].append(layer)
    fields = (
        "delta_mean_change",
        "timescale_median_change",
        "effective_projection_norm_change",
    )
    return [
        {
            "layer_index": index,
            "n_seeds": len(layers),
            **{
                field: _mean([float(layer[field]) for layer in layers if isinstance(layer.get(field), (int, float))])
                for field in fields
            },
        }
        for index, layers in sorted(grouped.items())
    ]


def compact_receipt(
    manifest: dict[str, Any], aggregate_result: dict[str, Any] | None,
    output_dir: Path, execution_error: str | None, aggregation_error: str | None,
) -> dict[str, Any]:
    """Return a bounded receipt without histories, run IDs, or per-seed telemetry."""
    completeness = (aggregate_result or {}).get("completeness", {})
    expected_ids = {str(run["run_id"]) for run in manifest["runs"]}
    failed = 0
    for run_id in expected_ids:
        status_path = run_status_path(output_dir, run_id)
        if status_path.exists():
            try:
                failed += read_json(status_path).get("state") == "failed"
            except (OSError, ValueError, TypeError):
                pass
    conditions = []
    for condition in (aggregate_result or {}).get("conditions", []):
        conditions.append({
            "task": condition["task"],
            "variant": condition["variant"],
            "final_return": condition["final_return"],
            "auc": condition["auc"],
            "delta_drift": _compact_delta_drift(condition.get("delta_drift_by_seed", {})),
        })
    paired = [
        {
            "task": value["task"],
            "contrast": value["contrast"],
            "final_return": value["final_return"],
            "auc": value["auc"],
        }
        for value in (aggregate_result or {}).get("paired_differences", [])
    ]
    completed = int(completeness.get("completed_runs", 0))
    missing = len(completeness.get("missing_run_ids", expected_ids))
    invalid = (
        len(completeness.get("invalid_run_ids", []))
        + len(completeness.get("invalid_pairs", []))
        + len(completeness.get("unexpected_run_ids", []))
    )
    passed = (
        execution_error is None and aggregation_error is None
        and bool(completeness.get("complete")) and failed == 0
    )
    return {
        "passed": passed,
        "campaign": CAMPAIGN_NAME,
        "source_commit": source_commit(),
        "canonical_run_count": manifest["run_count"],
        "status": {"completed": completed, "missing": missing, "invalid": invalid, "failed": failed},
        "conditions": conditions,
        "paired_differences": paired,
        "errors": {
            "execution": execution_error,
            "aggregation": aggregation_error,
        },
    }


def execute_campaign(args: argparse.Namespace) -> int:
    manifest = canonical_primary_manifest(args.reset_dt)
    if not args.authorize_full_execution:
        raise RuntimeError("Full execution requires --authorize-full-execution")
    matrix_args = argparse.Namespace(
        output_dir=str(args.output_dir), reset_dt=args.reset_dt, device=EXECUTION_DEVICE,
        use_wandb=False, execute=True, authorize_full_execution=True, resume=args.resume,
    )
    execute_matrix(matrix_args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.set_defaults(reset_dt=5.0)
    parser.add_argument("--authorize-full-execution", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = canonical_primary_manifest(args.reset_dt)
    install_signal_handlers()
    execution_error = None
    aggregation_error = None
    aggregate_result = None
    try:
        execute_campaign(args)
    except BaseException as error:
        execution_error = f"{type(error).__name__}: {error}"
    try:
        aggregate_result = aggregate(args.output_dir)
    except Exception as error:
        aggregation_error = f"{type(error).__name__}: {error}"
    receipt = compact_receipt(
        manifest, aggregate_result, args.output_dir, execution_error, aggregation_error
    )
    print("H200_CAMPAIGN_RESULT=" + json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
