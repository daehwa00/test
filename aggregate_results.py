"""Aggregate corrected-v2 run artifacts with training seed as the unit."""

from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from artifacts import (
    canonical_json,
    matrix_manifest_path,
    read_json,
    run_config_path,
    run_history_path,
    run_status_path,
    write_json,
)
from experiment_config import (
    PROTOCOL,
    expected_delta_telemetry_steps,
    expected_evaluation_plan,
)
from metrics import evaluation_events_match_plan, summarize_history
from run_brax_matrix import matrix_manifest

_T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def _critical_t(degrees_of_freedom: int) -> float:
    return _T_975.get(degrees_of_freedom, 1.96)


def _provenance_projection(value: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    return {
        side: {
            key: item
            for key, item in fields.items()
            if not key.endswith("_path")
        }
        for side, fields in value.items()
    }




def _summary(values: list[float]) -> dict[str, float | int | None]:
    count = len(values)
    mean = statistics.fmean(values) if values else None
    sample_std = statistics.stdev(values) if count > 1 else None
    sem = sample_std / math.sqrt(count) if sample_std is not None else None
    half_width = _critical_t(count - 1) * sem if sem is not None else None
    return {
        "n_seeds": count,
        "mean": mean,
        "sample_std": sample_std,
        "sem": sem,
        "ci95_low": mean - half_width if half_width is not None else None,
        "ci95_high": mean + half_width if half_width is not None else None,
    }


def aggregate(output_dir: Path) -> dict[str, Any]:
    """Read completed runs and aggregate one final-return/AUC value per seed."""
    runs_root = output_dir / "runs"
    manifest_path = matrix_manifest_path(output_dir)
    expected_run_ids = []
    expected_by_id = {}
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        try:
            canonical_manifest = matrix_manifest(float(manifest["reset_dt"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("matrix manifest lacks a valid reset_dt") from error
        if canonical_json(manifest) != canonical_json(canonical_manifest):
            raise ValueError("matrix manifest does not match the canonical 80-run matrix")
        expected_by_id = {
            str(run["run_id"]): run for run in canonical_manifest["runs"]
        }
        expected_run_ids = list(expected_by_id)
    completed_run_ids = set()
    seen_completed_ids = set()
    invalid_run_ids = set()
    by_condition: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    current_provenance = None
    invalid_pairs: list[str] = []
    if runs_root.exists():
        for directory in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            status_path = run_status_path(output_dir, directory.name)
            if not status_path.exists():
                continue
            status = read_json(status_path)
            if status.get("state") != "completed":
                continue
            seen_completed_ids.add(directory.name)
            config = read_json(run_config_path(output_dir, directory.name))
            history = read_json(run_history_path(output_dir, directory.name))
            attempt_id = status.get("attempt_id")
            if not isinstance(attempt_id, str) or not attempt_id:
                invalid_run_ids.add(directory.name)
                continue
            events = [
                event
                for event in history.get("events", [])
                if event.get("attempt_id") == attempt_id
            ]
            metrics = summarize_history(events)
            delta_telemetry = [
                event for event in events if event.get("event") == "delta_telemetry"
            ]
            delta_drift = None
            if delta_telemetry:
                from TiME.delta_telemetry import summarize_delta_drift

                delta_drift = summarize_delta_drift(delta_telemetry)
            initializations = [
                event
                for event in events
                if event.get("event") == "shared_initialization_verified"
                and event.get("backbone_provenance")
            ]
            digest = (
                initializations[0].get("shared_initialization_digest")
                if len(initializations) == 1
                else None
            )
            if manifest_path.exists():
                expected = expected_by_id.get(directory.name)
                evaluations = [
                    event for event in events if event.get("event") == "evaluation"
                ]
                attempt_starts = [
                    event for event in events if event.get("event") == "attempt_started"
                ]
                valid_initialization = len(initializations) == 1
                if valid_initialization:
                    from TiME.provenance import backbone_provenance

                    if current_provenance is None:
                        current_provenance = _provenance_projection(
                            backbone_provenance()
                        )
                    initialization = initializations[0]
                    digest = initialization.get("shared_initialization_digest")
                    expected_rebalanced = 4 if expected and expected.get("mr_enabled") else 0
                    valid_initialization = (
                        _provenance_projection(
                            initialization["backbone_provenance"]
                        )
                        == current_provenance
                        and isinstance(digest, str)
                        and len(digest) == 64
                        and all(
                            character in "0123456789abcdef"
                            for character in digest
                        )
                        and initialization.get("shared_tensor_count", 0) > 0
                        and initialization.get("rebalanced_buffer_count")
                        == expected_rebalanced
                    )
                evaluation_steps = [
                    event.get("total_steps", event.get("step")) for event in evaluations
                ]
                valid_attempt = (
                    len(attempt_starts) == 1
                    and expected is not None
                    and attempt_starts[0].get("device") == expected.get("device")
                )
                if (
                    expected is None
                    or canonical_json(config) != canonical_json(expected)
                    or not valid_attempt
                    or not valid_initialization
                    or evaluation_steps != sorted(evaluation_steps)
                    or not evaluation_events_match_plan(
                        evaluations, expected_evaluation_plan(expected)
                    )
                    or metrics["final_return"] is None
                    or metrics["auc"] is None
                    or [
                        event.get("total_steps", event.get("step"))
                        for event in delta_telemetry
                    ]
                    != sorted(expected_delta_telemetry_steps(config))
                ):
                    invalid_run_ids.add(directory.name)
                    continue
            if config.get("protocol") != PROTOCOL:
                invalid_run_ids.add(directory.name)
                continue
            completed_run_ids.add(directory.name)
            seed = int(config["seed"])
            condition = (str(config["task"]), str(config["variant"]))
            if seed in by_condition[condition]:
                raise ValueError(f"Duplicate completed seed {seed} for {condition}")
            by_condition[condition][seed] = {
                **metrics,
                "_shared_initialization_digest": digest,
                "delta_drift": delta_drift,
            }

    conditions = []
    for (task, variant), seed_metrics in sorted(by_condition.items()):
        ordered = [seed_metrics[seed] for seed in sorted(seed_metrics)]
        final_values = [value["final_return"] for value in ordered if value["final_return"] is not None]
        auc_values = [value["auc"] for value in ordered if value["auc"] is not None]
        conditions.append(
            {
                "task": task,
                "variant": variant,
                "seeds": sorted(seed_metrics),
                "final_return": _summary(final_values),
                "auc": _summary(auc_values),
                "delta_drift_by_seed": {
                    str(seed): seed_metrics[seed]["delta_drift"]
                    for seed in sorted(seed_metrics)
                },
            }
        )
    paired_differences = []
    tasks = sorted({task for task, _ in by_condition})
    for task in tasks:
        time_runs = by_condition.get((task, "time"), {})
        vanilla_runs = by_condition.get((task, "vanilla-mamba2"), {})
        candidate_seeds = sorted(time_runs.keys() & vanilla_runs.keys())
        shared_seeds = []
        for seed in candidate_seeds:
            time_digest = time_runs[seed]["_shared_initialization_digest"]
            vanilla_digest = vanilla_runs[seed]["_shared_initialization_digest"]
            if (
                not isinstance(time_digest, str)
                or time_digest != vanilla_digest
            ):
                invalid_pairs.append(f"{task}__seed-{seed}")
                continue
            shared_seeds.append(seed)
        metric_summaries = {}
        for metric in ("final_return", "auc"):
            differences = [
                time_runs[seed][metric] - vanilla_runs[seed][metric]
                for seed in shared_seeds
                if time_runs[seed][metric] is not None
                and vanilla_runs[seed][metric] is not None
            ]
            metric_summaries[metric] = _summary(differences)
        paired_differences.append(
            {
                "task": task,
                "contrast": "time-minus-vanilla-mamba2",
                "seeds": shared_seeds,
                **metric_summaries,
            }
        )
    missing_run_ids = sorted(set(expected_run_ids) - completed_run_ids)
    unexpected_run_ids = sorted(seen_completed_ids - set(expected_run_ids))
    completeness = {
        "manifest_present": manifest_path.exists(),
        "expected_runs": len(expected_run_ids) if manifest_path.exists() else None,
        "completed_runs": len(completed_run_ids),
        "missing_run_ids": missing_run_ids,
        "unexpected_run_ids": unexpected_run_ids,
        "invalid_run_ids": sorted(invalid_run_ids),
        "invalid_pairs": sorted(invalid_pairs),
        "complete": bool(manifest_path.exists())
        and not missing_run_ids
        and not unexpected_run_ids
        and not invalid_run_ids
        and not invalid_pairs,
    }
    return {
        "protocol": PROTOCOL,
        "seed_unit": "training_seed",
        "conditions": conditions,
        "paired_differences": paired_differences,
        "completeness": completeness,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/time-brax-corrected-v2")
    parser.add_argument("--output", default="aggregate_results.json")
    args = parser.parse_args()
    result = aggregate(Path(args.output_dir))
    destination = Path(args.output_dir) / args.output
    write_json(destination, result)
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
