"""Canonical JSON artifacts for corrected-v2 matrix runs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

MATRIX_MANIFEST = "matrix_manifest.json"
RUNS_DIRECTORY = "runs"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    """Atomically write a deterministic, human-readable JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def json_equal(left: Any, right: Any) -> bool:
    return canonical_json(left) == canonical_json(right)


def matrix_manifest_path(output_dir: Path) -> Path:
    return output_dir / MATRIX_MANIFEST


def run_directory(output_dir: Path, run_id: str) -> Path:
    return output_dir / RUNS_DIRECTORY / run_id


def run_config_path(output_dir: Path, run_id: str) -> Path:
    return run_directory(output_dir, run_id) / "config.json"


def run_history_path(output_dir: Path, run_id: str) -> Path:
    return run_directory(output_dir, run_id) / "history.json"


def run_status_path(output_dir: Path, run_id: str) -> Path:
    return run_directory(output_dir, run_id) / "status.json"


def run_status_history_path(output_dir: Path, run_id: str) -> Path:
    return run_directory(output_dir, run_id) / "status_history.json"


def write_matrix_manifest(output_dir: Path, manifest: Mapping[str, Any]) -> Path:
    path = matrix_manifest_path(output_dir)
    value = dict(manifest)
    if path.exists():
        if not json_equal(read_json(path), value):
            raise RuntimeError(f"matrix manifest mismatch at {path}")
        return path
    write_json(path, value)
    return path


def initialize_run(output_dir: Path, config: Mapping[str, Any]) -> None:
    run_id = str(config["run_id"])
    config_path = run_config_path(output_dir, run_id)
    history_path = run_history_path(output_dir, run_id)
    status_path = run_status_path(output_dir, run_id)
    value = dict(config)
    if config_path.exists():
        if not json_equal(read_json(config_path), value):
            raise RuntimeError(f"run config mismatch for {run_id}")
    else:
        write_json(config_path, value)
    if not history_path.exists():
        write_json(history_path, {"run_id": run_id, "events": []})
    if not status_path.exists():
        write_status(output_dir, run_id, "pending")


def write_history(output_dir: Path, run_id: str, events: list[Mapping[str, Any]]) -> None:
    write_json(run_history_path(output_dir, run_id), {"run_id": run_id, "events": events})


def append_history_event(output_dir: Path, run_id: str, event: Mapping[str, Any]) -> None:
    path = run_history_path(output_dir, run_id)
    history = read_json(path) if path.exists() else {"run_id": run_id, "events": []}
    history.setdefault("events", []).append(dict(event))
    write_json(path, history)


class BufferedHistoryRecorder:
    """Append events in memory and periodically persist the canonical history."""

    def __init__(self, output_dir: Path, run_id: str, flush_every: int = 16):
        if flush_every <= 0:
            raise ValueError("flush_every must be positive")
        self.output_dir = output_dir
        self.run_id = run_id
        self.flush_every = flush_every
        path = run_history_path(output_dir, run_id)
        self.events = (
            list(read_json(path).get("events", []))
            if path.exists()
            else []
        )
        self.pending = 0

    def __call__(self, event: Mapping[str, Any]) -> None:
        value = dict(event)
        self.events.append(value)
        self.pending += 1
        if self.pending >= self.flush_every or value.get("event") == "evaluation":
            self.flush()

    def flush(self) -> None:
        if self.pending:
            write_history(self.output_dir, self.run_id, self.events)
            self.pending = 0


def write_status(output_dir: Path, run_id: str, state: str, **details: Any) -> None:
    if state not in {"pending", "running", "completed", "failed"}:
        raise ValueError(f"Unsupported run state: {state}")
    current = read_status(output_dir, run_id)
    allowed = {
        None: {"pending"},
        "pending": {"running", "failed"},
        "running": {"completed", "failed"},
        "failed": {"running"},
        "completed": set(),
    }
    current_state = current["state"] if current is not None else None
    if state not in allowed[current_state]:
        raise ValueError(f"Illegal run-state transition: {current_state!r} -> {state!r}")
    value = {"run_id": run_id, "state": state, "updated_at": utc_now(), **details}
    history_path = run_status_history_path(output_dir, run_id)
    history = read_json(history_path) if history_path.exists() else {"events": []}
    history["events"].append(value)
    write_json(history_path, history)
    write_json(run_status_path(output_dir, run_id), value)


def read_status(output_dir: Path, run_id: str) -> Mapping[str, Any] | None:
    path = run_status_path(output_dir, run_id)
    return read_json(path) if path.exists() else None
