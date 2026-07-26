"""Build or reuse the pinned, isolated qlab H200 runtime and run its smoke probe.

The H200 base image is deliberately never modified.  A versioned virtual environment
under the persistent scratch cache is considered ready only after its exact pins have
been checked and an atomic marker has been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

CACHE_ROOT = Path("/app/scratch/cache")
RUNTIME_NAME = "time-h200-qlab"
READY_MARKER = "READY.json"
FINGERPRINT_SCHEMA = 1
PYTORCH_CU124_INDEX = "https://download.pytorch.org/whl/cu124"
VIRTUALENV_VERSION = "20.31.2"
CAMPAIGN_TIMEOUT_SECONDS = 12 * 60 * 60
STANDARD_PRIMARY_CAMPAIGN = "standard-primary"
BATCH_CAMPAIGNS = tuple(f"batch-{index}" for index in range(1, 9))
TAIL_LINES = 40
TAIL_CHARS = 512
BOOTSTRAP_REPORT_LIMIT = 25_000


def campaign_command(interpreter: Path, output_dir: Path, batch: str) -> tuple[str, ...]:
    """Build an explicitly authorized H200 batch invocation."""
    return (
        str(interpreter),
        str(Path(__file__).with_name("h200_campaign.py")),
        "--output-dir",
        str(output_dir),
        "--batch",
        batch,
        "--authorize-full-execution",
    )

# This is intentionally the single source of the scientific runtime contract.
QLAB_PINS = {
    "brax": "0.12.3",
    "causal-conv1d": "1.5.0.post8",
    "einops": "0.8.1",
    "jax": "0.6.0",
    "jaxlib": "0.6.0",
    "mamba-ssm": "2.2.4",
    "ninja": "1.11.1.4",
    "nvidia-cudnn-cu12": "9.8.0.87",
    "torch": "2.6.0+cu124",
    "triton": "3.2.0",
}


def bootstrap_tool_path(cache_root: Path) -> Path:
    return cache_root / "bootstrap" / f"virtualenv-{VIRTUALENV_VERSION}"


def bootstrap_tool_command(cache_root: Path, pip_cache: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        str(bootstrap_tool_path(cache_root)),
        "--cache-dir",
        str(pip_cache),
        f"virtualenv=={VIRTUALENV_VERSION}",
    )


def create_venv_command(venv: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "virtualenv",
        "--python",
        sys.executable,
        str(venv),
    )


def virtualenv_environment(cache_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    tool_path = str(bootstrap_tool_path(cache_root))
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = tool_path if not current else f"{tool_path}{os.pathsep}{current}"
    return environment


def environment_fingerprint(pins: dict[str, str] = QLAB_PINS) -> str:
    """Return a stable cache key for the explicit scientific runtime contract."""
    payload = {
        "schema": FINGERPRINT_SCHEMA,
        "pins": dict(sorted(pins.items())),
        "virtualenv": VIRTUALENV_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def runtime_path(cache_root: Path, fingerprint: str) -> Path:
    return cache_root / "venvs" / f"{RUNTIME_NAME}-{fingerprint}"


def venv_interpreter(venv: Path) -> Path:
    return venv / "bin" / "python"


def _text(value: str | bytes | None) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else (value or "")


def _tail(value: str | bytes | None) -> list[str]:
    return [line[-TAIL_CHARS:] for line in _text(value).splitlines()[-TAIL_LINES:]]


def _checkpoint(value: str | bytes | None) -> str | None:
    checkpoints = [line[len("H200_BATCH_CHECKPOINT="):] for line in _text(value).splitlines()
                   if line.startswith("H200_BATCH_CHECKPOINT=")]
    return checkpoints[-1] if checkpoints else None


def run_bounded(command: tuple[str, ...], timeout: int, *, env: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        stdout, stderr, returncode, timed_out = exc.stdout, exc.stderr, None, True
    else:
        stdout, stderr, returncode, timed_out = completed.stdout, completed.stderr, completed.returncode, False
    result = {"command": list(command), "returncode": returncode, "stdout_tail": _tail(stdout),
              "stderr_tail": _tail(stderr), "timed_out": timed_out}
    checkpoint = _checkpoint(stdout)
    if checkpoint is not None:
        result["latest_batch_checkpoint"] = checkpoint
    return result


def _compact_command_result(value: dict[str, Any]) -> dict[str, Any]:
    compact = {key: value[key] for key in ("stage", "status", "returncode", "timed_out",
                                           "latest_batch_checkpoint") if key in value}
    if "command" in value:
        compact["command"] = [str(argument)[-128:] for argument in value["command"][-16:]]
    for key in ("stdout_tail", "stderr_tail"):
        if value.get(key):
            compact[key] = [str(line)[-128:] for line in value[key][-2:]]
    for key in ("versions", "venv"):
        if key in value:
            compact[key] = value[key]
    return compact


def compact_bootstrap_result(result: dict[str, Any]) -> dict[str, Any]:
    compact = {key: value for key, value in result.items() if key not in {"audit", "smoke"}}
    compact["audit"] = [_compact_command_result(item) for item in result.get("audit", [])]
    if "smoke" in result:
        compact["smoke"] = _compact_command_result(result["smoke"])
    campaign = compact.get("campaign")
    if isinstance(campaign, dict) and isinstance(campaign.get("result"), dict):
        compact["campaign"] = {**campaign, "result": _compact_command_result(campaign["result"])}
    return compact


def print_result(result: dict[str, Any]) -> None:
    encoded = json.dumps(compact_bootstrap_result(result), sort_keys=True, separators=(",", ":"))
    if len(encoded) >= BOOTSTRAP_REPORT_LIMIT:
        raise RuntimeError("compacted H200 bootstrap result exceeds FaaS report cap")
    print("H200_BOOTSTRAP_RESULT=" + encoded)


def installation_commands(interpreter: Path, pip_cache: Path) -> tuple[tuple[str, ...], ...]:
    pip = (str(interpreter), "-m", "pip", "install", "--cache-dir", str(pip_cache))
    return (
        pip + ("--index-url", PYTORCH_CU124_INDEX, f"torch=={QLAB_PINS['torch']}", f"triton=={QLAB_PINS['triton']}"),
        pip + (f"jax[cuda12]=={QLAB_PINS['jax']}", f"jaxlib=={QLAB_PINS['jaxlib']}", f"brax=={QLAB_PINS['brax']}",
               f"einops=={QLAB_PINS['einops']}", f"ninja=={QLAB_PINS['ninja']}"),
        pip + ("--no-build-isolation", f"causal-conv1d=={QLAB_PINS['causal-conv1d']}",
               f"mamba-ssm=={QLAB_PINS['mamba-ssm']}"),
        pip + (
            "--no-deps",
            f"nvidia-cudnn-cu12=={QLAB_PINS['nvidia-cudnn-cu12']}",
        ),
    )


def installed_versions(interpreter: Path) -> dict[str, str | None] | None:
    probe = """import importlib.metadata as m, json
pins=json.loads(%r)
out={}
for name in pins:
 try: out[name]=m.version(name)
 except m.PackageNotFoundError: out[name]=None
try:
 import torch
 out['torch']=torch.__version__
except Exception: pass
print(json.dumps(out, sort_keys=True))""" % json.dumps(QLAB_PINS)
    result = run_bounded((str(interpreter), "-c", probe), timeout=60)
    if result["returncode"] != 0 or not result["stdout_tail"]:
        return None
    try:
        return json.loads(result["stdout_tail"][-1])
    except json.JSONDecodeError:
        return None


def exact_versions(interpreter: Path) -> tuple[bool, dict[str, str | None] | None]:
    versions = installed_versions(interpreter)
    return versions is not None and versions == QLAB_PINS, versions


def read_ready(venv: Path, fingerprint: str) -> dict[str, Any] | None:
    try:
        marker = json.loads((venv / READY_MARKER).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if marker.get("fingerprint") != fingerprint or marker.get("pins") != QLAB_PINS:
        return None
    return marker


def write_ready_atomic(venv: Path, fingerprint: str, versions: dict[str, str | None]) -> None:
    marker = venv / READY_MARKER
    payload = {"schema": FINGERPRINT_SCHEMA, "fingerprint": fingerprint, "pins": QLAB_PINS,
               "versions": versions, "interpreter": str(venv_interpreter(venv))}
    with tempfile.NamedTemporaryFile("w", dir=venv, prefix=".ready-", delete=False) as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, marker)


def interpreter_has_pip(interpreter: Path) -> bool:
    if not interpreter.exists():
        return False
    result = run_bounded((str(interpreter), "-m", "pip", "--version"), timeout=30)
    return result["returncode"] == 0


def ensure_runtime(cache_root: Path, *, dry_run: bool = False) -> tuple[Path, str, list[dict[str, Any]], str | None]:
    fingerprint = environment_fingerprint()
    venv = runtime_path(cache_root, fingerprint)
    interpreter = venv_interpreter(venv)
    pip_cache = cache_root / "pip" / "wheels"
    audit: list[dict[str, Any]] = []
    marker = read_ready(venv, fingerprint)
    exact, versions = exact_versions(interpreter) if marker and interpreter.exists() else (False, None)
    if marker and exact:
        audit.append({"stage": "cache-reuse", "status": "passed", "versions": versions})
        return venv, fingerprint, audit, None
    if marker:
        audit.append({"stage": "stale-marker", "status": "invalidated", "versions": versions})
        (venv / READY_MARKER).unlink(missing_ok=True)
    if dry_run:
        commands = (
            bootstrap_tool_command(cache_root, pip_cache),
            create_venv_command(venv),
            *installation_commands(interpreter, pip_cache),
        )
        audit.append({
            "stage": "build",
            "status": "would-build",
            "venv": str(venv),
            "commands": [list(command) for command in commands],
        })
        return venv, fingerprint, audit, None
    cache_root.mkdir(parents=True, exist_ok=True)
    pip_cache.mkdir(parents=True, exist_ok=True)
    tool_path = bootstrap_tool_path(cache_root)
    if not (tool_path / "virtualenv" / "__init__.py").is_file():
        result = run_bounded(bootstrap_tool_command(cache_root, pip_cache), timeout=300)
        audit.append({"stage": "install-virtualenv-bootstrap", **result})
        if result["returncode"] != 0:
            return venv, fingerprint, audit, "virtualenv-bootstrap-install-failed"
    if interpreter.exists() and not interpreter_has_pip(interpreter):
        audit.append({
            "stage": "partial-venv",
            "status": "invalidated",
            "venv": str(venv),
        })
        shutil.rmtree(venv)
    if not interpreter.exists():
        result = run_bounded(
            create_venv_command(venv),
            timeout=300,
            env=virtualenv_environment(cache_root),
        )
        audit.append({"stage": "create-venv", **result})
        if result["returncode"] != 0:
            return venv, fingerprint, audit, "create-venv-failed"
    for index, command in enumerate(installation_commands(interpreter, pip_cache), start=1):
        result = run_bounded(command, timeout=1_200)
        audit.append({"stage": f"install-{index}", **result})
        if result["returncode"] != 0:
            if index == 3:
                return venv, fingerprint, audit, "cuda12.4-extension-build-failed"
            if index == 4:
                return venv, fingerprint, audit, "qlab-cudnn-runtime-install-failed"
            return venv, fingerprint, audit, "dependency-install-failed"
    exact, versions = exact_versions(interpreter)
    audit.append({"stage": "exact-version-validation", "status": "passed" if exact else "failed", "versions": versions})
    if not exact:
        return venv, fingerprint, audit, "exact-version-validation-failed"
    write_ready_atomic(venv, fingerprint, versions)
    audit.append({"stage": "atomic-ready-marker", "status": "passed"})
    return venv, fingerprint, audit, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--campaign", choices=BATCH_CAMPAIGNS)
    parser.add_argument("--authorize-full-execution", action="store_true")
    parser.add_argument(
        "--campaign-output-dir", type=Path,
        default=Path("results/h200-standard-primary"),
    )
    args = parser.parse_args()
    venv, fingerprint, audit, reason = ensure_runtime(args.cache_root, dry_run=args.dry_run)
    interpreter = venv_interpreter(venv)
    result: dict[str, Any] = {
        "passed": False,
        "fingerprint": fingerprint,
        "venv": str(venv),
        "interpreter": str(interpreter),
        "audit": audit,
    }
    if args.campaign:
        command = campaign_command(interpreter, args.campaign_output_dir, args.campaign)
        result["campaign"] = {"name": STANDARD_PRIMARY_CAMPAIGN, "batch": args.campaign, "command": list(command)}
    if reason:
        result["reason"] = reason
        print_result(result)
        return 1
    if args.dry_run:
        if args.campaign:
            result["campaign"]["status"] = "would-run"
        result["passed"] = True
        print_result(result)
        return 0
    if args.campaign and not args.authorize_full_execution:
        result["reason"] = "campaign-requires-authorize-full-execution"
        print_result(result)
        return 1
    smoke = run_bounded(
        (str(interpreter), str(Path(__file__).with_name("h200_smoke.py")),
         "--scientific", "--cache-fingerprint", fingerprint, "--venv", str(venv)),
        timeout=1_800,
    )
    result["smoke"] = smoke
    if smoke["returncode"] != 0:
        result["passed"] = False
        print_result(result)
        return 1
    if args.campaign:
        campaign = run_bounded(command, timeout=CAMPAIGN_TIMEOUT_SECONDS)
        result["campaign"]["result"] = campaign
        result["passed"] = campaign["returncode"] == 0
    else:
        result["passed"] = True
    print_result(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
