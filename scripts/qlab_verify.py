#!/usr/bin/env python3
"""Stage an immutable TiME source bundle and run bounded verification on qlab."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

SOURCE_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__", "results", "verification"}
APPROVED_HOSTS = {"qlab"}
FORBIDDEN_COMMAND_FRAGMENTS = {
    "run_brax_matrix",
    "--authorize-full-execution",
    "--full-campaign",
    "--campaign",
}


def _source_files() -> list[Path]:
    files = []
    for path in SOURCE_ROOT.rglob("*"):
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def _inventory() -> bytes:
    entries = []
    for path in _source_files():
        relative = path.relative_to(SOURCE_ROOT).as_posix()
        entries.append({
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return (json.dumps(entries, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _bundle() -> tuple[bytes, str]:
    inventory = _inventory()
    source_digest = hashlib.sha256(inventory).hexdigest()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for path in _source_files():
            archive.add(path, arcname=path.relative_to(SOURCE_ROOT).as_posix(), recursive=False)
        info = tarfile.TarInfo("SOURCE_INVENTORY.json")
        info.size = len(inventory)
        info.mtime = 0
        archive.addfile(info, io.BytesIO(inventory))
    return buffer.getvalue(), source_digest


def _remote(
    command: str, host: str, *, input_data: bytes | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, command],
        input=input_data,
        check=False,
    )


def _validate_command(command: list[str], host: str) -> None:
    if not command:
        raise ValueError("a bounded verification command is required after --")
    if any(
        fragment in argument
        for argument in command
        for fragment in FORBIDDEN_COMMAND_FRAGMENTS
    ):
        raise ValueError("full-campaign commands are forbidden")
    executable = Path(command[0]).name
    if executable not in {"python", "python3", "python3.10"}:
        raise ValueError("qlab verification only permits a Python test runner")
    expected_arguments = [
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_*.py",
        "-v",
    ]
    if command[1:] != expected_arguments:
        raise ValueError(
            "qlab verification only permits the canonical repository unittest suite"
        )
    if host not in APPROVED_HOSTS:
        raise ValueError("dynamic verification is restricted to the qlab SSH host")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-release-root", required=True)
    parser.add_argument("--remote-output-root", required=True)
    parser.add_argument("--remote-cache-root", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        _validate_command(command, args.host)
    except ValueError as error:
        parser.error(str(error))

    bundle, source_digest = _bundle()
    release = f"{args.remote_release_root.rstrip('/')}/{source_digest}"
    quoted_release = shlex.quote(release)
    setup = (
        f"set -eu; mkdir -p {shlex.quote(args.remote_release_root)} "
        f"{shlex.quote(args.remote_output_root)} {shlex.quote(args.remote_cache_root)}; "
        f"if [ -e {quoted_release} ]; then "
        f"test -f {quoted_release}/SOURCE_INVENTORY.json; "
        f"test \"$(sha256sum {quoted_release}/SOURCE_INVENTORY.json | cut -d' ' -f1)\" = "
        f"\"$(tar -xzOf - SOURCE_INVENTORY.json | sha256sum | cut -d' ' -f1)\"; "
        f"else mkdir {quoted_release}; tar -xzf - -C {quoted_release}; fi"
    )
    # A release directory is content addressed and never overwritten. Feed the
    # bundle only once; an existing release is validated by its inventory.
    result = _remote(setup, args.host, input_data=bundle)
    if result.returncode:
        return result.returncode

    remote_command = " ".join(shlex.quote(part) for part in command)
    execute = (
        f"set -eu; cd {quoted_release}; "
        f"export TIME_OUTPUT_ROOT={shlex.quote(args.remote_output_root)} "
        f"XDG_CACHE_HOME={shlex.quote(args.remote_cache_root)} "
        f"TRITON_CACHE_DIR={shlex.quote(args.remote_cache_root + '/triton')} "
        f"CUDA_VISIBLE_DEVICES=0 "
        f"PYTHONPATH={quoted_release}${{PYTHONPATH:+:$PYTHONPATH}}; "
        f"exec timeout --signal=TERM --kill-after=30s 1800s {remote_command}"
    )
    return _remote(execute, args.host).returncode


if __name__ == "__main__":
    raise SystemExit(main())
