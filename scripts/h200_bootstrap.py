"""Install only missing H200 smoke dependencies, then run the bounded probe.

The Aerodrone base image owns Torch, CUDA, and Triton. This bootstrap never upgrades
or replaces them. Build logs are bounded so the issue-comment output stays useful.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


BOOTSTRAP_COMMANDS = (
    (
        sys.executable,
        "-m",
        "pip",
        "install",
        "jax[cuda12]==0.6.0",
        "brax==0.12.3",
        "ninja==1.11.1.4",
        "einops==0.8.1",
    ),
    (
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "mamba-ssm==2.3.2.post1",
    ),
)


def run_bounded(command: tuple[str, ...], timeout: int) -> dict:
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout.splitlines()[-40:],
        "stderr_tail": completed.stderr.splitlines()[-40:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"bootstrap_commands": BOOTSTRAP_COMMANDS}))
        return 0

    installs = []
    for command in BOOTSTRAP_COMMANDS:
        result = run_bounded(command, timeout=1_200)
        installs.append(result)
        if result["returncode"] != 0:
            print("H200_BOOTSTRAP_RESULT=" + json.dumps({
                "passed": False,
                "installs": installs,
                "reason": "dependency-install-failed",
            }, sort_keys=True))
            return 1

    smoke = run_bounded(
        (sys.executable, str(Path(__file__).with_name("h200_smoke.py"))),
        timeout=1_800,
    )
    print("H200_BOOTSTRAP_RESULT=" + json.dumps({
        "passed": smoke["returncode"] == 0,
        "installs": installs,
        "smoke": smoke,
    }, sort_keys=True))
    return smoke["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
