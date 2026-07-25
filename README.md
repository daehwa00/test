## TiME Brax experiments

This repository contains the submitted TiME implementation and the corrected protocol
`time-brax-corrected-v2`. Corrected-v2 changes sequence-axis handling, reset masks,
random-number isolation, and evaluation. It is not numerically equivalent to submitted
or reported results; historical results must never be relabeled as corrected-v2.

### Installation

Use a clean, repository-local environment. Do not copy or overwrite files in
`site-packages`: the vanilla Mamba-2 branch imports the pinned upstream package directly,
while TiME feature branches import the repository-local implementation.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --no-deps -r requirements-torch.txt
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Declared runtime versions are pinned in `requirements.txt`. Native CUDA/Triton builds
and dynamic verification are not supported on developer workstations.
The Torch bootstrap is intentionally dependency-free; `requirements.txt` then installs
the cuDNN 9.8 runtime required by JAX 0.6.0 while retaining Torch 2.6.0+cu124.

### Running

The existing command remains the submitted TiME default (EF and MR enabled), but it
must be executed in the pinned qlab environment:

```bash
python main.py --env halfcheetah --seed 0
```

`python run_brax_matrix.py` is a portable dry run by default and emits the deterministic
80-run TiME-versus-vanilla manifest. Actual matrix execution requires both `--execute`
and `--authorize-full-execution`; the qlab verification wrapper deliberately blocks that
campaign path. Corrected-v2 runs record protocol identity and JSON history. The training
diagnostic is named `training_unroll_reward`; it is not the deterministic evaluation
metric.

### qlab-only dynamic verification

Use the repository wrapper to stage an immutable source bundle and run a bounded remote
verification command on qlab. It uses `ssh -o BatchMode=yes`, separate remote release,
output, and cache roots, and returns the remote command's exit status.

```bash
python scripts/qlab_verify.py \
  --host qlab \
  --remote-release-root /home/qlab/TiME-neurips2026/releases \
  --remote-output-root /home/qlab/TiME-neurips2026/results \
  --remote-cache-root /home/qlab/TiME-neurips2026/cache \
  -- /home/qlab/TiME-neurips2026/envs/torch260-cu124/bin/python \
     -m unittest discover -s tests -p 'test_*.py' -v
```

The wrapper rejects full-campaign commands. Dynamic CUDA, Triton, build, smoke, and
training commands must run through qlab; local source editing is portable but is not
scientific verification. qlab artifacts remain the authoritative evidence.

### Licenses

See `LICENSES.md` for third-party licenses.
