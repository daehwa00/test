"""Source identity and shared-initialization evidence for corrected-v2 backbones."""

from __future__ import annotations

import hashlib
import importlib
import json
import importlib.metadata
import inspect
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn


PINNED_MAMBA_VERSION = "2.2.4"


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def upstream_mamba2_identity(
    expected_version: str = PINNED_MAMBA_VERSION,
) -> Mapping[str, str]:
    """Validate installed upstream source against the reviewed 2.2.4 manifest."""
    from mamba_ssm.modules.mamba2 import Mamba2 as UpstreamMamba2

    manifest_path = Path(__file__).with_name("upstream_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = importlib.metadata.version("mamba-ssm")
    if version != expected_version or manifest.get("version") != expected_version:
        raise RuntimeError(
            f"mamba-ssm version {version!r} does not match pinned {expected_version!r}"
        )
    if UpstreamMamba2.__module__ != "mamba_ssm.modules.mamba2":
        raise RuntimeError("vanilla Mamba2 is not the direct upstream module")

    observed = {}
    for module_name, expected_hash in manifest["files"].items():
        module = importlib.import_module(module_name)
        module_path = Path(inspect.getfile(module)).resolve()
        observed_hash = sha256_file(module_path)
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"upstream source mismatch for {module_name}: "
                f"expected {expected_hash}, observed {observed_hash}"
            )
        observed[module_name] = (module_path, observed_hash)

    module_path, module_hash = observed["mamba_ssm.modules.mamba2"]
    fused_path, fused_hash = observed["mamba_ssm.ops.triton.ssd_combined"]
    return {
        "distribution": "mamba-ssm",
        "version": version,
        "class_module": UpstreamMamba2.__module__,
        "source_path": str(module_path),
        "source_sha256": module_hash,
        "fused_op_path": str(fused_path),
        "fused_op_sha256": fused_hash,
        "manifest_sha256": sha256_file(manifest_path),
    }


def local_mamba2_identity() -> Mapping[str, str]:
    from mamba2 import Mamba2 as TiMEMamba2

    module_path = Path(inspect.getfile(TiMEMamba2)).resolve()
    return {
        "class_module": TiMEMamba2.__module__,
        "source_path": str(module_path),
        "source_sha256": sha256_file(module_path),
    }


def backbone_provenance(expected_version: str = PINNED_MAMBA_VERSION) -> Mapping[str, Mapping[str, str]]:
    return {
        "upstream": upstream_mamba2_identity(expected_version),
        "local": local_mamba2_identity(),
    }


def shared_tensor_names(left: nn.Module, right: nn.Module) -> tuple[str, ...]:
    """Return deterministic names with exactly compatible state-dict tensors."""
    left_state = left.state_dict()
    right_state = right.state_dict()
    return tuple(
        name
        for name in sorted(left_state.keys() & right_state.keys())
        if left_state[name].shape == right_state[name].shape
        and left_state[name].dtype == right_state[name].dtype
    )


def copy_shared_tensors(source: nn.Module, target: nn.Module) -> tuple[str, ...]:
    """Copy only compatible common tensors, leaving treatment-specific state intact."""
    names = shared_tensor_names(source, target)
    source_state = source.state_dict()
    target_state = target.state_dict()
    with torch.no_grad():
        for name in names:
            target_state[name].copy_(source_state[name])
    return names


def refresh_memory_rebalancer_buffers(module: nn.Module) -> int:
    """Rebuild treatment-only singular-vector buffers after shared-weight copying."""
    refreshed = 0
    with torch.no_grad():
        for child in module.modules():
            if child.__class__.__name__ != "MemoryRebalancer":
                continue
            u = torch.linalg.svd(child.weight.T, full_matrices=False)[-1][0]
            v = torch.linalg.svd(child.weight, full_matrices=False)[-1][0]
            child.u.copy_(u)
            child.v.copy_(v)
            sigma = torch.einsum("i,ij,j->", child.u, child.weight, child.v)
            if not torch.isfinite(sigma) or sigma.abs().item() <= 1e-12:
                raise RuntimeError("MemoryRebalancer has an invalid spectral denominator")
            refreshed += 1
    return refreshed
def assert_shared_tensors_equal(left: nn.Module, right: nn.Module) -> tuple[str, ...]:
    """Fail unless every compatible common tensor has identical values."""
    names = shared_tensor_names(left, right)
    left_state = left.state_dict()
    right_state = right.state_dict()
    unequal = [
        name for name in names if not torch.equal(left_state[name], right_state[name])
    ]
    if unequal:
        raise RuntimeError(f"shared initialization mismatch: {unequal}")
    return names



def shared_initialization_digest(
    modules: Iterable[nn.Module], names: Iterable[str] | None = None
) -> str:
    """Hash named common tensors in a device-independent canonical byte order."""
    module_list = tuple(modules)
    if not module_list:
        raise ValueError("at least one module is required")
    if names is None:
        common = set(module_list[0].state_dict())
        for module in module_list[1:]:
            common.intersection_update(module.state_dict())
        names = sorted(
            name
            for name in common
            if all(
                module.state_dict()[name].shape == module_list[0].state_dict()[name].shape
                and module.state_dict()[name].dtype == module_list[0].state_dict()[name].dtype
                for module in module_list[1:]
            )
        )
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8") + b"\0")
        for module in module_list:
            tensor = module.state_dict()[name].detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
