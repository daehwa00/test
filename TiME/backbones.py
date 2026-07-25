"""Explicit TiME/vanilla-Mamba2 backbone selection."""

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch.nn as nn


@dataclass(frozen=True)
class BackboneSpec:
    label: str
    ef_enabled: bool
    mr_enabled: bool
    reset_dt: float = 5.0


def resolve_backbone_spec(
    ef_enabled: bool = True, mr_enabled: bool = True, reset_dt: float = 5.0
) -> BackboneSpec:
    if not isinstance(ef_enabled, bool) or not isinstance(mr_enabled, bool):
        raise TypeError("ef_enabled and mr_enabled must be bool")
    if not isinstance(reset_dt, (int, float)) or isinstance(reset_dt, bool):
        raise TypeError("reset_dt must be a finite number")
    reset_dt = float(reset_dt)
    if not math.isfinite(reset_dt):
        raise ValueError("reset_dt must be finite")
    if reset_dt <= 0.0:
        raise ValueError("reset_dt must be positive")
    if not ef_enabled and reset_dt != 5.0:
        raise ValueError("reset_dt is only valid when ef_enabled is True")
    labels = {
        (True, True): "time",
        (True, False): "ef-only",
        (False, True): "mr-only",
        (False, False): "vanilla-mamba2",
    }
    return BackboneSpec(labels[(ef_enabled, mr_enabled)], ef_enabled, mr_enabled, reset_dt)


def _assert_vanilla_has_no_time_state(module: nn.Module) -> None:
    forbidden_attributes = {"gamma", "u", "v", "ef_enabled", "mr_enabled", "reset_dt"}
    for name, child in module.named_modules():
        if child.__class__.__name__ == "MemoryRebalancer":
            raise AssertionError(f"vanilla backbone contains MemoryRebalancer at {name}")
        present = forbidden_attributes.intersection(vars(child))
        if present:
            raise AssertionError(f"vanilla backbone retains TiME state at {name}: {sorted(present)}")


def build_backbone(
    *,
    d_model: int,
    d_state: int = 128,
    d_conv: int = 4,
    expand: int = 2,
    ef_enabled: bool = True,
    mr_enabled: bool = True,
    reset_dt: float = 5.0,
    **kwargs: Any,
) -> nn.Module:
    """Build one of the four corrected-v2 Mamba2 states.

    The false/false state deliberately imports the installed upstream class directly;
    it never passes TiME-only arguments to that constructor.
    """
    spec = resolve_backbone_spec(ef_enabled, mr_enabled, reset_dt)
    if kwargs.get("process_group") is not None:
        raise ValueError("process groups are not supported by corrected-v2 backbones")
    common: Mapping[str, Any] = {
        "d_model": d_model,
        "d_state": d_state,
        "d_conv": d_conv,
        "expand": expand,
        **kwargs,
    }
    if not (spec.ef_enabled or spec.mr_enabled):
        from TiME.provenance import upstream_mamba2_identity

        upstream_mamba2_identity()
        from mamba_ssm.modules.mamba2 import Mamba2 as UpstreamMamba2

        backbone = UpstreamMamba2(**common)
        _assert_vanilla_has_no_time_state(backbone)
        return backbone

    from mamba2 import Mamba2 as TiMEMamba2

    return TiMEMamba2(
        **common,
        ef_enabled=spec.ef_enabled,
        mr_enabled=spec.mr_enabled,
        reset_dt=spec.reset_dt,
    )
