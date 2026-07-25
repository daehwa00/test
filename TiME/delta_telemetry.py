"""Read-only telemetry for Mamba2 delta generators."""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
import torch.nn.functional as F


_STAT_KEYS = ("count", "mean", "std", "min", "max", "p05", "p50", "p95")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _summary(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("telemetry values must be non-empty and finite")
    quantiles = torch.quantile(values, torch.tensor((0.05, 0.5, 0.95), dtype=values.dtype))
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "p05": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p95": float(quantiles[2].item()),
    }


def _spectral_norm(weight: torch.Tensor, iterations: int = 16) -> float:
    """Deterministically estimate a matrix norm without touching module state."""
    matrix = weight.detach().to(device="cpu", dtype=torch.float64)
    if matrix.ndim != 2:
        raise ValueError("in_proj.weight must be a matrix")
    vector = torch.ones(matrix.shape[1], dtype=matrix.dtype)
    vector = F.normalize(vector, dim=0)
    estimate = 0.0
    for _ in range(iterations):
        left = matrix @ vector
        length = torch.linalg.vector_norm(left)
        if length.item() == 0.0:
            return 0.0
        left = left / length
        vector = matrix.T @ left
        length = torch.linalg.vector_norm(vector)
        if length.item() == 0.0:
            return 0.0
        vector = vector / length
        estimate = float(torch.linalg.vector_norm(matrix @ vector).item())
    return estimate


def _policy_mambas(agent: Any) -> list[tuple[int, Any]]:
    try:
        wrappers: Iterable[Any] = agent.policy_encoder.mambas
    except AttributeError as error:
        raise TypeError("agent must expose policy_encoder.mambas") from error
    layers = []
    for index, wrapper in enumerate(wrappers):
        mamba = getattr(wrapper, "mamba", None)
        in_proj = getattr(mamba, "in_proj", None)
        if mamba is None or in_proj is None:
            raise TypeError("each policy encoder entry must expose mamba.in_proj")
        for attribute in ("nheads", "dt_bias", "A_log"):
            if not hasattr(mamba, attribute):
                raise TypeError(f"policy Mamba layer {index} lacks {attribute}")
        layers.append((index, mamba))
    if not layers:
        raise ValueError("agent has no policy Mamba layers")
    return layers


def collect_policy_delta_telemetry(
    agent: Any, normalized_observation: torch.Tensor, total_steps: int
) -> dict[str, Any]:
    """Collect delta-generator telemetry without changing model or process state."""
    if not isinstance(normalized_observation, torch.Tensor):
        raise TypeError("normalized_observation must be a torch.Tensor")
    if normalized_observation.ndim != 3:
        raise ValueError("normalized_observation must have shape [B, T, O]")
    if not normalized_observation.is_floating_point():
        raise TypeError("normalized_observation must have a floating dtype")
    if normalized_observation.shape[0] <= 0 or normalized_observation.shape[1] <= 0:
        raise ValueError("normalized_observation requires positive B and T")
    if not bool(torch.isfinite(normalized_observation).all()):
        raise ValueError("normalized_observation must be finite")
    if isinstance(total_steps, bool) or not isinstance(total_steps, int):
        raise TypeError("total_steps must be an integer")
    if total_steps < 0:
        raise ValueError("total_steps must be non-negative")

    layers = _policy_mambas(agent)
    captured: dict[int, list[torch.Tensor]] = {index: [] for index, _ in layers}
    handles = []
    training_modes = [(module, module.training) for module in agent.modules()]

    def make_hook(index: int, mamba: Any):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            projected = output[0] if isinstance(output, tuple) else output
            if not isinstance(projected, torch.Tensor) or projected.shape[-1] < mamba.nheads:
                raise ValueError("in_proj output does not contain projected dt channels")
            captured[index].append(projected[..., -mamba.nheads :].detach().to("cpu", torch.float64))
        return hook

    try:
        for index, mamba in layers:
            handles.append(mamba.in_proj.register_forward_hook(make_hook(index, mamba)))
        agent.eval()
        resets = torch.zeros(
            normalized_observation.shape[:2], dtype=torch.bool, device=normalized_observation.device
        )
        with torch.inference_mode():
            agent.policy(normalized_observation, rollout=False, resets=resets)
    finally:
        for handle in handles:
            handle.remove()
        for module, was_training in training_modes:
            module.training = was_training

    layer_events = []
    for index, mamba in layers:
        if not captured[index]:
            raise RuntimeError(f"policy Mamba layer {index} did not execute in_proj")
        raw_dt = torch.cat(captured[index], dim=0)
        dt_bias = mamba.dt_bias.detach().to(device="cpu", dtype=torch.float64)
        a_log = mamba.A_log.detach().to(device="cpu", dtype=torch.float64)
        if dt_bias.numel() != mamba.nheads or a_log.numel() != mamba.nheads:
            raise ValueError("Mamba delta parameters do not match nheads")
        delta = F.softplus(raw_dt + dt_bias.reshape(*((1,) * (raw_dt.ndim - 1)), -1))
        timescale = 1.0 / (delta * torch.exp(a_log.reshape(*((1,) * (delta.ndim - 1)), -1)))
        projection = mamba.in_proj
        raw_norm = _spectral_norm(projection.weight)
        is_rebalancer = projection.__class__.__name__ == "MemoryRebalancer"
        gamma = None
        sigma = None
        effective_norm = raw_norm
        if is_rebalancer:
            gamma_tensor = projection.gamma.detach().to(device="cpu", dtype=torch.float64)
            u = projection.u.detach().to(device="cpu", dtype=torch.float64)
            v = projection.v.detach().to(device="cpu", dtype=torch.float64)
            weight = projection.weight.detach().to(device="cpu", dtype=torch.float64)
            gamma = _finite_number(gamma_tensor.reshape(-1)[0].item(), "gamma")
            sigma = _finite_number(
                torch.dot(u, weight @ v).item(), "sigma_estimate"
            )
            if abs(sigma) <= 1e-12:
                raise ValueError("MemoryRebalancer projection has zero spectral estimate")
            effective_norm = abs(gamma / sigma) * raw_norm
        layer_events.append(
            {
                "layer_index": index,
                "delta": _summary(delta),
                "timescale_steps": _summary(timescale),
                "generator": {
                    "nheads": int(mamba.nheads),
                    "dt_bias_mean": _finite_number(dt_bias.mean().item(), "dt_bias_mean"),
                    "raw_projection_spectral_norm": _finite_number(
                        raw_norm, "raw_projection_spectral_norm"
                    ),
                    "effective_projection_spectral_norm": _finite_number(effective_norm, "effective_projection_spectral_norm"),
                    "gamma": gamma,
                    "sigma_estimate": sigma,
                },
            }
        )
    return {
        "event": "delta_telemetry",
        "step": int(total_steps),
        "total_steps": int(total_steps),
        "probe": {
            "batch_size": int(normalized_observation.shape[0]),
            "sequence_length": int(normalized_observation.shape[1]),
            "observation_size": int(normalized_observation.shape[2]),
            "dtype": str(normalized_observation.dtype),
            "source": "first_training_rollout_after_normalization",
        },
        "reset_policy": "all_false_fixed_probe",
        "layers": layer_events,
    }


def summarize_delta_drift(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Validate delta telemetry events and summarize layer-wise drift."""
    events = list(events)
    if not events:
        raise ValueError("at least one delta telemetry event is required")
    expected_layers: set[int] | None = None
    expected_probe: dict[str, Any] | None = None
    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for event_index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("event") != "delta_telemetry":
            raise ValueError(f"event {event_index} is not delta telemetry")
        step = event.get("total_steps")
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("total_steps must be an integer")
        event_step = event.get("step")
        if isinstance(event_step, bool) or not isinstance(event_step, int):
            raise TypeError("step must be an integer")
        if event_step != step:
            raise ValueError("step and total_steps must agree")
        layers = event.get("layers")
        if event.get("reset_policy") != "all_false_fixed_probe":
            raise ValueError("delta telemetry reset_policy must be all_false_fixed_probe")
        probe = event.get("probe")
        if not isinstance(probe, dict):
            raise ValueError("delta telemetry probe metadata is required")
        if expected_probe is None:
            expected_probe = probe
        elif probe != expected_probe:
            raise ValueError("probe metadata must remain fixed across telemetry events")
        if not isinstance(layers, list) or not layers:
            raise ValueError("each event must contain non-empty layers")
        indexes = set()
        for layer in layers:
            if not isinstance(layer, dict) or not isinstance(layer.get("layer_index"), int):
                raise ValueError("layer_index must be an integer")
            index = layer["layer_index"]
            if index in indexes:
                raise ValueError("layer indexes must be unique per event")
            indexes.add(index)
            for field in ("delta", "timescale_steps"):
                summary = layer.get(field)
                if not isinstance(summary, dict) or set(_STAT_KEYS) - set(summary):
                    raise ValueError(f"layer {index} has malformed {field} summary")
                if not isinstance(summary["count"], int) or summary["count"] <= 0:
                    raise ValueError(f"layer {index} has invalid {field} count")
                for key in _STAT_KEYS[1:]:
                    _finite_number(summary[key], f"{field}.{key}")
            generator = layer.get("generator")
            if not isinstance(generator, dict):
                raise ValueError(f"layer {index} has malformed generator metadata")
            nheads = generator.get("nheads")
            if isinstance(nheads, bool) or not isinstance(nheads, int) or nheads <= 0:
                raise ValueError(f"layer {index} has invalid nheads")
            _finite_number(generator.get("dt_bias_mean"), "dt_bias_mean")
            norm = _finite_number(generator.get("effective_projection_spectral_norm"), "effective_projection_spectral_norm")
            _finite_number(
                generator.get("raw_projection_spectral_norm"),
                "raw_projection_spectral_norm",
            )
            gamma, sigma = generator.get("gamma"), generator.get("sigma_estimate")
            if (gamma is None) != (sigma is None):
                raise ValueError("gamma and sigma_estimate must both be null or both be finite")
            if gamma is not None:
                _finite_number(gamma, "gamma")
                _finite_number(sigma, "sigma_estimate")
            grouped.setdefault(index, []).append((step, {"layer": layer, "norm": norm}))
        if expected_layers is None:
            expected_layers = indexes
        elif indexes != expected_layers:
            raise ValueError("layer sets must be consistent across events")

    output_layers = []
    for index in sorted(grouped):
        points = sorted(grouped[index], key=lambda point: point[0])
        steps = [point[0] for point in points]
        if any(right <= left for left, right in zip(steps, steps[1:])):
            raise ValueError(f"layer {index} total_steps must be strictly increasing and unique")
        first, last = points[0][1], points[-1][1]
        first_delta, last_delta = first["layer"]["delta"], last["layer"]["delta"]
        first_time, last_time = first["layer"]["timescale_steps"], last["layer"]["timescale_steps"]
        span = steps[-1] - steps[0]
        mean_change = float(last_delta["mean"]) - float(first_delta["mean"])
        slope = 0.0 if span == 0 else mean_change * 1_000_000.0 / span
        output_layers.append(
            {
                "layer_index": index,
                "points": len(points),
                "first_step": steps[0],
                "last_step": steps[-1],
                "delta_mean_first": float(first_delta["mean"]),
                "delta_mean_last": float(last_delta["mean"]),
                "delta_mean_change": mean_change,
                "delta_std_first": float(first_delta["std"]),
                "delta_std_last": float(last_delta["std"]),
                "delta_mean_slope_per_million_steps": slope,
                "timescale_median_first": float(first_time["p50"]),
                "timescale_median_last": float(last_time["p50"]),
                "timescale_median_change": float(last_time["p50"]) - float(first_time["p50"]),
                "effective_projection_norm_first": first["norm"],
                "effective_projection_norm_last": last["norm"],
                "effective_projection_norm_change": last["norm"] - first["norm"],
            }
        )
    return {"event": "delta_drift_summary", "layers": output_layers}
