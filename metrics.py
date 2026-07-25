"""Return-curve metrics used by corrected-v2 result aggregation."""

from __future__ import annotations
import math

from typing import Any, Iterable, Mapping, Sequence

RETURN_KEYS = (
    "evaluation_mean_return",
    "episode_return",
    "eval/episode_return",
    "eval_return",
    "mean_return",
    "return",
)
STEP_KEYS = ("total_steps", "step", "timesteps")


def return_curve(events: Iterable[Mapping[str, Any]]) -> list[tuple[float, float]]:
    """Extract finite return observations in recorded order.

    Events without an explicit step use their ordinal position, retaining a
    useful metric for externally supplied evaluation histories.
    """
    curve: list[tuple[float, float]] = []
    for ordinal, event in enumerate(events):
        if event.get("event") != "evaluation":
            continue
        value = next((event[key] for key in RETURN_KEYS if key in event), None)
        if value is None:
            continue
        try:
            y = float(value)
            x = float(next((event[key] for key in STEP_KEYS if key in event), ordinal))
        except (TypeError, ValueError):
            continue
        if x == x and y == y and abs(x) != float("inf") and abs(y) != float("inf"):
            curve.append((x, y))
    return curve


def evaluation_events_match_plan(
    events: Sequence[Mapping[str, Any]],
    plan: Mapping[int, Mapping[str, Any]],
) -> bool:
    if len(events) != len(plan):
        return False
    seen_steps = set()
    for event in events:
        step = event.get("total_steps", event.get("step"))
        if not isinstance(step, int) or isinstance(step, bool) or step in seen_steps:
            return False
        expected = plan.get(step)
        if expected is None:
            return False
        seen_steps.add(step)
        if (
            event.get("evaluation_kind") != expected["evaluation_kind"]
            or event.get("evaluation_episodes") != expected["evaluation_episodes"]
        ):
            return False
        episode_returns = event.get("episode_return")
        if (
            not isinstance(episode_returns, list)
            or len(episode_returns) != expected["evaluation_episodes"]
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in episode_returns
            )
        ):
            return False
        mean_return = event.get("evaluation_mean_return")
        if (
            not isinstance(mean_return, (int, float))
            or isinstance(mean_return, bool)
            or not math.isfinite(float(mean_return))
            or not math.isclose(
                float(mean_return),
                sum(float(value) for value in episode_returns) / len(episode_returns),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            return False
    return seen_steps == set(plan)


def final_return(events: Iterable[Mapping[str, Any]]) -> float | None:
    curve = return_curve(events)
    return curve[-1][1] if curve else None


def auc(events: Iterable[Mapping[str, Any]]) -> float | None:
    """Return endpoint-normalized trapezoidal AUC."""
    curve = return_curve(events)
    if not curve:
        return None
    if len(curve) == 1:
        return 0.0
    area = 0.0
    for (left_x, left_y), (right_x, right_y) in zip(curve, curve[1:]):
        if right_x < left_x:
            raise ValueError("Return history steps must be nondecreasing")
        area += (right_x - left_x) * (left_y + right_y) / 2.0
    horizon = curve[-1][0] - curve[0][0]
    if horizon <= 0:
        raise ValueError("Return history must span a positive step interval")
    return area / horizon


def summarize_history(history: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    events = history.get("events", []) if isinstance(history, Mapping) else history
    return {"final_return": final_return(events), "auc": auc(events)}
