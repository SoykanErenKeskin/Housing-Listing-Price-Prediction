"""Target profile helpers for V19.

Smoke stage: residual_log is the active path (existing pipeline target_mode=residual).
direct_price / hybrid are declared for later ablation wiring.
"""

from __future__ import annotations

from typing import Any


SUPPORTED_TARGET_PROFILES = ("residual_log", "direct_price", "hybrid")


def resolve_pipeline_target_mode(target_profile: str) -> str:
    """Map V19 target profile → existing pipeline target_mode."""
    p = str(target_profile or "residual_log").lower()
    if p == "residual_log":
        return "residual"
    if p == "direct_price":
        return "raw"
    if p == "hybrid":
        # Stage-1 smoke: hybrid falls back to residual until full hybrid blend lands.
        return "residual"
    raise ValueError(f"Unknown target profile: {target_profile}")


def parse_hybrid_lambda_grid(raw: str) -> list[float]:
    vals: list[float] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals or [0.0, 0.1, 0.2, 0.3, 0.5]


def target_profile_notes(target_profile: str) -> list[str]:
    p = str(target_profile or "residual_log").lower()
    if p == "residual_log":
        return ["target = log(unit_price) - log(location_baseline)"]
    if p == "direct_price":
        return ["target = unit_price_gross (pipeline target_mode=raw)"]
    if p == "hybrid":
        return [
            "hybrid residual+direct blend planned",
            "stage1 smoke uses residual path until hybrid ablation is enabled",
        ]
    return [f"target_profile={p}"]


def build_target_profile_report(
    *,
    target_profile: str,
    hybrid_lambda_grid: str,
    selected_lambda: float | None = None,
) -> dict[str, Any]:
    return {
        "target_profile": target_profile,
        "pipeline_target_mode": resolve_pipeline_target_mode(target_profile),
        "hybrid_lambda_grid": parse_hybrid_lambda_grid(hybrid_lambda_grid),
        "selected_lambda": selected_lambda,
        "notes": target_profile_notes(target_profile),
    }
