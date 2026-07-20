"""Ensemble profile helpers for V19."""

from __future__ import annotations

from typing import Any


def resolve_selected_models(profile: str, fast_mode: bool = False) -> list[str]:
    profile = str(profile or "balanced").lower()
    if profile == "balanced":
        return ["ridge", "gradient_boosting", "extra_trees", "random_forest"]
    if profile == "no_ridge":
        return ["gradient_boosting", "extra_trees", "random_forest"]
    if profile == "tree_heavy":
        return ["extra_trees", "random_forest", "gradient_boosting"]
    if profile == "extra_trees_heavy":
        return ["extra_trees", "random_forest"]
    if profile == "gb_heavy":
        return ["gradient_boosting", "extra_trees"]
    raise ValueError(f"Unknown ensemble profile: {profile}")


def profile_notes(profile: str) -> list[str]:
    profile = str(profile or "balanced").lower()
    notes = {
        "balanced": ["V18-like balanced mix including ridge"],
        "no_ridge": ["Ridge removed to test shrink/mean-pulling reduction"],
        "tree_heavy": ["Tree models prioritized; ridge excluded"],
        "extra_trees_heavy": ["ExtraTrees/RF focused; watch overfit on n≈900"],
        "gb_heavy": ["GB + ExtraTrees; watch overfit on n≈900"],
    }
    return list(notes.get(profile, [f"profile={profile}"]))


def filter_model_comparison_for_profile(comparison_rows: list[dict[str, Any]], profile: str) -> list[dict[str, Any]]:
    allowed = set(resolve_selected_models(profile))
    return [r for r in comparison_rows if str(r.get("model", "")) in allowed]
