"""V3 shim: re-export repo-root ``shared_scripts.env_loader`` (no circular import)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

# v3/shared_scripts/env_loader.py -> parents[2] == repo root
_ROOT_ENV = Path(__file__).resolve().parents[2] / "shared_scripts" / "env_loader.py"
if not _ROOT_ENV.is_file():
    raise ImportError(f"Canonical env_loader not found at {_ROOT_ENV}")

_spec = importlib.util.spec_from_file_location("ml_pipeline_root_env_loader", _ROOT_ENV)
if _spec is None or _spec.loader is None:
    raise ImportError("Could not load canonical env_loader module spec")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

find_project_root = _mod.find_project_root
load_root_env = _mod.load_root_env

__all__ = ["find_project_root", "load_root_env"]


if __name__ == "__main__":
    path = load_root_env()
    print(f"loaded_env_name={path.name}")
