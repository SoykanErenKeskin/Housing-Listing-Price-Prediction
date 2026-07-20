"""Load project-root ``.env`` by walking parent directories.

Use from V19+ / active scripts::

    from env_loader import load_root_env
    load_root_env()

Or::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared_scripts"))
    from env_loader import load_root_env
    load_root_env()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def find_project_root(start: Optional[Path] = None) -> Optional[Path]:
    """Walk parents until a directory containing ``.env`` or ``MANIFEST.json``+``data/``."""
    cur = (start or Path.cwd()).resolve()
    if cur.is_file():
        cur = cur.parent
    for p in [cur, *cur.parents]:
        if (p / "MANIFEST.json").is_file() and (p / "data").is_dir():
            return p
        if (p / ".env").is_file() and (p / "data").is_dir():
            return p
    return None


def load_root_env(*, start: Optional[Path] = None, override: bool = False) -> Path:
    """Load the nearest project-root ``.env`` via python-dotenv.

    Returns the path that was loaded.

    Raises:
        FileNotFoundError: if no ``.env`` is found while walking parents.
        RuntimeError: if DATABASE_URL is still missing after load.
    """
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "python-dotenv is required. pip install python-dotenv"
        ) from exc

    start_path = (start or Path.cwd()).resolve()
    root = find_project_root(start_path)
    env_path: Optional[Path] = None
    if root is not None and (root / ".env").is_file():
        env_path = root / ".env"
    else:
        # fallback: any .env while walking from start
        cur = start_path if start_path.is_dir() else start_path.parent
        for p in [cur, *cur.parents]:
            cand = p / ".env"
            if cand.is_file():
                env_path = cand
                break

    if env_path is None:
        raise FileNotFoundError(
            "DATABASE_URL missing. Put .env in project root "
            "(same folder as MANIFEST.json / data/)."
        )

    load_dotenv(env_path, override=override)
    if not (os.getenv("DATABASE_URL") or os.getenv("DB_URL")):
        raise RuntimeError(
            "DATABASE_URL missing. Put .env in project root "
            f"(looked at {env_path})."
        )
    return env_path


if __name__ == "__main__":
    path = load_root_env()
    print(f"loaded_env={path}")
    print(f"DATABASE_URL_set={bool(os.getenv('DATABASE_URL') or os.getenv('DB_URL'))}")
