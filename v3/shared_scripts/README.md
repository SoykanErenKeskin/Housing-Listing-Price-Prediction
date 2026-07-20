# Shared scripts (V3)

Helpers used by V19+ packages under `v3/source_versions/`.

---

## `env_loader.py`

Loads the **repo-root** `.env` by walking parent directories from the current working
directory / call site until it finds `.env`.

```python
from env_loader import load_root_env
load_root_env()
```

Requirements:

- `python-dotenv` installed
- Root `.env` present with `DATABASE_URL`

Raises a clear error if `.env` or `DATABASE_URL` is missing. Never prints the secret value.

Also available at repo root: `shared_scripts/env_loader.py` (same idea for analysis tools).

---

## Secrets policy

- Keep a single root `.env` (see `.env.example`)
- Do not commit `.env`
- Do not recreate per-version secret files under archived trees
