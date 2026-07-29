# Shared scripts (V3)

Helpers used by V19+ packages under `v3/source_versions/` (through V24.1).

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
- Root `.env` present with password-less `DATABASE_URL` (user `ml_pipeline`)
  and `DB_ROLE_PASSWORD`

Raises a clear error if `.env`, `DATABASE_URL`, or `DB_ROLE_PASSWORD` is missing.
Never prints secret values.

Also available at repo root: `shared_scripts/env_loader.py` and
`shared_scripts/db_url.py` (central connection resolver).

---

## Secrets policy

- Keep a single root `.env` (see `.env.example`)
- Do not commit `.env`
- Do not recreate per-version secret files under archived trees
