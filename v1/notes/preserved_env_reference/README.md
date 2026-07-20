# Preserved env reference (no secret values)

All former per-version `.env` files under `_preserved_env/` shared **one identical SHA256**.
They were consolidated into a **single repo-root `.env`** on 2026-07-20.

## Policy

- Do **not** commit `.env`
- Use [`.env.example`](../../../.env.example) as the template
- Active scripts should load root `.env` via `shared_scripts/env_loader.py` or `v3/shared_scripts/env_loader.py`
- Duplicate `*.env` copies were removed after consolidation (identical content)

This folder documents the consolidation; it must never contain live credentials.
