"""Central Neon/PostgreSQL URL resolver for the ML pipeline.

Combines password-less ``DATABASE_URL`` with ``DB_ROLE_PASSWORD`` using
``urllib.parse`` (no manual string concat / replace / regex mutation of secrets).

Never log or return secrets from helpers intended for display.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import quote, unquote, urlparse, urlunparse

EXPECTED_DB_USER = "ml_pipeline"

_ALLOWED_SCHEMES = frozenset(
    {
        "postgres",
        "postgresql",
        "postgresql+psycopg2",
        "postgresql+psycopg",
        "postgresql+psycopg2cffi",
    }
)

# Patterns that may appear in driver/SQLAlchemy exception strings.
_SECRET_RE = re.compile(
    r"(?i)("
    r"password\s*=\s*[^;\s]+|"
    r"pwd\s*=\s*[^;\s]+|"
    r":[^:@/\s]+@"  # :password@ in URLs
    r")"
)
_URL_RE = re.compile(
    r"(?i)\b(?:postgres|postgresql)(?:\+[\w]+)?://[^\s'\"<>]+"
)


class DatabaseUrlError(RuntimeError):
    """Fail-fast configuration / URL resolution error (never includes secrets)."""


@dataclass(frozen=True)
class ResolvedDatabaseTarget:
    """Safe, non-secret view of the resolved target."""

    user: str
    host: str
    port: Optional[int]
    database: str
    query: str
    scheme: str

    @property
    def label(self) -> str:
        return f"{self.user}@{self.host}/{self.database}"


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def sanitize_db_error(exc: BaseException | str) -> str:
    """Redact credentials / DSNs from exception text for logs and warnings."""
    text = str(exc)
    text = _URL_RE.sub("[REDACTED_DATABASE_URL]", text)
    text = _SECRET_RE.sub("[REDACTED]", text)
    # Belt-and-suspenders: if role password leaked into the message, blank it.
    pwd = _env("DB_ROLE_PASSWORD")
    if pwd and pwd in text:
        text = text.replace(pwd, "[REDACTED]")
    return text


def safe_database_label(
    *,
    database_url: str | None = None,
    database_url_override: str | None = None,
) -> str:
    """Return ``user@host/database`` from the password-less URL template."""
    raw = (
        (database_url_override or "").strip()
        or (database_url or "").strip()
        or _env("DATABASE_URL")
        or _env("DB_URL")
    )
    if not raw:
        return "ml_pipeline@unknown/unknown"
    try:
        parsed = urlparse(raw)
        user = unquote(parsed.username or EXPECTED_DB_USER) or EXPECTED_DB_USER
        host = parsed.hostname or "unknown"
        database = (parsed.path or "/").lstrip("/") or "unknown"
        return f"{user}@{host}/{database}"
    except Exception:
        return "ml_pipeline@unknown/unknown"


def describe_database_target(
    *,
    database_url: str | None = None,
    database_url_override: str | None = None,
) -> ResolvedDatabaseTarget:
    """Parse the template URL into a safe metadata object (no password)."""
    raw = (
        (database_url_override or "").strip()
        or (database_url or "").strip()
        or _env("DATABASE_URL")
        or _env("DB_URL")
    )
    if not raw:
        raise DatabaseUrlError("DATABASE_URL is missing.")
    parsed = urlparse(raw)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise DatabaseUrlError("DATABASE_URL has an unsupported or invalid scheme.")
    if not parsed.hostname:
        raise DatabaseUrlError("DATABASE_URL is invalid (missing host).")
    user = unquote(parsed.username or "")
    if user != EXPECTED_DB_USER:
        raise DatabaseUrlError(
            f"DATABASE_URL username must be '{EXPECTED_DB_USER}', got '{user or ''}'."
        )
    database = (parsed.path or "/").lstrip("/")
    if not database:
        raise DatabaseUrlError("DATABASE_URL is invalid (missing database name).")
    return ResolvedDatabaseTarget(
        user=user,
        host=parsed.hostname,
        port=parsed.port,
        database=database,
        query=parsed.query or "",
        scheme=parsed.scheme,
    )


def resolve_database_url(
    *,
    database_url: str | None = None,
    database_url_override: str | None = None,
    role_password: str | None = None,
    expected_user: str = EXPECTED_DB_USER,
) -> str:
    """Build a driver-ready URL: template ``DATABASE_URL`` + ``DB_ROLE_PASSWORD``.

    Parameters
    ----------
    database_url:
        Explicit template (rarely needed). Defaults to env ``DATABASE_URL`` / ``DB_URL``.
    database_url_override:
        Optional CLI ``--db-url`` template. Still receives ``DB_ROLE_PASSWORD``.
    role_password:
        Explicit password. Defaults to env ``DB_ROLE_PASSWORD``.
    expected_user:
        Must match the URL username (default ``ml_pipeline``).
    """
    raw = (
        (database_url_override or "").strip()
        or (database_url or "").strip()
        or _env("DATABASE_URL")
        or _env("DB_URL")
    )
    password = (
        role_password
        if role_password is not None
        else _env("DB_ROLE_PASSWORD")
    )

    if not raw:
        raise DatabaseUrlError(
            "DATABASE_URL is missing. Set a password-less URL for user "
            f"'{expected_user}' in the project-root .env."
        )
    if not password:
        raise DatabaseUrlError(
            "DB_ROLE_PASSWORD is missing. Set the ml_pipeline role password "
            "in the project-root .env (separate from DATABASE_URL)."
        )

    parsed = urlparse(raw)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise DatabaseUrlError("DATABASE_URL has an unsupported or invalid scheme.")
    if not parsed.hostname:
        raise DatabaseUrlError("DATABASE_URL is invalid (missing host).")

    username = unquote(parsed.username or "")
    if username != expected_user:
        raise DatabaseUrlError(
            f"DATABASE_URL username must be '{expected_user}', got '{username or ''}'."
        )

    database = (parsed.path or "/").lstrip("/")
    if not database:
        raise DatabaseUrlError("DATABASE_URL is invalid (missing database name).")

    # Optional host gate for test-branch safety (no secrets).
    must_contain = _env("DB_HOST_MUST_CONTAIN")
    if must_contain and must_contain.lower() not in (parsed.hostname or "").lower():
        raise DatabaseUrlError(
            "DATABASE_URL host failed DB_HOST_MUST_CONTAIN check "
            "(refusing to connect)."
        )

    user_q = quote(username, safe="")
    pass_q = quote(password, safe="")
    host = parsed.hostname
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{user_q}:{pass_q}@{host}{port}"

    return urlunparse(
        (
            parsed.scheme,
            netloc,
            parsed.path if parsed.path else f"/{database}",
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def create_sqlalchemy_engine(
    database_url_override: str | None = None,
    *,
    pool_pre_ping: bool = True,
    **engine_kwargs: Any,
):
    """Create a SQLAlchemy engine via ``resolve_database_url`` (secrets never logged)."""
    try:
        from sqlalchemy import create_engine as _create_engine
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "sqlalchemy is not installed. Install sqlalchemy and psycopg2-binary."
        ) from None

    try:
        url = resolve_database_url(database_url_override=database_url_override)
        return _create_engine(url, pool_pre_ping=pool_pre_ping, **engine_kwargs)
    except DatabaseUrlError:
        raise
    except Exception as exc:
        label = safe_database_label(database_url_override=database_url_override)
        raise RuntimeError(
            f"Failed to create DB engine for {label}: {sanitize_db_error(exc)}"
        ) from None


def ensure_shared_scripts_on_path(start: Optional[Any] = None) -> str:
    """Insert repo ``shared_scripts/`` on ``sys.path`` if needed; return that path."""
    import sys
    from pathlib import Path

    start_path = Path(start or Path.cwd()).resolve()
    if start_path.is_file():
        start_path = start_path.parent
    for p in [start_path, *start_path.parents]:
        cand = p / "shared_scripts"
        if (cand / "db_url.py").is_file():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
            return s
    raise DatabaseUrlError(
        "Could not locate shared_scripts/db_url.py from the current path."
    )
