"""Unit tests for shared_scripts.db_url (no live DB, no real secrets)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from db_url import (  # noqa: E402
    EXPECTED_DB_USER,
    DatabaseUrlError,
    resolve_database_url,
    safe_database_label,
    sanitize_db_error,
)


class DbUrlResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = {
            k: os.environ.get(k)
            for k in ("DATABASE_URL", "DB_URL", "DB_ROLE_PASSWORD", "DB_HOST_MUST_CONTAIN")
        }
        for k in ("DATABASE_URL", "DB_URL", "DB_ROLE_PASSWORD", "DB_HOST_MUST_CONTAIN"):
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        for k, v in self._env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_missing_database_url(self) -> None:
        os.environ["DB_ROLE_PASSWORD"] = "secret-pass"
        with self.assertRaises(DatabaseUrlError) as ctx:
            resolve_database_url()
        self.assertIn("DATABASE_URL", str(ctx.exception))
        self.assertNotIn("secret-pass", str(ctx.exception))

    def test_missing_role_password(self) -> None:
        os.environ["DATABASE_URL"] = (
            "postgresql://ml_pipeline@ep-test-pooler.example/neondb?sslmode=require"
        )
        with self.assertRaises(DatabaseUrlError) as ctx:
            resolve_database_url()
        self.assertIn("DB_ROLE_PASSWORD", str(ctx.exception))

    def test_invalid_scheme(self) -> None:
        with self.assertRaises(DatabaseUrlError):
            resolve_database_url(
                database_url="mysql://ml_pipeline@host/db",
                role_password="x",
            )

    def test_wrong_username(self) -> None:
        with self.assertRaises(DatabaseUrlError) as ctx:
            resolve_database_url(
                database_url="postgresql://neondb_owner@host/neondb?sslmode=require",
                role_password="x",
            )
        self.assertIn(EXPECTED_DB_USER, str(ctx.exception))
        self.assertIn("neondb_owner", str(ctx.exception))

    def test_password_url_encoding_and_query_preserved(self) -> None:
        template = (
            "postgresql://ml_pipeline@ep-test-pooler.example:5432/neondb"
            "?sslmode=require&channel_binding=require"
        )
        password = "p@ss:word/with spaces?"
        resolved = resolve_database_url(database_url=template, role_password=password)
        parsed = urlparse(resolved)
        self.assertEqual(parsed.scheme, "postgresql")
        self.assertEqual(unquote(parsed.username or ""), "ml_pipeline")
        self.assertEqual(unquote(parsed.password or ""), password)
        self.assertEqual(parsed.hostname, "ep-test-pooler.example")
        self.assertEqual(parsed.port, 5432)
        self.assertEqual(parsed.path, "/neondb")
        qs = parse_qs(parsed.query)
        self.assertEqual(qs.get("sslmode"), ["require"])
        self.assertEqual(qs.get("channel_binding"), ["require"])
        # Resolved URL must not appear in safe label
        label = safe_database_label(database_url=template)
        self.assertEqual(label, "ml_pipeline@ep-test-pooler.example/neondb")
        self.assertNotIn(password, label)

    def test_override_template_still_injects_password(self) -> None:
        os.environ["DATABASE_URL"] = "postgresql://ml_pipeline@other-host/neondb?sslmode=require"
        os.environ["DB_ROLE_PASSWORD"] = "env-pass"
        resolved = resolve_database_url(
            database_url_override=(
                "postgresql://ml_pipeline@override-pooler.example/neondb?sslmode=require"
            )
        )
        parsed = urlparse(resolved)
        self.assertEqual(parsed.hostname, "override-pooler.example")
        self.assertEqual(unquote(parsed.password or ""), "env-pass")

    def test_host_must_contain_gate(self) -> None:
        os.environ["DB_HOST_MUST_CONTAIN"] = "test-branch"
        with self.assertRaises(DatabaseUrlError):
            resolve_database_url(
                database_url="postgresql://ml_pipeline@prod-pooler.example/neondb?sslmode=require",
                role_password="x",
            )
        ok = resolve_database_url(
            database_url=(
                "postgresql://ml_pipeline@ep-test-branch-pooler.example/neondb?sslmode=require"
            ),
            role_password="x",
        )
        self.assertIn("test-branch", urlparse(ok).hostname or "")

    def test_sanitize_db_error_redacts_url_and_password(self) -> None:
        os.environ["DB_ROLE_PASSWORD"] = "super-secret-role-pass"
        msg = (
            "could not connect to server: "
            "postgresql://ml_pipeline:super-secret-role-pass@ep-test.example/neondb?sslmode=require "
            "password=super-secret-role-pass"
        )
        cleaned = sanitize_db_error(msg)
        self.assertNotIn("super-secret-role-pass", cleaned)
        self.assertIn("[REDACTED", cleaned)


if __name__ == "__main__":
    unittest.main()
