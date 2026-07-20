# Preserved .env files

These secrets were **removed from version trees during reorg** (not deleted).
They are **not** inside `v1/` or `v2/` source trees.

To run an archived pipeline, copy the matching file back as `.env` into that version folder, e.g.:

`Copy-Item .\_preserved_env\v18_basiskele.env .\v2\source_versions\v18_basiskele\.env`

Do not commit these files.
