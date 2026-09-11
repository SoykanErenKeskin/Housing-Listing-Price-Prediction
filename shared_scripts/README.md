# shared_scripts/

Root-level **shared analysis / maintenance** utilities.

- **Not** a model-generation package (`v1` / `v2` / `v3` / `v4` are the model eras).
- Does **not** train price models; it pulls listings from the DB and writes data-health / audit reports.
- Requires repo-root `.env` with password-less `DATABASE_URL` (user `ml_pipeline`)
  and separate `DB_ROLE_PASSWORD`. The loader / resolver never logs secret values.
- Connection strings are built by `shared_scripts/db_url.py` (URL parse + password inject).
- Canonical Neon relations + legacy DF aliases: `shared_scripts/canonical_db.py`
  (`market.sale_listings`, `market.rental_listings`, `market.price_observations`,
  `geo.neighborhood_demographics`). Training pipelines still see legacy
  `city` / `county` / `district` column names after fetch for artifact compatibility.

Active model checkpoints live under `v3/` (best Kocaeli global = V24.1; best refreshed
Başiskele-only = V23). Visual/satellite work is under `v4/`.

---

## Listing inventory analysis

Run from **repo root**:

```powershell
python shared_scripts/analyze_listing_inventory.py --city Kocaeli

python shared_scripts/analyze_listing_inventory.py --city Kocaeli --county Başiskele

python shared_scripts/analyze_listing_inventory.py --city Kocaeli --purpose sale

python shared_scripts/analyze_listing_inventory.py --city Kocaeli --purpose rental

python shared_scripts/analyze_listing_inventory.py --city Kocaeli --county Başiskele --export-samples
```

Default output directory: `analysis_outputs/listing_inventory/<YYYY-MM-DD_HHMM>/` (gitignored).

Example artifacts:

- `inventory_summary.json`, `summary.md`
- `county_distribution.csv`, `district_distribution.csv`
- `sale_price_distribution.csv` / `rental_price_distribution.csv`
- `feature_missingness.csv`, `categorical_cardinality.csv`, `categorical_value_counts.csv`
- `location_quality_report.csv`, `model_readiness_report.csv`
- `basiskele_special_report.csv` (when scoped to Kocaeli / Başiskele)
- `duplicates_report.csv`, `suspicious_rows.csv`
- `plots/*.png` (disable with `--no-plots`)
- `samples/*` (with `--export-samples`)

---

## Related audit scripts

Also in this folder (analysis only; not training):

| File | Role |
|---|---|
| `audit_feature_reactivation.py` | Feature reactivation vs V24.1 + EDER app input coverage |
| `audit_basiskele_premium_residuals.py` | Başiskele premium residual / site-candidate audit |
| `audit_demographics_income_ses_features.py` | External SES / income feature audit helper |

Feature reactivation example:

```powershell
python shared_scripts/audit_feature_reactivation.py --city Kocaeli --source-site sahibinden
```

Default output: `analysis_outputs/feature_reactivation_audit_<YYYY-MM-DD_HHMM>/`.

---

## Files

| File | Role |
|---|---|
| `analyze_listing_inventory.py` | Main inventory analysis entrypoint |
| `audit_feature_reactivation.py` | V25 prep: DB vs model vs app feature gap audit |
| `db_utils.py` | DB engine + column-safe fetch helpers |
| `db_url.py` | Central `DATABASE_URL` + `DB_ROLE_PASSWORD` resolver |
| `canonical_db.py` | Canonical schema allowlist, fetch helpers, DF aliases |
| `smoke_ml_pipeline_db.py` | Read-only role/table smoke |
| `smoke_canonical_fetch_parity.py` | Bounded fetch + alias contract smoke |
| `env_loader.py` | Walk-to-root `.env` loader |
| `smoke_ml_pipeline_db.py` | Safe connectivity / permission smoke checks |
| `test_db_url.py` | Unit tests for URL resolver / redaction |

---

## Notes

This folder is for reusable data-quality tooling only. Prefer running from repo root
so `.env` and relative output paths resolve correctly.
