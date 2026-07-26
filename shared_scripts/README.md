# shared_scripts/

Root-level **shared analysis / maintenance** utilities.

- **Not** a model-generation package (`v1` / `v2` / `v3` / `v4` are the model eras).
- Does **not** train price models; it pulls listings from the DB and writes data-health / audit reports.
- Requires repo-root `.env` with `DATABASE_URL`. The loader never logs secret values.

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
- `feature_missingness.csv`, `categorical_cardinality.csv`
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
| `audit_basiskele_premium_residuals.py` | Başiskele premium residual / site-candidate audit |
| `audit_demographics_income_ses_features.py` | External SES / income feature audit helper |

---

## Files

| File | Role |
|---|---|
| `analyze_listing_inventory.py` | Main inventory analysis entrypoint |
| `db_utils.py` | DB engine + column-safe fetch helpers |
| `env_loader.py` | Walk-to-root `.env` loader |

---

## Notes

This folder is for reusable data-quality tooling only. Prefer running from repo root
so `.env` and relative output paths resolve correctly.
