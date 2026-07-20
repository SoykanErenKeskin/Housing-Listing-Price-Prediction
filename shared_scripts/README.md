# shared_scripts/

Root-level **shared analysis / maintenance** utilities.

- **Not** a model-generation package (`v1` / `v2` / `v3` are the model eras).
- Does **not** train price models; it pulls listings from the DB and writes data-health reports.
- Requires repo-root `.env` with `DATABASE_URL`. The loader never logs secret values.

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

---

## Outputs

Default directory:

`analysis_outputs/listing_inventory/<YYYY-MM-DD_HHMM>/`

(gitignored)

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

## Files

| File | Role |
|---|---|
| `analyze_listing_inventory.py` | Main inventory analysis entrypoint |
| `db_utils.py` | DB engine + column-safe fetch helpers |
| `env_loader.py` | Walk-to-root `.env` loader |

---

## Notes

Active model development remains under `v3/`. This folder is for reusable data-quality tooling only.
