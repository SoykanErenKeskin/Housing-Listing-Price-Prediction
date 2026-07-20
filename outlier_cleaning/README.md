# Listing Sales + Rental Outlier Cleaning Package

Standalone utility that cleans outliers from Neon/CSV listing exports and routes
suspicious rows into a separate review file.

This package is **independent** of the V17–V19 training pipelines (those have their
own in-pipeline cleaners). Use it for offline CSV QC / thesis-era exports.

---

## Inputs

```text
data/input/sale_listings.csv
data/input/rental_listings.csv
```

(Local CSV dumps under `outlier_cleaning/data/input|output/` are gitignored.)

---

## Run

From `outlier_cleaning/`:

```bash
python src/clean_outliers.py
```

Custom paths:

```bash
python src/clean_outliers.py \
  --sales data/input/sale_listings.csv \
  --rental data/input/rental_listings.csv
```

---

## Outputs

```text
data/output/sales_with_outlier_flags.csv
data/output/sales_cleaned.csv
data/output/sales_removed_outliers.csv
data/output/sales_review_needed.csv

data/output/rental_with_outlier_flags.csv
data/output/rental_cleaned.csv
data/output/rental_removed_outliers.csv
data/output/rental_review_needed.csv

reports/sales_quality_report.json
reports/rental_quality_report.json
reports/combined_quality_report.json
reports/sales_group_outlier_summary.csv
reports/rental_group_outlier_summary.csv
```

---

## Logic

Rows are split into:

1. **Removed outliers** — clear problems recommended for exclusion from training.
2. **Review needed** — not auto-deleted, but worth manual inspection.

### Sale checks (high level)

- Extreme total price / unit price
- Implausible gross/net m² relationships
- Group IQR outliers (district + rooms + m² segment)
- Global MAD/IQR outliers
- Special title patterns (duplex / villa / luxury / urgent / opportunity, …)

### Rental checks (high level)

- Extreme monthly rent / rent per m²
- Implausible gross/net m²
- Suspicious deposit/rent ratios
- Group IQR + special furnished/luxury/daily/villa/duplex titles

---

## Raw JSON support

Some helper columns may live inside a `raw` JSON blob in Neon exports. The script
lifts common detail prefixes to top-level fields when present, for example:

```text
front_*, view_*, transport_*, near_*, out_*, in_*, subtype_*
building_age_raw, building_age_group, detail_*
```

---

## Thresholds

See `config.json` in this package for numeric gates and group settings.
