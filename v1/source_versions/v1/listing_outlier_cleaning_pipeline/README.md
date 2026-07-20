# Listing Outlier Cleaning Pipeline

**Era:** V1 companion to the thesis pilot  
**Path:** `v1/source_versions/v1/listing_outlier_cleaning_pipeline/`

Cleans a listing portal listing CSV **before** model training.

## Purpose

- Drop missing or unrealistic price / m² values
- Flag extreme gross ₺/m² outliers
- Mark duplicate listings when the same id appears again
- Emit cleaned data, removed outliers, and a JSON report

## Input

```text
data/input/listing_dataset.csv
```

## Outputs

```text
data/output/listing_dataset_cleaned.csv
data/output/listing_dataset_removed_outliers.csv
data/output/listing_dataset_marked_with_outlier_reason.csv
reports/outlier_cleaning_report.json
```

## How to run

```bash
python src/clean_outliers.py
```

## Cleaning logic

1. Numeric coercion for `price`, `gross_m2`, `unit_price_gross`
2. If `unit_price_gross` is empty: `unit_price_gross = price / gross_m2`
3. Basic validity filters (extreme gross m², ₺/m², total price, missing essentials)
4. Duplicate filter: keep first `classified_id`, send repeats to removed set
5. Global quantile filter: mark bottom/top **1%** of ₺/m²
6. District IQR: within `city + county + district` when enough rows exist
7. County IQR: wider check at `city + county`

## Example result (historical small CSV)

| Stage | Count |
|---|---:|
| Raw rows | 200 |
| Clean remaining | 190 |
| Removed / marked | 10 |
| Removal rate | 5.00% |

## Tuning

Edit the `CONFIG` block in `src/clean_outliers.py`.

Softer:

```python
"lower_quantile": 0.005,
"upper_quantile": 0.995
```

Stricter:

```python
"lower_quantile": 0.02,
"upper_quantile": 0.98
```

## Thesis note

Every removed row keeps an `outlier_reason`, so the thesis can report criteria and
counts transparently. Cleaning runs at both global distribution and
district/county IQR levels.
