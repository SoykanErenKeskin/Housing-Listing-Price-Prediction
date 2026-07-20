from pathlib import Path
import json
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "input" / "listing_dataset.csv"
OUTPUT_DIR = ROOT / "data" / "output"
REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


CONFIG = {
    "target_unit_price_col": "unit_price_gross",
    "price_col": "price",
    "gross_m2_col": "gross_m2",
    "location_cols": ["city", "county", "district"],
    "soft_numeric_limits": {
        "gross_m2_min": 25,
        "gross_m2_max": 500,
        "unit_price_gross_min": 5000,
        "unit_price_gross_max": 300000,
        "price_min": 250000,
        "price_max": 100000000
    },
    "global_quantile_filter": {
        "enabled": True,
        "lower_quantile": 0.01,
        "upper_quantile": 0.99
    },
    "district_iqr_filter": {
        "enabled": True,
        "min_group_size": 8,
        "iqr_multiplier": 1.5
    },
    "county_iqr_filter": {
        "enabled": True,
        "min_group_size": 20,
        "iqr_multiplier": 1.75
    },
    "duplicate_filter": {
        "enabled": True,
        "subset": ["classified_id"]
    }
}


def clean_str(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    return np.nan if s == "" or s.lower() in {"nan", "none", "null"} else s


def to_num(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)

    s = str(x).strip()
    s = s.replace("TL", "").replace("₺", "").replace("m²", "").replace("m2", "")
    s = s.replace(".", "").replace(",", ".")
    s = "".join(ch for ch in s if ch.isdigit() or ch in ".-")

    try:
        return float(s)
    except Exception:
        return np.nan


def normalize_text_cols(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(clean_str)
    return df


def add_unit_price_if_needed(df, config):
    df = df.copy()

    price_col = config["price_col"]
    gross_col = config["gross_m2_col"]
    target_col = config["target_unit_price_col"]

    for col in [price_col, gross_col, target_col]:
        if col in df.columns:
            df[col] = df[col].map(to_num)

    if target_col not in df.columns:
        df[target_col] = np.nan

    missing_target = df[target_col].isna()
    can_calculate = df[price_col].notna() & df[gross_col].notna() & (df[gross_col] > 0)

    df.loc[missing_target & can_calculate, target_col] = (
        df.loc[missing_target & can_calculate, price_col] / df.loc[missing_target & can_calculate, gross_col]
    )

    return df


def mark_reason(mask, reason, reason_series):
    reason_series.loc[mask] = reason_series.loc[mask].apply(
        lambda old: reason if not old else f"{old}; {reason}"
    )
    return reason_series


def apply_basic_validity_filters(df, config, reason):
    price_col = config["price_col"]
    gross_col = config["gross_m2_col"]
    target_col = config["target_unit_price_col"]
    limits = config["soft_numeric_limits"]

    reason = mark_reason(df[price_col].isna(), "missing_price", reason)
    reason = mark_reason(df[gross_col].isna(), "missing_gross_m2", reason)
    reason = mark_reason(df[target_col].isna(), "missing_unit_price_gross", reason)

    reason = mark_reason(df[gross_col] < limits["gross_m2_min"], "gross_m2_too_low", reason)
    reason = mark_reason(df[gross_col] > limits["gross_m2_max"], "gross_m2_too_high", reason)

    reason = mark_reason(df[target_col] < limits["unit_price_gross_min"], "unit_price_gross_too_low", reason)
    reason = mark_reason(df[target_col] > limits["unit_price_gross_max"], "unit_price_gross_too_high", reason)

    reason = mark_reason(df[price_col] < limits["price_min"], "price_too_low", reason)
    reason = mark_reason(df[price_col] > limits["price_max"], "price_too_high", reason)

    return reason


def apply_duplicate_filter(df, config, reason):
    duplicate_config = config["duplicate_filter"]
    if not duplicate_config["enabled"]:
        return reason

    subset = [c for c in duplicate_config["subset"] if c in df.columns]
    if not subset:
        return reason

    duplicated = df.duplicated(subset=subset, keep="first")
    reason = mark_reason(duplicated, "duplicate_classified_id", reason)
    return reason


def apply_global_quantile_filter(df, config, reason):
    qconf = config["global_quantile_filter"]
    if not qconf["enabled"]:
        return reason

    target_col = config["target_unit_price_col"]
    valid = reason.eq("") & df[target_col].notna()

    if valid.sum() < 20:
        return reason

    lower = df.loc[valid, target_col].quantile(qconf["lower_quantile"])
    upper = df.loc[valid, target_col].quantile(qconf["upper_quantile"])

    mask = valid & ((df[target_col] < lower) | (df[target_col] > upper))
    reason = mark_reason(mask, f"global_quantile_outlier_{qconf['lower_quantile']}_{qconf['upper_quantile']}", reason)

    return reason


def apply_group_iqr_filter(df, config, reason, group_cols, filter_key, reason_prefix):
    fconf = config[filter_key]
    if not fconf["enabled"]:
        return reason

    target_col = config["target_unit_price_col"]
    available_group_cols = [c for c in group_cols if c in df.columns]

    if not available_group_cols:
        return reason

    active = reason.eq("") & df[target_col].notna()
    grouped = df.loc[active].groupby(available_group_cols, dropna=False)

    for group_key, group in grouped:
        if len(group) < fconf["min_group_size"]:
            continue

        q1 = group[target_col].quantile(0.25)
        q3 = group[target_col].quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr <= 0:
            continue

        lower = q1 - fconf["iqr_multiplier"] * iqr
        upper = q3 + fconf["iqr_multiplier"] * iqr

        idx = group.index[(group[target_col] < lower) | (group[target_col] > upper)]
        reason.loc[idx] = reason.loc[idx].apply(
            lambda old: f"{reason_prefix}_iqr_outlier" if not old else f"{old}; {reason_prefix}_iqr_outlier"
        )

    return reason


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if pd.isna(obj):
        return None
    return obj


def build_summary(raw_df, cleaned_df, removed_df, reason_col="outlier_reason"):
    reason_counts = {}
    if len(removed_df) > 0:
        exploded = (
            removed_df[reason_col]
            .fillna("")
            .str.split("; ")
            .explode()
            .replace("", np.nan)
            .dropna()
        )
        reason_counts = exploded.value_counts().to_dict()

    before_stats = raw_df["unit_price_gross"].describe().to_dict() if "unit_price_gross" in raw_df else {}
    after_stats = cleaned_df["unit_price_gross"].describe().to_dict() if "unit_price_gross" in cleaned_df else {}

    summary = {
        "rows_raw": int(len(raw_df)),
        "rows_cleaned": int(len(cleaned_df)),
        "rows_removed": int(len(removed_df)),
        "removed_ratio": float(len(removed_df) / len(raw_df)) if len(raw_df) else 0,
        "reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
        "unit_price_stats_before": before_stats,
        "unit_price_stats_after": after_stats,
        "config": CONFIG
    }
    return make_json_safe(summary)


def clean_outliers(df, config=CONFIG):
    df = normalize_text_cols(df)
    df = add_unit_price_if_needed(df, config)

    reason = pd.Series([""] * len(df), index=df.index, dtype="object")

    reason = apply_basic_validity_filters(df, config, reason)
    reason = apply_duplicate_filter(df, config, reason)
    reason = apply_global_quantile_filter(df, config, reason)

    reason = apply_group_iqr_filter(
        df,
        config,
        reason,
        group_cols=["city", "county", "district"],
        filter_key="district_iqr_filter",
        reason_prefix="district"
    )

    reason = apply_group_iqr_filter(
        df,
        config,
        reason,
        group_cols=["city", "county"],
        filter_key="county_iqr_filter",
        reason_prefix="county"
    )

    df["outlier_reason"] = reason
    cleaned = df[df["outlier_reason"].eq("")].copy()
    removed = df[~df["outlier_reason"].eq("")].copy()

    return cleaned, removed, df


def main():
    raw = pd.read_csv(INPUT_PATH)
    cleaned, removed, marked = clean_outliers(raw, CONFIG)

    cleaned.to_csv(OUTPUT_DIR / "listing_dataset_cleaned.csv", index=False, encoding="utf-8-sig")
    removed.to_csv(OUTPUT_DIR / "listing_dataset_removed_outliers.csv", index=False, encoding="utf-8-sig")
    marked.to_csv(OUTPUT_DIR / "listing_dataset_marked_with_outlier_reason.csv", index=False, encoding="utf-8-sig")

    summary = build_summary(marked, cleaned, removed)
    (REPORT_DIR / "outlier_cleaning_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
