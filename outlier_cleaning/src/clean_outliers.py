import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

DETAIL_PREFIXES = ("front_", "view_", "transport_", "near_", "out_", "in_", "subtype_")
DETAIL_EXACT = {
    "building_age_raw", "building_age_group",
    "detail_cephe", "detail_manzara", "detail_konut_tipi",
    "detail_ic_ozellikler", "detail_dis_ozellikler", "detail_muhit",
    "detail_ulasim", "detail_engelli_yasli_uygun",
    "detail_selected_count", "detail_quality_score",
}


def safe_json_loads(x: Any) -> Dict[str, Any]:
    if pd.isna(x):
        return {}
    if isinstance(x, dict):
        return x
    if not isinstance(x, str) or not x.strip():
        return {}
    try:
        obj = json.loads(x)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def expand_raw_json_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Neon export'ta helper feature'ları bazen raw JSON içinde geliyor.
    Bu fonksiyon raw içindeki değerleri üst seviyeye çıkarır.

    Pandas dtype hatasını engellemek için:
    - Numeric/binary kolonlara yalnızca numeric değer yazar
    - Text kolonları object dtype'a çevirir
    - Boş stringleri NaN gibi ele alır
    """
    if "raw" not in df.columns:
        return df.copy()

    out = df.copy()
    raw_objects = out["raw"].map(safe_json_loads)

    numeric_like_columns = {
        "price",
        "monthly_rent",
        "unit_price_gross",
        "unit_price_net",
        "rent_per_m2_gross",
        "rent_per_m2_net",
        "gross_m2",
        "net_m2",
        "building_age",
        "floor_num",
        "total_floors",
        "bathroom_count",
        "dues",
        "deposit",
        "detail_selected_count",
        "detail_quality_score",
    }

    text_like_columns = {
        "building_age_raw",
        "building_age_group",
        "room_count",
        "district",
        "county",
        "city",
        "floor",
        "floor_segment",
        "title",
        "detail_cephe",
        "detail_manzara",
        "detail_konut_tipi",
        "detail_ic_ozellikler",
        "detail_dis_ozellikler",
        "detail_muhit",
        "detail_ulasim",
        "detail_engelli_yasli_uygun",
    }

    keys = set()
    for obj in raw_objects:
        if not isinstance(obj, dict):
            continue
        for key in obj:
            if (
                key in DETAIL_EXACT
                or key.startswith(DETAIL_PREFIXES)
                or key in numeric_like_columns
                or key in text_like_columns
            ):
                keys.add(key)

    for key in sorted(keys):
        raw_values = raw_objects.map(
            lambda obj, k=key: obj.get(k, np.nan) if isinstance(obj, dict) else np.nan
        )
        raw_values = raw_values.replace("", np.nan)

        is_numeric_like = key in numeric_like_columns or key.startswith(DETAIL_PREFIXES)

        if is_numeric_like:
            raw_values = pd.to_numeric(raw_values, errors="coerce")

            if key not in out.columns:
                out[key] = raw_values
            else:
                current = out[key].replace("", np.nan)
                out[key] = pd.to_numeric(current, errors="coerce")
                mask = out[key].isna() & raw_values.notna()
                out.loc[mask, key] = raw_values.loc[mask]
        else:
            raw_values = raw_values.astype("object")

            if key not in out.columns:
                out[key] = raw_values
            else:
                out[key] = out[key].astype("object")
                current_text = out[key].fillna("").astype(str).str.strip()
                mask = current_text.eq("") & raw_values.notna()
                out.loc[mask, key] = raw_values.loc[mask]

    return out

def normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).replace("\u00a0", " ")).strip()


def to_num(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x)
    s = s.replace("TL", "").replace("₺", "").replace("m²", "").replace("m2", "")
    s = s.replace(".", "").replace(",", ".")
    s = "".join(ch for ch in s if ch.isdigit() or ch in ".-")
    if not s or s in {"-", "."}:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_building_age(value: Any) -> Tuple[float, str, str]:
    raw = normalize_text(value)
    low = raw.lower().replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c")
    if not raw:
        return np.nan, "", ""
    m = re.search(r"(\d+)\s*-\s*(\d+)", low)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return round((a + b) / 2, 1), raw, raw
    m = re.search(r"(\d+)\s*(ve)?\s*(uzeri|ustu|\+)", low)
    if m:
        a = float(m.group(1))
        return a + 4, raw, raw
    m = re.search(r"^\s*(\d+)\s*$", low)
    if m:
        a = float(m.group(1))
        return a, raw, raw
    m = re.search(r"(\d+)", low)
    if m:
        a = float(m.group(1))
        return a, raw, raw
    return np.nan, raw, raw


def make_m2_group(s: pd.Series) -> pd.Series:
    return pd.cut(
        s,
        bins=[0, 75, 100, 125, 150, 200, 10_000],
        labels=["0-75", "76-100", "101-125", "126-150", "151-200", "200+"],
        include_lowest=True,
    ).astype(object)


def robust_mad_z(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    med = x.median()
    mad = np.median(np.abs(x.dropna() - med)) if x.notna().any() else np.nan
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.nan, index=values.index)
    return 0.6745 * (x - med) / mad


def iqr_bounds(values: pd.Series, multiplier: float) -> Tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if len(x) < 4:
        return -np.inf, np.inf
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return -np.inf, np.inf
    return q1 - multiplier * iqr, q3 + multiplier * iqr


def contains_keyword(title: Any, keywords: List[str]) -> bool:
    t = normalize_text(title).lower()
    t_ascii = t.replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c")
    for kw in keywords:
        k = kw.lower()
        k_ascii = k.replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c")
        if k in t or k_ascii in t_ascii:
            return True
    return False


def prepare_common(df: pd.DataFrame, kind: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    out = expand_raw_json_columns(df)
    out = out.copy()

    numeric_cols = [
        "price", "monthly_rent", "unit_price_gross", "unit_price_net", "rent_per_m2_gross", "rent_per_m2_net",
        "gross_m2", "net_m2", "building_age", "floor_num", "total_floors", "bathroom_count", "dues", "deposit",
        "detail_selected_count", "detail_quality_score",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = out[col].map(to_num)

    if "building_age_raw" not in out.columns:
        out["building_age_raw"] = np.nan
    if "building_age_group" not in out.columns:
        out["building_age_group"] = np.nan
    age_source = out["building_age_raw"].where(out["building_age_raw"].notna() & (out["building_age_raw"].astype(str).str.strip() != ""), out.get("building_age", np.nan))
    parsed = age_source.map(parse_building_age)
    # Fill numeric age only when missing or bad.
    parsed_age = parsed.map(lambda x: x[0])
    out["building_age"] = out.get("building_age", np.nan)
    out["building_age"] = out["building_age"].where(out["building_age"].notna(), parsed_age)
    out["building_age_raw"] = out["building_age_raw"].where(out["building_age_raw"].notna() & (out["building_age_raw"].astype(str).str.strip() != ""), parsed.map(lambda x: x[1]))
    out["building_age_group"] = out["building_age_group"].where(out["building_age_group"].notna() & (out["building_age_group"].astype(str).str.strip() != ""), parsed.map(lambda x: x[2]))

    if "gross_m2" in out.columns and "net_m2" in out.columns:
        out["net_gross_ratio"] = out["net_m2"] / out["gross_m2"].replace(0, np.nan)
    else:
        out["net_gross_ratio"] = np.nan

    if kind == "sales":
        if "unit_price_gross" not in out.columns:
            out["unit_price_gross"] = np.nan
        if "price" in out.columns and "gross_m2" in out.columns:
            calc = out["price"] / out["gross_m2"].replace(0, np.nan)
            out["unit_price_gross"] = out["unit_price_gross"].where(out["unit_price_gross"].notna(), calc)
    else:
        if "rent_per_m2_gross" not in out.columns:
            out["rent_per_m2_gross"] = np.nan
        if "monthly_rent" in out.columns and "gross_m2" in out.columns:
            calc = out["monthly_rent"] / out["gross_m2"].replace(0, np.nan)
            out["rent_per_m2_gross"] = out["rent_per_m2_gross"].where(out["rent_per_m2_gross"].notna(), calc)

    if "gross_m2" in out.columns:
        out["m2_group"] = make_m2_group(out["gross_m2"])
    else:
        out["m2_group"] = "missing"

    for col in ["district", "county", "room_count", "title"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].map(normalize_text)

    return out


def add_flags(df: pd.DataFrame, kind: str, cfg_all: Dict[str, Any]) -> pd.DataFrame:
    cfg = cfg_all[kind]
    general = cfg_all["general"]
    out = df.copy()
    hard = cfg["hard_limits"]
    review = cfg["review_flags_only"]
    target = cfg["target_col"]

    flags_remove = []
    flags_review = []

    def flag(name: str, condition: pd.Series, remove: bool = True):
        nonlocal flags_remove, flags_review, out
        condition = condition.fillna(False)
        out[name] = condition.astype(int)
        if remove:
            flags_remove.append(name)
        else:
            flags_review.append(name)

    if kind == "sales":
        flag("flag_price_hard_limit", (out["price"] < hard["price_min"]) | (out["price"] > hard["price_max"]))
        flag("flag_unit_price_hard_limit", (out[target] < hard["unit_price_gross_min"]) | (out[target] > hard["unit_price_gross_max"]))
    else:
        flag("flag_rent_hard_limit", (out["monthly_rent"] < hard["monthly_rent_min"]) | (out["monthly_rent"] > hard["monthly_rent_max"]))
        flag("flag_rent_m2_hard_limit", (out[target] < hard["rent_per_m2_gross_min"]) | (out[target] > hard["rent_per_m2_gross_max"]))
        if "deposit" in out.columns and "monthly_rent" in out.columns:
            deposit_months = out["deposit"] / out["monthly_rent"].replace(0, np.nan)
            out["deposit_months"] = deposit_months
            flag("flag_deposit_too_high_review", deposit_months > review["very_high_deposit_months"], remove=False)

    flag("flag_gross_m2_hard_limit", (out["gross_m2"] < hard["gross_m2_min"]) | (out["gross_m2"] > hard["gross_m2_max"]))
    flag("flag_net_m2_hard_limit", (out["net_m2"] < hard["net_m2_min"]) | (out["net_m2"] > hard["net_m2_max"]))
    flag("flag_net_gross_suspicious", (out["net_gross_ratio"] < hard["net_gross_ratio_min"]) | (out["net_gross_ratio"] > hard["net_gross_ratio_max"]))
    flag("flag_net_gt_gross", out["net_m2"] > out["gross_m2"])

    global_low, global_high = iqr_bounds(out[target], general["iqr_multiplier"])
    flag("flag_global_iqr_outlier", (out[target] < global_low) | (out[target] > global_high))
    z = robust_mad_z(out[target])
    out["global_mad_z"] = z
    flag("flag_global_mad_outlier", z.abs() > general["mad_z_threshold"])

    # Group outliers: district+room+m2_group, then district+room, then district.
    out["group_outlier_level"] = ""
    out["group_outlier_reason"] = ""
    group_flag = pd.Series(False, index=out.index)
    for group_cols, level_name in [
        (["district", "room_count", "m2_group"], "district_room_m2"),
        (["district", "room_count"], "district_room"),
        (["district"], "district"),
    ]:
        for keys, idx in out.groupby(group_cols, dropna=False).groups.items():
            idx = list(idx)
            if len(idx) < general["min_group_n_for_group_outlier"]:
                continue
            vals = out.loc[idx, target]
            low, high = iqr_bounds(vals, general["district_iqr_multiplier"])
            cond_idx = vals[(vals < low) | (vals > high)].index
            # Only set if not already set by more specific grouping.
            cond_idx = [i for i in cond_idx if not group_flag.loc[i]]
            if cond_idx:
                group_flag.loc[cond_idx] = True
                out.loc[cond_idx, "group_outlier_level"] = level_name
                out.loc[cond_idx, "group_outlier_reason"] = f"{target} outside {level_name} IQR bounds"
    flag("flag_group_iqr_outlier", group_flag)

    flag("flag_large_property_review", out["gross_m2"] > review["large_property_m2"], remove=False)
    flag("flag_old_building_review", out["building_age"] >= review["very_old_age"], remove=False)
    if "dues" in out.columns:
        flag("flag_high_dues_review", out["dues"] > review["very_high_dues"], remove=False)
    if "detail_selected_count" in out.columns:
        flag("flag_low_detail_review", out["detail_selected_count"].fillna(0) <= review["low_detail_selected_count"], remove=False)
    flag("flag_special_title_review", out["title"].map(lambda x: contains_keyword(x, cfg["title_special_keywords"])), remove=False)

    out["remove_flag_count"] = out[flags_remove].sum(axis=1) if flags_remove else 0
    out["review_flag_count"] = out[flags_review].sum(axis=1) if flags_review else 0
    out["remove_reasons"] = out[flags_remove].apply(lambda r: "|".join([c for c, v in r.items() if int(v) == 1]), axis=1) if flags_remove else ""
    out["review_reasons"] = out[flags_review].apply(lambda r: "|".join([c for c, v in r.items() if int(v) == 1]), axis=1) if flags_review else ""
    out["is_removed_outlier"] = (out["remove_flag_count"] > 0).astype(int)
    out["needs_review"] = ((out["review_flag_count"] > 0) | (out["remove_flag_count"] > 0)).astype(int)
    return out


def summarize(df: pd.DataFrame, kind: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    target = cfg[kind]["target_col"]
    out = {
        "kind": kind,
        "rows_raw_or_prepared": int(len(df)),
        "rows_removed": int(df["is_removed_outlier"].sum()),
        "rows_cleaned": int((df["is_removed_outlier"] == 0).sum()),
        "rows_needing_review": int(df["needs_review"].sum()),
        "target": target,
        "target_summary_all": df[target].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).replace({np.nan: None}).to_dict(),
        "target_summary_cleaned": df.loc[df["is_removed_outlier"] == 0, target].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).replace({np.nan: None}).to_dict(),
        "remove_flag_counts": {c: int(df[c].sum()) for c in df.columns if c.startswith("flag_") and not c.endswith("_review") and df[c].dropna().isin([0,1]).all()},
        "review_flag_counts": {c: int(df[c].sum()) for c in df.columns if c.startswith("flag_") and c.endswith("_review") and df[c].dropna().isin([0,1]).all()},
    }
    return out


def group_report(df: pd.DataFrame, kind: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    target = cfg[kind]["target_col"]
    rows = []
    for col in ["district", "room_count", "m2_group", "floor_segment", "building_age_group"]:
        if col not in df.columns:
            continue
        g = df.groupby(col, dropna=False).agg(
            n=(target, "size"),
            median_target=(target, "median"),
            mean_target=(target, "mean"),
            removed=("is_removed_outlier", "sum"),
            review=("needs_review", "sum"),
        ).reset_index()
        g.insert(0, "group_column", col)
        g = g.rename(columns={col: "group_value"})
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def clean_one(input_path: Path, kind: str, cfg: Dict[str, Any], output_dir: Path, report_dir: Path) -> Dict[str, Any]:
    df_raw = pd.read_csv(input_path)
    prepared = prepare_common(df_raw, kind, cfg)
    flagged = add_flags(prepared, kind, cfg)

    cleaned = flagged.loc[flagged["is_removed_outlier"] == 0].copy()
    removed = flagged.loc[flagged["is_removed_outlier"] == 1].copy()
    review = flagged.loc[flagged["needs_review"] == 1].copy()

    prefix = "sales" if kind == "sales" else "rental"
    flagged.to_csv(output_dir / f"{prefix}_with_outlier_flags.csv", index=False, encoding="utf-8-sig")
    cleaned.to_csv(output_dir / f"{prefix}_cleaned.csv", index=False, encoding="utf-8-sig")
    removed.to_csv(output_dir / f"{prefix}_removed_outliers.csv", index=False, encoding="utf-8-sig")
    review.to_csv(output_dir / f"{prefix}_review_needed.csv", index=False, encoding="utf-8-sig")

    report = summarize(flagged, kind, cfg)
    (report_dir / f"{prefix}_quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    group_report(flagged, kind, cfg).to_csv(report_dir / f"{prefix}_group_outlier_summary.csv", index=False, encoding="utf-8-sig")
    return report


def main():
    parser = argparse.ArgumentParser(description="Clean outliers from listing portal sales and rental CSV exports.")
    parser.add_argument("--sales", default="data/input/sale_listings.csv")
    parser.add_argument("--rental", default="data/input/rental_listings.csv")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output-dir", default="data/output")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sales_path = Path(args.sales)
    rental_path = Path(args.rental)
    cfg_path = Path(args.config)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    if not sales_path.is_absolute(): sales_path = root / sales_path
    if not rental_path.is_absolute(): rental_path = root / rental_path
    if not cfg_path.is_absolute(): cfg_path = root / cfg_path
    if not output_dir.is_absolute(): output_dir = root / output_dir
    if not report_dir.is_absolute(): report_dir = root / report_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    combined = {}
    if sales_path.exists():
        combined["sales"] = clean_one(sales_path, "sales", cfg, output_dir, report_dir)
    else:
        combined["sales_error"] = f"Sales file not found: {sales_path}"
    if rental_path.exists():
        combined["rental"] = clean_one(rental_path, "rental", cfg, output_dir, report_dir)
    else:
        combined["rental_error"] = f"Rental file not found: {rental_path}"

    (report_dir / "combined_quality_report.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(combined, indent=2, ensure_ascii=False, default=str))

if __name__ == "__main__":
    main()
