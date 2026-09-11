"""V23 controlled duplex / large-home feature extraction.

Primary duplex source: detail_konut_tipi (structured).
Secondary: title (fallback / auxiliary).
On conflict, detail wins. Raw text never enters the model matrix.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

LARGE_HOME_M2 = 180.0

DUPLEX_TYPES = ("none", "roof", "garden", "middle_floor", "standard", "unknown")
MATCH_SOURCES = ("detail", "title", "detail_and_title", "none", "conflict_detail_wins")

DUPLEX_FLAG_NUMERIC = [
    "is_duplex",
    "is_roof_duplex",
    "is_garden_duplex",
    "is_middle_floor_duplex",
    "is_standard_duplex",
    "duplex_source_conflict",
]
DUPLEX_FLAG_CATEGORICAL = [
    "duplex_type",
    "duplex_match_source",
]
LARGEHOME_NUMERIC = [
    "is_large_home",
]
LARGEHOME_CATEGORICAL = [
    "large_home_bucket",
]
DUPLEX_INTERACTION_NUMERIC = [
    "duplex_or_large_home",
    "duplex_x_gross_m2",
    "duplex_x_site_inside",
    "duplex_x_site_project_known_premium",
    "garden_duplex_x_private_garden",
]
DUPLEX_INTERACTION_CATEGORICAL = [
    "duplex_x_district",
    "duplex_x_county",
    "duplex_x_site_quality_tier",
    "duplex_x_site_project",
    "roof_duplex_x_floor_segment",
    "middle_floor_duplex_x_floor_segment",
    "large_home_x_site_project_id_bucket",
]

# Extract-only columns (never model inputs via FeatureColumnKeeper).
DUPLEX_TEXT_SOURCE_COLUMNS = {
    "title",
    "detail_konut_tipi",
    "detail_ic_ozellikler",
    "detail_dis_ozellikler",
    "site_name",
    "address_text",
}

_TR_FOLD = str.maketrans(
    {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "â": "a",
        "î": "i",
        "û": "u",
    }
)

_ROOF_PATTERNS = [
    re.compile(r"cati\s*dubleksi?"),
    re.compile(r"cati\s*dublex"),
    re.compile(r"roof\s*duplex"),
    re.compile(r"teras\s*dubleksi?"),
]
_GARDEN_PATTERNS = [
    re.compile(r"bahce\s*dubleksi?"),
    re.compile(r"bahce\s*dublex"),
    re.compile(r"bahceli\s*dubleksi?"),
    re.compile(r"bahce\s*kati\s*dubleksi?"),
    re.compile(r"garden\s*duplex"),
]
_MIDDLE_PATTERNS = [
    re.compile(r"ara\s*kat\s*dubleksi?"),
    re.compile(r"arakat\s*dubleksi?"),
    re.compile(r"middle\s*floor\s*duplex"),
]
_GENERIC_PATTERNS = [
    re.compile(r"\bdubleksi?\b"),
    re.compile(r"\bdublex\b"),
    re.compile(r"\bduplex\b"),
]
_PRIVATE_GARDEN_PATTERNS = [
    re.compile(r"ozel\s*bahce"),
    re.compile(r"mustakil\s*bahce"),
    re.compile(r"private\s*garden"),
    re.compile(r"\bbahceli\b"),
    re.compile(r"bahce\s*kullanim"),
]


def normalize_tr(text: Any) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).strip().lower()
    if not s or s in {"nan", "none", "null", "<na>"}:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_TR_FOLD)
    # fold remaining combining marks
    s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _match_type(norm: str) -> str | None:
    """Return typed duplex class or 'standard' for generic, or None if no duplex."""
    if not norm:
        return None
    for pat in _ROOF_PATTERNS:
        if pat.search(norm):
            return "roof"
    for pat in _GARDEN_PATTERNS:
        if pat.search(norm):
            return "garden"
    for pat in _MIDDLE_PATTERNS:
        if pat.search(norm):
            return "middle_floor"
    for pat in _GENERIC_PATTERNS:
        if pat.search(norm):
            return "standard"
    return None


def _has_private_garden(norm: str) -> int:
    if not norm:
        return 0
    return int(any(p.search(norm) for p in _PRIVATE_GARDEN_PATTERNS))


def classify_duplex_row(
    detail_konut_tipi: Any,
    title: Any,
) -> dict[str, Any]:
    detail_raw = "" if detail_konut_tipi is None else str(detail_konut_tipi)
    title_raw = "" if title is None else str(title)
    detail_n = normalize_tr(detail_konut_tipi)
    title_n = normalize_tr(title)
    detail_type = _match_type(detail_n)
    title_type = _match_type(title_n)

    notes: list[str] = []
    conflict = 0
    match_source = "none"
    duplex_type = "none"

    if detail_type is not None and title_type is not None:
        if detail_type == title_type:
            duplex_type = detail_type
            match_source = "detail_and_title"
        elif detail_type == "standard" and title_type != "standard":
            # Detail only says generic duplex; title subtype is more specific → keep detail-primary
            # but prefer subtype from title only when detail is generic? Spec: detail wins on conflict.
            # Treat generic vs typed as non-conflict agreement on duplex; keep detail type (standard)
            # unless we consider subtype enrichment. Spec says conflict when types conflict;
            # standard vs roof is a conflict → detail wins → standard.
            duplex_type = detail_type
            match_source = "conflict_detail_wins"
            conflict = 1
            notes.append(f"conflict detail={detail_type} title={title_type}")
        elif title_type == "standard" and detail_type != "standard":
            duplex_type = detail_type
            match_source = "detail_and_title"
            notes.append("title_generic_confirms_detail_subtype")
        else:
            duplex_type = detail_type
            match_source = "conflict_detail_wins"
            conflict = 1
            notes.append(f"conflict detail={detail_type} title={title_type}")
    elif detail_type is not None:
        duplex_type = detail_type
        match_source = "detail"
    elif title_type is not None:
        duplex_type = title_type
        match_source = "title"
    else:
        duplex_type = "none"
        match_source = "none"

    is_duplex = int(duplex_type != "none")
    return {
        "detail_konut_tipi_raw": detail_raw if detail_raw.lower() not in {"nan", "none"} else "",
        "title_raw": title_raw if title_raw.lower() not in {"nan", "none"} else "",
        "duplex_type": duplex_type,
        "is_duplex": is_duplex,
        "is_roof_duplex": int(duplex_type == "roof"),
        "is_garden_duplex": int(duplex_type == "garden"),
        "is_middle_floor_duplex": int(duplex_type == "middle_floor"),
        "is_standard_duplex": int(duplex_type == "standard"),
        "duplex_match_source": match_source,
        "duplex_source_conflict": conflict,
        "extraction_notes": "|".join(notes),
        "_detail_type": detail_type,
        "_title_type": title_type,
    }


def large_home_bucket(gross: float) -> str:
    if not np.isfinite(gross) or gross < LARGE_HOME_M2:
        return "lt180"
    if gross < 220:
        return "180_220"
    if gross < 280:
        return "220_280"
    return "280p"


def site_id_bucket(site_id: Any, freq_map: dict[str, int] | None = None) -> str:
    sid = str(site_id or "").strip()
    if not sid or sid.lower() in {"nan", "none", "missing", "other", ""}:
        return "missing"
    n = int((freq_map or {}).get(sid, 0))
    if n <= 0:
        return "rare"
    if n < 3:
        return "n1_2"
    if n < 10:
        return "n3_9"
    if n < 30:
        return "n10_29"
    return "n30p"


def get_duplex_numeric_feature_names(mode: str) -> list[str]:
    m = str(mode or "none").lower()
    if m in {"", "none"}:
        return []
    names: list[str] = []
    if m in {"flags", "interactions", "full"}:
        names += list(DUPLEX_FLAG_NUMERIC)
    if m in {"largehome", "full"}:
        names += list(LARGEHOME_NUMERIC)
    if m in {"interactions", "full"}:
        names += list(DUPLEX_INTERACTION_NUMERIC)
        if m == "full":
            # duplex_or_large_home needs is_large_home; already listed in LARGEHOME for full
            pass
        else:
            # interactions without largehome mode: still need is_large_home for duplex_or_large_home
            if "is_large_home" not in names:
                names.append("is_large_home")
    # full already has largehome + interactions
    if m == "full" and "duplex_or_large_home" not in names:
        names += list(DUPLEX_INTERACTION_NUMERIC)
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def get_duplex_categorical_feature_names(mode: str) -> list[str]:
    m = str(mode or "none").lower()
    if m in {"", "none"}:
        return []
    names: list[str] = []
    if m in {"flags", "interactions", "full"}:
        names += list(DUPLEX_FLAG_CATEGORICAL)
    if m in {"largehome", "full"}:
        names += list(LARGEHOME_CATEGORICAL)
    if m in {"interactions", "full"}:
        names += list(DUPLEX_INTERACTION_CATEGORICAL)
        if "large_home_bucket" not in names and m == "interactions":
            names.append("large_home_bucket")
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def extract_duplex_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level duplex classification + audit columns (no site interactions)."""
    detail_col = "detail_konut_tipi" if "detail_konut_tipi" in df.columns else None
    title_col = "title" if "title" in df.columns else None
    rows = []
    for idx in df.index:
        detail_v = df.at[idx, detail_col] if detail_col else ""
        title_v = df.at[idx, title_col] if title_col else ""
        row = classify_duplex_row(detail_v, title_v)
        row["classified_id"] = df.at[idx, "classified_id"] if "classified_id" in df.columns else idx
        rows.append(row)
    return pd.DataFrame(rows, index=df.index)


def duplex_source_coverage_metrics(audit: pd.DataFrame) -> dict[str, int]:
    src = audit["duplex_match_source"].astype(str) if "duplex_match_source" in audit.columns else pd.Series(dtype=str)
    dtype = audit["duplex_type"].astype(str) if "duplex_type" in audit.columns else pd.Series(dtype=str)
    detail_hit = src.isin(["detail", "detail_and_title", "conflict_detail_wins"])
    title_hit = src.isin(["title", "detail_and_title", "conflict_detail_wins"])
    # Also count raw type presence via helper cols if present
    return {
        "detail_duplex_count": int(detail_hit.sum()),
        "title_duplex_count": int(title_hit.sum()),
        "detail_and_title_duplex_count": int((src == "detail_and_title").sum()),
        "title_only_duplex_count": int((src == "title").sum()),
        "detail_only_duplex_count": int((src == "detail").sum()),
        "conflict_count": int((src == "conflict_detail_wins").sum()),
        "roof_duplex_count": int((dtype == "roof").sum()),
        "garden_duplex_count": int((dtype == "garden").sum()),
        "middle_floor_duplex_count": int((dtype == "middle_floor").sum()),
        "generic_duplex_count": int((dtype == "standard").sum()),
    }


class DuplexLargeHomeFeatureAdder(BaseEstimator, TransformerMixin):
    """Add controlled duplex / large-home flags and interactions.

    Modes:
      none | flags | largehome | interactions | full
    """

    def __init__(self, duplex_feature_mode: str = "full"):
        self.duplex_feature_mode = duplex_feature_mode
        self.site_id_freq_: dict[str, int] = {}

    def fit(self, X: pd.DataFrame, y: Any = None):
        mode = str(self.duplex_feature_mode or "none").lower()
        self.site_id_freq_ = {}
        if mode in {"interactions", "full"} and isinstance(X, pd.DataFrame) and "site_project_id" in X.columns:
            vc = X["site_project_id"].astype(str).value_counts()
            self.site_id_freq_ = {str(k): int(v) for k, v in vc.items()}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DuplexLargeHomeFeatureAdder expects a pandas DataFrame")
        mode = str(self.duplex_feature_mode or "none").lower()
        df = X.copy()
        if mode in {"", "none"}:
            return df

        need_duplex = mode in {"flags", "interactions", "full"}
        need_lh = mode in {"largehome", "interactions", "full"}
        need_inter = mode in {"interactions", "full"}

        audit = extract_duplex_frame(df) if need_duplex or need_inter else None

        if need_duplex and audit is not None:
            for col in DUPLEX_FLAG_NUMERIC + DUPLEX_FLAG_CATEGORICAL:
                df[col] = audit[col].values

        gross = pd.to_numeric(df["gross_m2"], errors="coerce") if "gross_m2" in df.columns else pd.Series(np.nan, index=df.index)
        if need_lh or need_inter:
            is_lh = (gross >= LARGE_HOME_M2).fillna(False).astype(float)
            df["is_large_home"] = is_lh
            df["large_home_bucket"] = [large_home_bucket(float(v) if np.isfinite(v) else 0.0) for v in gross.fillna(0.0)]

        if need_inter and audit is not None:
            is_dup = pd.to_numeric(df.get("is_duplex", 0), errors="coerce").fillna(0.0).clip(0, 1)
            is_lh = pd.to_numeric(df.get("is_large_home", 0), errors="coerce").fillna(0.0).clip(0, 1)
            df["duplex_or_large_home"] = ((is_dup > 0) | (is_lh > 0)).astype(float)
            df["duplex_x_gross_m2"] = is_dup * gross.fillna(0.0)

            site_inside = pd.Series(0.0, index=df.index)
            if "site_inside" in df.columns:
                raw = df["site_inside"]
                if raw.dtype == object:
                    site_inside = raw.astype(str).str.lower().isin(["evet", "var", "1", "true", "yes"]).astype(float)
                else:
                    site_inside = pd.to_numeric(raw, errors="coerce").fillna(0.0).clip(0, 1)
            if "attr_is_site_inside" in df.columns:
                site_inside = np.maximum(site_inside, pd.to_numeric(df["attr_is_site_inside"], errors="coerce").fillna(0.0))
            df["duplex_x_site_inside"] = is_dup * site_inside

            known = pd.to_numeric(df["site_project_known_premium_flag"], errors="coerce").fillna(0.0) if "site_project_known_premium_flag" in df.columns else 0.0
            df["duplex_x_site_project_known_premium"] = is_dup * known

            district = df["district"].astype(str) if "district" in df.columns else pd.Series("missing", index=df.index)
            district = district.replace({"nan": "missing", "None": "missing", "": "missing"})
            df["duplex_x_district"] = np.where(is_dup > 0, "duplex__" + district, "nonduplex")

            county = df["county"].astype(str) if "county" in df.columns else pd.Series("missing", index=df.index)
            county = county.replace({"nan": "missing", "None": "missing", "": "missing"})
            df["duplex_x_county"] = np.where(is_dup > 0, "duplex__" + county, "nonduplex")

            tier = df["site_quality_tier"].astype(str) if "site_quality_tier" in df.columns else pd.Series("unknown", index=df.index)
            tier = tier.replace({"nan": "unknown", "None": "unknown", "": "unknown"})
            df["duplex_x_site_quality_tier"] = np.where(is_dup > 0, "duplex__" + tier, "nonduplex")

            sid = df["site_project_id"].astype(str) if "site_project_id" in df.columns else pd.Series("missing", index=df.index)
            sid = sid.replace({"nan": "missing", "None": "missing", "": "missing"})
            df["duplex_x_site_project"] = np.where(is_dup > 0, "duplex__" + sid, "nonduplex")

            # private garden from detail_dis_ozellikler (+ title as weak aux for garden cue only)
            dis = df["detail_dis_ozellikler"] if "detail_dis_ozellikler" in df.columns else ""
            ic = df["detail_ic_ozellikler"] if "detail_ic_ozellikler" in df.columns else ""
            title = df["title"] if "title" in df.columns else ""
            priv = []
            for i in df.index:
                blob = normalize_tr(dis[i] if hasattr(dis, "__getitem__") else "") + " " + normalize_tr(ic[i] if hasattr(ic, "__getitem__") else "")
                priv.append(_has_private_garden(blob))
            is_garden = pd.to_numeric(df.get("is_garden_duplex", 0), errors="coerce").fillna(0.0)
            df["garden_duplex_x_private_garden"] = is_garden * np.asarray(priv, dtype=float)

            floor = df["floor_segment"].astype(str) if "floor_segment" in df.columns else pd.Series("unknown", index=df.index)
            floor = floor.replace({"nan": "unknown", "None": "unknown", "": "unknown"})
            is_roof = pd.to_numeric(df.get("is_roof_duplex", 0), errors="coerce").fillna(0.0)
            is_mid = pd.to_numeric(df.get("is_middle_floor_duplex", 0), errors="coerce").fillna(0.0)
            df["roof_duplex_x_floor_segment"] = np.where(is_roof > 0, "roof__" + floor, "nonroof")
            df["middle_floor_duplex_x_floor_segment"] = np.where(is_mid > 0, "mid__" + floor, "nonmid")

            sid = df["site_project_id"] if "site_project_id" in df.columns else pd.Series("missing", index=df.index)
            buckets = [site_id_bucket(v, self.site_id_freq_) for v in sid]
            df["large_home_x_site_project_id_bucket"] = np.where(is_lh > 0, "lh__" + pd.Series(buckets, index=df.index), "non_lh")

        # Ensure declared columns exist for keeper
        for col in get_duplex_numeric_feature_names(mode):
            if col not in df.columns:
                df[col] = 0.0
        for col in get_duplex_categorical_feature_names(mode):
            if col not in df.columns:
                df[col] = "missing"
        return df


def write_duplex_extraction_reports(df: pd.DataFrame, reports_dir) -> dict[str, Any]:
    """Write audit + source coverage CSVs; return coverage metrics dict."""
    from pathlib import Path

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    audit = extract_duplex_frame(df)
    # Drop internal helper columns from CSV
    drop_cols = [c for c in audit.columns if c.startswith("_")]
    audit_out = audit.drop(columns=drop_cols, errors="ignore")
    audit_path = reports_dir / "duplex_extraction_audit_v23.csv"
    audit_out.to_csv(audit_path, index=False, encoding="utf-8-sig")
    cov = duplex_source_coverage_metrics(audit)
    cov_df = pd.DataFrame([cov])
    cov_path = reports_dir / "duplex_source_coverage_v23.csv"
    cov_df.to_csv(cov_path, index=False, encoding="utf-8-sig")
    return cov
