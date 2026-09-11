"""V20 Başiskele controlled premium-signal features.

Free text is never passed raw into the model. Only boolean / count / score /
normalized site-project identity features are produced.

Fold-safe site-project target encoding (site_project_encoding=foldsafe_target)
must be fit inside CV folds only — validation targets never enter the encoder.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Injected by LocationResidualRegressor (same pattern as comparable features).
PREMIUM_UNIT_PRICE_COL = "__premium_unit_price__"

KNOWN_PREMIUM_PROJECTS: tuple[str, ...] = (
    "zeray perla",
    "zeray korupark",
    "zeray gunesi",
    "orka life 2",
    "evimiz kocaeli",
    "alizepark city",
    "ustgrup mezire",
    "mamik life",
    "royal life",
    "panoramakent",
)

# Text flag keyword dictionaries (Turkish-heavy; applied on folded text).
KEYWORD_DICT: dict[str, tuple[str, ...]] = {
    "text_sea_view": (
        r"deniz\s*manzar",
        r"korfez\s*manzar",
        r"ful+\s*deniz",
        r"full\s*deniz",
        r"denize\s*sifir",
        r"deniz\s*goren",
        r"\bdeniz\b",
        r"\bmanzara\b",
        r"sea\s*view",
    ),
    "text_has_pool": (
        r"yuzme\s*havuz",
        r"acik\s*havuz",
        r"kapali\s*havuz",
        r"\bhavuzlu\b",
        r"\bhavuz\b",
        r"\bpool\b",
    ),
    "text_luxury_signal": (
        r"ultra\s*luks",
        r"\bluks\b",
        r"\blux\b",
        r"\bultra\b",
        r"\bpremium\b",
        r"ozel\s*yapim",
        r"\bkaliteli\b",
        r"\bprestij\b",
        r"ust\s*grup",
    ),
    "text_project_signal": (
        r"\bproje\b",
        r"\bproject\b",
        r"\bsitesi\b",
        r"\bsite\b",
        r"\bresidence\b",
        r"\brezidans\b",
        r"konaklari",
        r"\byasam\b",
        r"\blife\b",
        r"\bpark\b",
        r"\bvadi\b",
    ),
    "text_villa_like": (
        r"\bvilla\b",
        r"\bvila\b",
        r"\bmustakil\b",
        r"\btripleks\b",
        r"\btriplex\b",
        r"bahceli\s*villa",
        r"villa\s*tipi",
    ),
    "text_duplex": (
        r"\bdubleks\b",
        r"\bduplex\b",
    ),
    "text_garden_duplex": (
        r"bahce\s*dubleks",
        r"bahce\s*kati\s*dubleks",
        r"bahceli\s*dubleks",
        r"garden\s*duplex",
    ),
    "text_roof_duplex": (
        r"cati\s*dubleks",
        r"teras\s*dubleks",
        r"\bpenthouse\b",
        r"roof\s*duplex",
    ),
    "text_private_garden": (
        r"ozel\s*bahce",
        r"mustakil\s*bahce",
        r"private\s*garden",
        r"\bbahceli\b",
        r"bahce\s*kullanim",
    ),
    "text_terrace": (
        r"\bterasli\b",
        r"\bteras\b",
        r"\bterrace\b",
    ),
    "text_low_rise": (
        r"az\s*katli",
        r"dusuk\s*katli",
        r"bahce\s*kat",
        r"3\s*katli",
        r"4\s*katli",
        r"villa\s*site",
    ),
    "text_high_floor": (
        r"en\s*ust\s*kat",
        r"ust\s*kat",
        r"yuksek\s*kat",
        r"cati\s*kat",
        r"\bpenthouse\b",
    ),
    "text_new_project": (
        r"yeni\s*proje",
        r"sifir\s*proje",
        r"yeni\s*teslim",
        r"bitmeye\s*yakin\s*proje",
        r"\bsifir\b",
    ),
    "text_premium_finish": (
        r"hilton\s*banyo",
        r"ebeveyn\s*banyo",
        r"akilli\s*ev",
        r"smart\s*home",
        r"yerden\s*isitma",
        r"jakuzi",
        r"amerikan\s*mutfak",
        r"ful+\s*yapili",
        r"full\s*yapili",
    ),
}

TEXT_FLAG_FEATURES: list[str] = list(KEYWORD_DICT.keys())

SITE_NUMERIC_FEATURES = [
    "has_site_project_name",
    "site_project_listing_count",
    "site_project_freq_bucket",
    "site_project_known_premium_flag",
]

SITE_CATEGORICAL_FEATURES = [
    "site_project_name_normalized",
]

SCORE_FEATURES = [
    "premium_signal_score",
]

SCORE_CATEGORICAL = [
    "premium_signal_level",
]

INTERACTION_NUMERIC = [
    "premium_score_x_gross_m2",
    "premium_score_x_large_home",
    "premium_score_x_site_inside",
    "premium_score_x_distance_to_coastline_inv",
    "sea_view_x_distance_to_coastline_inv",
    "pool_x_site_inside",
    "luxury_x_known_project",
    "known_project_x_district_code",
    "site_project_freq_x_premium_score",
    "distance_to_coastline_inv",
]

INTERACTION_CATEGORICAL = [
    "premium_level_x_district",
    "site_project_bucket_x_district",
    "premium_level_x_m2_group",
    "premium_level_x_room_count",
]

FOLDSAFE_NUMERIC = [
    "site_project_oof_price_level",
    "site_project_oof_residual_mean",
    "site_project_oof_count",
    "site_project_oof_confidence",
]

SITE_NAME_STOP = {
    "basiskele",
    "kocaeli",
    "izmit",
    "satilik",
    "kiralik",
    "daire",
    "konut",
    "arsa",
    "firsat",
    "acil",
    "guzel",
    "luks",
    "yeni",
    "merkez",
    "mahalle",
    "mah",
    "kat",
    "arakat",
    "belirtilmemis",
    "yok",
    "var",
    "nan",
    "none",
    "missing",
    "other",
}

GENERIC_SINGLE = {
    "site",
    "sitesi",
    "sit",
    "proje",
    "projesi",
    "residence",
    "rezidans",
    "life",
    "park",
    "evleri",
    "vadi",
    "perla",
    "konaklari",
}


def fold_text(s: Any) -> str:
    text = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("ı", "i").replace("İ", "i").lower()
    text = re.sub(r"[^\w\s|+./'-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _combine_parts(parts: list[str]) -> str:
    out = []
    for p in parts:
        if not p:
            continue
        s = str(p).strip()
        if not s or fold_text(s) in {"", "nan", "none", "belirtilmemis", "yok"}:
            continue
        out.append(s)
    return " | ".join(out)


def build_description_proxy(row: pd.Series) -> str:
    detail_cols = [
        c
        for c in row.index
        if str(c).startswith("detail_") and str(c) not in {"detail_selected_count", "detail_quality_score"}
    ]
    # Prefer explicit description if present
    for name in ("description", "listing_description", "aciklama", "ilan_aciklama"):
        if name in row.index and pd.notna(row.get(name)):
            return str(row.get(name))
    parts = []
    for c in detail_cols:
        v = row.get(c)
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s and fold_text(s) not in {"", "nan", "none"}:
            parts.append(s)
    return _combine_parts(parts)


def build_audit_text(row: pd.Series) -> str:
    desc = row.get("description")
    if pd.isna(desc) or not str(desc).strip():
        desc = build_description_proxy(row)
    return _combine_parts(
        [
            str(row.get("title") or ""),
            str(row.get("site_name") or ""),
            str(row.get("address_text") or ""),
            str(desc or ""),
        ]
    )


def normalize_site_project_name(raw: str) -> str:
    """Normalize site/project string without wiping meaningful tokens."""
    t = fold_text(raw)
    if not t:
        return "missing"
    # drop common listing junk
    t = re.sub(r"\b\d+\+\d+\b", " ", t)
    t = re.sub(r"\b\d+\s*m2\b", " ", t)
    t = re.sub(r"\b\d+\b", " ", t)
    # normalize suffix spellings but keep stem+family
    replacements = [
        (r"\bsitesinde\b", "sitesi"),
        (r"\bsite'?sinde\b", "sitesi"),
        (r"\bsit\b", "sitesi"),
        (r"\bsite\b", "sitesi"),
        (r"\brezidans\b", "residence"),
        (r"\bprojesi\b", "proje"),
        (r"\bkonaklari\b", "konaklari"),
        (r"\bevl eri\b", "evleri"),
        (r"\bevl eri\b", "evleri"),
    ]
    for pat, rep in replacements:
        t = re.sub(pat, rep, t)
    t = re.sub(r"\s+", " ", t).strip()
    tokens = [tok for tok in t.split() if tok and tok not in SITE_NAME_STOP]
    if not tokens:
        return "missing"
    if len(tokens) == 1 and tokens[0] in GENERIC_SINGLE:
        return "missing"
    # collapse known premium aliases
    joined = " ".join(tokens)
    for known in KNOWN_PREMIUM_PROJECTS:
        if known in joined or joined.startswith(known):
            return known
    # keep up to 4 tokens
    return " ".join(tokens[:4])


def extract_site_project_raw(row: pd.Series) -> str:
    site = row.get("site_name")
    if pd.notna(site):
        folded = fold_text(site)
        if folded and folded not in {"belirtilmemis", "yok", "nan", "none", "missing"}:
            return str(site).strip()

    blob = _combine_parts([str(row.get("title") or ""), str(row.get("address_text") or "")])
    if not blob:
        return ""
    m = re.search(
        r"([A-ZÇĞİÖŞÜ0-9][A-Za-zÇĞİÖŞÜçğıöşü0-9'’.\-]{1,}(?:\s+[A-ZÇĞİÖŞÜ0-9][A-Za-zÇĞİÖŞÜçğıöşü0-9'’.\-]{1,}){0,4})\s+"
        r"(S[İI]TES[İI]|S[İI]TE|S[İI]T|REZ[İI]DANS|RESIDENCE|PROJES[İI]|PROJE)\b",
        blob,
        flags=re.IGNORECASE,
    )
    if m:
        return f"{m.group(1)} {m.group(2)}"
    m2 = re.search(
        r"([A-ZÇĞİÖŞÜ0-9][A-Za-zÇĞİÖŞÜçğıöşü0-9'’.\-]{1,}(?:\s+[A-ZÇĞİÖŞÜ0-9][A-Za-zÇĞİÖŞÜçğıöşü0-9'’.\-]{1,}){0,3})\s+"
        r"S[İI]TES[İI]",
        blob,
        flags=re.IGNORECASE,
    )
    if m2:
        return f"{m2.group(1)} Sitesi"
    return ""


def match_flags(text_folded: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, patterns in KEYWORD_DICT.items():
        hit = 0
        for pat in patterns:
            if re.search(pat, text_folded):
                hit = 1
                break
        out[name] = hit
    return out


def is_known_premium(norm_name: str) -> int:
    n = fold_text(norm_name)
    if not n or n in {"missing", "other"}:
        return 0
    for known in KNOWN_PREMIUM_PROJECTS:
        if known in n or n in known:
            return 1
    return 0


def premium_score_from_flags(flags: dict[str, int], known_project: int) -> int:
    score = 0
    score += 2 * int(flags.get("text_sea_view", 0))
    score += 2 * int(flags.get("text_has_pool", 0))
    score += 2 * int(known_project)
    score += 1 * int(flags.get("text_luxury_signal", 0))
    score += 1 * int(flags.get("text_project_signal", 0))
    score += 1 * int(flags.get("text_villa_like", 0))
    score += 1 * int(flags.get("text_garden_duplex", 0))
    score += 1 * int(flags.get("text_private_garden", 0))
    score += 1 * int(flags.get("text_terrace", 0))
    return int(max(0, min(10, score)))


def score_to_level(score: int) -> str:
    if score <= 0:
        return "none"
    if score <= 2:
        return "weak"
    if score <= 5:
        return "medium"
    return "strong"


def freq_bucket(count: int) -> int:
    if count <= 0:
        return 0
    if count < 3:
        return 1
    if count < 8:
        return 2
    if count < 20:
        return 3
    return 4


def get_premium_feature_names(
    premium_feature_mode: str = "full",
    site_project_encoding: str = "frequency",
) -> list[str]:
    mode = str(premium_feature_mode or "none").lower().strip()
    enc = str(site_project_encoding or "none").lower().strip()
    if mode in {"", "none"}:
        return []

    names: list[str] = []
    if mode in {"flags", "full"}:
        names += TEXT_FLAG_FEATURES + SCORE_FEATURES
    if mode in {"site", "full"}:
        names += SITE_NUMERIC_FEATURES
        if enc == "foldsafe_target":
            names += FOLDSAFE_NUMERIC
    if mode in {"interactions", "full"}:
        # interactions need score/flags/site pieces available
        if mode == "interactions":
            names += TEXT_FLAG_FEATURES + SCORE_FEATURES + SITE_NUMERIC_FEATURES
            if enc == "foldsafe_target":
                names += FOLDSAFE_NUMERIC
        names += INTERACTION_NUMERIC
    # de-dupe
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def get_premium_categorical_feature_names(
    premium_feature_mode: str = "full",
    site_project_encoding: str = "frequency",
) -> list[str]:
    mode = str(premium_feature_mode or "none").lower().strip()
    if mode in {"", "none"}:
        return []
    names: list[str] = []
    if mode in {"flags", "full", "interactions"}:
        names += SCORE_CATEGORICAL
    if mode in {"site", "full", "interactions"}:
        names += SITE_CATEGORICAL_FEATURES
    if mode in {"interactions", "full"}:
        names += INTERACTION_CATEGORICAL
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _district_code(series: pd.Series) -> pd.Series:
    folded = series.astype(str).map(fold_text)
    # stable small int codes for numeric interaction (not one-hot)
    cats = {v: i + 1 for i, v in enumerate(sorted(set(folded.fillna("missing"))))}
    return folded.map(lambda x: float(cats.get(x, 0)))


def _m2_group(gross: pd.Series) -> pd.Series:
    g = pd.to_numeric(gross, errors="coerce")
    out = pd.Series("missing", index=g.index, dtype=object)
    out = out.mask(g.isna(), "missing")
    out = out.mask(g <= 75, "0-75")
    out = out.mask((g > 75) & (g <= 100), "76-100")
    out = out.mask((g > 100) & (g <= 125), "101-125")
    out = out.mask((g > 125) & (g <= 150), "126-150")
    out = out.mask((g > 150) & (g <= 200), "151-200")
    out = out.mask(g > 200, "200+")
    return out


def _coast_inv(df: pd.DataFrame) -> pd.Series:
    for col in (
        "distance_to_coastline_m",
        "coast_distance_m",
        "dist_to_coast_m",
        "distance_to_coast_m",
    ):
        if col in df.columns:
            d = pd.to_numeric(df[col], errors="coerce")
            return 1.0 / (1.0 + d.fillna(5000.0) / 1000.0)
    return pd.Series(0.0, index=df.index, dtype=float)


def _site_inside01(df: pd.DataFrame) -> pd.Series:
    if "site_inside" not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    s = df["site_inside"].astype(str).str.strip().str.lower()
    return s.isin({"evet", "var", "yes", "true", "1", "site içi", "site ici"}).astype(float)


def _large_home01(df: pd.DataFrame) -> pd.Series:
    g = pd.to_numeric(df.get("gross_m2"), errors="coerce")
    return (g >= 180).astype(float)


class PremiumSignalFeatureAdder(BaseEstimator, TransformerMixin):
    """Deterministic (non-target) premium features + frequency site encoding.

    Fold-safe target stats are handled by SiteProjectFoldSafeEncoder.
    """

    def __init__(
        self,
        premium_feature_mode: str = "full",
        site_project_encoding: str = "frequency",
        min_site_freq: int = 3,
    ):
        self.premium_feature_mode = premium_feature_mode
        self.site_project_encoding = site_project_encoding
        self.min_site_freq = int(min_site_freq)
        self.site_counts_: dict[str, int] = {}
        self.fitted_ = False

    def fit(self, X, y=None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        mode = str(self.premium_feature_mode or "none").lower()
        enc = str(self.site_project_encoding or "none").lower()
        self.site_counts_ = {}
        if mode in {"site", "full", "interactions"} and enc in {"frequency", "foldsafe_target"}:
            norms = []
            for _, row in df.iterrows():
                raw = extract_site_project_raw(row)
                norms.append(normalize_site_project_name(raw) if raw else "missing")
            vc = pd.Series(norms).value_counts()
            self.site_counts_ = {str(k): int(v) for k, v in vc.items() if str(k) not in {"missing"}}
        self.fitted_ = True
        return self

    def _map_site_id(self, norm: str) -> str:
        if not norm or norm == "missing":
            return "missing"
        for known in KNOWN_PREMIUM_PROJECTS:
            if known in norm or norm == known:
                return known
        cnt = int(self.site_counts_.get(norm, 0))
        if cnt < self.min_site_freq:
            return "other"
        return norm

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        out = df.copy()
        mode = str(self.premium_feature_mode or "none").lower()
        enc = str(self.site_project_encoding or "none").lower()
        if mode in {"", "none"}:
            return out

        n = len(out)
        flag_mat = {k: np.zeros(n, dtype=int) for k in TEXT_FLAG_FEATURES}
        site_norm = np.array(["missing"] * n, dtype=object)
        has_site = np.zeros(n, dtype=int)
        site_count = np.zeros(n, dtype=float)
        site_bucket = np.zeros(n, dtype=float)
        known = np.zeros(n, dtype=int)
        scores = np.zeros(n, dtype=float)
        levels = np.array(["none"] * n, dtype=object)

        need_flags = mode in {"flags", "full", "interactions"}
        need_site = mode in {"site", "full", "interactions"}
        need_inter = mode in {"interactions", "full"}

        for i, (_, row) in enumerate(out.iterrows()):
            text = fold_text(build_audit_text(row))
            flags = match_flags(text) if need_flags or need_inter or need_site else {}
            if need_flags or need_inter:
                for k, v in flags.items():
                    flag_mat[k][i] = int(v)

            raw = extract_site_project_raw(row) if need_site or need_inter else ""
            norm = normalize_site_project_name(raw) if raw else "missing"
            mapped = self._map_site_id(norm) if need_site or need_inter else "missing"
            if enc == "none" and mode == "flags":
                mapped = "missing"
            site_norm[i] = mapped
            has_site[i] = int(mapped not in {"missing", "other", ""})
            cnt = float(self.site_counts_.get(norm, 0)) if mapped not in {"missing"} else 0.0
            # for "other", keep raw count for freq feature but category is other
            if mapped == "other":
                cnt = float(self.site_counts_.get(norm, 0))
            site_count[i] = cnt
            site_bucket[i] = float(freq_bucket(int(cnt)))
            known[i] = is_known_premium(norm)
            sc = premium_score_from_flags(flags if flags else match_flags(text), int(known[i]))
            scores[i] = float(sc)
            levels[i] = score_to_level(sc)

        block: dict[str, Any] = {}
        if need_flags or need_inter:
            for k, arr in flag_mat.items():
                block[k] = arr
            block["premium_signal_score"] = scores
            block["premium_signal_level"] = levels

        if need_site or need_inter:
            block["site_project_name_normalized"] = site_norm
            block["has_site_project_name"] = has_site
            block["site_project_listing_count"] = site_count
            block["site_project_freq_bucket"] = site_bucket
            block["site_project_known_premium_flag"] = known

        if need_inter:
            gross = pd.to_numeric(out.get("gross_m2"), errors="coerce").fillna(0.0).to_numpy()
            large = _large_home01(out).to_numpy()
            site_in = _site_inside01(out).to_numpy()
            coast_inv = _coast_inv(out).to_numpy()
            district = out["district"] if "district" in out.columns else pd.Series(["missing"] * n)
            room = out["room_count"] if "room_count" in out.columns else pd.Series(["missing"] * n)
            m2g = _m2_group(out["gross_m2"] if "gross_m2" in out.columns else pd.Series(np.nan, index=out.index))
            dcode = _district_code(district).to_numpy()

            block["distance_to_coastline_inv"] = coast_inv
            block["premium_score_x_gross_m2"] = scores * gross
            block["premium_score_x_large_home"] = scores * large
            block["premium_score_x_site_inside"] = scores * site_in
            block["premium_score_x_distance_to_coastline_inv"] = scores * coast_inv
            block["sea_view_x_distance_to_coastline_inv"] = flag_mat["text_sea_view"] * coast_inv
            block["pool_x_site_inside"] = flag_mat["text_has_pool"] * site_in
            block["luxury_x_known_project"] = flag_mat["text_luxury_signal"] * known
            block["known_project_x_district_code"] = known * dcode
            block["site_project_freq_x_premium_score"] = site_count * scores

            # categorical interactions
            dist_f = district.astype(str).fillna("missing").map(fold_text)
            room_f = room.astype(str).fillna("missing").map(fold_text)
            block["premium_level_x_district"] = [f"{a}__{b}" for a, b in zip(levels, dist_f)]
            bucket_lab = [f"b{int(b)}" for b in site_bucket]
            block["site_project_bucket_x_district"] = [f"{a}__{b}" for a, b in zip(bucket_lab, dist_f)]
            block["premium_level_x_m2_group"] = [f"{a}__{b}" for a, b in zip(levels, m2g.astype(str))]
            block["premium_level_x_room_count"] = [f"{a}__{b}" for a, b in zip(levels, room_f)]

        # assign without fragmentation / avoid duplicate column names
        drop_cols = [c for c in block.keys() if c in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)
        return pd.concat([out, pd.DataFrame(block, index=out.index)], axis=1)


class SiteProjectFoldSafeEncoder(BaseEstimator, TransformerMixin):
    """Fold-safe site_project target encoding (price level + residual mean).

    Fit uses train fold unit prices only. Transform never peeks at y.
    """

    def __init__(
        self,
        enabled: bool = False,
        min_count: int = 3,
        alpha: float = 20.0,
        premium_feature_mode: str = "full",
    ):
        self.enabled = bool(enabled)
        self.min_count = int(min_count)
        self.alpha = float(alpha)
        self.premium_feature_mode = premium_feature_mode
        self.global_mean_: float = np.nan
        self.stats_: dict[str, dict[str, float]] = {}
        self.leakage_guard_: dict[str, Any] = {}

    def fit(self, X, y=None):
        self.stats_ = {}
        self.global_mean_ = np.nan
        self.leakage_guard_ = {
            "enabled": bool(self.enabled),
            "uses_train_pool_only": True,
            "validation_targets_used": False,
            "outer_validation_targets_used_in_encoder": False,
            "min_count": self.min_count,
            "alpha": self.alpha,
            "pass": True,
            "notes": [],
        }
        if not self.enabled:
            self.leakage_guard_["notes"].append("foldsafe encoder disabled")
            return self

        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if PREMIUM_UNIT_PRICE_COL in df.columns:
            y_arr = pd.to_numeric(df[PREMIUM_UNIT_PRICE_COL], errors="coerce").to_numpy(dtype=float)
        elif y is not None:
            y_arr = np.asarray(y, dtype=float)
        elif "unit_price_gross" in df.columns:
            y_arr = pd.to_numeric(df["unit_price_gross"], errors="coerce").to_numpy(dtype=float)
        else:
            self.leakage_guard_["pass"] = False
            self.leakage_guard_["notes"].append("no unit price available for foldsafe fit")
            return self

        # Ensure site ids exist (PremiumSignalFeatureAdder should run before this).
        if "site_project_name_normalized" not in df.columns:
            tmp = PremiumSignalFeatureAdder(
                premium_feature_mode="site",
                site_project_encoding="frequency",
            ).fit_transform(df)
            site_id = tmp["site_project_name_normalized"].astype(str)
        else:
            site_id = df["site_project_name_normalized"].astype(str)

        mask = np.isfinite(y_arr)
        if int(mask.sum()) < 5:
            self.leakage_guard_["pass"] = False
            self.leakage_guard_["notes"].append("insufficient train rows for foldsafe")
            return self

        y_ok = y_arr[mask]
        id_ok = site_id.to_numpy()[mask]
        self.global_mean_ = float(np.mean(y_ok))
        # residual vs global mean as simple level residual
        for sid in sorted(set(id_ok)):
            if sid in {"missing", "other", "nan", "None"}:
                continue
            vals = y_ok[id_ok == sid]
            n = int(len(vals))
            if n < self.min_count:
                continue
            mean = float(np.mean(vals))
            # smoothed toward global
            w = n / (n + self.alpha)
            level = w * mean + (1.0 - w) * self.global_mean_
            resid = level - self.global_mean_
            conf = float(min(1.0, n / (n + self.alpha)))
            self.stats_[str(sid)] = {
                "price_level": level,
                "residual_mean": resid,
                "count": float(n),
                "confidence": conf,
            }
        self.leakage_guard_["n_sites_encoded"] = len(self.stats_)
        self.leakage_guard_["global_mean"] = self.global_mean_
        self.leakage_guard_["notes"].append("fit on train fold unit prices only")
        return self

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        out = df.copy()
        n = len(out)
        if not self.enabled:
            return out

        if "site_project_name_normalized" in out.columns:
            site_id = out["site_project_name_normalized"].astype(str)
        else:
            site_id = pd.Series(["missing"] * n, index=out.index)

        price = np.full(n, self.global_mean_ if np.isfinite(self.global_mean_) else np.nan, dtype=float)
        resid = np.zeros(n, dtype=float)
        count = np.zeros(n, dtype=float)
        conf = np.zeros(n, dtype=float)
        for i, sid in enumerate(site_id.tolist()):
            st = self.stats_.get(str(sid))
            if not st:
                continue
            price[i] = float(st["price_level"])
            resid[i] = float(st["residual_mean"])
            count[i] = float(st["count"])
            conf[i] = float(st["confidence"])

        block = pd.DataFrame(
            {
                "site_project_oof_price_level": price,
                "site_project_oof_residual_mean": resid,
                "site_project_oof_count": count,
                "site_project_oof_confidence": conf,
            },
            index=out.index,
        )
        return pd.concat([out, block], axis=1)

    def leakage_guard_report(self) -> dict[str, Any]:
        return dict(self.leakage_guard_ or {})


def write_leakage_guard(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def premium_feature_coverage(df: pd.DataFrame, flag_cols: list[str] | None = None) -> pd.DataFrame:
    cols = flag_cols or (
        TEXT_FLAG_FEATURES
        + SITE_NUMERIC_FEATURES
        + SCORE_FEATURES
        + ["site_project_name_normalized", "premium_signal_level"]
    )
    rows = []
    n = len(df)
    for c in cols:
        if c not in df.columns:
            rows.append({"feature": c, "non_null": 0, "coverage": 0.0, "mean_or_share": np.nan})
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            non = int(s.notna().sum())
            rows.append(
                {
                    "feature": c,
                    "non_null": non,
                    "coverage": non / n if n else np.nan,
                    "mean_or_share": float(pd.to_numeric(s, errors="coerce").fillna(0).mean()),
                }
            )
        else:
            non = int(s.notna().sum())
            share_known = float((s.astype(str).map(fold_text) != "missing").mean()) if n else np.nan
            rows.append(
                {
                    "feature": c,
                    "non_null": non,
                    "coverage": non / n if n else np.nan,
                    "mean_or_share": share_known,
                }
            )
    return pd.DataFrame(rows)


def site_project_candidates_table(df: pd.DataFrame) -> pd.DataFrame:
    if "site_project_name_normalized" not in df.columns:
        return pd.DataFrame()
    agg_map: dict[str, tuple[str, str]] = {
        "n_listings": ("site_project_name_normalized", "size"),
    }
    if "site_project_known_premium_flag" in df.columns:
        agg_map["known_premium"] = ("site_project_known_premium_flag", "max")
    if "premium_signal_score" in df.columns:
        agg_map["mean_premium_score"] = ("premium_signal_score", "mean")
    g = (
        df.groupby("site_project_name_normalized", dropna=False)
        .agg(**{k: v for k, v in agg_map.items()})
        .reset_index()
        .sort_values("n_listings", ascending=False)
    )
    return g
