"""V21 Başiskele site/project extraction (anti-overmerge).

Free text is never passed raw into the model. Produces controlled
canonical site IDs, dictionary hits, quality tiers (no target leakage),
frequency features, and optional fold-safe encodings.

Hard rules:
- Do not strip brand tokens: life, park, kent, vadi, city, royal, koru, perla
- Only strip clearly generic suffixes: sitesi, site, konutlari, evleri, ...
- Do not auto-merge phase numbers (orka_life != orka_life_2)
- Conservative alias dictionary; uncertain -> separate IDs + review CSV
- Quality tier from dictionary/curation only (not target means)
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Injected by LocationResidualRegressor (same pattern as V20 premium/comparable).
SITE_UNIT_PRICE_COL = "__premium_unit_price__"

# Brand tokens that must NEVER be globally stripped.
BRAND_TOKENS = {
    "life",
    "park",
    "kent",
    "vadi",
    "city",
    "royal",
    "koru",
    "perla",
}

# Generic suffixes only (folded ASCII forms).
GENERIC_SUFFIXES = (
    "sitesi",
    "site",
    "sit",
    "konutlari",
    "evleri",
    "rezidans",
    "residence",
    "projesi",
    "proje",
    "blok",
)

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

TIER_CODE = {"unknown": 0, "mass": 1, "mid": 2, "premium": 3}


@dataclass(frozen=True)
class SiteDictEntry:
    canonical_id: str
    aliases: tuple[str, ...]
    quality_tier: str  # premium | mid | mass | unknown
    known_premium: bool
    notes: str = ""


# Conservative curated dictionary. Phase numbers are explicit; no auto-merge.
BASISKELE_SITE_DICTIONARY: tuple[SiteDictEntry, ...] = (
    SiteDictEntry("zeray_perla", ("zeray perla", "zeray perla sitesi"), "premium", True, "curated premium"),
    SiteDictEntry("zeray_korupark", ("zeray korupark", "zeray koru park", "zeray korupark sitesi"), "premium", True, "curated premium"),
    SiteDictEntry("zeray_gunesi", ("zeray gunesi", "zeray gunes", "zeray gunesi sitesi"), "premium", True, "curated premium"),
    SiteDictEntry("orka_life_2", ("orka life 2", "orka life ii", "orka life 2 sitesi"), "premium", True, "phase kept; not orka_life"),
    SiteDictEntry("orka_life", ("orka life", "orka life sitesi"), "mid", False, "separate from orka_life_2"),
    SiteDictEntry("evimiz_kocaeli", ("evimiz kocaeli", "evimiz", "evimiz sitesi"), "premium", True, "suffix-only merge"),
    SiteDictEntry("alizepark_city", ("alizepark city", "alize park city", "alizepark city sitesi"), "premium", True, "curated premium"),
    SiteDictEntry("ustgrup_mezire", ("ustgrup mezire", "ust grup mezire", "mezire"), "premium", True, "curated premium"),
    SiteDictEntry("mamik_life", ("mamik life", "mamik life sitesi"), "premium", True, "distinct from mamik_reform"),
    SiteDictEntry("mamik_reform", ("mamik reform", "mamik reform sitesi"), "mid", False, "distinct from mamik_life"),
    SiteDictEntry("royal_life", ("royal life", "royal life sitesi"), "premium", True, "distinct from royal_city / royal_country"),
    SiteDictEntry("royal_city_akkurt", ("royal city akkurt", "royal city", "royal city sitesi"), "mid", False, "distinct from royal_life"),
    SiteDictEntry("royal_country", ("royal country", "royal country sitesi"), "mid", False, "distinct from royal_life"),
    SiteDictEntry("panoramakent", ("panoramakent", "panorama kent", "panoramakent sitesi"), "premium", True, "unnumbered; panoramakent_3 separate if seen"),
    SiteDictEntry("bahcekent_konutlari", ("bahcekent konutlari", "bahcekent", "bahce kent"), "mid", False, "suffix strip only"),
    SiteDictEntry("kirazli_kent", ("kirazli kent", "kirazli kent sitesi"), "mid", False, "brand kent preserved"),
    SiteDictEntry("yakamoz_konutlari", ("yakamoz konutlari", "yakamoz"), "mid", False, "curated mid"),
    SiteDictEntry("yuvana_evleri", ("yuvana evleri", "yuvana"), "mid", False, "curated mid"),
    SiteDictEntry("zeraykent_yuvacik", ("zeraykent yuvacik", "zeray kent yuvacik"), "mid", False, "kent preserved"),
    SiteDictEntry("ustpark_afraze", ("ustpark afraze", "ust park afraze"), "mid", False, "park preserved"),
    SiteDictEntry("ritim_vadi_evleri", ("ritim vadi evleri", "ritim vadi"), "mid", False, "vadi preserved"),
)


SITE_NUMERIC_FEATURES = [
    "has_site_project_id",
    "site_project_listing_count",
    "site_project_freq_bucket",
    "site_project_known_premium_flag",
    "site_is_premium_tier",
    "site_is_mid_tier",
    "site_tier_code",
]

SITE_CATEGORICAL_FEATURES = [
    "site_project_id",
    "site_project_match_source",
    "site_quality_tier",
]

INTERACTION_NUMERIC = [
    "known_premium_x_district_code",
    "site_tier_x_large_home",
    "site_tier_x_coast_inv",
    "canonical_count_x_large_home",
    "has_site_x_site_inside",
    "distance_to_coastline_inv",
]

INTERACTION_CATEGORICAL = [
    "site_tier_x_district",
    "site_id_bucket_x_district",
]

FOLDSAFE_NUMERIC = [
    "site_project_oof_price_level",
    "site_project_oof_residual_mean",
    "site_project_oof_count",
    "site_project_oof_confidence",
]


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


def build_audit_text(row: pd.Series) -> str:
    return _combine_parts(
        [
            str(row.get("title") or ""),
            str(row.get("site_name") or ""),
            str(row.get("address_text") or ""),
        ]
    )


def _build_alias_index() -> dict[str, SiteDictEntry]:
    idx: dict[str, SiteDictEntry] = {}
    for e in BASISKELE_SITE_DICTIONARY:
        for a in e.aliases:
            idx[fold_text(a)] = e
        idx[fold_text(e.canonical_id.replace("_", " "))] = e
    return idx


ALIAS_INDEX = _build_alias_index()


def strip_generic_suffixes(folded: str) -> str:
    """Strip only clearly generic suffixes; never strip brand tokens."""
    t = fold_text(folded)
    if not t:
        return ""
    tokens = t.split()
    # Drop trailing generic suffixes repeatedly, but never if token is brand-critical alone wrongly
    while tokens:
        last = tokens[-1]
        if last in GENERIC_SUFFIXES and last not in BRAND_TOKENS:
            tokens = tokens[:-1]
            continue
        # "etap" only when not needed for identity (drop trailing etap / etap N when N alone)
        if last == "etap":
            tokens = tokens[:-1]
            continue
        if re.fullmatch(r"etap\d+", last):
            # keep phase if dictionary might need it — leave for dict match; don't auto-strip numbered etap
            break
        break
    return " ".join(tokens).strip()


def normalize_site_stem(raw: str) -> str:
    """Conservative normalize: fold + generic suffix strip; keep brand tokens & phase nums."""
    t = fold_text(raw)
    if not t:
        return "missing"
    t = re.sub(r"\b\d+\+\d+\b", " ", t)
    t = re.sub(r"\b\d+\s*m2\b", " ", t)
    # roman ii -> 2 only as standalone phase token after life/park-like stems handled later via dict
    t = re.sub(r"\s+", " ", t).strip()
    t = strip_generic_suffixes(t)
    tokens = [tok for tok in t.split() if tok and tok not in SITE_NAME_STOP]
    if not tokens:
        return "missing"
    # Keep trailing phase digits (panoramakent 3, orka life 2). Do NOT auto-drop.
    # Dict match decides whether unnumbered vs numbered are the same ID.
    # Map life ii -> life 2 (phase synonym only)
    out_toks: list[str] = []
    for tok in tokens:
        if tok in {"ii", "ii."}:
            out_toks.append("2")
        else:
            out_toks.append(tok)
    return " ".join(out_toks[:6])


def stem_to_id(stem: str) -> str:
    s = fold_text(stem)
    if not s or s == "missing":
        return "missing"
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_") or "missing"


def extract_site_project_raw(row: pd.Series) -> tuple[str, str]:
    """Return (raw_string, match_source).

    Order:
    1) site_name if usable
    2) dictionary alias phrase in title/address audit text
    3) broadened site/project pattern on folded text (no Capital-letter requirement)
    """
    site = row.get("site_name")
    if pd.notna(site):
        folded = fold_text(site)
        if folded and folded not in {"belirtilmemis", "yok", "nan", "none", "missing", "var"}:
            return str(site).strip(), "site_name"

    title = str(row.get("title") or "")
    addr = str(row.get("address_text") or "")
    blob = _combine_parts([title, addr])
    if not blob:
        return "", "missing"
    audit = fold_text(blob)

    # Dictionary phrase hit in free text (longest alias first) — still phase-safe via match_dictionary later.
    best_alias = ""
    best_len = 0
    for alias in ALIAS_INDEX:
        if not alias or len(alias) < 4:
            continue
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", audit):
            # Prefer numbered alias when present (orka life 2 over orka life)
            if len(alias) > best_len:
                best_alias = alias
                best_len = len(alias)
    if best_alias:
        src = "title" if title and best_alias in fold_text(title) else ("address" if addr else "dict")
        if src == "address" and best_alias not in fold_text(addr):
            src = "dict"
        return best_alias, src

    # Broad pattern on folded text: "<name tokens> <generic|brand suffix>"
    m = re.search(
        r"\b([a-z0-9][a-z0-9'./-]{1,}(?:\s+[a-z0-9][a-z0-9'./-]{1,}){0,5})\s+"
        r"(sitesi|site|sit|rezidans|residence|projesi|proje|konutlari|evleri|life|park|kent|vadi|city)\b",
        audit,
    )
    if m:
        name = m.group(1).strip()
        suf = m.group(2).strip()
        # Reject pure stop / too-generic heads
        head_toks = [t for t in name.split() if t not in SITE_NAME_STOP]
        if head_toks and not (len(head_toks) == 1 and head_toks[0] in GENERIC_SUFFIXES):
            raw = f"{name} {suf}"
            src = "title" if title and name in fold_text(title) else "address"
            return raw, src

    # Trailing brand-like bigram without suffix: e.g. ".... mamik life end"
    m2 = re.search(
        r"\b([a-z0-9]{3,}(?:\s+[a-z0-9]{2,}){0,3}\s+(?:life|park|kent|vadi|city|perla|korupark|gunesi))\b",
        audit,
    )
    if m2:
        raw = m2.group(1).strip()
        if raw not in SITE_NAME_STOP and len(raw.split()) >= 2:
            src = "title" if title and raw in fold_text(title) else "address"
            return raw, src

    return "", "missing"


def match_dictionary(stem_folded: str, audit_folded: str) -> SiteDictEntry | None:
    """Conservative dict match. Prefer exact alias; avoid phase over-merge."""
    stem = fold_text(stem_folded)
    if stem and stem in ALIAS_INDEX:
        return ALIAS_INDEX[stem]

    # Exact token-boundary match of aliases against stem (longest first).
    # Reject if stem has an extra trailing phase token not present in the alias
    # (e.g. alias "panoramakent" must NOT absorb "panoramakent 3").
    best: SiteDictEntry | None = None
    best_len = 0
    stem_tokens = stem.split() if stem else []
    for alias, entry in ALIAS_INDEX.items():
        if not alias or len(alias) < 4:
            continue
        alias_tokens = alias.split()
        if stem_tokens[: len(alias_tokens)] == alias_tokens:
            # leftover tokens after alias
            rest = stem_tokens[len(alias_tokens) :]
            if rest and all(re.fullmatch(r"\d+", t) or t in {"ii"} for t in rest):
                # phase remainder -> do not use unnumbered alias
                continue
            if len(alias) > best_len:
                best = entry
                best_len = len(alias)
                continue
        # containment in audit text only when alias appears as whole phrase
        if audit_folded and re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", audit_folded):
            # if audit also has alias + digit phase nearby, skip unnumbered
            if re.search(rf"(?<!\w){re.escape(alias)}\s+\d+\b", audit_folded):
                # only accept if this alias itself ends with that phase
                if not re.search(rf"(?<!\w){re.escape(alias)}(?!\s+\d)", audit_folded):
                    continue
            if len(alias) > best_len:
                best = entry
                best_len = len(alias)
    return best


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


def get_site_feature_names(
    site_extraction_mode: str = "full",
    site_project_encoding: str = "foldsafe_target",
) -> list[str]:
    mode = str(site_extraction_mode or "none").lower().strip()
    enc = str(site_project_encoding or "none").lower().strip()
    if mode in {"", "none"}:
        return []
    names: list[str] = []
    # all site modes that produce identity features
    if mode in {"v20_parity", "alias", "dict", "tier", "interactions", "full"}:
        names += [
            "has_site_project_id",
            "site_project_listing_count",
            "site_project_freq_bucket",
            "site_project_known_premium_flag",
        ]
    if mode in {"tier", "interactions", "full"}:
        names += ["site_is_premium_tier", "site_is_mid_tier", "site_tier_code"]
    if mode in {"interactions", "full"}:
        names += list(INTERACTION_NUMERIC)
    if enc == "foldsafe_target" and mode not in {"", "none"}:
        names += list(FOLDSAFE_NUMERIC)
    # de-dupe
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def get_site_categorical_feature_names(
    site_extraction_mode: str = "full",
    site_project_encoding: str = "foldsafe_target",
) -> list[str]:
    mode = str(site_extraction_mode or "none").lower().strip()
    if mode in {"", "none"}:
        return []
    names: list[str] = []
    if mode in {"v20_parity", "alias", "dict", "tier", "interactions", "full"}:
        names += ["site_project_id", "site_project_match_source"]
    if mode in {"tier", "interactions", "full"}:
        names += ["site_quality_tier"]
    if mode in {"interactions", "full"}:
        names += list(INTERACTION_CATEGORICAL)
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _district_code(series: pd.Series) -> pd.Series:
    folded = series.astype(str).map(fold_text)
    cats = {v: i + 1 for i, v in enumerate(sorted(set(folded.fillna("missing"))))}
    return folded.map(lambda x: float(cats.get(x, 0)))


def _coast_inv(df: pd.DataFrame) -> pd.Series:
    for col in ("distance_to_coastline_m", "coast_distance_m", "distance_to_coast_m"):
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


class SiteProjectExtractionAdder(BaseEstimator, TransformerMixin):
    """Deterministic site/project features with conservative aliasing."""

    def __init__(
        self,
        site_extraction_mode: str = "full",
        site_project_encoding: str = "frequency",
        min_site_freq: int = 1,
        merge_gap_warning_tl: float = 8000.0,
    ):
        self.site_extraction_mode = site_extraction_mode
        self.site_project_encoding = site_project_encoding
        self.min_site_freq = int(min_site_freq)
        self.merge_gap_warning_tl = float(merge_gap_warning_tl)
        self.site_counts_: dict[str, int] = {}
        self.raw_to_canonical_: dict[str, str] = {}
        self.review_rows_: list[dict[str, Any]] = []
        self.merge_rows_: list[dict[str, Any]] = []
        self.coverage_: dict[str, float] = {}
        self.fitted_ = False

    def fit(self, X, y=None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        mode = str(self.site_extraction_mode or "none").lower()
        self.site_counts_ = {}
        self.raw_to_canonical_ = {}
        self.review_rows_ = []
        self.merge_rows_ = []
        self.coverage_ = {}
        if mode in {"", "none"}:
            self.fitted_ = True
            return self

        rows_meta: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            raw, src = extract_site_project_raw(row)
            stem = normalize_site_stem(raw) if raw else "missing"
            audit = fold_text(build_audit_text(row))
            entry = None
            if mode in {"dict", "tier", "interactions", "full"}:
                entry = match_dictionary(stem, audit)
            elif mode in {"alias", "v20_parity"}:
                # alias mode: suffix-normalized stem only; known premium via dict known list without forced merge of phases
                entry = match_dictionary(stem, "") if mode == "alias" else None
                if mode == "v20_parity":
                    # V20-like: known premium match on stem containment of premium ids only
                    entry = match_dictionary(stem, stem)

            if entry is not None:
                cid = entry.canonical_id
                match_src = "dict" if src != "missing" or entry else src
                if src == "missing" and entry is not None:
                    match_src = "dict"
                else:
                    match_src = src if src != "missing" else "dict"
            else:
                cid = stem_to_id(stem) if stem not in {"missing", ""} else "missing"
                match_src = src

            rows_meta.append(
                {
                    "raw": raw or "",
                    "stem": stem,
                    "canonical_id": cid,
                    "match_source": match_src,
                    "entry": entry,
                    "unit_price": pd.to_numeric(row.get("unit_price_gross"), errors="coerce"),
                }
            )

        # frequency on provisional canonicals (before rare->other)
        vc = pd.Series([r["canonical_id"] for r in rows_meta]).value_counts()
        self.site_counts_ = {str(k): int(v) for k, v in vc.items() if str(k) not in {"missing"}}

        # Rare handling: keep distinct canonical IDs (min_site_freq controls collapse).
        # freq < 3 non-dict still flagged for manual review, but not force-merged when min_freq<=1.
        variant_map: dict[str, list[str]] = {}
        for r in rows_meta:
            cid = r["canonical_id"]
            if cid not in {"missing"} and r["entry"] is None:
                freq_i = int(self.site_counts_.get(cid, 0))
                if freq_i < self.min_site_freq:
                    self.review_rows_.append(
                        {
                            "raw_name": r["raw"] or r["stem"],
                            "proposed_canonical_id": cid,
                            "reason": "freq_lt_min_separate_other",
                            "frequency": freq_i,
                            "mean_price": np.nan,
                            "median_price": np.nan,
                            "mean_residual": np.nan,
                            "needs_manual_review": True,
                        }
                    )
                    cid = "other"
                    r["canonical_id"] = cid
                elif freq_i < 3:
                    self.review_rows_.append(
                        {
                            "raw_name": r["raw"] or r["stem"],
                            "proposed_canonical_id": cid,
                            "reason": "low_freq_kept_separate_review",
                            "frequency": freq_i,
                            "mean_price": np.nan,
                            "median_price": np.nan,
                            "mean_residual": np.nan,
                            "needs_manual_review": True,
                        }
                    )
            # uncertain near-matches: stem close to a dict id but not matched
            if r["entry"] is None and r["stem"] not in {"missing", ""}:
                for e in BASISKELE_SITE_DICTIONARY:
                    stem_id = stem_to_id(r["stem"])
                    if stem_id != e.canonical_id and (
                        e.canonical_id in stem_id or stem_id in e.canonical_id
                    ):
                        self.review_rows_.append(
                            {
                                "raw_name": r["raw"] or r["stem"],
                                "proposed_canonical_id": e.canonical_id,
                                "reason": "near_dict_not_merged_conservative",
                                "frequency": int(self.site_counts_.get(stem_to_id(r["stem"]), 0)),
                                "mean_price": np.nan,
                                "median_price": np.nan,
                                "mean_residual": np.nan,
                                "needs_manual_review": True,
                            }
                        )
            key = fold_text(r["raw"] or r["stem"])
            if key:
                self.raw_to_canonical_[key] = r["canonical_id"]
                variant_map.setdefault(r["canonical_id"], [])
                if key not in variant_map[r["canonical_id"]]:
                    variant_map[r["canonical_id"]].append(key)

        # recompute counts after rare remap
        vc2 = pd.Series([r["canonical_id"] for r in rows_meta]).value_counts()
        self.site_counts_ = {str(k): int(v) for k, v in vc2.items() if str(k) not in {"missing"}}

        # merge audit with price gaps
        price_by_variant: dict[str, dict[str, list[float]]] = {}
        for r in rows_meta:
            cid = r["canonical_id"]
            raw_k = fold_text(r["raw"] or r["stem"]) or "missing"
            p = r["unit_price"]
            if cid in {"missing"}:
                continue
            price_by_variant.setdefault(cid, {}).setdefault(raw_k, [])
            if pd.notna(p):
                price_by_variant[cid][raw_k].append(float(p))

        for cid, variants in price_by_variant.items():
            if cid in {"missing", "other"}:
                continue
            medians = {}
            for v, ps in variants.items():
                if ps:
                    medians[v] = float(np.median(ps))
            gap = float(max(medians.values()) - min(medians.values())) if len(medians) >= 2 else 0.0
            warning = "possible_bad_merge" if gap >= self.merge_gap_warning_tl else ""
            self.merge_rows_.append(
                {
                    "canonical_id": cid,
                    "raw_variants": " | ".join(sorted(variants.keys())),
                    "n_variants": len(variants),
                    "total_count": int(sum(len(ps) for ps in variants.values()) or self.site_counts_.get(cid, 0)),
                    "median_unit_price_by_variant": json.dumps(medians, ensure_ascii=False),
                    "max_variant_price_gap": gap,
                    "warning": warning,
                }
            )

        n = max(len(rows_meta), 1)
        extracted = sum(1 for r in rows_meta if r["raw"])
        alias_hit = sum(1 for r in rows_meta if r["stem"] not in {"", "missing"} and r["stem"] != fold_text(r["raw"]))
        # alias_hit_rate: stem differs from raw fold after suffix strip OR dict/stem assigned
        alias_hit = sum(
            1
            for r in rows_meta
            if r["stem"] not in {"", "missing"}
            and strip_generic_suffixes(fold_text(r["raw"])) == r["stem"]
            and r["raw"]
        )
        dict_hit = sum(1 for r in rows_meta if r["entry"] is not None)
        canon_nm = sum(1 for r in rows_meta if r["canonical_id"] not in {"missing", "other", ""})
        self.coverage_ = {
            "extracted_raw_rate": extracted / n,
            "alias_hit_rate": alias_hit / n,
            "dict_hit_rate": dict_hit / n,
            "canonical_non_missing_rate": canon_nm / n,
            "n_rows": float(n),
            "n_canonical_sites": float(len([k for k in self.site_counts_ if k not in {"other"}])),
        }
        self._fit_rows_meta_ = rows_meta  # type: ignore[attr-defined]
        self.fitted_ = True
        return self

    def _resolve_one(self, row: pd.Series) -> dict[str, Any]:
        mode = str(self.site_extraction_mode or "none").lower()
        raw, src = extract_site_project_raw(row)
        stem = normalize_site_stem(raw) if raw else "missing"
        audit = fold_text(build_audit_text(row))
        entry = None
        if mode in {"dict", "tier", "interactions", "full", "alias", "v20_parity"}:
            if mode == "v20_parity":
                entry = match_dictionary(stem, stem)
            elif mode == "alias":
                entry = match_dictionary(stem, "")
            else:
                entry = match_dictionary(stem, audit)

        if entry is not None:
            cid = entry.canonical_id
            match_source = src if src != "missing" else "dict"
            if src == "missing":
                match_source = "dict"
            known = int(entry.known_premium)
            tier = entry.quality_tier if mode in {"tier", "interactions", "full"} else "unknown"
        else:
            cid = stem_to_id(stem) if stem not in {"missing", ""} else "missing"
            if cid not in {"missing"} and int(self.site_counts_.get(cid, 0)) < self.min_site_freq:
                cid = "other"
            match_source = src
            known = 0
            tier = "unknown"

        cnt = float(self.site_counts_.get(cid, 0)) if cid not in {"missing"} else 0.0
        return {
            "raw": raw,
            "stem": stem,
            "site_project_id": cid,
            "has_site_project_id": int(cid not in {"missing", "other", ""}),
            "site_project_listing_count": cnt,
            "site_project_freq_bucket": float(freq_bucket(int(cnt))),
            "site_project_known_premium_flag": known,
            "site_project_match_source": match_source,
            "site_quality_tier": tier,
            "site_is_premium_tier": int(tier == "premium"),
            "site_is_mid_tier": int(tier == "mid"),
            "site_tier_code": float(TIER_CODE.get(tier, 0)),
        }

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        out = df.copy()
        mode = str(self.site_extraction_mode or "none").lower()
        if mode in {"", "none"}:
            return out

        n = len(out)
        resolved = [self._resolve_one(row) for _, row in out.iterrows()]

        block: dict[str, Any] = {
            "site_project_id": [r["site_project_id"] for r in resolved],
            "has_site_project_id": [r["has_site_project_id"] for r in resolved],
            "site_project_listing_count": [r["site_project_listing_count"] for r in resolved],
            "site_project_freq_bucket": [r["site_project_freq_bucket"] for r in resolved],
            "site_project_known_premium_flag": [r["site_project_known_premium_flag"] for r in resolved],
            "site_project_match_source": [r["site_project_match_source"] for r in resolved],
        }
        if mode in {"tier", "interactions", "full"}:
            block["site_quality_tier"] = [r["site_quality_tier"] for r in resolved]
            block["site_is_premium_tier"] = [r["site_is_premium_tier"] for r in resolved]
            block["site_is_mid_tier"] = [r["site_is_mid_tier"] for r in resolved]
            block["site_tier_code"] = [r["site_tier_code"] for r in resolved]

        if mode in {"interactions", "full"}:
            coast_inv = _coast_inv(out).to_numpy()
            large = _large_home01(out).to_numpy()
            site_in = _site_inside01(out).to_numpy()
            district = out["district"] if "district" in out.columns else pd.Series(["missing"] * n)
            dcode = _district_code(district).to_numpy()
            known = np.array([r["site_project_known_premium_flag"] for r in resolved], dtype=float)
            tier_c = np.array([r["site_tier_code"] for r in resolved], dtype=float)
            cnt = np.array([r["site_project_listing_count"] for r in resolved], dtype=float)
            has = np.array([r["has_site_project_id"] for r in resolved], dtype=float)
            buckets = [f"b{int(r['site_project_freq_bucket'])}" for r in resolved]
            tiers = [r.get("site_quality_tier", "unknown") for r in resolved]
            dist_f = district.astype(str).fillna("missing").map(fold_text)

            block["distance_to_coastline_inv"] = coast_inv
            block["known_premium_x_district_code"] = known * dcode
            block["site_tier_x_large_home"] = tier_c * large
            block["site_tier_x_coast_inv"] = tier_c * coast_inv
            block["canonical_count_x_large_home"] = cnt * large
            block["has_site_x_site_inside"] = has * site_in
            block["site_tier_x_district"] = [f"{a}__{b}" for a, b in zip(tiers, dist_f)]
            block["site_id_bucket_x_district"] = [f"{a}__{b}" for a, b in zip(buckets, dist_f)]

        drop_cols = [c for c in block.keys() if c in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)
        return pd.concat([out, pd.DataFrame(block, index=out.index)], axis=1)


class SiteProjectFoldSafeEncoder(BaseEstimator, TransformerMixin):
    """Fold-safe target encoding on canonical site_project_id."""

    def __init__(
        self,
        enabled: bool = False,
        min_count: int = 3,
        alpha: float = 20.0,
        site_extraction_mode: str = "full",
    ):
        self.enabled = bool(enabled)
        self.min_count = int(min_count)
        self.alpha = float(alpha)
        self.site_extraction_mode = site_extraction_mode
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
        if SITE_UNIT_PRICE_COL in df.columns:
            y_arr = pd.to_numeric(df[SITE_UNIT_PRICE_COL], errors="coerce").to_numpy(dtype=float)
        elif y is not None:
            y_arr = np.asarray(y, dtype=float)
        elif "unit_price_gross" in df.columns:
            y_arr = pd.to_numeric(df["unit_price_gross"], errors="coerce").to_numpy(dtype=float)
        else:
            self.leakage_guard_["pass"] = False
            self.leakage_guard_["notes"].append("no unit price available for foldsafe fit")
            return self

        if "site_project_id" not in df.columns:
            tmp = SiteProjectExtractionAdder(
                site_extraction_mode=str(self.site_extraction_mode or "dict"),
                site_project_encoding="frequency",
            ).fit_transform(df)
            site_id = tmp["site_project_id"].astype(str)
        else:
            site_id = df["site_project_id"].astype(str)

        mask = np.isfinite(y_arr)
        if int(mask.sum()) < 5:
            self.leakage_guard_["pass"] = False
            self.leakage_guard_["notes"].append("insufficient train rows for foldsafe")
            return self

        y_ok = y_arr[mask]
        id_ok = site_id.to_numpy()[mask]
        self.global_mean_ = float(np.mean(y_ok))
        for sid in sorted(set(id_ok)):
            if sid in {"missing", "other", "nan", "None"}:
                continue
            vals = y_ok[id_ok == sid]
            n = int(len(vals))
            if n < self.min_count:
                continue
            mean = float(np.mean(vals))
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
        if not self.enabled:
            return out
        n = len(out)
        site_id = (
            out["site_project_id"].astype(str)
            if "site_project_id" in out.columns
            else pd.Series(["missing"] * n, index=out.index)
        )
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
        block = {
            "site_project_oof_price_level": price,
            "site_project_oof_residual_mean": resid,
            "site_project_oof_count": count,
            "site_project_oof_confidence": conf,
        }
        drop_cols = [c for c in block if c in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)
        return pd.concat([out, pd.DataFrame(block, index=out.index)], axis=1)

    def leakage_guard_report(self) -> dict[str, Any]:
        return dict(self.leakage_guard_ or {})


def write_leakage_guard(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def dictionary_table() -> pd.DataFrame:
    rows = []
    for e in BASISKELE_SITE_DICTIONARY:
        rows.append(
            {
                "canonical_id": e.canonical_id,
                "aliases": " | ".join(e.aliases),
                "quality_tier": e.quality_tier,
                "known_premium": int(e.known_premium),
                "notes": e.notes,
            }
        )
    return pd.DataFrame(rows)


def alias_map_table(adder: SiteProjectExtractionAdder) -> pd.DataFrame:
    rows = [{"raw_folded": k, "canonical_id": v} for k, v in sorted(adder.raw_to_canonical_.items())]
    return pd.DataFrame(rows)


def coverage_table(adder: SiteProjectExtractionAdder) -> pd.DataFrame:
    cov = dict(adder.coverage_ or {})
    return pd.DataFrame([{"metric": k, "value": v} for k, v in cov.items()])


def candidates_table(df: pd.DataFrame) -> pd.DataFrame:
    if "site_project_id" not in df.columns:
        return pd.DataFrame()
    g = (
        df.groupby("site_project_id", dropna=False)
        .agg(
            n_listings=("site_project_id", "size"),
            known_premium=("site_project_known_premium_flag", "max")
            if "site_project_known_premium_flag" in df.columns
            else ("site_project_id", "size"),
        )
        .reset_index()
        .sort_values("n_listings", ascending=False)
    )
    return g


def enrich_review_with_prices(review_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    if review_df is None or review_df.empty or "unit_price_gross" not in df.columns:
        return review_df
    out = review_df.copy()
    # best-effort: group by proposed id if present on frame
    if "site_project_id" in df.columns:
        stats = (
            df.assign(_p=pd.to_numeric(df["unit_price_gross"], errors="coerce"))
            .groupby("site_project_id")["_p"]
            .agg(["mean", "median", "count"])
            .reset_index()
        )
        stats = stats.rename(columns={"site_project_id": "proposed_canonical_id", "mean": "mean_price", "median": "median_price"})
        out = out.drop(columns=["mean_price", "median_price"], errors="ignore")
        out = out.merge(stats[["proposed_canonical_id", "mean_price", "median_price"]], on="proposed_canonical_id", how="left")
    return out


def count_severe_bad_merges(merge_df: pd.DataFrame) -> int:
    if merge_df is None or merge_df.empty:
        return 0
    m = merge_df.copy()
    warn = m["warning"].astype(str).eq("possible_bad_merge")
    severe = warn & ((pd.to_numeric(m["total_count"], errors="coerce").fillna(0) >= 8) | (pd.to_numeric(m["n_variants"], errors="coerce").fillna(0) >= 3))
    return int(severe.sum())
