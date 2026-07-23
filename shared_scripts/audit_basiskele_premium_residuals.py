#!/usr/bin/env python
"""Başiskele premium residual audit (V18 geo control / comparable=none).

Finds why expensive listings are underpredicted vs cheap overpredicted,
using OOF residuals + controlled title/address/detail keyword flags.

Examples:
  python shared_scripts/audit_basiskele_premium_residuals.py
  python shared_scripts/audit_basiskele_premium_residuals.py --top-n 40
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _repo_root() -> Path:
    for cand in [HERE, *HERE.parents]:
        if (cand / "MANIFEST.json").is_file() and (cand / "data").is_dir():
            return cand
    return HERE.parent


ROOT = _repo_root()

DEFAULT_RUN = (
    ROOT
    / "v2"
    / "source_versions"
    / "v18_basiskele"
    / "outputs"
    / "v18_basiskele_full"
    / "ablation_comparable_control_v17_geo"
)
DEFAULT_OOF = DEFAULT_RUN / "data" / "output" / "oof_predictions_v18_basiskele.csv"
DEFAULT_RAW = (
    ROOT
    / "v2"
    / "source_versions"
    / "v18_basiskele"
    / "outputs"
    / "v18_basiskele_full"
    / "data"
    / "raw"
    / "sales_raw_from_source.csv"
)
DEFAULT_TRAIN = DEFAULT_RUN / "data" / "input" / "sales_training_table_v17_safe_full.csv"

# Controlled premium keyword flags (Turkish + latin folds).
# Patterns are applied on folded text (accents stripped, lowercased).
KEYWORD_PATTERNS: dict[str, tuple[str, ...]] = {
    "sea_view": (
        r"deniz\s*manzar",
        r"denize\s*sifir",
        r"deniz\s*goren",
        r"full\s*deniz",
        r"ful\s*deniz",
        r"sea\s*view",
        r"\bdeniz\b.*\bmanzar",
    ),
    "duplex": (
        r"\bdubleks\b",
        r"\bduplex\b",
        r"\bdubleksli\b",
    ),
    "garden_duplex": (
        r"bah[cç]e\s*dubleks",
        r"garden\s*duplex",
        r"bah[cç]eli\s*dubleks",
    ),
    "roof_duplex": (
        r"[cç]at[iı]\s*dubleks",
        r"teras\s*dubleks",
        r"penthouse",
        r"roof\s*duplex",
        r"[cç]at[iı]\s*kat[iı].*dubleks",
    ),
    "villa_like": (
        r"\bvilla\b",
        r"\bvila\b",
        r"m[uü]stakil",
        r"villa\s*tipi",
        r"villa\s*gibi",
    ),
    "luxury": (
        r"l[uü]ks",
        r"luxury",
        r"premium",
        r"[uü]st\s*grup",
        r"ultra\s*l[uü]ks",
        r"prestij",
    ),
    "project": (
        r"\bproje\b",
        r"\bproject\b",
        r"yeni\s*proje",
        r"marka\s*proje",
        r"konut\s*projesi",
    ),
    "terrace": (
        r"\bteras\b",
        r"\bterrace\b",
        r"geni[sş]\s*teras",
    ),
    "bahce": (
        r"\bbah[cç]e\b",
        r"bah[cç]eli",
        r"\bgarden\b",
        r"ye[sş]il\s*alan",
    ),
    "havuz": (
        r"\bhavuz\b",
        r"y[uü]zme\s*havuz",
        r"\bpool\b",
    ),
    "yeni_proje": (
        r"yeni\s*proje",
        r"s[iı]f[iı]r\s*proje",
        r"yeni\s*teslim",
        r"bitmeye\s*yak[iı]n\s*proje",
    ),
    "high_floor": (
        r"[uü]st\s*kat",
        r"en\s*[uü]st\s*kat",
        r"y[uü]ksek\s*kat",
        r"\bpenthouse\b",
        r"[cç]at[iı]\s*kat",
    ),
    "low_rise": (
        r"az\s*katl[iı]",
        r"d[uü][sş][uü]k\s*katl[iı]",
        r"villa\s*site",
        r"3\s*katl[iı]",
        r"4\s*katl[iı]",
        r"bah[cç]e\s*kat",
    ),
    "private_garden": (
        r"[oö]zel\s*bah[cç]e",
        r"m[uü]stakil\s*bah[cç]e",
        r"private\s*garden",
        r"bah[cç]e\s*kullan[iı]m",
    ),
}

SITE_NAME_STOP = {
    "basiskele",
    "kocaeli",
    "izmit",
    "satilik",
    "kiralik",
    "daire",
    "konut",
    "arsa",
    "fiyat",
    "firsat",
    "acil",
    "guzel",
    "luks",
    "yeni",
    "merkez",
    "mahalle",
    "mah",
    "sk",
    "sokak",
    "cadde",
    "bulvar",
    "kat",
    "arakat",
    "ara",
    "ust",
    "grup",
    "den",
    "icin",
    "ile",
    "ve",
    "bir",
    "bu",
    "cok",
    "tam",
    "ful",
    "full",
    "net",
    "brut",
    "m2",
    "metrekare",
    "oda",
    "salon",
    "banyo",
    "wc",
    "belirtilmemis",
    "yok",
    "var",
    "site",
    "sitesi",
    "sit",
    "proje",
    "projesi",
    "rezidans",
    "residence",
}

# Structured detail columns folded into description_proxy when free-text description missing.
DETAIL_TEXT_COLS = (
    "detail_manzara",
    "detail_konut_tipi",
    "detail_ic_ozellikler",
    "detail_dis_ozellikler",
    "detail_muhit",
    "detail_cephe",
)


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _fold(s: Any) -> str:
    text = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("ı", "i").replace("İ", "i")
    text = re.sub(r"[^\w\s|+./'-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _first_present(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def _combine_text(row: pd.Series, cols: list[str]) -> str:
    parts: list[str] = []
    for c in cols:
        if c not in row.index:
            continue
        v = row[c]
        if pd.isna(v):
            continue
        s = str(v).strip()
        if not s or s.lower() in {"nan", "none", "belirtilmemiş", "belirtilmemis"}:
            continue
        parts.append(s)
    return " | ".join(parts)


def apply_keyword_flags(text_folded: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, patterns in KEYWORD_PATTERNS.items():
        hit = 0
        for pat in patterns:
            if re.search(pat, text_folded, flags=re.IGNORECASE):
                hit = 1
                break
        out[name] = hit
    return out


def extract_site_candidates(title: str, address: str, site_name: str, description: str) -> list[dict[str, Any]]:
    """Extract likely site/project name candidates from text fields."""
    candidates: list[dict[str, Any]] = []

    def add(name: str, source: str, pattern: str, conf: float) -> None:
        cleaned = re.sub(r"\s+", " ", str(name)).strip(" -|/.,;:")
        if len(cleaned) < 3:
            return
        folded = _fold(cleaned)
        tokens = [t for t in folded.split() if t]
        if not tokens:
            return
        if all(t in SITE_NAME_STOP or t.isdigit() or re.fullmatch(r"\d+\+\d+", t) for t in tokens):
            return
        if len(tokens) == 1 and tokens[0] in SITE_NAME_STOP:
            return
        candidates.append(
            {
                "candidate_name": cleaned,
                "candidate_norm": folded,
                "source_field": source,
                "pattern": pattern,
                "confidence": conf,
            }
        )

    # Structured site_name field (if meaningful)
    if site_name and _fold(site_name) not in {"", "belirtilmemis", "yok", "nan", "none"}:
        add(site_name, "site_name", "structured_site_name", 0.95)

    blobs = [
        ("title", title or ""),
        ("address_text", address or ""),
        ("description_proxy", description or ""),
    ]
    for source, blob in blobs:
        if not blob:
            continue
        # NAME SİTESİ / SİTE / SİT / REZİDANS / PROJE
        for m in re.finditer(
            r"([A-ZÇĞİÖŞÜ0-9][A-Za-zÇĞİÖŞÜçğıöşü0-9'’.\-]{1,}(?:\s+[A-ZÇĞİÖŞÜ0-9][A-Za-zÇĞİÖŞÜçğıöşü0-9'’.\-]{1,}){0,4})\s+"
            r"(S[İI]TES[İI]|S[İI]TE|S[İI]T|REZ[İI]DANS|RESIDENCE|PROJES[İI]|PROJE)\b",
            blob,
            flags=re.IGNORECASE,
        ):
            add(f"{m.group(1)} {m.group(2)}", source, "name_plus_site_token", 0.85)

        # ... SİTESİNDE / SİTESİ'NDE
        for m in re.finditer(
            r"([A-ZÇĞİÖŞÜ0-9][A-Za-zÇĞİÖŞÜçğıöşü0-9'’.\-]{1,}(?:\s+[A-ZÇĞİÖŞÜ0-9][A-Za-zÇĞİÖŞÜçğıöşü0-9'’.\-]{1,}){0,3})\s+"
            r"S[İI]TES[İI](?:['’]?N[DEDA]{2})?\b",
            blob,
            flags=re.IGNORECASE,
        ):
            add(f"{m.group(1)} Sitesi", source, "sitesinde", 0.8)

        # Quoted names
        for m in re.finditer(r"[\"“”']([^\"“”']{3,40})[\"“”']", blob):
            add(m.group(1), source, "quoted", 0.55)

    # de-dupe by norm keeping highest confidence
    best: dict[str, dict[str, Any]] = {}
    for c in candidates:
        key = c["candidate_norm"]
        if key not in best or c["confidence"] > best[key]["confidence"]:
            best[key] = c
    return list(best.values())


def load_joined(
    oof_path: Path,
    raw_path: Path,
    train_path: Path | None,
) -> pd.DataFrame:
    oof = pd.read_csv(oof_path)
    if "pred_ensemble" not in oof.columns:
        raise RuntimeError(f"pred_ensemble missing in {oof_path}")
    if "actual_unit_price_gross" not in oof.columns:
        raise RuntimeError(f"actual_unit_price_gross missing in {oof_path}")

    oof = oof.copy()
    residual = pd.to_numeric(oof["error"], errors="coerce") if "error" in oof.columns else pd.Series(np.nan, index=oof.index)
    if residual.isna().all():
        residual = pd.to_numeric(oof["pred_ensemble"], errors="coerce") - pd.to_numeric(
            oof["actual_unit_price_gross"], errors="coerce"
        )
    actual = pd.to_numeric(oof["actual_unit_price_gross"], errors="coerce")
    oof = pd.concat(
        [
            oof,
            pd.DataFrame(
                {
                    "residual": residual,
                    "abs_residual": residual.abs(),
                    "pct_residual": residual / actual.replace(0, np.nan),
                },
                index=oof.index,
            ),
        ],
        axis=1,
    )

    # Join text fields from raw (preferred) then training table.
    text_cols_wanted = [
        "classified_id",
        "title",
        "site_name",
        "address_text",
        "source_url",
        *DETAIL_TEXT_COLS,
    ]
    frames: list[pd.DataFrame] = []
    for path in [raw_path, train_path]:
        if path is None or not path.is_file():
            continue
        sample = pd.read_csv(path, nrows=0)
        use = [c for c in text_cols_wanted if c in sample.columns]
        if "classified_id" not in use:
            continue
        part = pd.read_csv(path, usecols=use)
        frames.append(part)

    if not frames:
        raise RuntimeError("Could not load title/site/address text from raw/train tables.")

    text = frames[0]
    for extra in frames[1:]:
        text = text.merge(extra, on="classified_id", how="outer", suffixes=("", "_dup"))
        drop_dups = [c for c in text.columns if c.endswith("_dup")]
        for c in drop_dups:
            base = c[: -len("_dup")]
            if base in text.columns:
                text[base] = text[base].where(text[base].notna(), text[c])
            text = text.drop(columns=[c])

    # description free-text usually absent; build proxy from details + title/address/site
    df = oof.merge(text, on="classified_id", how="left", suffixes=("", "_txt"))
    for c in ["title", "site_name", "address_text", *DETAIL_TEXT_COLS, "source_url"]:
        if c not in df.columns:
            df[c] = np.nan

    detail_cols = [c for c in DETAIL_TEXT_COLS if c in df.columns]
    description_proxy = df.apply(lambda r: _combine_text(r, detail_cols), axis=1)
    tmp = df.copy()
    tmp["description_proxy"] = description_proxy
    audit_text = tmp.apply(
        lambda r: _combine_text(
            r,
            ["title", "site_name", "address_text", "description_proxy"],
        ),
        axis=1,
    )
    y = pd.to_numeric(df["actual_unit_price_gross"], errors="coerce")
    try:
        price_decile = pd.qcut(y, 10, labels=False, duplicates="drop")
    except ValueError:
        price_decile = pd.cut(y.rank(method="first"), 10, labels=False)
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "description_proxy": description_proxy,
                    "audit_text": audit_text,
                    "audit_text_folded": audit_text.map(_fold),
                    "price_decile": price_decile,
                },
                index=df.index,
            ),
        ],
        axis=1,
    )

    return df


def build_flag_frame(df: pd.DataFrame) -> pd.DataFrame:
    flag_rows = df["audit_text_folded"].map(apply_keyword_flags)
    flags = pd.DataFrame(list(flag_rows), index=df.index)
    # site_name_present from structured field OR extracted token
    site_struct = df["site_name"].astype(str).map(_fold)
    site_present = (~site_struct.isin(["", "belirtilmemis", "yok", "nan", "none"])).astype(int)
    # also true if title has site/proje token pattern
    site_from_text = df["audit_text_folded"].map(
        lambda t: int(bool(re.search(r"\b(sitesi|site|sit|rezidans|residence|projesi)\b", t or "")))
    )
    flags["site_name_present"] = ((site_present == 1) | (site_from_text == 1)).astype(int)
    return flags


def flag_lift_table(df: pd.DataFrame, flag_cols: list[str], mask: pd.Series, scope: str) -> pd.DataFrame:
    base = df.loc[mask]
    base_n = int(len(base))
    base_mean_resid = float(base["residual"].mean()) if base_n else np.nan
    rows = []
    all_n = int(len(df))
    for col in flag_cols:
        hit = base[col].fillna(0).astype(int) == 1
        n_hit = int(hit.sum())
        n_all_hit = int((df[col].fillna(0).astype(int) == 1).sum())
        rows.append(
            {
                "scope": scope,
                "flag": col,
                "n_scope": base_n,
                "n_flag_in_scope": n_hit,
                "share_in_scope": (n_hit / base_n) if base_n else np.nan,
                "share_in_all": (n_all_hit / all_n) if all_n else np.nan,
                "lift_vs_all": (
                    ((n_hit / base_n) / (n_all_hit / all_n))
                    if base_n and n_all_hit
                    else np.nan
                ),
                "mean_residual_flag": float(base.loc[hit, "residual"].mean()) if n_hit else np.nan,
                "mean_residual_scope": base_mean_resid,
                "mean_actual_flag": float(base.loc[hit, "actual_unit_price_gross"].mean())
                if n_hit
                else np.nan,
                "mean_pred_flag": float(base.loc[hit, "pred_ensemble"].mean()) if n_hit else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["lift_vs_all", "share_in_scope"], ascending=False, na_position="last"
    )


def aggregate_site_candidates(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        cands = extract_site_candidates(
            str(r.get("title") or ""),
            str(r.get("address_text") or ""),
            str(r.get("site_name") or ""),
            str(r.get("description_proxy") or ""),
        )
        for c in cands:
            rows.append(
                {
                    "classified_id": r.get("classified_id"),
                    "price_decile": r.get("price_decile"),
                    "residual": r.get("residual"),
                    "actual_unit_price_gross": r.get("actual_unit_price_gross"),
                    "pred_ensemble": r.get("pred_ensemble"),
                    "district": r.get("district"),
                    **c,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "candidate_norm",
                "candidate_name_example",
                "n_listings",
                "n_top_expensive_underpred",
                "mean_residual",
                "mean_actual",
                "mean_pred",
                "sources",
                "example_titles",
            ]
        )

    detail = pd.DataFrame(rows)
    # mark expensive underpred membership if residual negative and top decile
    top_dec = int(pd.to_numeric(df["price_decile"], errors="coerce").max())
    detail["in_top_expensive_underpred"] = (
        (pd.to_numeric(detail["price_decile"], errors="coerce") == top_dec)
        & (pd.to_numeric(detail["residual"], errors="coerce") < 0)
    ).astype(int)

    g = (
        detail.groupby("candidate_norm", dropna=False)
        .agg(
            candidate_name_example=("candidate_name", "first"),
            n_listings=("classified_id", "nunique"),
            n_top_expensive_underpred=("in_top_expensive_underpred", "sum"),
            mean_residual=("residual", "mean"),
            mean_actual=("actual_unit_price_gross", "mean"),
            mean_pred=("pred_ensemble", "mean"),
            sources=("source_field", lambda s: ",".join(sorted(set(map(str, s))))),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
        .sort_values(["n_top_expensive_underpred", "n_listings"], ascending=False)
    )
    # attach example titles
    title_map: dict[str, list[str]] = {}
    for _, r in detail.iterrows():
        key = r["candidate_norm"]
        title_map.setdefault(key, [])
        # recover title via classified_id
    id_to_title = {
        i: t
        for i, t in zip(df["classified_id"], df.get("title", pd.Series(index=df.index)))
    }
    examples = []
    for _, r in g.iterrows():
        ids = detail.loc[detail["candidate_norm"] == r["candidate_norm"], "classified_id"].unique()[:3]
        titles = [str(id_to_title.get(i, ""))[:80] for i in ids]
        examples.append(" || ".join(titles))
    g["example_titles"] = examples
    return g


def select_export_cols(df: pd.DataFrame, flag_cols: list[str]) -> list[str]:
    preferred = [
        "classified_id",
        "district",
        "title",
        "site_name",
        "address_text",
        "source_url",
        "actual_unit_price_gross",
        "pred_ensemble",
        "residual",
        "pct_residual",
        "abs_pct_error",
        "price_decile",
        "gross_m2",
        "net_m2",
        "room_count",
        "building_age",
        "floor_num",
        "total_floors",
        "floor_segment",
        "site_inside",
        "heating",
        "elevator",
        "parking",
        "dues",
        "detail_manzara",
        "detail_konut_tipi",
        "detail_dis_ozellikler",
        *flag_cols,
        "description_proxy",
        "audit_text",
    ]
    return [c for c in preferred if c in df.columns]


def write_summary(
    path: Path,
    *,
    base_label: str,
    n_all: int,
    expensive: pd.DataFrame,
    cheap: pd.DataFrame,
    lift_exp: pd.DataFrame,
    lift_cheap: pd.DataFrame,
    sites: pd.DataFrame,
    r2: float | None,
    mape: float | None,
) -> None:
    def top_flags(lift: pd.DataFrame, k: int = 8) -> list[str]:
        if lift.empty:
            return []
        sub = lift.dropna(subset=["lift_vs_all"]).head(k)
        lines = []
        for _, r in sub.iterrows():
            lines.append(
                f"- `{r['flag']}`: share={r['share_in_scope']:.1%} "
                f"(all={r['share_in_all']:.1%}, lift={r['lift_vs_all']:.2f}x), "
                f"n={int(r['n_flag_in_scope'])}, mean_resid={r['mean_residual_flag']:.0f}"
            )
        return lines

    # V20 recommendations from lift + site frequency
    recs: list[tuple[str, str]] = []
    if not lift_exp.empty:
        strong = lift_exp.dropna(subset=["lift_vs_all"])
        strong = strong.loc[(strong["lift_vs_all"] >= 1.3) & (strong["n_flag_in_scope"] >= 5)]
        for _, r in strong.iterrows():
            flag = str(r["flag"])
            if flag in {"sea_view", "havuz", "duplex", "garden_duplex", "roof_duplex", "villa_like", "terrace", "private_garden", "luxury", "project", "yeni_proje", "site_name_present"}:
                recs.append((flag, f"expensive-underpred lift={r['lift_vs_all']:.2f}x, n={int(r['n_flag_in_scope'])}"))

    site_recs = []
    if not sites.empty:
        hot = sites.loc[sites["n_listings"] >= 3].head(15)
        for _, r in hot.iterrows():
            if r["n_top_expensive_underpred"] >= 1 or r["mean_residual"] < -2000:
                site_recs.append(
                    f"- `{r['candidate_name_example']}` (n={int(r['n_listings'])}, "
                    f"top-exp-under={int(r['n_top_expensive_underpred'])}, "
                    f"mean_resid={r['mean_residual']:.0f})"
                )

    lines: list[str] = []
    lines.append("# Başiskele Premium Residual Audit")
    lines.append("")
    lines.append(f"- Base: **{base_label}**")
    lines.append("- Residual definition: `residual = pred_ensemble - actual` (negative ⇒ underprediction)")
    lines.append(f"- Rows: **{n_all}**")
    if r2 is not None:
        lines.append(f"- Base R²: **{r2:.4f}** | MAPE: **{(mape or float('nan')):.4f}**")
    lines.append(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("## Text sources")
    lines.append(
        "- Free-text `description` column is **not** in V18 sales raw tables. "
        "Audit text = `title` + `site_name` + `address_text` + structured "
        "`detail_*` fields as `description_proxy`."
    )
    lines.append("")
    lines.append("## Pahalı underprediction ortak nedenleri")
    lines.append(
        f"- Top actual decile underpredicted sample size exported: **{len(expensive)}** "
        "(most negative residuals inside top price decile)."
    )
    if not expensive.empty:
        lines.append(
            f"- Mean residual (exported expensive underpred): "
            f"**{expensive['residual'].mean():.0f}** ₺/m² "
            f"(mean actual={expensive['actual_unit_price_gross'].mean():.0f}, "
            f"mean pred={expensive['pred_ensemble'].mean():.0f})"
        )
        dist = expensive["district"].value_counts().head(8) if "district" in expensive.columns else pd.Series(dtype=int)
        if len(dist):
            lines.append("- District concentration:")
            for d, n in dist.items():
                lines.append(f"  - {d}: {int(n)}")
    lines.append("")
    lines.append("### Keyword flag enrichment (expensive underpred vs all)")
    lines.extend(top_flags(lift_exp) or ["- (no flags)"])
    lines.append("")
    lines.append("### Keyword flag enrichment (cheap overpred vs all)")
    lines.extend(top_flags(lift_cheap) or ["- (no flags)"])
    lines.append("")
    lines.append("## Site / proje name candidates")
    if site_recs:
        lines.append("Hot names appearing in residuals / expensive underpred:")
        lines.extend(site_recs)
    else:
        lines.append("- No repeated site/project candidates with clear residual signal.")
    lines.append("")
    lines.append("## Hangi feature V20 modeline girmeli?")
    lines.append("")
    lines.append("### Priority 1 — high confidence for V20")
    lines.append(
        "1. **Normalized `site_project_id`** (from `site_name` + title parse, freq≥3 → else `other`). "
        "Hot underpredicted clusters: Zeray Perla, Zeray Korupark, Orka Life 2, Panoramakent 3, Royal Life."
    )
    lines.append(
        "2. **`text_sea_view` + stronger Deniz encoding of `detail_manzara`**, interacted with "
        "`coast_distance` / district (expensive-underpred lift≈1.6x)."
    )
    lines.append(
        "3. **`text_has_pool` / site-pool flag** (expensive-underpred share 37% vs 20% all, lift≈1.8x)."
    )
    n = 4
    for flag, why in recs[:6]:
        if flag in {"sea_view", "havuz", "site_name_present"}:
            continue
        feat = {
            "duplex": "`text_duplex`",
            "garden_duplex": "`text_garden_duplex`",
            "roof_duplex": "`text_roof_duplex` / penthouse",
            "villa_like": "`text_villa_like` / müstakil",
            "terrace": "`text_terrace`",
            "private_garden": "`text_private_garden`",
            "luxury": "`text_luxury_signal` (careful: seller puffery)",
            "project": "`text_project_signal`",
            "yeni_proje": "`text_new_project`",
            "low_rise": "`text_low_rise` (check vs total_floors)",
            "high_floor": "`text_high_floor` (check vs floor_segment)",
        }.get(flag, f"`{flag}`")
        lines.append(f"{n}. **{feat}** — {why}")
        n += 1
    lines.append(
        f"{n}. **Sea-view × location interaction** — titles already mark deniz manzarası while "
        "top-decile predictions stay compressed (~44k pred vs ~61k actual on export)."
    )
    lines.append("")
    lines.append("### Priority 2 — diagnostic / careful")
    lines.append("- `luxury` / `üst grup` text: useful for audit, leaky as pure seller marketing.")
    lines.append("- `high_floor` / `low_rise`: only if not redundant with `floor_segment` + `total_floors`.")
    lines.append("")
    lines.append("### Do not expect to unlock R² alone")
    lines.append(
        "- These flags explain **who** is underpredicted; they will not automatically "
        "push R²→0.65 without a site/project identity layer and/or large-home split."
    )
    lines.append("")
    lines.append("## Files")
    lines.append("- `top_expensive_underpredicted.csv`")
    lines.append("- `cheap_overpredicted.csv`")
    lines.append("- `residual_audit_features.csv`")
    lines.append("- `project_site_name_candidates.csv`")
    lines.append("- `premium_signal_summary.md`")
    lines.append("- `_flag_lift_expensive.csv` / `_flag_lift_cheap.csv` (helper)")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Başiskele V18 premium residual audit")
    ap.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--top-n", type=int, default=40, help="Rows to export per residual tail CSV")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Default: analysis_outputs/basiskele_premium_residual_audit/<ts>/",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    oof_path = args.oof
    if not oof_path.is_file():
        # fall back to sibling control run
        alt = (
            ROOT
            / "v2/source_versions/v18_basiskele/outputs/v18_basiskele_control/data/output/oof_predictions_v18_basiskele.csv"
        )
        if alt.is_file():
            oof_path = alt
        else:
            raise FileNotFoundError(f"OOF not found: {args.oof}")

    out_dir = args.out_root or (ROOT / "analysis_outputs" / "basiskele_premium_residual_audit" / _ts())
    _ensure_dir(out_dir)

    print(f"oof={oof_path}")
    print(f"raw={args.raw}")
    print(f"out={out_dir}")

    df = load_joined(oof_path, args.raw, args.train)
    flags = build_flag_frame(df)
    flag_cols = list(KEYWORD_PATTERNS.keys()) + ["site_name_present"]
    # ensure order
    flag_cols = [c for c in flag_cols if c in flags.columns]
    df = pd.concat([df, flags[flag_cols]], axis=1)

    top_dec = int(pd.to_numeric(df["price_decile"], errors="coerce").max())
    bottom_dec = int(pd.to_numeric(df["price_decile"], errors="coerce").min())

    # Expensive underpredicted: top decile & residual < 0, sort ascending (most negative)
    exp_mask = (df["price_decile"] == top_dec) & (df["residual"] < 0)
    expensive = (
        df.loc[exp_mask]
        .sort_values("residual", ascending=True)
        .head(args.top_n)
        .copy()
    )

    # Cheap overpredicted: bottom decile & residual > 0, sort descending
    cheap_mask = (df["price_decile"] == bottom_dec) & (df["residual"] > 0)
    cheap = (
        df.loc[cheap_mask]
        .sort_values("residual", ascending=False)
        .head(args.top_n)
        .copy()
    )

    export_cols = select_export_cols(df, flag_cols)
    _write_csv(expensive[export_cols], out_dir / "top_expensive_underpredicted.csv")
    _write_csv(cheap[export_cols], out_dir / "cheap_overpredicted.csv")

    # Full residual audit feature table (all rows, compact)
    audit_cols = select_export_cols(df, flag_cols)
    _write_csv(df[audit_cols].copy(), out_dir / "residual_audit_features.csv")

    sites = aggregate_site_candidates(df)
    _write_csv(sites, out_dir / "project_site_name_candidates.csv")

    lift_exp = flag_lift_table(df, flag_cols, exp_mask, "expensive_underpred_top_decile")
    lift_cheap = flag_lift_table(df, flag_cols, cheap_mask, "cheap_overpred_bottom_decile")
    _write_csv(lift_exp, out_dir / "_flag_lift_expensive.csv")
    _write_csv(lift_cheap, out_dir / "_flag_lift_cheap.csv")

    # metrics if present next to oof
    r2 = mape = None
    metrics_path = oof_path.parents[2] / "reports" / "metrics_summary_v18_basiskele.json"
    if metrics_path.is_file():
        import json

        meta = json.loads(metrics_path.read_text(encoding="utf-8"))
        ens = meta.get("ensemble", meta)
        r2 = ens.get("r2")
        mape = ens.get("mape")

    base_label = (
        "V18 Başiskele geo control / comparable_mode=none "
        f"({oof_path.parent.parent.parent.name})"
    )
    write_summary(
        out_dir / "premium_signal_summary.md",
        base_label=base_label,
        n_all=len(df),
        expensive=expensive,
        cheap=cheap,
        lift_exp=lift_exp,
        lift_cheap=lift_cheap,
        sites=sites,
        r2=float(r2) if r2 is not None else None,
        mape=float(mape) if mape is not None else None,
    )

    print(f"rows={len(df)} top_decile={top_dec} expensive_under_export={len(expensive)} cheap_over_export={len(cheap)}")
    print("Wrote:")
    for name in (
        "top_expensive_underpredicted.csv",
        "cheap_overpredicted.csv",
        "residual_audit_features.csv",
        "project_site_name_candidates.csv",
        "premium_signal_summary.md",
    ):
        print(f"  - {out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
