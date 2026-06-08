"""
build_nascdi_final.py
=====================
Production-grade News-Based Apple Supply Chain Disruption Index (NASCDI)
builder for the NASCDI-Apple-Price-Transmission research project.

Journal-ready features
----------------------
* Robust column inference across GDELT, ProQuest, Factiva, manual exports
* Four-stage diagnostic output — pinpoints exactly where a pipeline empties
* Full commodity-context gate with override flag
* Dual normalization options (z-score → [mean=100, sd=10] or min-max)
* Per-article component scores saved for audit / replication
* Optional date-range restriction (--start / --end)
* Sensitivity mode: runs both lexicons and reports correlation between them

Usage
-----
python build_nascdi_final.py \
    --news_dir  data/news/raw \
    --lexicon   config/lexicon.yaml \
    --out_dir   data/nascdi \
    [--start 2010-01-01] [--end 2025-12-31] \
    [--no_require_commodity_context] \
    [--sensitivity_lexicon config/lexicon_sensitivity.yaml] \
    [--normalize z_to_100_10 | minmax_0_100]
"""

from __future__ import annotations

import os
import re
import sys
import glob
import yaml
import json
import hashlib
import argparse
import warnings
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
except AttributeError:
    pass

# ─────────────────────────────────────────────────────────────
# 1. TEXT UTILITIES
# ─────────────────────────────────────────────────────────────

def robust_parse_date(s: pd.Series) -> pd.Series:
    """Parse mixed date formats (ISO, DD/MM/YYYY, MM/DD/YYYY, etc.)."""
    d1 = pd.to_datetime(s, errors="coerce", dayfirst=False, utc=False)
    mask = d1.isna()
    if mask.any():
        d2 = pd.to_datetime(s[mask], errors="coerce", dayfirst=True, utc=False)
        d1.loc[mask] = d2
    # Try GDELT compact format: 20140907000000
    mask2 = d1.isna()
    if mask2.any():
        d3 = pd.to_datetime(s[mask2], errors="coerce", format="%Y%m%d%H%M%S", utc=False)
        d1.loc[mask2] = d3
    mask3 = d1.isna()
    if mask3.any():
        d4 = pd.to_datetime(s[mask3], errors="coerce", format="%Y%m%d", utc=False)
        d1.loc[mask3] = d4
    if hasattr(d1, "dt"):
        try:
            d1 = d1.dt.tz_localize(None)
        except Exception:
            d1 = d1.dt.tz_convert(None)
    return d1


def clean_text(text) -> str:
    """Normalise article text for lexicon matching."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text).lower()
    text = text.replace("nh 44", "nh-44").replace("nh44", "nh-44")
    text = text.replace("nh 1a", "nh-44").replace("nh-1a", "nh-44")  # old designation
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────
# 2. LEXICON
# ─────────────────────────────────────────────────────────────

def load_lexicon(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        lex = yaml.safe_load(f)
    for k in ("disruption_terms", "mitigation_terms", "commodity_terms"):
        if k not in lex or lex[k] is None:
            lex[k] = {}
    return lex


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_row_count(path: str) -> int:
    try:
        return max(sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore")) - 1, 0)
    except OSError:
        return 0


def source_file_manifest(news_dir: str) -> List[Dict[str, object]]:
    files = sorted(glob.glob(os.path.join(news_dir, "*.csv")))
    manifest = []
    for fp in files:
        manifest.append({
            "file": fp,
            "bytes": os.path.getsize(fp),
            "rows": csv_row_count(fp),
            "sha256": file_sha256(fp),
        })
    return manifest


def count_term_occurrences(text: str, term: str) -> int:
    """Count whole-phrase occurrences with pseudo-word-boundary matching."""
    term = term.strip().lower()
    if not term:
        return 0
    pattern = r"(?<![a-z0-9\-])" + re.escape(term) + r"(?![a-z0-9\-])"
    return len(re.findall(pattern, text))


def score_article_components(
    text: str, lex: Dict[str, Dict[str, float]]
) -> Tuple[float, float, float]:
    """Return (disruption_score, mitigation_score, commodity_score)."""
    d = sum(
        count_term_occurrences(text, t) * float(w)
        for t, w in lex["disruption_terms"].items()
    )
    m = sum(
        count_term_occurrences(text, t) * float(w)
        for t, w in lex["mitigation_terms"].items()
    )
    c = sum(
        count_term_occurrences(text, t) * float(w)
        for t, w in lex["commodity_terms"].items()
    )
    return d, m, c


# ─────────────────────────────────────────────────────────────
# 3. DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────

# Accepted column-name aliases (case-insensitive)
_ALIASES: Dict[str, List[str]] = {
    "date":   ["date", "published_date", "publish_date", "publishdate",
               "time", "datetime", "seendate", "dateadded", "pub_date",
               "article_date", "news_date"],
    "title":  ["title", "headline", "news_title", "subject", "head"],
    "text":   ["text", "content", "body", "article", "article_text",
               "news_text", "excerpt", "summary", "abstract", "description"],
    "source": ["source", "newspaper", "site", "outlet", "publication",
               "sourcecountry", "sourcecommonname", "domain"],
    "url":    ["url", "link", "article_url", "documentidentifier",
               "sourceurl", "articleurl"],
}


def _pick(cols_lower: List[str], cands: List[str]) -> Optional[int]:
    for c in cands:
        if c in cols_lower:
            return cols_lower.index(c)
    return None


def infer_columns(df: pd.DataFrame, filepath: str = "") -> pd.DataFrame:
    """
    Map arbitrary input column names to a standard schema.
    Raises ValueError with a helpful message if required columns are missing.
    """
    cols_lower = [c.lower().strip() for c in df.columns]

    def pick(key: str) -> Optional[str]:
        idx = _pick(cols_lower, _ALIASES[key])
        return df.columns[idx] if idx is not None else None

    date_col   = pick("date")
    title_col  = pick("title")
    text_col   = pick("text")
    source_col = pick("source")
    url_col    = pick("url")

    if date_col is None:
        raise ValueError(
            f"No date column found in {filepath}.\n"
            f"  Columns present: {df.columns.tolist()}\n"
            f"  Expected one of: {_ALIASES['date']}"
        )
    if text_col is None and title_col is None:
        raise ValueError(
            f"No text/title column found in {filepath}.\n"
            f"  Columns present: {df.columns.tolist()}\n"
            f"  Expected one of: {_ALIASES['text']} or {_ALIASES['title']}"
        )

    out = pd.DataFrame()
    out["date_raw"] = df[date_col].astype(str)
    out["date"]     = robust_parse_date(out["date_raw"])
    out["title"]    = df[title_col].fillna("")  if title_col  else ""
    out["text"]     = df[text_col].fillna("")   if text_col   else ""
    out["source"]   = df[source_col].fillna("") if source_col else ""
    out["url"]      = df[url_col].fillna("")    if url_col    else ""
    for audit_col in (
        "query_block",
        "query",
        "window_start",
        "window_end",
        "source_country",
        "language",
    ):
        out[audit_col] = df[audit_col].fillna("") if audit_col in df.columns else ""
    return out


def dedupe_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate articles (URL-first; fallback date+title)."""
    df = df.copy()
    df["title_clean"] = df["title"].fillna("").astype(str).map(clean_text)
    df["text_clean"]  = df["text"].fillna("").astype(str).map(clean_text)
    df["url_norm"] = df["url"].fillna("").astype(str).str.strip()
    with_url = df["url_norm"].str.len() > 5

    if with_url.any():
        aggregations = {
            col: "first"
            for col in df.columns
            if col not in {"url_norm", "query_block"}
        }
        if "query_block" in df.columns:
            aggregations["query_block"] = lambda s: "|".join(
                sorted(set(v for v in s.dropna().astype(str) if v))
            )
        by_url = (
            df[with_url]
            .sort_values("date")
            .groupby("url_norm", as_index=False)
            .agg(aggregations)
        )
        no_url = df[~with_url].copy()
        no_url["_key"] = no_url["date"].astype(str) + "||" + no_url["title_clean"].str[:120]
        no_url = no_url.sort_values("date").drop_duplicates(subset=["_key"], keep="first")
        df = pd.concat([by_url, no_url], ignore_index=True, sort=False)
    else:
        df["_key"] = df["date"].astype(str) + "||" + df["title_clean"].str[:120]
        df = df.sort_values("date").drop_duplicates(subset=["_key"], keep="first")
    return df.sort_values("date").reset_index(drop=True)


def load_news_corpus(
    news_dir: str, verbose: bool = True
) -> pd.DataFrame:
    """
    Load all CSVs from news_dir into a single deduplicated article DataFrame.
    Prints diagnostic info at each stage.
    """
    files = sorted(glob.glob(os.path.join(news_dir, "*.csv")))
    if verbose:
        print(f"\n{'='*60}")
        print(f"STAGE 1 — File discovery")
        print(f"  Directory : {news_dir}")
        print(f"  CSV files : {len(files)}")
        for f in files:
            print(f"    {os.path.basename(f)}")

    if not files:
        raise FileNotFoundError(
            f"\n[EMPTY] No CSV files in {news_dir}.\n"
            "  → Place your GDELT/ProQuest/Factiva exports here.\n"
            "  → Each CSV must have columns: date, title/text, [url, source]."
        )

    frames = []
    for fp in files:
        try:
            raw = pd.read_csv(fp, low_memory=False)
        except Exception as e:
            print(f"  [WARN] Cannot read {fp}: {e}")
            continue
        if raw.empty:
            print(f"  [WARN] Empty file: {fp}")
            continue
        try:
            std = infer_columns(raw, fp)
        except ValueError as e:
            print(f"  [WARN] Column inference failed for {fp}: {e}")
            continue
        std["_file"] = os.path.basename(fp)
        frames.append(std)
        if verbose:
            print(f"  Loaded: {os.path.basename(fp):50s}  rows={len(std)}")

    if not frames:
        raise ValueError(
            "\n[EMPTY] Zero articles loaded. Common causes:\n"
            "  1. CSV files are empty.\n"
            "  2. Column names don't match any expected alias.\n"
            "  3. All date values failed to parse.\n"
            "  Run with --verbose for per-file diagnostics."
        )

    df = pd.concat(frames, ignore_index=True)
    n_before = len(df)

    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.normalize()
    n_after_date = len(df)

    df = dedupe_articles(df)
    n_after_dedup = len(df)

    if verbose:
        print(f"\nSTAGE 2 — Cleaning")
        print(f"  Total rows loaded     : {n_before:>7,}")
        print(f"  After date cleaning   : {n_after_date:>7,}  "
              f"({'OK' if n_after_date > 0 else 'EMPTY — date parsing failed'})")
        print(f"  After deduplication   : {n_after_dedup:>7,}")

    if df.empty:
        raise ValueError(
            "\n[EMPTY] No articles remain after date parsing.\n"
            "  → Check that your date column contains parseable dates.\n"
            "  → Accepted formats: YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY, YYYYMMDDHHMMSS."
        )
    return df


# ─────────────────────────────────────────────────────────────
# 4. SCORING & INDEX CONSTRUCTION
# ─────────────────────────────────────────────────────────────

def score_corpus(
    articles: pd.DataFrame,
    lex: Dict[str, Dict[str, float]],
    min_score_threshold: float = 1.0,
    clip_raw: Optional[float] = 30.0,
    require_commodity_context: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Score articles and build daily NASCDI.

    Returns
    -------
    scored_articles : DataFrame with per-article scores
    daily           : DataFrame indexed by date with NASCDI columns
    """
    df = articles.copy()

    # ── Text assembly ──────────────────────────────────────────
    title = df["title_clean"].fillna("")
    text  = df["text_clean"].fillna("")
    # Avoid double-counting identical title/text (GDELT ArtList case)
    df["full_text"] = np.where(
        title.str.strip() == text.str.strip(),
        title,
        (title + " " + text).str.strip(),
    )

    # ── Component scoring ──────────────────────────────────────
    components = df["full_text"].apply(
        lambda t: score_article_components(t, lex)
    )
    df["disruption_score"] = components.apply(lambda x: x[0])
    df["mitigation_score"] = components.apply(lambda x: x[1])
    df["commodity_score"]  = components.apply(lambda x: x[2])
    df["score_raw"]        = df["disruption_score"] + df["mitigation_score"]
    df["has_commodity_context"] = df["commodity_score"] > 0
    if clip_raw is not None:
        df["score_raw_clipped"] = df["score_raw"].clip(
            lower=-abs(clip_raw), upper=abs(clip_raw)
        )
    else:
        df["score_raw_clipped"] = df["score_raw"]
    df["is_index_article"] = df["score_raw_clipped"].abs() >= min_score_threshold
    df["is_disruption_hit"] = (
        df["score_raw_clipped"] >= min_score_threshold
    ).astype(int)

    if verbose:
        print(f"\nSTAGE 3 — Scoring")
        print(f"  Articles scored                   : {len(df):>7,}")
        non_zero = (df["score_raw"] != 0).sum()
        print(f"  Articles with non-zero score_raw  : {non_zero:>7,}  "
              f"({non_zero/len(df)*100:.1f}%)")
        comm_hit = df["has_commodity_context"].sum()
        print(f"  Articles with commodity context   : {comm_hit:>7,}  "
              f"({comm_hit/len(df)*100:.1f}%)")

    # ── Commodity context filter ───────────────────────────────
    if require_commodity_context:
        df_daily = df[df["has_commodity_context"] & df["is_index_article"]].copy()
        if verbose:
            print(f"  After context and score filters   : {len(df_daily):>7,}")
        if df_daily.empty:
            raise ValueError(
                "\n[EMPTY] No articles passed the context and score filters.\n"
                "  This means no articles contained both a context term and a material score.\n"
                "  Fixes:\n"
                "  A) Add more commodity_terms to your lexicon (e.g., 'kashmir', 'fruit').\n"
                "  B) Rerun with --no_require_commodity_context if your corpus\n"
                "     is already filtered to apple/supply-chain articles.\n"
                "  C) Check your news CSV — is it actually about Kashmir apples?"
            )
    else:
        df_daily = df[df["is_index_article"]].copy()
        if verbose:
            print("  Commodity filter : DISABLED (--no_require_commodity_context)")

    # ── Clipping ───────────────────────────────────────────────
    if clip_raw is not None:
        df_daily["score_raw_clipped"] = df_daily["score_raw"].clip(
            lower=-abs(clip_raw), upper=abs(clip_raw)
        )
    else:
        df_daily["score_raw_clipped"] = df_daily["score_raw"]

    df_daily["is_disruption_hit"] = (
        df_daily["score_raw_clipped"] >= min_score_threshold
    ).astype(int)

    # ── Daily aggregation ──────────────────────────────────────
    daily = (
        df_daily.groupby("date", as_index=True)
        .agg(
            raw_nascdi              = ("score_raw_clipped", "mean"),
            total_score             = ("score_raw_clipped", "sum"),
            news_volume             = ("score_raw_clipped", "count"),
            disruption_article_count= ("is_disruption_hit", "sum"),
            disruption_score_mean   = ("disruption_score",  "mean"),
            mitigation_score_mean   = ("mitigation_score",  "mean"),
        )
        .sort_index()
    )

    if verbose:
        print(f"\nSTAGE 4 — Daily aggregation")
        print(f"  Daily observations     : {len(daily):>7,}")
        if len(daily) > 0:
            print(f"  Date range             : {daily.index.min().date()} — "
                  f"{daily.index.max().date()}")
            print(f"  Mean raw_nascdi        : {daily['raw_nascdi'].mean():.4f}")
            print(f"  Std  raw_nascdi        : {daily['raw_nascdi'].std():.4f}")

    if daily.empty:
        raise ValueError("\n[EMPTY] Daily NASCDI is empty after grouping. "
                         "Check article date range.")
    return df, df_daily, daily


def normalize_index(
    daily: pd.DataFrame,
    method: str = "z_to_100_10",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Optionally restrict date range, fill gaps, then normalise.
    Returns a complete daily series with NASCDI column.
    """
    start = pd.to_datetime(start_date) if start_date else daily.index.min()
    end   = pd.to_datetime(end_date)   if end_date   else daily.index.max()
    full_index = pd.date_range(
        start.normalize(), end.normalize(), freq="D"
    )
    daily = daily.reindex(full_index)
    daily.index.name = "date"
    daily[["raw_nascdi", "total_score"]] = (
        daily[["raw_nascdi", "total_score"]].fillna(0.0)
    )
    int_cols = ["news_volume", "disruption_article_count"]
    daily[int_cols] = daily[int_cols].fillna(0).astype(int)
    daily[["disruption_score_mean", "mitigation_score_mean"]] = (
        daily[["disruption_score_mean", "mitigation_score_mean"]].fillna(0.0)
    )

    if method == "z_to_100_10":
        mu = daily["raw_nascdi"].mean()
        sd = daily["raw_nascdi"].std(ddof=0)
        if sd == 0 or np.isnan(sd):
            daily["NASCDI"] = 100.0
        else:
            daily["NASCDI"] = ((daily["raw_nascdi"] - mu) / sd) * 10.0 + 100.0
    elif method == "minmax_0_100":
        mn, mx = daily["raw_nascdi"].min(), daily["raw_nascdi"].max()
        if mx == mn:
            daily["NASCDI"] = 50.0
        else:
            daily["NASCDI"] = (
                (daily["raw_nascdi"] - mn) / (mx - mn) * 100.0
            )
    else:
        raise ValueError(
            f"Unknown normalize_method '{method}'. "
            "Choose 'z_to_100_10' or 'minmax_0_100'."
        )
    return daily


# ─────────────────────────────────────────────────────────────
# 5. PLOTTING
# ─────────────────────────────────────────────────────────────

def plot_index(
    daily: pd.DataFrame,
    out_path: str,
    title: str = "NASCDI — Daily News-Based Supply Chain Disruption Index",
) -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=(16, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    ax = axes[0]
    ax.plot(daily.index, daily["NASCDI"], color="#1a1a2e", linewidth=0.9,
            alpha=0.9, label="NASCDI (daily)")
    # 90-day rolling mean
    roll = daily["NASCDI"].rolling(90, min_periods=14).mean()
    ax.plot(daily.index, roll, color="#e94560", linewidth=1.8,
            label="90-day moving average")
    ax.axhline(100, color="#6c757d", linestyle="--", linewidth=1.0,
               label="Baseline (mean = 100)")
    ax.fill_between(daily.index, 100, daily["NASCDI"],
                    where=daily["NASCDI"] > 100,
                    alpha=0.12, color="#e94560", label="Above baseline")
    ax.fill_between(daily.index, 100, daily["NASCDI"],
                    where=daily["NASCDI"] < 100,
                    alpha=0.12, color="#0f3460", label="Below baseline")
    ax.set_ylabel("NASCDI", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(
        max(50, daily["NASCDI"].min() - 10),
        min(160, daily["NASCDI"].max() + 10)
    )

    ax2 = axes[1]
    ax2.bar(daily.index, daily["news_volume"],
            color="#457b9d", alpha=0.7, width=1.0, label="Article count")
    ax2.set_ylabel("Articles/day", fontsize=10)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax2.grid(axis="y", alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {out_path}")


def plot_weekly(
    daily: pd.DataFrame,
    out_path: str,
) -> None:
    weekly = daily.resample("W").agg(
        NASCDI      = ("NASCDI",       "mean"),
        news_volume = ("news_volume",  "sum"),
    )
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(weekly.index, weekly["NASCDI"], color="#1a1a2e",
            linewidth=1.4, label="NASCDI (weekly mean)")
    roll = weekly["NASCDI"].rolling(13, min_periods=4).mean()
    ax.plot(weekly.index, roll, color="#e94560", linewidth=2.0,
            label="13-week moving average")
    ax.axhline(100, color="#6c757d", linestyle="--", linewidth=1.0,
               label="Baseline (100)")
    ax.fill_between(weekly.index, 100, weekly["NASCDI"],
                    where=weekly["NASCDI"] > 100,
                    alpha=0.15, color="#e94560")
    ax.fill_between(weekly.index, 100, weekly["NASCDI"],
                    where=weekly["NASCDI"] < 100,
                    alpha=0.15, color="#0f3460")
    ax.set_ylabel("NASCDI", fontsize=11)
    ax.set_title(
        "NASCDI — Weekly Aggregated (News-Based)", fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# 6. SENSITIVITY COMPARISON
# ─────────────────────────────────────────────────────────────

def run_sensitivity_comparison(
    articles: pd.DataFrame,
    lex_main: Dict[str, Dict[str, float]],
    lex_sens: Dict[str, Dict[str, float]],
    min_score: float,
    clip_raw: float,
    require_commodity: bool,
    start_date: Optional[str],
    end_date: Optional[str],
    out_dir: str,
) -> None:
    """Build NASCDI with both lexicons and report correlation."""
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS — Lexicon Robustness")
    print("="*60)

    results = {}
    for label, lex in [("main", lex_main), ("sensitivity", lex_sens)]:
        try:
            _, _, daily = score_corpus(
                articles, lex, min_score, clip_raw,
                require_commodity, verbose=False
            )
            daily = normalize_index(daily, "z_to_100_10", start_date, end_date)
            results[label] = daily["NASCDI"]
            print(f"  [{label:>12}] mean={daily['NASCDI'].mean():.2f}  "
                  f"sd={daily['NASCDI'].std():.2f}  n={len(daily):,}")
        except ValueError as e:
            print(f"  [{label:>12}] FAILED: {e}")
            return

    # Align on common dates
    combined = pd.DataFrame(results).dropna()
    if len(combined) < 10:
        print("  [SKIP] Too few overlapping observations for correlation.")
        return

    corr = combined["main"].corr(combined["sensitivity"])
    print(f"\n  Pearson r (main vs sensitivity) = {corr:.4f}")
    if corr >= 0.90:
        print("  ✓ ROBUST — correlation ≥ 0.90 supports lexicon validity.")
    elif corr >= 0.75:
        print("  ~ MODERATE — consider refining sensitivity lexicon terms.")
    else:
        print("  ✗ LOW — revisit term weights or sensitivity lexicon scope.")

    # Save comparison CSV
    combined.columns = ["NASCDI_main", "NASCDI_sensitivity"]
    combined.to_csv(
        os.path.join(out_dir, "nascdi_lexicon_sensitivity.csv")
    )

    # Plot comparison
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(combined.index, combined["NASCDI_main"],
            color="#1a1a2e", label="Main lexicon", linewidth=1.2)
    ax.plot(combined.index, combined["NASCDI_sensitivity"],
            color="#e94560", linestyle="--",
            label="Sensitivity lexicon", linewidth=1.0, alpha=0.8)
    ax.axhline(100, color="#aaa", linestyle=":", linewidth=1.0)
    ax.set_title(
        f"Lexicon Robustness Check (r = {corr:.3f})", fontsize=12
    )
    ax.set_ylabel("NASCDI")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "nascdi_sensitivity_comparison.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"  Sensitivity plot: {fig_path}")


def save_descriptive_table(daily: pd.DataFrame, weekly: pd.DataFrame, out_dir: str) -> str:
    out_path = os.path.join(out_dir, "table1_nascdi_descriptive.csv")
    s = daily["NASCDI"]
    rows = [
        ("Observations (daily)", len(daily)),
        ("Observations (weekly)", len(weekly)),
        ("Mean", round(float(s.mean()), 6)),
        ("Std Dev", round(float(s.std(ddof=1)), 6)),
        ("Minimum", round(float(s.min()), 6)),
        ("Maximum", round(float(s.max()), 6)),
        ("P25", round(float(s.quantile(0.25)), 6)),
        ("Median (P50)", round(float(s.quantile(0.50)), 6)),
        ("P75", round(float(s.quantile(0.75)), 6)),
        ("Days NASCDI >= 110", int((s >= 110).sum())),
        ("Days NASCDI <= 90", int((s <= 90).sum())),
    ]
    pd.DataFrame(rows, columns=["Statistic", "Value"]).to_csv(out_path, index=False)
    return out_path


def save_reproducibility_metadata(
    args: argparse.Namespace,
    articles: pd.DataFrame,
    scored: pd.DataFrame,
    df_daily: pd.DataFrame,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    outputs: Dict[str, str],
) -> str:
    out_path = os.path.join(args.out_dir, "nascdi_build_metadata.json")
    metadata = {
        "script": "src/nascdi/build_nascdi_final.py",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "args": vars(args),
        "source_files": source_file_manifest(args.news_dir),
        "lexicon": {
            "path": args.lexicon,
            "sha256": file_sha256(args.lexicon) if os.path.exists(args.lexicon) else "",
        },
        "sensitivity_lexicon": {
            "path": args.sensitivity_lexicon or "",
            "sha256": (
                file_sha256(args.sensitivity_lexicon)
                if args.sensitivity_lexicon and os.path.exists(args.sensitivity_lexicon)
                else ""
            ),
        },
        "counts": {
            "articles_loaded_after_dedup": int(len(articles)),
            "articles_scored": int(len(scored)),
            "articles_used_for_daily": int(len(df_daily)),
            "daily_rows": int(len(daily)),
            "weekly_rows": int(len(weekly)),
        },
        "date_ranges": {
            "articles_min": str(articles["date"].min()) if "date" in articles and not articles.empty else "",
            "articles_max": str(articles["date"].max()) if "date" in articles and not articles.empty else "",
            "daily_min": str(daily.index.min()) if not daily.empty else "",
            "daily_max": str(daily.index.max()) if not daily.empty else "",
            "weekly_min": str(weekly["week_end"].min()) if "week_end" in weekly and not weekly.empty else "",
            "weekly_max": str(weekly["week_end"].max()) if "week_end" in weekly and not weekly.empty else "",
        },
        "nascdi_stats": {
            "mean": float(daily["NASCDI"].mean()),
            "std": float(daily["NASCDI"].std(ddof=1)),
            "min": float(daily["NASCDI"].min()),
            "max": float(daily["NASCDI"].max()),
            "days_ge_110": int((daily["NASCDI"] >= 110).sum()),
            "days_le_90": int((daily["NASCDI"] <= 90).sum()),
        },
        "outputs": {
            name: {
                "path": path,
                "sha256": file_sha256(path) if os.path.exists(path) else "",
            }
            for name, path in outputs.items()
        },
        "journal_note": (
            "This metadata file is intended for replication. It records source CSV hashes, "
            "lexicon hash, row counts, date ranges, and output hashes."
        ),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return out_path


# ─────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build journal-ready NASCDI from news CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--news_dir",  default="data/news/raw",
                        help="Folder containing news CSV files.")
    parser.add_argument("--lexicon",   default="config/lexicon.yaml",
                        help="Primary lexicon YAML.")
    parser.add_argument("--out_dir",   default="data/nascdi",
                        help="Output directory.")
    parser.add_argument("--min_score", type=float, default=1.0,
                        help="Min raw score to count as disruption article.")
    parser.add_argument("--clip_raw",  type=float, default=30.0,
                        help="Clip per-article score at this absolute value.")
    parser.add_argument("--start",     default=None,
                        help="Daily index start date (YYYY-MM-DD).")
    parser.add_argument("--end",       default=None,
                        help="Daily index end date (YYYY-MM-DD).")
    parser.add_argument("--normalize", default="z_to_100_10",
                        choices=["z_to_100_10", "minmax_0_100"],
                        help="Normalization method.")
    parser.add_argument("--no_require_commodity_context",
                        action="store_true",
                        help="Score all articles (bypass commodity filter).")
    parser.add_argument("--sensitivity_lexicon", default=None,
                        help="Optional path to sensitivity lexicon YAML for robustness check.")
    parser.add_argument("--no_verbose", action="store_true",
                        help="Suppress diagnostic output.")
    args = parser.parse_args()

    verbose = not args.no_verbose
    os.makedirs(args.out_dir, exist_ok=True)

    if verbose:
        print("\n" + "="*60)
        print("NASCDI Builder — Apple Supply Chain Disruption Index")
        print("="*60)

    # ── Load news corpus ───────────────────────────────────────
    try:
        articles = load_news_corpus(args.news_dir, verbose=verbose)
    except (FileNotFoundError, ValueError) as e:
        print(str(e))
        sys.exit(1)

    # ── Load lexicon ───────────────────────────────────────────
    lex_main = load_lexicon(args.lexicon)
    if verbose:
        print(f"\nLexicon loaded: {args.lexicon}")
        print(f"  disruption_terms  : {len(lex_main['disruption_terms'])}")
        print(f"  mitigation_terms  : {len(lex_main['mitigation_terms'])}")
        print(f"  commodity_terms   : {len(lex_main['commodity_terms'])}")

    # ── Score & build daily index ──────────────────────────────
    try:
        scored, df_daily, daily_raw = score_corpus(
            articles,
            lex_main,
            min_score_threshold=args.min_score,
            clip_raw=args.clip_raw,
            require_commodity_context=not args.no_require_commodity_context,
            verbose=verbose,
        )
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    daily = normalize_index(
        daily_raw,
        method=args.normalize,
        start_date=args.start,
        end_date=args.end,
    )

    # ── Weekly aggregation ─────────────────────────────────────
    weekly = (
        daily.resample("W")
        .agg(
            raw_nascdi               = ("raw_nascdi",               "mean"),
            NASCDI                   = ("NASCDI",                   "mean"),
            news_volume              = ("news_volume",               "sum"),
            disruption_article_count = ("disruption_article_count", "sum"),
            total_score              = ("total_score",               "sum"),
        )
        .reset_index()
        .rename(columns={"date": "week_end"})
    )

    # ── Save outputs ───────────────────────────────────────────
    scored_path    = os.path.join(args.out_dir, "news_scored.csv")
    daily_path     = os.path.join(args.out_dir, "nascdi_daily.csv")
    weekly_path    = os.path.join(args.out_dir, "nascdi_news_weekly.csv")
    fig_daily_path = os.path.join(args.out_dir, "nascdi_daily.png")
    fig_wkly_path  = os.path.join(args.out_dir, "nascdi_news_weekly.png")

    if verbose:
        print(f"\nSTAGE 5 — Saving outputs")

    scored.to_csv(scored_path, index=False)
    daily.reset_index().to_csv(daily_path, index=False)
    weekly.to_csv(weekly_path, index=False)

    if verbose:
        print(f"  Scored articles : {scored_path}")
        print(f"  Daily NASCDI    : {daily_path}  ({len(daily):,} rows)")
        print(f"  Weekly NASCDI   : {weekly_path}  ({len(weekly):,} rows)")

    plot_index(daily, fig_daily_path)
    plot_weekly(daily, fig_wkly_path)
    desc_path = save_descriptive_table(daily, weekly, args.out_dir)

    # ── Sensitivity analysis ───────────────────────────────────
    if args.sensitivity_lexicon and os.path.exists(args.sensitivity_lexicon):
        lex_sens = load_lexicon(args.sensitivity_lexicon)
        run_sensitivity_comparison(
            articles=articles,
            lex_main=lex_main,
            lex_sens=lex_sens,
            min_score=args.min_score,
            clip_raw=args.clip_raw,
            require_commodity=not args.no_require_commodity_context,
            start_date=args.start,
            end_date=args.end,
            out_dir=args.out_dir,
        )

    metadata_path = save_reproducibility_metadata(
        args=args,
        articles=articles,
        scored=scored,
        df_daily=df_daily,
        daily=daily,
        weekly=weekly,
        outputs={
            "news_scored": scored_path,
            "daily_nascdi": daily_path,
            "weekly_nascdi": weekly_path,
            "daily_plot": fig_daily_path,
            "weekly_plot": fig_wkly_path,
            "descriptive_table": desc_path,
        },
    )

    # ── Final summary ──────────────────────────────────────────
    if verbose:
        print("\n" + "="*60)
        print("NASCDI SUMMARY")
        print("="*60)
        print(f"  Date range : {daily.index.min().date()} — {daily.index.max().date()}")
        print(f"  Mean       : {daily['NASCDI'].mean():.2f}")
        print(f"  Std        : {daily['NASCDI'].std():.2f}")
        print(f"  Min        : {daily['NASCDI'].min():.2f}")
        print(f"  Max        : {daily['NASCDI'].max():.2f}")
        print(f"  Obs (daily): {len(daily):,}")
        print(f"  Obs (weekly): {len(weekly):,}")
        print(f"  Descriptives: {desc_path}")
        print(f"  Metadata    : {metadata_path}")
        print("\n✓ Done. NASCDI built successfully.\n")


if __name__ == "__main__":
    main()
