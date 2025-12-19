import os
import re
import glob
import yaml
import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def robust_parse_date(s: pd.Series) -> pd.Series:
    """Parse mixed date formats safely."""
    # First try ISO / normal parsing
    d1 = pd.to_datetime(s, errors="coerce", dayfirst=False, utc=False)
    # For the failures, try dayfirst
    mask = d1.isna()
    if mask.any():
        d2 = pd.to_datetime(s[mask], errors="coerce", dayfirst=True, utc=False)
        d1.loc[mask] = d2
    return d1.dt.tz_localize(None)


def clean_text(text: str) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text).lower()
    # normalize common highway variants
    text = text.replace("nh 44", "nh-44").replace("nh44", "nh-44")
    # remove urls
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # keep letters/numbers/hyphen, convert rest to space
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_lexicon(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        lex = yaml.safe_load(f)
    for k in ["disruption_terms", "mitigation_terms", "commodity_terms"]:
        if k not in lex:
            lex[k] = {}
    return lex


def count_term_occurrences(text: str, term: str) -> int:
    """
    Count occurrences of a term as a phrase, reasonably robust.
    For phrases, we search substring with word boundary-ish behavior.
    """
    term = term.strip().lower()
    if not term:
        return 0
    # phrase boundary: allow hyphens and spaces, but avoid partial word matches
    # Example: 'landslide' should not match 'landslides' unless explicitly included
    pattern = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
    return len(re.findall(pattern, text))


def score_article(text: str, lex: Dict[str, Dict[str, float]]) -> float:
    score = 0.0
    # disruption
    for term, w in lex["disruption_terms"].items():
        score += count_term_occurrences(text, term) * float(w)
    # commodity context
    for term, w in lex["commodity_terms"].items():
        score += count_term_occurrences(text, term) * float(w)
    # mitigation reduces
    for term, w in lex["mitigation_terms"].items():
        score += count_term_occurrences(text, term) * float(w)
    return score


def infer_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common column variants to standard names."""
    colmap = {}
    cols = [c.lower().strip() for c in df.columns]

    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in cols:
                return df.columns[cols.index(c)]
        return None

    date_col = pick(["date", "published_date", "publish_date", "time", "datetime"])
    title_col = pick(["title", "headline", "news_title"])
    text_col = pick(["text", "content", "body", "article", "article_text", "news_text"])
    source_col = pick(["source", "newspaper", "site"])
    url_col = pick(["url", "link", "article_url"])

    if date_col is None:
        raise ValueError("No date column found. Please include a 'date' column in your news CSV(s).")
    if text_col is None and title_col is None:
        raise ValueError("No text/body column found. Please include 'text'/'content' or at least 'title'.")

    out = pd.DataFrame()
    out["date_raw"] = df[date_col]
    out["date"] = robust_parse_date(out["date_raw"].astype(str))
    out["title"] = df[title_col] if title_col else ""
    out["text"] = df[text_col] if text_col else ""
    out["source"] = df[source_col] if source_col else ""
    out["url"] = df[url_col] if url_col else ""
    return out


def dedupe_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by URL if present; else by (date, normalized title hash)."""
    df = df.copy()
    df["title_clean"] = df["title"].fillna("").astype(str).map(clean_text)
    df["text_clean"] = df["text"].fillna("").astype(str).map(clean_text)

    # Prefer URL dedupe
    if (df["url"].fillna("").str.len() > 0).any():
        df["url_norm"] = df["url"].fillna("").astype(str).str.strip()
        df = df.sort_values(["date"]).drop_duplicates(subset=["url_norm"], keep="first")
    else:
        df["key"] = df["date"].astype(str) + "||" + df["title_clean"].str[:120]
        df = df.sort_values(["date"]).drop_duplicates(subset=["key"], keep="first")
    return df


def build_daily_index(
    articles: pd.DataFrame,
    lex: Dict[str, Dict[str, float]],
    min_score_threshold: float = 1.0,
    normalize_method: str = "z_to_100_10",
    clip_raw: Optional[float] = 30.0,
) -> pd.DataFrame:
    """
    Compute daily NASCDI:
      - score each article
      - optionally drop very low signal pieces
      - daily aggregation: mean score per day + volume features
      - normalize to mean 100, sd 10 (default)
    """
    df = articles.copy()

    # combine title + text for scoring (title weighted implicitly by inclusion)
    df["full_text"] = (df["title_clean"].fillna("") + " " + df["text_clean"].fillna("")).str.strip()
    df["score_raw"] = df["full_text"].apply(lambda t: score_article(t, lex))

    # keep weak signals for volume counts, but also track "disruption articles"
    df["is_disruption_hit"] = (df["score_raw"] >= min_score_threshold).astype(int)

    # optional clipping to prevent one very long article dominating
    if clip_raw is not None:
        df["score_raw"] = df["score_raw"].clip(lower=-abs(clip_raw), upper=abs(clip_raw))

    daily = (
        df.groupby("date", as_index=True)
          .agg(
              raw_nascdi=("score_raw", "mean"),
              total_score=("score_raw", "sum"),
              news_volume=("score_raw", "count"),
              disruption_article_count=("is_disruption_hit", "sum"),
          )
          .sort_index()
    )

    # Normalize
    if normalize_method == "z_to_100_10":
        mu = daily["raw_nascdi"].mean()
        sd = daily["raw_nascdi"].std(ddof=0)
        if sd == 0 or np.isnan(sd):
            daily["NASCDI"] = 100.0
        else:
            daily["NASCDI"] = ((daily["raw_nascdi"] - mu) / sd) * 10.0 + 100.0
    elif normalize_method == "minmax_0_100":
        mn, mx = daily["raw_nascdi"].min(), daily["raw_nascdi"].max()
        if mx == mn:
            daily["NASCDI"] = 50.0
        else:
            daily["NASCDI"] = (daily["raw_nascdi"] - mn) / (mx - mn) * 100.0
    else:
        raise ValueError("Unknown normalize_method. Use 'z_to_100_10' or 'minmax_0_100'.")

    return df, daily


def plot_index(daily: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(14, 4))
    plt.plot(daily.index, daily["NASCDI"])
    plt.title("NASCDI (Daily)")
    plt.xlabel("Date")
    plt.ylabel("Index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Build NASCDI from local news CSV files.")
    parser.add_argument("--news_dir", default="data/news/raw", help="Folder with news CSV files.")
    parser.add_argument("--lexicon", default="config/lexicon.yaml", help="Path to lexicon YAML.")
    parser.add_argument("--out_dir", default="data/nascdi", help="Output directory.")
    parser.add_argument("--min_score", type=float, default=1.0, help="Min score to count as disruption-hit article.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = glob.glob(os.path.join(args.news_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {args.news_dir}")

    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        std = infer_columns(df)
        std["file"] = os.path.basename(fp)
        frames.append(std)

    articles = pd.concat(frames, ignore_index=True)
    articles = articles.dropna(subset=["date"])
    articles = dedupe_articles(articles)

    lex = load_lexicon(args.lexicon)

    scored_articles, daily = build_daily_index(
        articles=articles,
        lex=lex,
        min_score_threshold=args.min_score,
        normalize_method="z_to_100_10",
        clip_raw=30.0
    )

    scored_path = os.path.join(args.out_dir, "news_scored.csv")
    daily_path = os.path.join(args.out_dir, "nascdi_daily.csv")
    fig_path = os.path.join(args.out_dir, "nascdi_daily.png")

    scored_articles.to_csv(scored_path, index=False)
    daily.reset_index().rename(columns={"index": "date"}).to_csv(daily_path, index=False)
    plot_index(daily, fig_path)

    print(f"Saved scored articles: {scored_path}")
    print(f"Saved daily NASCDI:     {daily_path}")
    print(f"Saved plot:            {fig_path}")
    print("Done.")


if __name__ == "__main__":
    main()
