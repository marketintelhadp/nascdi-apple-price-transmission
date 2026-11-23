"""
Rebuild the NASCDI on the COMPLETE corpus.
 - 2015-02..2018-02: rows (and validated scores) from the existing scored corpus (monthly BigQuery files; originals no longer on disk)
 - 2018-03..2025-12: the 94 newly exported monthly BigQuery files (same query)
Scoring, clipping, thresholds, commodity gate, daily mean, z-normalisation and weekly aggregation replicate src/nascdi/build_nascdi_final.py
(validated to 1e-14 on the old corpus).  Deduplication keeps the earliest-dated row per URL, as the pipeline does.
"""
import sys, glob, os, re, time, yaml
from pathlib import Path
import numpy as np, pandas as pd
from multiprocessing import Pool

import os
REPO = Path(os.environ.get("NASCDI_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO))
from src.nascdi import build_nascdi_final as B          # infer_columns, clean_text, normalize_index

OUT = REPO / "data" / "nascdi_news_v2"; OUT.mkdir(exist_ok=True)
SCR = REPO / "data" / "news_scored"
LEX = yaml.safe_load(open(REPO / "config/lexicon_news.yaml", encoding="utf-8"))
def comp(t): return re.compile(r"(?<![a-z0-9\-])" + re.escape(t) + r"(?![a-z0-9\-])")
D = [(t.strip().lower(), float(w), comp(t.strip().lower())) for t, w in LEX["disruption_terms"].items()]
M = [(t.strip().lower(), float(w), comp(t.strip().lower())) for t, w in LEX["mitigation_terms"].items()]
C = [(t.strip().lower(), float(w), comp(t.strip().lower())) for t, w in LEX["commodity_terms"].items()]

def score(text):
    dm = de = mm = me = cm = 0.0
    for t, w, r in D:
        if t in text:
            n = len(r.findall(text))
            if n: dm += n * w; de += n
    for t, w, r in M:
        if t in text:
            n = len(r.findall(text))
            if n: mm += n * w; me += n
    for t, w, r in C:
        if t in text:
            n = len(r.findall(text))
            if n: cm += n * w
    return dm, de, mm, me, cm

def work(args):
    path, raw = args
    std = B.infer_columns(raw, path)
    std["date"] = std["date"].dt.normalize()
    tc = std["title"].fillna("").astype(str).map(B.clean_text); xc = std["text"].fillna("").astype(str).map(B.clean_text)
    full = np.where(tc.str.strip() == xc.str.strip(), tc, (tc + " " + xc).str.strip())
    a = np.array([score(t) for t in full])
    return pd.DataFrame({"date": std["date"].values, "url": std["url"].astype(str).str.strip().values, "d_main": a[:, 0], "d_eq": a[:, 1], "m_main": a[:, 2], "m_eq": a[:, 3], "comm": a[:, 4], "src": os.path.basename(path)})

def gen(files):
    for f in files:
        for ch in pd.read_csv(f, low_memory=False, chunksize=40000): yield f, ch

if __name__ == "__main__":
    t0 = time.time()
    new_files = sorted(glob.glob(str(REPO / "data/news/raw_gdelt_gkg_2018plus*/gdelt_gkg_*.csv")) + glob.glob(str(REPO / "data/news/raw_gdelt_gkg_fill*/gdelt_gkg_*.csv")))
    months = {re.search(r"gdelt_gkg_(\d{4}_\d{2})", f).group(1) for f in new_files}
    print(len(new_files), "new monthly files,", len(months), "distinct months", flush=True)
    assert len(new_files) == len(months) == 94, "expected 94 distinct monthly files"
    parts = []
    with Pool(8) as p:
        for i, o in enumerate(p.imap_unordered(work, gen(new_files), chunksize=1)):
            parts.append(o)
            if i % 10 == 0: print(f"  new chunks {i}, {time.time()-t0:.0f}s", flush=True)
    new = pd.concat(parts, ignore_index=True); print("new rows:", len(new), f"{time.time()-t0:.0f}s", flush=True)

    # old rows (2015-02..2018-02) : positional join of the scored corpus (urls) with the stored article scores
    sc = pd.read_csv(SCR / "article_scores.csv.gz", parse_dates=["date"])
    urls = []
    for ch in pd.read_csv(REPO / "data/nascdi_news/news_scored.csv", usecols=["date", "url", "_file"], chunksize=500000): urls.append(ch)
    uf = pd.concat(urls, ignore_index=True); assert len(uf) == len(sc), (len(uf), len(sc))
    old = pd.DataFrame({"date": sc["date"].dt.normalize().values, "url": uf["url"].astype(str).str.strip().values, "d_main": sc["d_main"].values, "d_eq": sc["d_eq"].values, "m_main": sc["m_main"].values, "m_eq": sc["m_eq"].values, "comm": sc["comm"].values, "src": uf["_file"].values})
    old = old[old["src"].str.startswith("gdelt_gkg_") & (old["date"] < "2018-03-01")]
    print("old rows kept (monthly files, < 2018-03):", len(old), flush=True)

    al = pd.concat([old, new], ignore_index=True)
    n0 = len(al); al = al.sort_values("date", kind="stable")
    has_url = al["url"].str.len() > 5
    al = pd.concat([al[has_url].drop_duplicates("url", keep="first"), al[~has_url]]).sort_values("date", kind="stable").reset_index(drop=True)
    print(f"after dedup: {len(al)} (removed {n0-len(al)})", flush=True)
    al.to_csv(SCR / "article_scores_v2.csv.gz", index=False)

    # index construction (same steps as score_corpus / main in build_nascdi_final.py)
    def build(d_col, m_col, weekly_name, daily_name):
        s = al.copy(); s["score_raw"] = s[d_col] + (s[m_col] if m_col == "m_main" else -s[m_col])
        s["clip"] = s["score_raw"].clip(-30, 30)
        s["is_idx"] = s["clip"].abs() >= 1.0
        s = s[(s["comm"] > 0) & s["is_idx"]].copy()
        s["is_hit"] = (s["clip"] >= 1.0).astype(int)
        s["dscore"] = s[d_col]; s["mscore"] = s[m_col] if m_col == "m_main" else -s[m_col]
        daily = s.groupby("date").agg(raw_nascdi=("clip", "mean"), total_score=("clip", "sum"), news_volume=("clip", "count"),
                                      disruption_article_count=("is_hit", "sum"), disruption_score_mean=("dscore", "mean"), mitigation_score_mean=("mscore", "mean")).sort_index()
        daily = B.normalize_index(daily, method="z_to_100_10", start_date="2015-02-19", end_date="2025-12-31")
        weekly = daily.resample("W").agg(raw_nascdi=("raw_nascdi", "mean"), NASCDI=("NASCDI", "mean"), news_volume=("news_volume", "sum"),
                                         disruption_article_count=("disruption_article_count", "sum"), total_score=("total_score", "sum")).reset_index().rename(columns={"date": "week_end"})
        daily.reset_index().to_csv(OUT / daily_name, index=False); weekly.to_csv(OUT / weekly_name, index=False)
        return daily, weekly
    daily, weekly = build("d_main", "m_main", "nascdi_news_weekly.csv", "nascdi_daily.csv")
    build("d_eq", "m_eq", "nascdi_news_weekly_equalweights.csv", "nascdi_daily_equalweights.csv")
    print("done", f"{time.time()-t0:.0f}s", "| weekly rows", len(weekly), "| daily rows", len(daily), flush=True)
