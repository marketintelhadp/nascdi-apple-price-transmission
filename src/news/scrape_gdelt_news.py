import os
import time
import random
import hashlib
from datetime import datetime, timedelta

import pandas as pd
import requests

# =========================
# Configuration
# =========================
OUT_DIR = "data/news/raw"
CACHE_DIR = "data/news/cache_gdelt"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (NASCDI academic research; contact: university)"
}

# =========================
# LEXICON-ALIGNED QUERIES
# =========================
QUERIES = {
    # Transport & connectivity (highest weight)
    "transport_nh44": (
        '"NH-44" OR "Jammu Srinagar highway" OR "Banihal tunnel" '
        'OR "Jawahar Tunnel" OR "Zoji La"'
    ),

    # Weather & natural hazards
    "weather_hazards": (
        'landslide OR avalanche OR "heavy snowfall" OR "shooting stones" '
        'OR cloudburst'
    ),

    # Market arrivals & logistics
    "market_arrivals": (
        '"low arrivals" OR "decline in arrivals" OR "truck shortage" '
        'OR "logistics bottleneck" OR "supply chain disruption"'
    ),

    # Price stress & volatility
    "price_stress": (
        '"price spike" OR "price volatility" OR "sharp price increase" '
        'OR "distress sale" OR glut'
    )
}

# Mandatory geographic & commodity filter
CONTEXT_FILTER = '(Kashmir OR Srinagar OR Parimpora OR Azadpur) AND (apple OR apples OR fruit)'

# =========================
# Helper functions
# =========================
def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def cache_file(qname, day):
    key = f"{qname}_{day.strftime('%Y-%m-%d')}"
    h = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.txt")

def is_json(text):
    return text and text.lstrip().startswith("{")

def fetch_day(query, day, maxrecords=100, timeout=25):
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": day.strftime("%Y%m%d000000"),
        "enddatetime": day.strftime("%Y%m%d235959"),
        "maxrecords": maxrecords,
        "sourcelang": "English",
    }
    r = requests.get(GDELT_API, params=params, headers=HEADERS, timeout=timeout)
    return r

# =========================
# Main scraper
# =========================
def main():
    start = datetime(2010, 1, 1)
    end = datetime(2025, 12, 31)

    rows = []

    for qname, qcore in QUERIES.items():
        final_query = f"({qcore}) AND {CONTEXT_FILTER}"
        print(f"\n▶ Running query block: {qname}")

        for day in daterange(start, end):
            cpath = cache_file(qname, day)
            if os.path.exists(cpath):
                continue  # already attempted

            success = False
            for attempt in range(1, 5):
                try:
                    r = fetch_day(final_query, day)
                    txt = r.text

                    if r.status_code != 200 or not is_json(txt):
                        raise RuntimeError("Non-JSON response")

                    data = r.json()
                    articles = data.get("articles", [])

                    for a in articles:
                        rows.append({
                            "date": a.get("seendate"),
                            "title": a.get("title"),
                            "text": a.get("title"),  # title-only acceptable
                            "source": a.get("domain"),
                            "url": a.get("url"),
                            "query_block": qname
                        })

                    with open(cpath, "w") as f:
                        f.write("OK")

                    success = True
                    break

                except Exception as e:
                    if attempt == 4:
                        print(f"FAILED: {day.date()} | {qname} | {e}")
                        with open(cpath, "w") as f:
                            f.write(str(e))
                    time.sleep((2 ** attempt) + random.uniform(0, 1))

            time.sleep(1.0 + random.uniform(0, 0.5))

    df = pd.DataFrame(rows)
    out_file = os.path.join(OUT_DIR, "gdelt_2010_2025.csv")
    df.to_csv(out_file, index=False)

    print("\n✔ Scraping complete")
    print("✔ Articles collected:", len(df))
    print("✔ Saved to:", out_file)


if __name__ == "__main__":
    main()
