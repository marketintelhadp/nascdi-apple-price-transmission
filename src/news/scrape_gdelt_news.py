import os
import time
import random
import hashlib
from datetime import datetime, timedelta

import pandas as pd
import requests

# ==============================
# Paths
# ==============================
OUT_DIR = "data/news/raw"
CACHE_DIR = "data/news/cache_gdelt"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"

HEADERS = {
    "User-Agent": "Academic research (NASCDI) – non-commercial"
}

# ==============================
# VERY FOCUSED QUERY BLOCKS
# ==============================
QUERY_BLOCKS = {
    "landslide": 'landslide OR "shooting stones"',
    "snowfall": 'snowfall OR "heavy snowfall" OR "untimely snowfall" OR rainfall',
    "nh44_updates": '"Jammu Srinagar National Highway" OR "NH-44 traffic"',
    "highway_closure": '"highway closed" OR "traffic suspended" OR "tunnel closed"',
    "political_unrest": 'shutdown OR hartal OR "stone pelting" OR "political unrest"',
    "security": '"terrorist encounter" OR "border tension" OR "Indo Pak"',
    "covid": 'coronavirus OR covid OR lockdown'
}

# Mandatory context filter
CONTEXT = '(Kashmir OR Srinagar OR Jammu OR Parimpora OR Azadpur) AND (apple OR fruit)'

# ==============================
def daterange(start, end):
    while start <= end:
        yield start
        start += timedelta(days=1)

def cache_marker(name, day):
    h = hashlib.md5(f"{name}_{day}".encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.done")

def is_json(text):
    return text and text.strip().startswith("{")

def fetch(query, day):
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": day.strftime("%Y%m%d000000"),
        "enddatetime": day.strftime("%Y%m%d235959"),
        "maxrecords": 50,
        "sourcelang": "English"
    }
    return requests.get(GDELT_API, params=params, headers=HEADERS, timeout=20)

# ==============================
def main():
    start = datetime(2010, 1, 1)
    end = datetime(2025, 12, 31)

    rows = []

    for qname, qtext in QUERY_BLOCKS.items():
        final_query = f"({qtext}) AND {CONTEXT}"
        print(f hookup: {qname}")

        for day in daterange(start, end):
            marker = cache_marker(qname, day.strftime("%Y-%m-%d"))
            if os.path.exists(marker):
                continue

            for attempt in range(1, 4):
                try:
                    r = fetch(final_query, day)
                    if r.status_code != 200 or not is_json(r.text):
                        raise ValueError("Non-JSON response")

                    data = r.json()
                    articles = data.get("articles", [])

                    for a in articles:
                        rows.append({
                            "date": a.get("seendate"),
                            "title": a.get("title"),
                            "text": a.get("title"),
                            "source": a.get("domain"),
                            "url": a.get("url"),
                            "query_block": qname
                        })

                    open(marker, "w").write("OK")
                    break

                except Exception as e:
                    if attempt == 3:
                        print(f"FAILED {day.date()} | {qname}")
                        open(marker, "w").write("FAIL")
                    time.sleep(2 ** attempt + random.random())

            time.sleep(0.7 + random.random())

    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "gdelt_2010_2025.csv")
    df.to_csv(out, index=False)

    print("\n Scraping complete")
    print(" Articles:", len(df))
    print(" Saved to:", out)

if __name__ == "__main__":
    main()
