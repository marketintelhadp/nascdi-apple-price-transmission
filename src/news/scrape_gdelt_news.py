import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

OUT_DIR = "data/news/raw"
os.makedirs(OUT_DIR, exist_ok=True)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# 🔑 Keep queries SHORT and focused
QUERIES = {
    "Jammu - Srinagar National Highway": 'NH-44 AND (National Highway)',
    "landslide": 'landslide AND (Kashmir OR Srinagar)',
    "Shooting Stones": 'Road Blockage due to shooting stones',
    "Snowfall": 'apple trucks, supply trucks',
    "Road Closure" : 'trucks, traffic',
    "Jammu Srinagar National Highway Trafic Update" : "Suspended for traffic movement"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (NASCDI academic research)"
}


def daterange(start, end):
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)


def fetch_gdelt(query, day, retries=3, sleep=2):
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": day.strftime("%Y%m%d000000"),
        "enddatetime": day.strftime("%Y%m%d235959"),
        "maxrecords": 100,
        "sourcelang": "English",
    }

    for attempt in range(retries):
        try:
            r = requests.get(GDELT_URL, params=params, headers=HEADERS, timeout=20)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            if not r.text.strip().startswith("{"):
                raise ValueError("Non-JSON response")
            data = r.json()
            return data.get("articles", [])
        except Exception as e:
            if attempt == retries - 1:
                print(f"FAILED: {day.date()} | {query} | {e}")
                return []
            time.sleep(sleep * (attempt + 1))


def main():
    start = datetime(2010, 1, 1)
    end = datetime(2025, 12, 31)

    rows = []

    for query_name, query in QUERIES.items():
        print(f"\n▶ Query: {query_name}")
        for day in daterange(start, end):
            articles = fetch_gdelt(query, day)
            for a in articles:
                rows.append({
                    "date": a.get("seendate"),
                    "title": a.get("title"),
                    "text": a.get("title"),  # title-only is acceptable
                    "source": a.get("domain"),
                    "url": a.get("url"),
                    "query": query_name
                })
            time.sleep(0.8)  # polite rate limit

    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "gdelt_2010_2025.csv")
    df.to_csv(out, index=False)
    print(f"\n✔ Saved: {out}")
    print(f"✔ Articles collected: {len(df)}")


if __name__ == "__main__":
    main()
