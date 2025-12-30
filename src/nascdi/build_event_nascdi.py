import pandas as pd
import numpy as np
import os

EVENT_FILE = "data/nascdi/event_calendar.csv"

if not os.path.exists(EVENT_FILE) or os.path.getsize(EVENT_FILE) == 0:
    raise RuntimeError(
        "event_calendar.csv is missing or empty. "
        "Please populate it with event data before running NASCDI."
    )

events = pd.read_csv(EVENT_FILE, parse_dates=["start_date", "end_date"])


# Load event calendar
events = pd.read_csv("data/nascdi/event_calendar.csv", parse_dates=["start_date", "end_date"])

# Create daily date range
dates = pd.date_range("2010-01-01", "2025-12-31", freq="D")
df = pd.DataFrame({"date": dates})
df["raw_score"] = 0.0

# Apply events
for _, row in events.iterrows():
    mask = (df["date"] >= row["start_date"]) & (df["date"] <= row["end_date"])
    df.loc[mask, "raw_score"] += row["weight"]

# Normalize: mean 100, SD 10
mu = df["raw_score"].mean()
sd = df["raw_score"].std(ddof=0)

if sd > 0:
    df["NASCDI"] = ((df["raw_score"] - mu) / sd) * 10 + 100
else:
    df["NASCDI"] = 100

df.to_csv("data/nascdi/nascdi_event_based.csv", index=False)
print("Event-based NASCDI created successfully")
