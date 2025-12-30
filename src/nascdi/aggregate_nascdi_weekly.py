import pandas as pd

# Load daily NASCDI
df = pd.read_csv(
    "data/nascdi/nascdi_event_based.csv",
    parse_dates=["date"]
)

# Set date as index
df = df.set_index("date")

# Weekly aggregation (mean is standard)
weekly = df.resample("W").mean()

# Save
weekly.reset_index().to_csv(
    "data/nascdi/nascdi_event_weekly.csv",
    index=False
)

print("Weekly NASCDI created successfully")
