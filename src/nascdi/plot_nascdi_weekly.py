import pandas as pd
import matplotlib.pyplot as plt

# Load weekly NASCDI
df = pd.read_csv(
    "data/nascdi/nascdi_event_weekly.csv",
    parse_dates=["date"]
)

# Summary check
print("Weekly NASCDI summary:")
print(df["NASCDI"].describe())

# Plot
plt.figure(figsize=(15, 5))
plt.plot(df["date"], df["NASCDI"], color="black", linewidth=1.5)

plt.axhline(100, color="red", linestyle="--", linewidth=1, label="Mean = 100")

plt.title("Weekly Event-Based NASCDI (2010–2025)", fontsize=14)
plt.xlabel("Year")
plt.ylabel("NASCDI Index")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("data/nascdi/nascdi_event_weekly.png", dpi=300)
plt.show()
