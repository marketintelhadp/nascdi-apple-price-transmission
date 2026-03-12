import pandas as pd
import os
from statsmodels.tsa.ardl import ardl_select_order

DATA_DIR = "data/model"
OUTPUT_FILE = "results/lag_selection_results.csv"

rows = []

for file in os.listdir(DATA_DIR):

    if not file.endswith(".csv"):
        continue

    path = os.path.join(DATA_DIR, file)
    df = pd.read_csv(path)

    # Required variables
    y = df["log_price_prod"]
    X = df[["NASCDI_pos", "NASCDI_neg"]]

    # ARDL lag selection
    sel = ardl_select_order(
        endog=y,
        maxlag=6,
        exog=X,
        maxorder=6,
        ic="aic",
        trend="c"
    )

    rows.append({
        "dataset": file,
        "optimal_lag_y": sel.model._maxlag,
        "optimal_lag_exog": sel.model._maxorder
    })

results = pd.DataFrame(rows)

os.makedirs("results", exist_ok=True)
results.to_csv(OUTPUT_FILE, index=False)

print("Lag selection results saved to:", OUTPUT_FILE)