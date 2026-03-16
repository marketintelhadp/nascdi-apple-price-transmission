import os
import pandas as pd
from statsmodels.tsa.ardl import ardl_select_order

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = "data/model"
OUTPUT_DIR = "results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "lag_selection_results.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

rows = []

# -----------------------------
# Loop through datasets
# -----------------------------
for file in os.listdir(DATA_DIR):

    if not file.endswith(".csv"):
        continue

    path = os.path.join(DATA_DIR, file)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Could not read {file}: {e}")
        continue

    required_cols = ["log_price_prod", "NASCDI_pos", "NASCDI_neg"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print(f"Skipping {file} (missing columns: {missing})")
        continue

    # Remove missing values
    df = df.dropna()

    y = df["log_price_prod"]
    X = df[["NASCDI_pos", "NASCDI_neg"]]

    try:

        # -----------------------------
        # ARDL Lag Selection
        # -----------------------------
        sel = ardl_select_order(
            endog=y,
            maxlag=6,
            exog=X,
            maxorder=6,
            ic="aic",
            trend="c"
        )

        model = sel.model

        # -----------------------------
        # Extract dependent variable lag
        # -----------------------------
        try:
            lag_y = max(model.ar_lags) if model.ar_lags else 0
        except:
            lag_y = None

        # -----------------------------
        # Extract exogenous lags
        # -----------------------------
        lag_pos = None
        lag_neg = None

        if hasattr(model, "dl_lags") and model.dl_lags:

            lag_pos = max(model.dl_lags.get("NASCDI_pos", [0]))
            lag_neg = max(model.dl_lags.get("NASCDI_neg", [0]))

        # -----------------------------
        # Extract Information Criteria
        # -----------------------------
        try:
            best_aic_idx = sel.aic.idxmin()
            best_aic_value = float(sel.aic.loc[best_aic_idx])
        except:
            best_aic_value = None

        try:
            best_bic_idx = sel.bic.idxmin()
            best_bic_value = float(sel.bic.loc[best_bic_idx])
        except:
            best_bic_value = None

        try:
            best_hqic_idx = sel.hqic.idxmin()
            best_hqic_value = float(sel.hqic.loc[best_hqic_idx])
        except:
            best_hqic_value = None

        rows.append({
            "dataset": file,
            "lag_y": lag_y,
            "lag_NASCDI_pos": lag_pos,
            "lag_NASCDI_neg": lag_neg,
            "AIC": round(best_aic_value, 4) if best_aic_value is not None else None,
            "BIC": round(best_bic_value, 4) if best_bic_value is not None else None,
            "HQIC": round(best_hqic_value, 4) if best_hqic_value is not None else None
        })

        print(f"Processed: {file}")

    except Exception as e:

        print(f"Error processing {file}: {e}")
        continue


# -----------------------------
# Save Results
# -----------------------------
results = pd.DataFrame(rows)

results.to_csv(OUTPUT_FILE, index=False)

# Optional Excel export
results.to_excel(os.path.join(OUTPUT_DIR, "lag_selection_results.xlsx"), index=False)

print("\nLag selection results saved to:")
print(OUTPUT_FILE)

print("\nPreview:")
print(results)