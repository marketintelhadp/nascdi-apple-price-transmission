import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RES = "results/nardl/nardl_results_table.csv"
DIAG = "results/nardl/nardl_diagnostics.csv"
OUT_DIR = "results/nardl/plots/plots_advanced"
os.makedirs(OUT_DIR, exist_ok=True)

def build_label(df):
    return df["producer_market"].astype(str) + " | " + df["variety"].astype(str) + " | G" + df["grade"].astype(str)

def minmax(s):
    s = s.astype(float)
    return (s - s.min()) / (s.max() - s.min() + 1e-12)

def main():
    r = pd.read_csv(RES)
    d = pd.read_csv(DIAG)
    df = r.merge(d, on=["producer_market","variety","grade"], how="left")

    df["label"] = build_label(df)

    # Build normalized policy dimensions
    df["vulnerability"] = (df["LR_effect_NASCDI_pos"].abs() * df["ECT_y_L1"].abs())
    df["fit"] = df["R2"]
    df["cointegration"] = 1 - df["bounds_F_pvalue"].clip(0,1)
    df["no_autocorr"] = df["ljungbox_p(8)"].clip(0,1)
    df["homosk"] = df["bp_heterosk_p"].clip(0,1)

    radar = pd.DataFrame({
        "label": df["label"],
        "Vulnerability": minmax(df["vulnerability"]),
        "Fit (R²)": minmax(df["fit"]),
        "Cointegration strength": minmax(df["cointegration"]),
        "No autocorr (LB p)": minmax(df["no_autocorr"]),
        "Homoskedasticity (BP p)": minmax(df["homosk"]),
    })

    categories = radar.columns[1:].tolist()
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    # Plot each series (small N)
    for _, row in radar.iterrows():
        values = row[categories].tolist()
        values += values[:1]

        fig = plt.figure(figsize=(7, 6))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_title(row["label"], pad=20)
        ax.set_ylim(0, 1)

        safe = row["label"].replace(" | ", "_").replace(" ", "_")
        out = os.path.join(OUT_DIR, f"radar_{safe}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=220)
        plt.close(fig)
        print("✅ Saved:", out)

if __name__ == "__main__":
    main()
