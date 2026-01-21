import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = "results/nardl/plots/plots_advanced"
os.makedirs(OUT_DIR, exist_ok=True)

def box(ax, xy, text, w=2.6, h=0.6):
    x,y = xy
    patch = FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0.02,rounding_size=0.07",
                           linewidth=1.2, facecolor="white")
    ax.add_patch(patch)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=10)

def arrow(ax, x1,y1,x2,y2):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="->", lw=1.2))

def main():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_axis_off()

    box(ax, (0.3, 3.4), "Raw price CSVs\n(producer + terminal)")
    box(ax, (3.3, 3.4), "Cleaning & standardization\n(prices_long.csv)")
    box(ax, (6.3, 3.4), "Weekly aggregation\n(week_end alignment)")
    box(ax, (9.3, 3.4), "Merge producer→terminal\n(per series files)")

    box(ax, (0.3, 2.1), "NASCDI weekly index\n(nascdi_event_weekly.csv)")
    box(ax, (3.3, 2.1), "Shin decomposition\nNASCDI_pos / NASCDI_neg")

    box(ax, (6.3, 2.1), "NARDL / ECM estimation\nΔy, lagged levels,\nlagged differences")
    box(ax, (9.3, 2.1), "Outputs\nTables + Diagnostics")

    box(ax, (6.3, 0.8), "Advanced visuals\nIRF, multipliers,\n3D surfaces, Sankey")

    arrow(ax, 2.9, 3.7, 3.3, 3.7)
    arrow(ax, 5.9, 3.7, 6.3, 3.7)
    arrow(ax, 8.9, 3.7, 9.3, 3.7)

    arrow(ax, 2.9, 2.4, 3.3, 2.4)
    arrow(ax, 5.9, 2.4, 6.3, 2.4)

    # merge NASCDI into model estimation
    arrow(ax, 4.6, 3.4, 6.3, 2.7)
    arrow(ax, 4.6, 2.1, 6.3, 2.1)

    arrow(ax, 8.9, 2.4, 9.3, 2.4)
    arrow(ax, 7.6, 2.1, 7.6, 1.4)

    ax.set_xlim(0, 12.3)
    ax.set_ylim(0, 4.3)

    out = os.path.join(OUT_DIR, "workflow_diagram.png")
    plt.tight_layout()
    plt.savefig(out, dpi=240)
    plt.close(fig)
    print("✅ Saved:", out)

if __name__ == "__main__":
    main()
