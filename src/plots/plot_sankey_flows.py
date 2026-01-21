import os
import pandas as pd

OUT_DIR = "results/nardl/plots/plots_advanced"
os.makedirs(OUT_DIR, exist_ok=True)

RES = "results/nardl/nardl_results_table.csv"

def main():
    df = pd.read_csv(RES)
    df["vulnerability"] = df["LR_effect_NASCDI_pos"].abs() * df["ECT_y_L1"].abs()

    # In your current setup, terminal market is implicitly Azadpur.
    # If you later add multiple terminals, replace this field accordingly.
    df["terminal_market"] = "Azadpur"

    nodes = sorted(set(df["producer_market"].tolist() + df["terminal_market"].tolist()))
    node_index = {n:i for i,n in enumerate(nodes)}

    sources = df["producer_market"].map(node_index).tolist()
    targets = df["terminal_market"].map(node_index).tolist()
    values = df["vulnerability"].fillna(0).tolist()

    try:
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Sankey(
            node=dict(label=nodes),
            link=dict(source=sources, target=targets, value=values)
        )])

        fig.update_layout(title_text="Sankey: Producer → Terminal vulnerability-weighted flows", font_size=12)

        html = os.path.join(OUT_DIR, "sankey_vulnerability_flows.html")
        fig.write_html(html)

        try:
            png = os.path.join(OUT_DIR, "sankey_vulnerability_flows.png")
            fig.write_image(png, scale=2)
            print("✅ Saved:", png)
        except Exception as e:
            print("⚠️ PNG export failed (install kaleido). Saved HTML instead.", e)

        print("✅ Saved:", html)

    except ImportError:
        print("Plotly not installed. Run: pip install plotly kaleido")

if __name__ == "__main__":
    main()
