# analyze_resolution_metrics.py
# - Scans height-step CSVs: 0cm.csv ... 10cm.csv (step 0.5) inside DATA_DIR
# - Extracts and summarizes (per height):
#       dCcal_pF_per_df, N_counts, bits_eq
# - Saves:
#   (1) log_plots/metrics_summary.csv  (per-height mean/std/min/max/N)
#   (2) PNG + CSV (x,y) for each plot:
#       - dCcal_mean_vs_h
#       - N_counts_mean_vs_h
#       - bits_eq_mean_vs_h
#       - bits_eq_vs_N_counts (scatter)
#
# No args. Just run:
#   python analyze_resolution_metrics.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = Path("ensaio_experimental")   # folder with 0cm.csv ... 10cm.csv
OUT_DIR  = Path("log_plots") / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

H_MIN, H_MAX, H_STEP = 0.0, 10.0, 0.5   # cm

# Columns we care about (must exist in your logs)
COL_DCDf   = "dCcal_pF_per_df"
COL_N      = "N_counts"
COL_BITS   = "bits_eq"

# -----------------------
# HELPERS
# -----------------------
def height_to_fname(h_cm: float) -> str:
    if abs(h_cm - round(h_cm)) < 1e-12:
        return f"{int(round(h_cm))}cm.csv"
    a = int(h_cm)
    b = int(round((h_cm - a) * 10))  # 0.5 -> 5
    return f"{a}_{b}cm.csv"

def safe_read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df[col], errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s

def save_xy_csv(x, y, out_csv: Path, xname="x", yname="y"):
    pd.DataFrame({xname: x, yname: y}).to_csv(out_csv, index=False, encoding="utf-8")

def plot_mean_with_std(h, mean, std, xlabel, ylabel, title, out_png: Path):
    plt.figure()
    plt.plot(h, mean, marker="o")
    # optional: show +-1 std as dashed
    if std is not None:
        plt.plot(h, mean + std, linestyle="--", linewidth=1.0)
        plt.plot(h, mean - std, linestyle="--", linewidth=1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_scatter(x, y, xlabel, ylabel, title, out_png: Path):
    plt.figure()
    plt.plot(x, y, linestyle="none", marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# -----------------------
# MAIN
# -----------------------
def main():
    rows = []
    scatter_bits = []
    scatter_N = []

    h_cm = H_MIN
    while h_cm <= H_MAX + 1e-9:
        h_cm = round(h_cm, 2)
        fpath = DATA_DIR / height_to_fname(h_cm)

        if not fpath.exists():
            print(f"[SKIP] missing {fpath}")
            h_cm += H_STEP
            continue

        df = safe_read_csv(fpath)

        s_dcdf = numeric_series(df, COL_DCDf)
        s_n    = numeric_series(df, COL_N)
        s_bits = numeric_series(df, COL_BITS)

        # collect scatter points (per-sample, across all heights)
        if len(s_n) and len(s_bits):
            n_common = min(len(s_n), len(s_bits))
            scatter_N.append(s_n.iloc[:n_common].to_numpy())
            scatter_bits.append(s_bits.iloc[:n_common].to_numpy())

        def stats(s: pd.Series):
            if len(s) == 0:
                return dict(N=0, mean=np.nan, std=np.nan, min=np.nan, max=np.nan)
            return dict(
                N=int(len(s)),
                mean=float(s.mean()),
                std=float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                min=float(s.min()),
                max=float(s.max()),
            )

        st_dcdf = stats(s_dcdf)
        st_n    = stats(s_n)
        st_bits = stats(s_bits)

        rows.append({
            "h_ref_cm": h_cm,

            "dCcal_pF_per_df_N": st_dcdf["N"],
            "dCcal_pF_per_df_mean": st_dcdf["mean"],
            "dCcal_pF_per_df_std": st_dcdf["std"],
            "dCcal_pF_per_df_min": st_dcdf["min"],
            "dCcal_pF_per_df_max": st_dcdf["max"],

            "N_counts_N": st_n["N"],
            "N_counts_mean": st_n["mean"],
            "N_counts_std": st_n["std"],
            "N_counts_min": st_n["min"],
            "N_counts_max": st_n["max"],

            "bits_eq_N": st_bits["N"],
            "bits_eq_mean": st_bits["mean"],
            "bits_eq_std": st_bits["std"],
            "bits_eq_min": st_bits["min"],
            "bits_eq_max": st_bits["max"],
        })

        print(f"[OK] {fpath.name}")
        h_cm += H_STEP

    if not rows:
        print("[ERR] No files processed.")
        return

    summary = pd.DataFrame(rows).sort_values("h_ref_cm").reset_index(drop=True)
    out_summary_csv = OUT_DIR / "metrics_summary.csv"
    summary.to_csv(out_summary_csv, index=False, encoding="utf-8")
    print(f"\n[OK] saved: {out_summary_csv}")

    # -----------------------
    # (1) dC/df vs h (mean ± std)
    # -----------------------
    h = summary["h_ref_cm"].to_numpy()

    dc_mean = summary["dCcal_pF_per_df_mean"].to_numpy()
    dc_std  = summary["dCcal_pF_per_df_std"].to_numpy()
    save_xy_csv(h, dc_mean, OUT_DIR / "dCcal_mean_vs_h.csv", "h_ref_cm", "dCcal_mean_pF_per_df")
    plot_mean_with_std(
        h, dc_mean, dc_std,
        xlabel="h (cm)",
        ylabel="mean dCcal/df (pF/Hz)",
        title="Sensitivity indicator: mean dCcal/df vs reference height",
        out_png=OUT_DIR / "dCcal_mean_vs_h.png"
    )

    # -----------------------
    # (2) N_counts vs h (mean ± std)
    # -----------------------
    n_mean = summary["N_counts_mean"].to_numpy()
    n_std  = summary["N_counts_std"].to_numpy()
    save_xy_csv(h, n_mean, OUT_DIR / "N_counts_mean_vs_h.csv", "h_ref_cm", "N_counts_mean")
    plot_mean_with_std(
        h, n_mean, n_std,
        xlabel="h (cm)",
        ylabel="mean N_counts (-)",
        title="Counter statistics: mean N_counts vs reference height",
        out_png=OUT_DIR / "N_counts_mean_vs_h.png"
    )

    # -----------------------
    # (3) bits_eq vs h (mean ± std)
    # -----------------------
    b_mean = summary["bits_eq_mean"].to_numpy()
    b_std  = summary["bits_eq_std"].to_numpy()
    save_xy_csv(h, b_mean, OUT_DIR / "bits_eq_mean_vs_h.csv", "h_ref_cm", "bits_eq_mean")
    plot_mean_with_std(
        h, b_mean, b_std,
        xlabel="h (cm)",
        ylabel="mean bits_eq (bits)",
        title="Effective resolution: mean bits_eq vs reference height",
        out_png=OUT_DIR / "bits_eq_mean_vs_h.png"
    )

    # -----------------------
    # (4) Scatter: bits_eq vs N_counts
    # -----------------------
    if scatter_N and scatter_bits:
        N_all = np.concatenate(scatter_N)
        B_all = np.concatenate(scatter_bits)

        # save scatter csv
        out_sc_csv = OUT_DIR / "bits_eq_vs_N_counts.csv"
        pd.DataFrame({"N_counts": N_all, "bits_eq": B_all}).to_csv(out_sc_csv, index=False, encoding="utf-8")

        plot_scatter(
            N_all, B_all,
            xlabel="N_counts (-)",
            ylabel="bits_eq (bits)",
            title="Consistency check: bits_eq vs N_counts (scatter)",
            out_png=OUT_DIR / "bits_eq_vs_N_counts.png"
        )
        print(f"[OK] saved: {out_sc_csv}")
    else:
        print("[WARN] not enough data for bits_eq vs N_counts scatter")

    print(f"\nDone. Outputs in: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
