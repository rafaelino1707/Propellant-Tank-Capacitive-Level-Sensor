# fill_percent_vs_time.py
# - Reads one log CSV
# - Applies FINAL calibration mapping:
#       C_cal_map = A_MAP * C_raw + B_MAP
# - Converts C_cal_map -> h_est (cm) using theoretical inverse model
# - Converts h_est -> fill percentage (%)
# - Saves CSV and PNG for direct TikZ usage

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# USER CONFIG
# -----------------------
INPUT_CSV = Path("log/titi.csv")   # <<< MUDA SE PRECISARES
OUT_BASE  = Path("log_plots")

TIME_COL_CANDIDATES = ["elapsed_ms", "t_ms", "t_s", "sample_index"]

SMOOTH_WIN = 0   # 0 = sem smoothing, >1 = moving average

# -----------------------
# MODEL CONSTANTS (report)
# -----------------------
C_AIR_PF    = 2.66
C_WATER_PF  = 213.0
H_ACTIVE_CM = 10.0  # 100 mm

# FINAL CALIBRATION MAPPING (DO REPORT)
A_MAP = 3.08276201
B_MAP = -32.16346968

# -----------------------
# HELPERS
# -----------------------
def pick_time_column(df: pd.DataFrame) -> str:
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError("No valid time column found.")

def to_seconds(series: pd.Series, colname: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if "ms" in colname.lower():
        return s / 1000.0
    return s

def capacitance_to_height_cm(C_pf: pd.Series) -> pd.Series:
    h_cm = H_ACTIVE_CM * (C_pf - C_AIR_PF) / (C_WATER_PF - C_AIR_PF)
    return h_cm.clip(0.0, H_ACTIVE_CM)

# -----------------------
# MAIN
# -----------------------
def main():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    df.columns = [c.strip() for c in df.columns]

    if "C_cal_pF" not in df.columns:
        raise KeyError("Column C_raw_pF not found in CSV.")

    # Time axis
    tcol = pick_time_column(df)
    t_s  = to_seconds(df[tcol], tcol)

    # Numeric conversion
    C_raw = pd.to_numeric(df["C_cal_pF"], errors="coerce")

    # -----------------------
    # FINAL calibrated capacitance (REPORT-DEFINITION)
    # -----------------------
    C_cal_map = A_MAP * C_raw + B_MAP

    # Height estimate
    h_est_cm = capacitance_to_height_cm(C_cal_map)

    # Fill percentage
    fill_pct = 100.0 * h_est_cm / H_ACTIVE_CM
    fill_pct = fill_pct.clip(0.0, 100.0)

    # Optional smoothing
    if SMOOTH_WIN and SMOOTH_WIN > 1:
        fill_pct_plot = fill_pct.rolling(
            SMOOTH_WIN, center=True, min_periods=1
        ).mean()
    else:
        fill_pct_plot = fill_pct

    # -----------------------
    # OUTPUT
    # -----------------------
    out_dir = OUT_BASE / INPUT_CSV.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "fill_percent_vs_time.csv"

    out_df = pd.DataFrame({
        "t_s": t_s,
        "C_cal_pF": C_raw,
        "C_cal_map_pF": C_cal_map,
        "h_est_cm": h_est_cm,
        "fill_percent": fill_pct,
        "fill_percent_plot": fill_pct_plot,
    }).replace([np.inf, -np.inf], np.nan).dropna()

    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    # -----------------------
    # PLOT
    # -----------------------
    plt.figure()
    plt.plot(out_df["t_s"], out_df["fill_percent_plot"])
    plt.xlabel("Time (s)")
    plt.ylabel("Fill level (%)")
    plt.title("Fill percentage vs time")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    out_png = out_dir / "fill_percent_vs_time.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    print("[OK] CSV saved :", out_csv)
    print("[OK] PNG saved :", out_png)

if __name__ == "__main__":
    main()
