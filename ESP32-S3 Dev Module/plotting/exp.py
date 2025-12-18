# plot_all_heights_with_plot_csvs.py
# Para cada altura: guarda PNGs + um CSV por plot (x,y) em log_plots/<altura>/

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = Path("ensaio_experimental")   # ajusta se necessário
OUT_DIR  = Path("log_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

H_MIN, H_MAX, H_STEP = 0.0, 10.0, 0.5

# Eixo X preferido
X_PRIMARY = ["elapsed_ms", "sample_index"]

# Variáveis a plotar (plota só as que existirem no CSV)
Y_COLS = [
    "f_meas_Hz",
    "f_from_Craw_Hz",
    "f_from_Ccal_Hz",
    "C_raw_pF",
    "C_cal_pF",
    "df_Hz",
    "dCraw_pF_per_df",
    "dCcal_pF_per_df",
    "N_counts",
    "bits_eq",
]

def height_to_fname(h: float) -> str:
    # 0.0 -> 0cm.csv ; 0.5 -> 0_5cm.csv ; 1.5 -> 1_5cm.csv
    if abs(h - round(h)) < 1e-9:
        return f"{int(round(h))}cm.csv"
    a = int(h)
    b = int(round((h - a) * 10))  # 0.5 -> 5
    return f"{a}_{b}cm.csv"

def pick_x_column(df: pd.DataFrame) -> str:
    for c in X_PRIMARY:
        if c in df.columns:
            return c
    return ""

def safe_read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]  # limpa espaços nos headers
    return df

def save_plot_and_xy_csv(df: pd.DataFrame, xcol: str, ycol: str, h: float, out_dir: Path):
    # CSV (x,y)
    xy = df[[xcol, ycol]].copy()
    xy.dropna(inplace=True)

    out_csv = out_dir / f"{ycol}_vs_{xcol}.csv"
    xy.to_csv(out_csv, index=False, encoding="utf-8")

    # PNG
    plt.figure()
    plt.plot(xy[xcol].to_numpy(), xy[ycol].to_numpy())
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(f"{ycol} vs {xcol}  ({h:.1f} cm)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    out_png = out_dir / f"{ycol}_vs_{xcol}.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    h = H_MIN
    while h <= H_MAX + 1e-9:
        h = round(h, 2)
        fname = height_to_fname(h)
        fpath = DATA_DIR / fname

        if not fpath.exists():
            print(f"[SKIP] Não encontrei: {fpath}")
            h += H_STEP
            continue

        df = safe_read_csv(fpath)
        xcol = pick_x_column(df)
        if not xcol:
            print(f"[SKIP] {fname}: sem 'elapsed_ms' nem 'sample_index'")
            h += H_STEP
            continue

        # pasta da altura
        h_dir = OUT_DIR / f"{h:.1f}cm".replace(".", "_")
        h_dir.mkdir(parents=True, exist_ok=True)

        # guarda PNG + CSV de cada plot
        any_saved = False
        for y in Y_COLS:
            if y in df.columns:
                save_plot_and_xy_csv(df, xcol, y, h, h_dir)
                any_saved = True

        # opcional: estatísticas
        stats_cols = [c for c in Y_COLS if c in df.columns]
        if stats_cols:
            df[stats_cols].describe().T.to_csv(h_dir / "summary_stats.csv", encoding="utf-8")

        if any_saved:
            print(f"[OK] {fname} -> {h_dir}")
        else:
            print(f"[WARN] {fname}: nenhuma coluna de Y_COLS encontrada")

        h += H_STEP

    print(f"\nFeito. Saída em: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
