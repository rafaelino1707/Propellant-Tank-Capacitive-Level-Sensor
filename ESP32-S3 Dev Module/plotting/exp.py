# plot_all_heights_with_height_and_C_of_h.py
# + Análise de erro (raw vs theoretical, calibrated vs theoretical)
#   - RMSE, MAE, erro máximo, R^2
#   - erros por ponto (absoluto e relativo) guardados em CSV
#   - plots de erro vs h guardados em log_plots/

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = Path("ensaio_experimental")   # ajusta se necessário
OUT_DIR  = Path("log_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

H_MIN, H_MAX, H_STEP = 0.0, 10.0, 0.5   # cm

X_PRIMARY = ["elapsed_ms", "sample_index"]

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

# --------- Teórico do report ----------
C_AIR_PF   = 2.66
C_WATER_PF = 213.0
H_ACTIVE_CM = 10.0  # 100 mm
K_TH_PF_PER_CM = (C_WATER_PF - C_AIR_PF) / H_ACTIVE_CM

C_FOR_HEIGHT = "C_cal_pF"


# -----------------------
# HELPERS
# -----------------------
def height_to_fname(h_cm: float) -> str:
    if abs(h_cm - round(h_cm)) < 1e-9:
        return f"{int(round(h_cm))}cm.csv"
    a = int(h_cm)
    b = int(round((h_cm - a) * 10))  # 0.5 -> 5
    return f"{a}_{b}cm.csv"


def safe_read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def pick_x_column(df: pd.DataFrame) -> str:
    for c in X_PRIMARY:
        if c in df.columns:
            return c
    return ""


def save_plot_and_xy_csv(df: pd.DataFrame, xcol: str, ycol: str, title: str, out_dir: Path):
    xy = df[[xcol, ycol]].copy()
    xy = xy.replace([np.inf, -np.inf], np.nan).dropna()

    out_csv = out_dir / f"{ycol}_vs_{xcol}.csv"
    xy.to_csv(out_csv, index=False, encoding="utf-8")

    plt.figure()
    plt.plot(xy[xcol].to_numpy(), xy[ycol].to_numpy())
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    out_png = out_dir / f"{ycol}_vs_{xcol}.png"
    plt.savefig(out_png, dpi=200)
    plt.close()


def C_theoretical_pF(h_cm: np.ndarray) -> np.ndarray:
    h_cm = np.clip(h_cm, 0.0, H_ACTIVE_CM)
    return C_AIR_PF + (C_WATER_PF - C_AIR_PF) * (h_cm / H_ACTIVE_CM)


def capacitance_to_height_cm(C_pf: pd.Series) -> pd.Series:
    denom = (C_WATER_PF - C_AIR_PF)
    h_cm = H_ACTIVE_CM * (C_pf - C_AIR_PF) / denom
    return h_cm.clip(lower=0.0, upper=H_ACTIVE_CM)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {"N": 0, "RMSE": np.nan, "MAE": np.nan, "MAX_ABS": np.nan, "R2": np.nan}

    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    max_abs = float(np.max(np.abs(err)))

    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else np.nan

    return {"N": int(y_true.size), "RMSE": rmse, "MAE": mae, "MAX_ABS": max_abs, "R2": r2}


def save_error_plot(h: np.ndarray, err: np.ndarray, title: str, ylabel: str, out_path: Path):
    plt.figure()
    plt.plot(h, err, marker="o")
    plt.xlabel("h (cm)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------
# MAIN
# -----------------------
def main():
    summary_rows = []

    h_cm = H_MIN
    while h_cm <= H_MAX + 1e-9:
        h_cm = round(h_cm, 2)
        fname = height_to_fname(h_cm)
        fpath = DATA_DIR / fname

        if not fpath.exists():
            print(f"[SKIP] Não encontrei: {fpath}")
            h_cm += H_STEP
            continue

        df = safe_read_csv(fpath)
        xcol = pick_x_column(df)
        if not xcol:
            print(f"[SKIP] {fname}: sem 'elapsed_ms' nem 'sample_index'")
            h_cm += H_STEP
            continue

        h_dir = OUT_DIR / f"{h_cm:.1f}cm".replace(".", "_")
        h_dir.mkdir(parents=True, exist_ok=True)

        # 1) Plots normais (PNG + CSV)
        for y in Y_COLS:
            if y in df.columns:
                save_plot_and_xy_csv(df, xcol, y, f"{y} vs {xcol}  ({h_cm:.1f} cm)", h_dir)

        # 2) Estimar altura a partir de C
        if C_FOR_HEIGHT in df.columns:
            df[C_FOR_HEIGHT] = pd.to_numeric(df[C_FOR_HEIGHT], errors="coerce")
            df["h_est_cm"] = capacitance_to_height_cm(df[C_FOR_HEIGHT])
            save_plot_and_xy_csv(df, xcol, "h_est_cm",
                                 f"h_est_cm (from {C_FOR_HEIGHT}) vs {xcol}  ({h_cm:.1f} cm)", h_dir)
            df.to_csv(h_dir / "data_with_h_est_cm.csv", index=False, encoding="utf-8")

            c_mean = float(df[C_FOR_HEIGHT].dropna().mean())
            summary_rows.append({"h_ref_cm": h_cm, "C_mean_pF": c_mean})
        else:
            print(f"[WARN] {fname}: não encontrei {C_FOR_HEIGHT} para estimar h.")

        stats_cols = [c for c in Y_COLS if c in df.columns]
        if stats_cols:
            df[stats_cols].describe().T.to_csv(h_dir / "summary_stats.csv", encoding="utf-8")

        print(f"[OK] {fname} -> {h_dir}")
        h_cm += H_STEP

    if not summary_rows:
        print("\n[ERR] Sem dados para C(h).")
        return

    # 3) C(h) médio por ficheiro
    ch = pd.DataFrame(summary_rows).sort_values("h_ref_cm").reset_index(drop=True)
    ch.to_csv(OUT_DIR / "C_vs_h.csv", index=False, encoding="utf-8")

    plt.figure()
    plt.plot(ch["h_ref_cm"].to_numpy(), ch["C_mean_pF"].to_numpy(), marker="o")
    plt.xlabel("h (cm)")
    plt.ylabel(f"mean {C_FOR_HEIGHT} (pF)")
    plt.title("C(h): mean per file")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "C_vs_h.png", dpi=200)
    plt.close()

    # 4) raw, calibrated, theoretical + erro
    h = ch["h_ref_cm"].to_numpy()
    C_raw = ch["C_mean_pF"].to_numpy()

    # Fit raw: C = a*h + b
    a_exp, b_exp = np.polyfit(h, C_raw, 1)
    C_exp_fit = a_exp * h + b_exp

    # Teórica
    C_th = C_theoretical_pF(h)

    # Calibração afim para encaixar na teórica
    alpha = K_TH_PF_PER_CM / a_exp
    beta  = C_AIR_PF - alpha * b_exp
    C_cal = alpha * C_raw + beta

    # Guardar CSV das 3 + fit
    out_ch = pd.DataFrame({
        "h_ref_cm": h,
        "C_raw_pF": C_raw,
        "C_cal_pF": C_cal,
        "C_theoretical_pF": C_th,
        "C_exp_fit_pF": C_exp_fit,
    })
    out_ch.to_csv(OUT_DIR / "C_h_raw_cal_theoretical.csv", index=False, encoding="utf-8")

    # Plot das curvas
    plt.figure()
    plt.plot(h, C_raw, "o", label="Experimental (raw mean)")
    plt.plot(h, C_cal, "s-", label="Experimental (calibrated)")
    plt.plot(h, C_th,  "--", label="Theoretical")
    plt.plot(h, C_exp_fit, ":", label="Experimental fit (raw)")
    plt.xlabel("h (cm)")
    plt.ylabel("Capacitance (pF)")
    plt.title("C(h): raw vs calibrated vs theoretical")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "C_h_raw_cal_theoretical.png", dpi=200)

    # -------------------------------
    # ERRO: raw e calibrated vs teórica
    # -------------------------------
    err_raw = C_raw - C_th
    err_cal = C_cal - C_th

    # relativo (%), evita divisão por zero
    rel_raw = 100.0 * err_raw / np.where(np.abs(C_th) > 1e-12, C_th, np.nan)
    rel_cal = 100.0 * err_cal / np.where(np.abs(C_th) > 1e-12, C_th, np.nan)

    # Guardar CSV de erros
    err_df = pd.DataFrame({
        "h_ref_cm": h,
        "C_theoretical_pF": C_th,
        "C_raw_pF": C_raw,
        "C_cal_pF": C_cal,
        "err_raw_pF": err_raw,
        "err_cal_pF": err_cal,
        "rel_err_raw_percent": rel_raw,
        "rel_err_cal_percent": rel_cal,
    })
    err_df.to_csv(OUT_DIR / "error_analysis_C_of_h.csv", index=False, encoding="utf-8")

    # Métricas globais
    m_raw = metrics(C_th, C_raw)
    m_cal = metrics(C_th, C_cal)

    metrics_df = pd.DataFrame([
        {"case": "raw_vs_theoretical", **m_raw},
        {"case": "cal_vs_theoretical", **m_cal},
    ])
    metrics_df.to_csv(OUT_DIR / "error_metrics_summary.csv", index=False, encoding="utf-8")

    print("\n=== EQUAÇÕES ===")
    print(f"Experimental fit (raw):  C = {a_exp:.6f}·h + {b_exp:.6f}    [pF, h em cm]")
    print(f"Theoretical:             C = {K_TH_PF_PER_CM:.6f}·h + {C_AIR_PF:.6f}  [pF, h em cm]")
    print(f"Calibration mapping:     C_cal = {alpha:.8f}·C_raw + {beta:.8f}")

    print("\n=== MÉTRICAS (vs teórica) ===")
    print(f"RAW: N={m_raw['N']}  RMSE={m_raw['RMSE']:.4f} pF  MAE={m_raw['MAE']:.4f} pF  "
          f"MAX={m_raw['MAX_ABS']:.4f} pF  R2={m_raw['R2']:.6f}")
    print(f"CAL: N={m_cal['N']}  RMSE={m_cal['RMSE']:.4f} pF  MAE={m_cal['MAE']:.4f} pF  "
          f"MAX={m_cal['MAX_ABS']:.4f} pF  R2={m_cal['R2']:.6f}")

    # Plots de erro vs h
    save_error_plot(h, err_raw, "Error vs h (raw - theoretical)", "error (pF)", OUT_DIR / "err_raw_pF_vs_h.png")
    save_error_plot(h, err_cal, "Error vs h (calibrated - theoretical)", "error (pF)", OUT_DIR / "err_cal_pF_vs_h.png")
    save_error_plot(h, rel_raw, "Relative error vs h (raw - theoretical)", "relative error (%)", OUT_DIR / "rel_err_raw_percent_vs_h.png")
    save_error_plot(h, rel_cal, "Relative error vs h (calibrated - theoretical)", "relative error (%)", OUT_DIR / "rel_err_cal_percent_vs_h.png")

    plt.show()  # mostra o gráfico final das curvas (e também o último aberto)
    print(f"\nFeito. Saída em: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
