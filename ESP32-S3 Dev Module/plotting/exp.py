# plot_all_heights_with_height_and_C_of_h.py
# - Para cada CSV (0cm.csv ... 10cm.csv de 0.5 em 0.5):
#   (1) Plota variáveis vs tempo e guarda PNG + CSV(x,y)
#   (2) Calcula h_est_cm a partir de C_cal_map_pF (mapeamento final) e guarda plots/CSV
# - No fim:
#   (3) Constrói C(h) com médias por altura:
#       - RAW: mean(C_raw_pF)
#       - CAL_MAP: mean(C_cal_map_pF) = a_map*C_raw + b_map
#       (opcional) também guarda mean(C_cal_pF) original do CSV como "C_fw_mean_pF"
#   (4) Linearity (best-fit e end-point) em %FS + R^2 para RAW e CAL_MAP
#       guarda linearity_summary.csv e resíduos (CSV+PNG)
#   (5) Erro vs teórico (RAW e CAL_MAP): absoluto (pF) e relativo (%) + métricas globais
#   (6) Repeatability da altura estimada por altura (CV e CD), usando h_est_cm (de CAL_MAP)

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
    "C_cal_pF",  # (firmware / intermediate) - NÃO é o "calibrated" final do report
    "df_Hz",
    "dCraw_pF_per_df",
    "dCcal_pF_per_df",
    "N_counts",
    "bits_eq",
]

# --------- Teórico do report ----------
C_AIR_PF    = 2.66
C_WATER_PF  = 213.0
H_ACTIVE_CM = 10.0  # 100 mm
K_TH_PF_PER_CM = (C_WATER_PF - C_AIR_PF) / H_ACTIVE_CM

# --------- Calibration mapping (FINAL) ----------
A_MAP = 3.08276201
B_MAP = -32.16346968

# -----------------------
# HELPERS
# -----------------------
def print_table(df: pd.DataFrame, cols, title: str, round_map=None):
    print("\n" + title)
    print("-" * len(title))
    if df.empty:
        print("(empty)")
        return
    df2 = df[cols].copy()
    if round_map:
        for c, nd in round_map.items():
            if c in df2.columns:
                df2[c] = df2[c].round(nd)
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(df2.to_string(index=False))


def height_to_fname(h_cm: float) -> str:
    if abs(h_cm - round(h_cm)) < 1e-9:
        return f"{int(round(h_cm))}cm.csv"
    a = int(h_cm)
    b = int(round((h_cm - a) * 10))
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
    if xcol not in df.columns or ycol not in df.columns:
        return
    xy = df[[xcol, ycol]].copy()
    xy = xy.replace([np.inf, -np.inf], np.nan).dropna()
    if xy.empty:
        return

    out_csv = out_dir / f"{ycol}_vs_{xcol}.csv"
    xy.to_csv(out_csv, index=False, encoding="utf-8")

    plt.figure()
    plt.plot(xy[xcol].to_numpy(), xy[ycol].to_numpy())
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / f"{ycol}_vs_{xcol}.png", dpi=200)
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


def save_error_plot(h: np.ndarray, y: np.ndarray, title: str, ylabel: str, out_path: Path):
    plt.figure()
    plt.plot(h, y, marker="o")
    plt.xlabel("h (cm)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def linearity_metrics(h, y):
    h = np.asarray(h, float)
    y = np.asarray(y, float)
    mask = np.isfinite(h) & np.isfinite(y)
    h, y = h[mask], y[mask]

    a, b = np.polyfit(h, y, 1)
    y_fit = a*h + b
    resid = y - y_fit

    FS = float(np.nanmax(y) - np.nanmin(y))
    FS = FS if FS > 1e-12 else np.nan

    bestfit_lin_pctFS = 100.0 * float(np.nanmax(np.abs(resid))) / FS

    a_ep = (y[-1] - y[0]) / (h[-1] - h[0])
    b_ep = y[0] - a_ep*h[0]
    y_ep = a_ep*h + b_ep
    resid_ep = y - y_ep
    endpoint_lin_pctFS = 100.0 * float(np.nanmax(np.abs(resid_ep))) / FS

    ss_res = float(np.sum((y - y_fit)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

    return {
        "a": a, "b": b, "R2": r2,
        "FS_pF": FS,
        "best_fit_linearity_percentFS": bestfit_lin_pctFS,
        "end_point_linearity_percentFS": endpoint_lin_pctFS
    }


# -----------------------
# MAIN
# -----------------------
def main():
    summary_rows = []   # C(h): médias por altura
    repeat_rows  = []   # repeatability: stats de h_est por altura

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
                df[y] = pd.to_numeric(df[y], errors="coerce")
                save_plot_and_xy_csv(df, xcol, y, f"{y} vs {xcol}  ({h_cm:.1f} cm)", h_dir)

        # 2) Criar C_cal_map_pF a partir de C_raw_pF e estimar h
        if "C_raw_pF" in df.columns:
            df["C_cal_pF"] = pd.to_numeric(df["C_cal_pF"], errors="coerce")
            df["C_cal_map_pF"] = A_MAP * df["C_cal_pF"] + B_MAP

            df["h_est_cm"] = capacitance_to_height_cm(df["C_cal_map_pF"])

            # guardar plots/CSV dedicados
            save_plot_and_xy_csv(df, xcol, "C_cal_map_pF",
                                 f"C_cal_map_pF (A*C_raw+B) vs {xcol}  ({h_cm:.1f} cm)", h_dir)
            save_plot_and_xy_csv(df, xcol, "h_est_cm",
                                 f"h_est_cm (from C_cal_map_pF) vs {xcol}  ({h_cm:.1f} cm)", h_dir)

            df.to_csv(h_dir / "data_with_C_cal_map_and_h_est.csv", index=False, encoding="utf-8")

            # --- stats para C(h) ---
            c_raw = df["C_cal_pF"].replace([np.inf, -np.inf], np.nan).dropna()
            c_map = df["C_cal_map_pF"].replace([np.inf, -np.inf], np.nan).dropna()

            row = {"h_ref_cm": h_cm}
            if len(c_raw) > 0:
                row["C_cal_mean_pF"] = float(c_raw.mean())
            if len(c_map) > 0:
                row["C_cal_map_mean_pF"] = float(c_map.mean())

            # opcional: guardar a média do C_cal_pF do firmware, só para debug
            if "C_cal_pF" in df.columns:
                c_fw = pd.to_numeric(df["C_cal_pF"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if len(c_fw) > 0:
                    row["C_fw_mean_pF"] = float(c_fw.mean())

            summary_rows.append(row)

            # --- repeatability de h_est ---
            h_series = df["h_est_cm"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(h_series) > 0:
                repeat_rows.append({
                    "h_ref_cm": h_cm,
                    "h_mean_cm": float(h_series.mean()),
                    "h_std_cm": float(h_series.std(ddof=1)) if len(h_series) > 1 else 0.0,
                    "h_min_cm": float(h_series.min()),
                    "h_max_cm": float(h_series.max()),
                    "N": int(len(h_series)),
                })

        else:
            print(f"[WARN] {fname}: não encontrei C_raw_pF (necessário para o mapping final).")

        # stats gerais por ficheiro
        stats_cols = [c for c in Y_COLS if c in df.columns]
        if stats_cols:
            df[stats_cols].describe().T.to_csv(h_dir / "summary_stats.csv", encoding="utf-8")

        print(f"[OK] {fname} -> {h_dir}")
        h_cm += H_STEP

    if not summary_rows:
        print("\n[ERR] Sem dados para C(h).")
        return

    # -------------------------------
    # C(h): médias por altura
    # -------------------------------
    ch = pd.DataFrame(summary_rows).sort_values("h_ref_cm").reset_index(drop=True)
    ch.to_csv(OUT_DIR / "C_vs_h.csv", index=False, encoding="utf-8")

    # prints
    print_table(
        ch,
        [c for c in ["h_ref_cm", "C_raw_mean_pF", "C_cal_map_mean_pF"] if c in ch.columns],
        "Mean capacitance per reference height (raw and CAL_MAP)",
        round_map={"C_raw_mean_pF": 3, "C_cal_map_mean_pF": 3}
    )

    # Plot C(h) raw + mapped calibrated + theoretical
    h = ch["h_ref_cm"].to_numpy()
    C_th = C_theoretical_pF(h)

    plt.figure()
    if "C_cal_mean_pF" in ch.columns:
        plt.plot(h, ch["C_cal_mean_pF"].to_numpy(), marker="o", label="Experimental (raw mean)")
    if "C_cal_map_mean_pF" in ch.columns:
        plt.plot(h, ch["C_cal_map_mean_pF"].to_numpy(), marker="s", label="Experimental (calibrated, mapping)")
    plt.plot(h, C_th, "--", label="Theoretical")
    plt.xlabel("h (cm)")
    plt.ylabel("Capacitance (pF)")
    plt.title("C(h): raw vs calibrated(mapping) vs theoretical")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "C_h_raw_calmap_theoretical.png", dpi=200)
    plt.close()

    # -------------------------------
    # Repeatability: CV / CD por altura (h_est)
    # -------------------------------
    if repeat_rows:
        rep = pd.DataFrame(repeat_rows).sort_values("h_ref_cm").reset_index(drop=True)
        eps = 1e-9
        rep["CV_percent"] = 100.0 * rep["h_std_cm"] / np.maximum(np.abs(rep["h_mean_cm"]), eps)
        rep["CD_percent"] = 100.0 * (rep["h_max_cm"] - rep["h_min_cm"]) / np.maximum(np.abs(rep["h_mean_cm"]), eps)
        rep.to_csv(OUT_DIR / "height_repeatability.csv", index=False, encoding="utf-8")

        save_error_plot(rep["h_ref_cm"].to_numpy(), rep["CV_percent"].to_numpy(),
                        "Coefficient of variation vs reference height", "CV (%)",
                        OUT_DIR / "CV_percent_vs_h.png")
        save_error_plot(rep["h_ref_cm"].to_numpy(), rep["CD_percent"].to_numpy(),
                        "Coefficient of dispersion vs reference height", "CD (%)",
                        OUT_DIR / "CD_percent_vs_h.png")

    # -------------------------------
    # Linearity (RAW e CAL_MAP) + residuals
    # -------------------------------
    lin_rows = []
    for col, name in [("C_cal_mean_pF", "raw_mean"), ("C_cal_map_mean_pF", "cal_map_mean")]:
        if col in ch.columns and ch[col].notna().any():
            y = ch[col].to_numpy()
            m = linearity_metrics(h, y)
            lin_rows.append({"case": name, **m})

            resid = y - (m["a"]*h + m["b"])
            pd.DataFrame({"h_ref_cm": h, "residual_pF": resid}).to_csv(
                OUT_DIR / f"residuals_{name}.csv", index=False, encoding="utf-8"
            )
            plt.figure()
            plt.plot(h, resid, marker="o")
            plt.axhline(0.0)
            plt.xlabel("h (cm)")
            plt.ylabel("Residual (pF)")
            plt.title(f"Linearity residuals ({name} vs best-fit line)")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"residuals_{name}.png", dpi=200)
            plt.close()

    lin_df = pd.DataFrame(lin_rows)
    lin_df.to_csv(OUT_DIR / "linearity_summary.csv", index=False, encoding="utf-8")

    if not lin_df.empty:
        print("\n=== LINEARITY (percent of full scale) ===")
        for _, r in lin_df.iterrows():
            print(f"{r['case']}: best-fit={r['best_fit_linearity_percentFS']:.3f}%FS, "
                  f"end-point={r['end_point_linearity_percentFS']:.3f}%FS, R2={r['R2']:.6f}, "
                  f"fit: C={r['a']:.6f}h+{r['b']:.6f}")

    # -------------------------------
    # Error vs theoretical (RAW e CAL_MAP)
    # -------------------------------
    err_out = {"h_ref_cm": h, "C_theoretical_pF": C_th}

    # raw errors
    if "C_cal_mean_pF" in ch.columns:
        C_raw = ch["C_cal_mean_pF"].to_numpy()
        err_raw = C_raw - C_th
        rel_raw = 100.0 * err_raw / np.where(np.abs(C_th) > 1e-12, C_th, np.nan)
        err_out.update({
            "C_cal_mean_pF": C_raw,
            "err_raw_pF": err_raw,
            "rel_err_raw_percent": rel_raw
        })
        m_raw = metrics(C_th, C_raw)

    # cal_map errors (o CORRETO para o report)
    if "C_cal_map_mean_pF" in ch.columns:
        C_calmap = ch["C_cal_map_mean_pF"].to_numpy()
        err_cal = C_calmap - C_th
        rel_cal = 100.0 * err_cal / np.where(np.abs(C_th) > 1e-12, C_th, np.nan)
        err_out.update({
            "C_cal_map_mean_pF": C_calmap,
            "err_cal_pF": err_cal,
            "rel_err_cal_percent": rel_cal
        })
        m_cal = metrics(C_th, C_calmap)

        # plots de erro (cal_map)
        save_error_plot(h, err_cal, "Absolute error vs h (calibrated mapping mean - theoretical)",
                        "error (pF)", OUT_DIR / "err_calmap_pF_vs_h.png")
        save_error_plot(h, rel_cal, "Relative error vs h (calibrated mapping mean - theoretical)",
                        "relative error (%)", OUT_DIR / "rel_err_calmap_percent_vs_h.png")

    pd.DataFrame(err_out).to_csv(OUT_DIR / "error_analysis_C_of_h.csv", index=False, encoding="utf-8")

    # resumo de métricas
    metrics_rows = []
    if "C_cal_mean_pF" in ch.columns:
        metrics_rows.append({"case": "raw_mean_vs_theoretical", **m_raw})
    if "C_cal_map_mean_pF" in ch.columns:
        metrics_rows.append({"case": "cal_map_mean_vs_theoretical", **m_cal})
    pd.DataFrame(metrics_rows).to_csv(OUT_DIR / "error_metrics_summary.csv", index=False, encoding="utf-8")

    print("\n=== EQUATIONS (for report) ===")
    print(f"Experimental fit (raw):  C = 6.823102·h + 11.296191    [pF, h in cm]")
    print(f"Theoretical:             C = {K_TH_PF_PER_CM:.6f}·h + {C_AIR_PF:.6f}  [pF, h in cm]")
    print(f"Calibration mapping:     C_cal = {A_MAP:.8f}·C_raw + {B_MAP:.8f}")

    if "C_cal_mean_pF" in ch.columns:
        print("\n=== ERROR METRICS vs THEORETICAL (RAW mean) ===")
        print(f"N={m_raw['N']}  RMSE={m_raw['RMSE']:.4f} pF  MAE={m_raw['MAE']:.4f} pF  "
              f"MAX={m_raw['MAX_ABS']:.4f} pF  R2={m_raw['R2']:.6f}")

    if "C_cal_map_mean_pF" in ch.columns:
        print("\n=== ERROR METRICS vs THEORETICAL (CALIBRATED mapping mean) ===")
        print(f"N={m_cal['N']}  RMSE={m_cal['RMSE']:.4f} pF  MAE={m_cal['MAE']:.4f} pF  "
              f"MAX={m_cal['MAX_ABS']:.4f} pF  R2={m_cal['R2']:.6f}")

    print(f"\nFeito. Saída em: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
