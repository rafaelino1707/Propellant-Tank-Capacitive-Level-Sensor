import serial
import os
from datetime import datetime

# -----------------------------------
# Porta série
# -----------------------------------
PORT = "COM5"
BAUD = 115200
TIMEOUT = 1

ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)

# -----------------------------------
# CSV
# -----------------------------------
os.makedirs("log", exist_ok=True)
fname = f"log/dados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
print("A gravar CSV em:", fname)

HEADER = (
    "sample_index,elapsed_ms,"
    "f_meas_Hz,f_from_Craw_Hz,f_from_Ccal_Hz,"
    "C_raw_pF,C_cal_pF,"
    "Tgate_s,df_Hz,"
    "dCraw_pF_per_df,dCcal_pF_per_df,"
    "N_counts,bits_eq"
)

f = open(fname, "w", encoding="utf-8")
f.write(HEADER + "\n")
f.flush()

try:
    while True:
        raw = ser.readline().decode(errors="ignore").strip()
        if not raw:
            continue

        # ignora headers / mensagens
        if raw.startswith("sample_index") or raw.startswith("==="):
            continue

        # opcional: garante nº de colunas
        parts = raw.split(",")
        if len(parts) != 13:
            # se quiseres ver o que está a falhar:
            print("IGNORED (cols != 13):", raw)
            continue

        print(raw)
        f.write(raw + "\n")
        f.flush()

except KeyboardInterrupt:
    print("Terminou (CTRL+C).")

finally:
    f.close()
    ser.close()
