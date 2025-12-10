import serial
import matplotlib
matplotlib.use("TkAgg")   # <- backend mais estável no Windows

import matplotlib.pyplot as plt
from datetime import datetime

PORT = "COM5"
BAUD = 115200
MAX_POINTS = 500
TIMEOUT = 1

ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)

fname = f"log/dados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
print("A gravar CSV em:", fname)

f = open(fname, "w", encoding="utf-8")
f.write("sample_index,elapsed_ms,freq_Hz,C_est_F\n")

plt.ion()
fig, ax = plt.subplots()

# ---- fullscreen seguro ----
# ---------------------------

time_data = []
freq_data = []

try:
    while True:
        raw = ser.readline().decode(errors="ignore").strip()
        if not raw:
            continue

        if raw.startswith("sample_index"):
            continue

        print(raw)
        f.write(raw + "\n")
        f.flush()

        parts = raw.split(",")
        if len(parts) != 4:
            continue

        try:
            elapsed_ms = float(parts[1])
            freq = float(parts[3])
        except:
            continue

        time_data.append(elapsed_ms / 1000.0)
        freq_data.append(freq)

        if len(time_data) > MAX_POINTS:
            time_data = time_data[-MAX_POINTS:]
            freq_data = freq_data[-MAX_POINTS:]

        # ---- update estável ----
        ax.clear()
        ax.plot(time_data, freq_data, marker="o")
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Frequência (Hz)")
        ax.set_title("Frequência em tempo real")
        ax.grid(True)

        fig.canvas.draw()
        fig.canvas.flush_events()
        # -------------------------

except KeyboardInterrupt:
    print("Terminou (CTRL+C).")

finally:
    f.close()
    ser.close()
