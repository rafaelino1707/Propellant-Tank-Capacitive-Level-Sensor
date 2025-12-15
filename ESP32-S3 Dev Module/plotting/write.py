import serial
import os
from datetime import datetime

# -----------------------------------
# Configuração da porta série
# -----------------------------------
PORT = "COM5"
BAUD = 115200
TIMEOUT = 1  # segundos

ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)

# -----------------------------------
# Nome e pasta do ficheiro CSV
# -----------------------------------
os.makedirs("log", exist_ok=True)
#fname = f"log/dados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
fname = f"log/aabapF.csv"
print("A gravar CSV em:", fname)

f = open(fname, "w", encoding="utf-8")

# Se quiseres forçar um header fixo, descomenta esta linha:
# f.write("sample_index,elapsed_ms,freq_Hz,C_est_F\n")

try:
    while True:
        # Lê uma linha da série
        raw = ser.readline().decode(errors="ignore").strip()
        if not raw:
            continue

        # Opcional: ignorar linha de header vinda da ESP32
        if raw.startswith("sample_index"):
            # Se quiseres escrever o header vindo da ESP32 no ficheiro,
            # podes fazer:
            # f.write(raw + "\n")
            # f.flush()
            continue

        print(raw)          # eco no terminal (podes comentar se não quiseres)
        f.write(raw + "\n") # grava no CSV
        f.flush()           # força escrita imediata em disco

except KeyboardInterrupt:
    print("Terminou (CTRL+C).")

finally:
    f.close()
    ser.close()
