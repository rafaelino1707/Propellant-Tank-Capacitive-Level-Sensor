import matplotlib.pyplot as plt

filename = "log/dados_20251206_020303.csv"

time_s = []
freq = []

with open(filename, "r", encoding="utf-8") as f:
    header = f.readline()  # lê cabeçalho e ignora

    for raw in f:
        line = raw.strip()

        if not line:
            continue  # linha vazia

        parts = line.split(",")

        if len(parts) != 4:
            print("Linha inválida:", line)
            continue

        try:
            elapsed_ms = float(parts[1])
            freq_Hz    = float(parts[3])
        except ValueError:
            print("Erro a converter linha:", line)
            continue

        time_s.append(elapsed_ms / 1000.0)
        freq.append(freq_Hz)

# Debug: ver listas
print("time_s:", time_s)
print("freq:", freq)

# Plot
plt.figure(figsize=(8,4))
plt.plot(time_s, freq, color="r")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.ylim(0, 1500)
plt.title("Frequency [Hz] vs Time [s]")
plt.grid(True)
plt.tight_layout()
plt.show()
