import matplotlib.pyplot as plt

filename = "log/aaaapF.csv"

time_s = []
freq = []
cmes=[]

with open(filename, "r", encoding="utf-8") as f:
    header = f.readline()  # lê cabeçalho e ignora

    for raw in f:
        line = raw.strip()

        if not line:
            continue  # linha vazia

        parts = line.split(",")

        if len(parts) != 5:
            print("Linha inválida:", line)
            continue

        try:
            elapsed_ms = float(parts[1])
            freq_Hz    = float(parts[2])
            cmeas = float(parts[4])
        except ValueError:
            print("Erro a converter linha:", line)
            continue

        time_s.append(elapsed_ms / 1000.0)
        freq.append(freq_Hz)
        cmes.append(cmeas*1e12)

# Debug: ver listas
print("time_s:", time_s)
print("freq:", freq)
print(cmes)

# Plot
plt.figure(figsize=(8,4))
plt.plot(time_s, freq, color="r")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.ylim(0, 5000)
plt.title("Frequency [Hz] vs Time [s]")
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,4))
plt.plot(time_s, cmes, color="g")
plt.xlabel("Time [s]")
plt.ylabel("Capacitance [pF]")
plt.ylim(0, 200)
plt.title("Capacitance [pF] vs Time [s]")
plt.grid(True)
plt.tight_layout()
plt.show()