import numpy as np

# Põe aqui os teus valores médios, MESMA unidade (por ex. pF)
C_nom = np.array([10, 47, 100, 220], dtype=float)      # nominal
C_est = np.array([C10m, C47m, C100m, C220m], dtype=float)  # medias medidas

N = len(C_nom)
Sx  = np.sum(C_est)
Sy  = np.sum(C_nom)
Sxx = np.sum(C_est**2)
Sxy = np.sum(C_est * C_nom)

a = (N*Sxy - Sx*Sy)/(N*Sxx - Sx**2)
b = (Sy - a*Sx)/N

print("a =", a)
print("b =", b)

# fator simples (sem offset), se quiseres comparar
k = np.sum(C_nom * C_est) / np.sum(C_est**2)
print("k =", k)
