import numpy as np

# Põe aqui os teus valores médios, MESMA unidade (por ex. pF)
C_nom = np.array([10, 15, 20, 33, 35, 47, 68, 100, 150, 220], dtype=float)      # nominal
C_est = np.array([26, 30, 36, 49, 50, 63, 85, 117, 163, 230], dtype=float)  # medias medidas

# regressão: C_est = m*C_nom + c
N   = len(C_nom)
Sx  = np.sum(C_nom)
Sy  = np.sum(C_est)
Sxx = np.sum(C_nom**2)
Sxy = np.sum(C_nom * C_est)

m = (N*Sxy - Sx*Sy) / (N*Sxx - Sx**2)
c = (Sy - m*Sx) / N

print("m =", m)
print("c =", c)

# verifica como fica a inversa:
C_est_test = C_est
C_real_est = (C_est_test - c) / m
print("nominal:", C_nom)
print("recovered:", C_real_est)
