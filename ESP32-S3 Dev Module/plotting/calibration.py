import numpy as np

# Põe aqui os teus valores médios, MESMA unidade (por ex. pF)

C_nom = np.array([150, 68, 100, 221, 47, 33, 15, 10, 15, 20, 25], dtype=float)
C_est = np.array([164.89, 85.63, 119.83, 233.97, 64.67, 49.95, 31.11, 26.37, 31.02, 39, 41.38], dtype=float)

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
