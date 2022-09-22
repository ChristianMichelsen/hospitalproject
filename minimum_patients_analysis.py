import numpy as np

# E = data_all['y'].sum()

E = 1476
n = 22017
phi = E / n
P = 33
MAPE = 0.05

B1 = (1.96 / 0.05) ** 2 * phi * (1 - phi)
B1

B2 = np.exp((-0.508 + 0.259 * np.log(phi) + 0.504 * np.log(P) - np.log(MAPE)) / 0.544)
B2

ln_L_null = E * np.log(E / n) + (n - E) * np.log((n - E) / n)
max_R2_CS = 1 - np.exp(2 * ln_L_null / n)
max_R2_CS

frac_variability = 0.20
R2_anticipated = frac_variability * max_R2_CS
S = 0.9

B3 = P / ((S - 1) * np.log(1 - R2_anticipated / S))
B3
