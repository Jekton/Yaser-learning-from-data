import numpy as np

def E(u, v):
    return (u * np.exp(v) - 2 * v * np.exp(-u)) ** 2


def E_derivative_u(u, v):
    return 2 * (u * np.exp(v) - 2 * v * np.exp(-u)) * (np.exp(v) + 2 * v * np.exp(-u))

def E_derivative_v(u, v):
    return 2 * (u * np.exp(v) - 2 * v * np.exp(-u)) * (u * np.exp(v) - 2 * np.exp(-u))


steps = 0
error = 1
eta = 0.1
u, v = 1, 1
while error > 1e-14:
    steps += 1
    du = E_derivative_u(u, v)
    dv = E_derivative_v(u, v)
    u -= eta * du;
    v -= eta * dv;
    error = E(u, v)

print("steps =", steps)
print((u, v))


u, v = 1, 1
for _ in range(30):
    du = E_derivative_u(u, v)
    u -= eta * du;
    dv = E_derivative_v(u, v)
    v -= eta * dv;
print("error =", E(u, v))
