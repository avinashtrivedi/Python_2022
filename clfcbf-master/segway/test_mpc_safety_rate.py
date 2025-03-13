import numpy as np
import matplotlib.pyplot as plt
import core
import os
import config
import time

is_safe_list = []
goal_reaching_error = []

num_testing_epochs = 1
num_steps = 1500
t1 = time.time()
for i in range(num_testing_epochs):
    x = np.array([-2, 0, 0, 0], dtype=np.float32).reshape(1, 4, 1)
    mass_noise = np.random.uniform(0.9, 1.3)
    inertia_noise = np.random.uniform(0.9, 1.3)
    is_safe = True
    for j in range(num_steps):
        u = core.mpc_solver(x.squeeze())
        dxdt = core.f_func_np_noise(x, mass_noise, inertia_noise) + core.B_func_np_noise(x, mass_noise, inertia_noise) * u
        x = x + dxdt * config.TIME_STEP

        top = [x[0, 0, 0] + np.sin(x[0, 1, 0]), np.cos(x[0, 1, 0])]
        if top[0]**2 + (top[1] - 1)**2 < 0.07**2:
            is_safe = False
            break
        if abs(x[0, 0, 0] - 1.0) < 0.05:
            break
    goal_reaching_error.append(abs(x[0, 0, 0] - 1.0))
    is_safe_list.append(is_safe)

    print('epoch {}, safety rate {:.4f}, goal reaching error {:.4f}'.format(i, np.mean(is_safe_list), np.mean(goal_reaching_error)))

t2 = time.time()
print(f'time = {t2-t1}')