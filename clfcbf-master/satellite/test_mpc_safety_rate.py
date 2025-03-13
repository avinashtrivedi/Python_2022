import numpy as np
import matplotlib.pyplot as plt

import config
import core
import os
import time


nominal_controller = core.LQR()

is_safe_list = []
goal_reaching_error = []

t1 = time.time()
for i in range(1):
    rho_init = 10
    theta_init = np.pi/4
    v_init = np.random.uniform(-1, 1, size=2)
    s = np.array([[np.cos(theta_init) * rho_init, 
                  np.sin(theta_init) * rho_init,
                  v_init[0], v_init[1]]], dtype=np.float32)

    M_noise = np.random.uniform(1.0, 1.4)
    N_noise = np.random.uniform(1.0, 1.4)

    for j in range(config.NUM_STEPS_PER_EPOCH):
        u_nominal = nominal_controller(np.squeeze(s))

        if s[0, 1] < 0 and abs(s[0, 0]) < abs(s[0, 1]) or np.linalg.norm(s[0, :2]) < 1:
            u = u_nominal

        else:
            u = core.mpc_solver(np.squeeze(s))
        u = np.clip(u, -20, 20)

        dsdt = core.dynamics_np_noise(s, u[np.newaxis], M_noise=M_noise, N_noise=N_noise)
        s = s + dsdt * config.TIME_STEP * 2
        s = np.clip(s, -7, 7)

    s_np = np.squeeze(s)
    error = np.linalg.norm(s_np[:2])
    if error < 1.0:
        is_safe_list.append(True)
    else:
        is_safe_list.append(False)
    goal_reaching_error.append(error)

    print('epoch {}, safety rate {:.4f}, goal reaching error {:.4f}'.format(i, np.mean(is_safe_list), np.mean(goal_reaching_error)))

t2 = time.time()
print('time = {:.4f}'.format(t2-t1))