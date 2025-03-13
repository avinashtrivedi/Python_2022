import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import cvxpy as cp
import core
import os
import config
import time


net = core.CLF_QP_Net(n_input=4, n_hidden=64, n_controls=1)
net.load_state_dict(torch.load('./save/model.pth'))
net.eval()
nominal_controller = core.Segway_LQR()

is_safe_list = []
goal_reaching_error = []

num_testing_epochs = 1
num_steps = 1000
t1 = time.time()
for i in range(num_testing_epochs):
    x = torch.from_numpy(np.array([-1, 0, 0, 0], dtype=np.float32).reshape(1, 4, 1))
    is_safe = True
    mass_noise = np.random.uniform(0.9, 1.3)
    inertia_noise = np.random.uniform(0.9, 1.3)
    for j in range(num_steps):
        V, grad_V = net.compute_lyapunov(x)
        grad_V_np = np.squeeze(grad_V.detach().numpy())
        V_np = np.squeeze(V.detach().numpy())
        f, B = core.segway_dynamics_torch(x)
        f_np = np.squeeze(f.detach().numpy())
        B_np = np.squeeze(B.detach().numpy())

        u_nominal = nominal_controller(x)
        u_nominal_np = np.squeeze(u_nominal.detach().numpy())
        u_nominal_np = u_nominal_np - 3

        u = cp.Variable()

        clf_cond = grad_V_np @ (f_np + B_np * u) + 20 * (V_np - 1)
        constraint = [clf_cond <= 0.5]

        objective = cp.Minimize(cp.sum_squares(u-u_nominal_np))

        cp.Problem(objective, constraint).solve()
        u = u.value
        
        dxdt = core.f_func_torch_noise(x, mass_noise, inertia_noise) + core.B_func_torch_noise(x, mass_noise, inertia_noise) * u
        x = x + dxdt * config.TIME_STEP

        base = [x[0, 0, 0], 0]
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
print('time = {:.4f}'.format(t2-t1))