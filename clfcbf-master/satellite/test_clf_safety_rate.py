import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import cvxpy as cp

import config
import core
import os
import time


net = core.CLF_QP_Net(4, 256, 2)
net.load_state_dict(torch.load('./logs/model.pth'))
net.eval()

is_safe_list = []
goal_reaching_error = []

nominal_controller = core.LQR()

t1 = time.time()
for i in range(1):
    rho_init = 10
    theta_init = np.pi/4
    v_init = np.random.uniform(-1, 1, size=2)
    s = np.array([[np.cos(theta_init) * rho_init, 
                  np.sin(theta_init) * rho_init,
                  v_init[0], v_init[1]]], dtype=np.float32)

    s = torch.from_numpy(s)
    M_noise = np.random.uniform(1.0, 1.4)
    N_noise = np.random.uniform(1.0, 1.4)

    for j in range(config.NUM_STEPS_PER_EPOCH):

        u_nominal = nominal_controller(np.squeeze(s.detach().numpy()))
        
        s_np = np.squeeze(s.numpy())
        if s_np[1] < 0 and abs(s_np[0]) < abs(s_np[1]) or np.linalg.norm(s_np[:2]) < 2:
            u = torch.from_numpy(np.expand_dims(u_nominal, 0).astype(np.float32))

        else:
            V, grad_V = net.compute_lyapunov(s)
            grad_V_np = np.squeeze(grad_V.detach().numpy())
            V_np = np.squeeze(V.detach().numpy())

            s_np = s.numpy()
            u = cp.Variable(2)
            clf_cond = grad_V_np @ (core.A.dot(s_np[0]) + core.B @ u) + 0.005 * V_np
            constraint = [clf_cond <= 0]

            objective = cp.Minimize(cp.sum_squares(u-u_nominal))

            cp.Problem(objective, constraint).solve()
            u = u.value
            u = torch.from_numpy(np.expand_dims(u, 0).astype(np.float32))

        dsdt = core.dynamics_torch_noise(s, u, M_noise=M_noise, N_noise=N_noise)
        s = torch.clip(s + dsdt * config.TIME_STEP, -7, 7)

    s_np = np.squeeze(s.numpy())
    error = np.linalg.norm(s_np[:2])
    if error < 1.0:
        is_safe_list.append(True)
    else:
        is_safe_list.append(False)
    goal_reaching_error.append(error)

    print('epoch {}, safety rate {:.4f}, goal reaching error {:.4f}'.format(i, np.mean(is_safe_list), np.mean(goal_reaching_error)))

t2 = time.time()
print('time = {:.4f}'.format(t2-t1))