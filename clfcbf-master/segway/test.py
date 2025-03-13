import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import cvxpy as cp
import core
import os
import config


net = core.CLF_QP_Net(n_input=4, n_hidden=64, n_controls=1)
net.load_state_dict(torch.load('./save/model.pth'))
net.eval()
nominal_controller = core.Segway_LQR()

plt.ion()
plt.close()
fig, ax = plt.subplots(figsize=(7, 7))

num_testing_epochs = 10
num_steps = 10000
for i in range(num_testing_epochs):
    x = torch.from_numpy(np.array([-1, 0, 0, 0], dtype=np.float32).reshape(1, 4, 1))
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
        
        dxdt = core.f_func_torch_noise(x, 1.0, 1.0) + core.B_func_torch_noise(x, 1.0, 1.0) * u
        x = x + dxdt * config.TIME_STEP

        base = [x[0, 0, 0], 0]
        top = [x[0, 0, 0] + np.sin(x[0, 1, 0]), np.cos(x[0, 1, 0])]

        if np.mod(j, 5) > 0:
            continue
        plt.clf()
        plt.plot([base[0], top[0]], [base[1], top[1]], color='black', alpha=0.7, linewidth=3)
        ax = fig.gca()
        ax.add_patch(plt.Circle((0, 1), 0.08, color='black', alpha=0.5, linewidth=3))

        plt.xlim(-1, 1)
        plt.ylim(0, 2)

        fig.canvas.draw()
        plt.pause(0.01)
 