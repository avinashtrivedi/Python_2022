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
num_steps = 1000
for i in range(num_testing_epochs):
    x = torch.from_numpy(np.array([-1, 0, 0, 0], dtype=np.float32).reshape(1, 4, 1))
    stage = 0
    traj = []
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

        print(u)
        
        dxdt = core.f_func_torch_noise(x) + core.B_func_torch_noise(x) * u
        x = x + dxdt * config.TIME_STEP

        base = [x[0, 0, 0], 0]
        top = [x[0, 0, 0] + np.sin(x[0, 1, 0]), np.cos(x[0, 1, 0])]

        if np.mod(j, 10) > 0:
            continue
        plt.clf()
        plt.plot([base[0], top[0]], [base[1], top[1]])
        ax = fig.gca()
        ax.add_patch(plt.Circle((0, 1), 0.1, color='r', alpha=0.3))

        plt.xlim(-2, 2)
        plt.ylim(-1, 3)

        fig.canvas.draw()
        plt.pause(0.01)

        if x[0, 0, 0] > -0.7:
            if stage == 0:
                stage = 1
        if x[0, 0, 0] > 0.7:
            if stage == 1:
                stage = 2

        if stage == 1:
            traj.append(np.array(top).reshape(1, -1))
        print(stage)

    traj = np.concatenate(traj, axis=0)
    np.save(open('./clf_traj.npy', 'wb'), traj)
