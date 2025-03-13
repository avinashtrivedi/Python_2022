import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import cvxpy as cp

import config
import core
import os

plt.ion()
plt.close()
fig, ax = plt.subplots(figsize=(7, 7))

net = core.CLF_QP_Net(4, 256, 2)
net.load_state_dict(torch.load('./logs/model.pth'))
net.eval()

nominal_controller = core.LQR()

for i in range(20):
    rho_init = 10 #np.random.uniform(10, 12)
    theta_init = np.pi/4 #np.random.uniform(np.pi/4, np.pi/3)
    v_init = np.random.uniform(-1, 1, size=2)
    s = np.array([[np.cos(theta_init) * rho_init, 
                  np.sin(theta_init) * rho_init,
                  v_init[0], v_init[1]]], dtype=np.float32)

    s = torch.from_numpy(s)

    s_traj = []

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

        dsdt = core.dynamics_torch_noise(s, u)
        s = torch.clip(s + dsdt * config.TIME_STEP, -7, 7)

        s_traj.append(s.detach().cpu().numpy().reshape(1, -1))

        plt.clf()
        plt.scatter(s[0, 0].detach().numpy(), s[0, 1].detach().numpy(), s=100)

        ax = fig.gca()
        ax.add_patch(plt.Circle((0, 0), 4, color='r', alpha=0.3, fill=False, linewidth=4))
        ax.plot([0, -2.8], [0, -2.8], color='r', alpha=0.3, linewidth=4)
        ax.plot([0, 2.8], [0, -2.8], color='r', alpha=0.3, linewidth=4)

        plt.xlim(-13, 13)
        plt.ylim(-13, 13)

        plt.axis('off')

        fig.canvas.draw()
        plt.pause(0.01)

    s_traj = np.concatenate(s_traj, axis=0)
    f = open('clf_traj.npy', 'wb')
    np.save(f, s_traj)