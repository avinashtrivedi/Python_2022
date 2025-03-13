import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import config
import core
import os


plt.ion()
plt.close()
fig, ax = plt.subplots(figsize=(7, 7))

cbf_net = core.CBFNetTest()
cbf_net.load_state_dict(torch.load('./runs/cbf_net.pth'))
cbf_net.eval()

for i in range(config.NUM_TRAINING_EPOCHS):

    rho_init = np.random.uniform(10, 12)
    theta_init = np.random.uniform(0, np.pi * 2)
    v_init = np.random.uniform(-10, 10, size=2)
    s = np.array([np.cos(theta_init) * rho_init, 
                      np.sin(theta_init) * rho_init,
                      v_init[0], v_init[1]], dtype=np.float32)

    acc_safe = 0
    acc_dang = 0
    acc_deriv = 0
    count_safe = 0
    count_dang = 0
    count_deriv = 0

    for j in range(config.NUM_STEPS_PER_EPOCH):
        h, par_h_par_s, u = cbf_net(torch.from_numpy(s))
        rho = np.linalg.norm(s[:2])
        in_triangle = s[1] < 0 and abs(s[0]) < abs(s[1])
        if rho > 6 or rho <= 1 or in_triangle:
            h_np = h.detach().numpy()
            if h_np > 0:
                acc_safe += 1
            count_safe += 1
        elif rho < 4 and rho > 1:
            h_np = h.detach().numpy()
            if h_np < 0:
                acc_dang += 1
            count_dang += 1
        else:
            pass

        dsdt = dsdt = core.dynamics_np(s, u)
        s = s + dsdt * config.TIME_STEP
        s = s.astype(np.float32)

        cbf_third_cond = torch.matmul(
                par_h_par_s, torch.from_numpy(dsdt)) + config.CBF_ALPHA * h.detach().numpy()

        plt.clf()
        plt.scatter(s[0], s[1], s=100)

        ax = fig.gca()
        ax.add_patch(plt.Circle((0, 0), 4, color='r', alpha=0.3))
        ax.add_patch(plt.Circle((0, 0), 6, color='blue', alpha=0.3, fill=False))
        plt.xlim(-13, 13)
        plt.ylim(-13, 13)

        plt.axis('off')

        fig.canvas.draw()
        plt.pause(0.01)

    acc_safe = acc_safe * 1.0/ (1e-5 + count_safe)
    acc_dang = acc_dang * 1.0/ (1e-5 + count_dang)
    print('{:.2f} {:.2f} {:.2f}'.format(acc_safe, acc_dang, acc_deriv))