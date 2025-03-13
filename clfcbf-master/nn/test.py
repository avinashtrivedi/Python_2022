import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import config
import core
import os

plt.ion()
plt.close()
fig, ax = plt.subplots(figsize=(7, 7))

cbf_net = core.CBFNet()
cbf_net.load_state_dict(torch.load('./runs/cbf_net.pth'))
cbf_net.eval()

action_net = core.ActionNet()
action_net.load_state_dict(torch.load('./runs/action_net.pth'))
action_net.eval()

nominal_controller = core.PnCLQR()

params = [p for p in cbf_net.parameters()] + [p for p in action_net.parameters()]

for i in range(config.NUM_TRAINING_EPOCHS):

    rho_init = np.random.uniform(10, 12)
    theta_init = np.random.uniform(np.pi/3, 2*np.pi/3)
    v_init = np.random.uniform(-10, 10, size=2)
    s = np.array([np.cos(theta_init) * rho_init, 
                      np.sin(theta_init) * rho_init,
                      v_init[0], v_init[1]], dtype=np.float32)
    nominal_controller.plan_trajectory(s)
    s = torch.from_numpy(s)

    acc_safe = 0
    acc_dang = 0
    acc_deriv = 0
    count_safe = 0
    count_dang = 0
    count_deriv = 0

    for j in range(config.NUM_STEPS_PER_EPOCH):
        h, par_h_par_s = cbf_net(s)
        u = action_net(s)
        #u_nominal = torch.from_numpy(nominal_controller(s.detach().numpy()))
        #u = u + u_nominal

        rho = np.linalg.norm(s[:2].detach().numpy())
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

        u_nominal = torch.from_numpy(nominal_controller(s.detach().numpy()))
        if rho > 6:
            u = u + u_nominal
        else:
            u = u + u_nominal * 0.3

        noise = np.random.normal(size=4)
        dsdt = core.dynamics_torch(s, u) + torch.from_numpy(noise.astype(np.float32))

        s_next = s + dsdt * config.TIME_STEP
        h_next, par_h_par_s_next = cbf_net(s_next)

    
        k = 0
        while h_next - config.CBF_ALPHA * h < 0 and k < 200:
            # print(h_next - config.CBF_ALPHA * h)
            u = u + torch.matmul(par_h_par_s_next, core.B_torch) * 10
            dsdt = core.dynamics_torch(s, u)
            s_next = s + dsdt * config.TIME_STEP
            h_next, par_h_par_s_next = cbf_net(s_next)
            k = k + 1

        
        s = s_next

        plt.clf()
        plt.scatter(s[0].detach().numpy(), s[1].detach().numpy(), s=100)

        ax = fig.gca()
        ax.add_patch(plt.Circle((0, 0), 4, color='r', alpha=0.3))
        ax.add_patch(plt.Circle((0, 0), 6, color='blue', alpha=0.3, fill=False))

        ax.scatter(nominal_controller.traj[:, 0], nominal_controller.traj[:, 1])

        plt.xlim(-13, 13)
        plt.ylim(-13, 13)

        plt.axis('off')

        fig.canvas.draw()
        plt.pause(0.01)

    acc_safe = acc_safe * 1.0/ (1e-5 + count_safe)
    acc_dang = acc_dang * 1.0/ (1e-5 + count_dang)
    print('{:.2f} {:.2f} {:.2f}'.format(acc_safe, acc_dang, acc_deriv))