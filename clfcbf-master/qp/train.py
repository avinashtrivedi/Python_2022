import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import config
import core
import os


#plt.ion()
#plt.close()
#fig = plt.figure(figsize=(7, 5))

cbf_net = core.CBFNet()

optimizer = torch.optim.SGD(cbf_net.parameters(), lr=0.001, momentum=0.9)

for i in range(config.NUM_TRAINING_EPOCHS):
    
    optimizer.zero_grad()
    loss = 0.0

    acc_safe = 0
    acc_dang = 0
    acc_deriv = 0
    count_safe = 0
    count_dang = 0
    count_deriv = 0

    for j in range(4):
        rho_init = np.random.uniform(8, 9)
        theta_init = np.random.uniform(0, np.pi * 2)
        v_init = np.random.uniform(-10, 10, size=2)
        s = np.array([np.cos(theta_init) * rho_init, 
                      np.sin(theta_init) * rho_init,
                      v_init[0], v_init[1]], dtype=np.float32)

        for j in range(config.NUM_STEPS_PER_EPOCH):
            h, par_h_par_s, u = cbf_net(torch.from_numpy(s))
            rho = np.linalg.norm(s[:2])
            in_triangle = s[1] < 0 and abs(s[0]) < abs(s[1])
            if rho > 6 or rho <= 1 or in_triangle:
                loss = loss + F.relu(0.01 - h)
                h_np = h.detach().numpy()
                if h_np > 0:
                    acc_safe += 1
                count_safe += 1
            elif rho < 4 and rho > 1:
                loss = loss + F.relu(0.01 + h) * 4.0
                h_np = h.detach().numpy()
                if h_np < 0:
                    acc_dang += 1
                count_dang += 1
            else:
                pass

            dsdt = core.dynamics_np(s, u)
            cbf_third_cond = torch.matmul(
                par_h_par_s, torch.from_numpy(dsdt)) + config.CBF_ALPHA * h.detach().numpy()
            loss = loss + F.relu(0.01 - cbf_third_cond) * 0.01

            if cbf_third_cond.detach().numpy() > 0:
                acc_deriv += 1
            count_deriv += 1

            s = np.array(s + dsdt * config.TIME_STEP, dtype=np.float32)

    loss = loss / config.NUM_STEPS_PER_EPOCH

    loss.backward()

    optimizer.step()

    print('{:.2f} {:.2f} {:.2f}'.format(acc_safe*1.0/count_safe, acc_dang* 1.0/count_dang, acc_deriv*1.0/count_deriv))


if not os.path.exists('./runs'):
    os.mkdir('./runs')

torch.save(cbf_net.state_dict(), './runs/cbf_net.pth')
