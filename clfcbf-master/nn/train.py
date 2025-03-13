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
action_net = core.ActionNet()
nominal_controller = core.PnCLQR()

params = [p for p in cbf_net.parameters()] + [p for p in action_net.parameters()]

params = [{'params': [p for p in cbf_net.parameters()], 'lr': 0.001, 'weight_decay': 1e-6},
          {'params': [p for p in action_net.parameters()], 'lr': 0.1, 'weight_decay': 1e-6}]

optimizer = torch.optim.SGD(params, momentum=0.9)

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
        rho_init = np.random.uniform(10, 12)
        theta_init = np.random.uniform(0, np.pi * 2)
        v_init = np.random.uniform(-10, 10, size=2)
        s = np.array([np.cos(theta_init) * rho_init, 
                      np.sin(theta_init) * rho_init,
                      v_init[0], v_init[1]], dtype=np.float32)
        nominal_controller.plan_trajectory(s)
        s = torch.from_numpy(s)

        h_prev = None

        for j in range(config.NUM_STEPS_PER_EPOCH):
            h, _ = cbf_net(s)
            u = action_net(s)
            print(s)
            u_nominal = torch.from_numpy(nominal_controller(s.detach().numpy()))
            loss = loss + torch.sum(u**2) * 1e-5
            
            u = u + u_nominal

            rho = np.linalg.norm(s[:2].detach().numpy())
            # The CBF barrier loss
            in_triangle = s[1] < 0 and abs(s[0]) < abs(s[1])
            if rho > 6 or rho <= 1 or in_triangle:
                loss = loss + F.relu(0.01 - h)
                h_np = h.detach().numpy()
                if h_np > 0:
                    acc_safe += 1
                count_safe += 1
            elif rho < 4 and rho > 1:
                loss = loss + F.relu(0.01 + h) * 5.0
                h_np = h.detach().numpy()
                if h_np < 0:
                    acc_dang += 1
                count_dang += 1
            else:
                pass
            dsdt = core.dynamics_torch(s, u)
            s = s + dsdt * config.TIME_STEP

            if h_prev is None:
                h_prev = h
            # The dynamics loss
            diff = h - config.CBF_ALPHA * h_prev
            loss = loss + F.relu(0.0005 - diff) * 20.0
            
            h_prev = h

            if diff.detach().numpy() > 0:
                acc_deriv += 1
            count_deriv += 1

    loss = loss / config.NUM_STEPS_PER_EPOCH

    loss.backward()

    optimizer.step()

    acc_safe = acc_safe * 1.0/ (1e-5 + count_safe)
    acc_dang = acc_dang * 1.0/ (1e-5 + count_dang)
    acc_deriv = acc_deriv * 1.0/ (1e-5 + count_deriv)

    print('{:.2f} {:.2f} {:.2f}'.format(acc_safe, acc_dang, acc_deriv))

if not os.path.exists('./runs'):
    os.mkdir('./runs')

torch.save(cbf_net.state_dict(), './runs/cbf_net.pth')
torch.save(action_net.state_dict(), './runs/action_net.pth')