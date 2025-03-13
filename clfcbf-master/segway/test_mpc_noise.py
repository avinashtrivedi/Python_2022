import numpy as np
import matplotlib.pyplot as plt
import core
import os
import config


plt.ion()
plt.close()
fig, ax = plt.subplots(figsize=(7, 7))

num_testing_epochs = 10
num_steps = 1500
for i in range(num_testing_epochs):
    x = np.array([-2, 0, 0, 0], dtype=np.float32).reshape(1, 4, 1)
    stage = 0
    traj = []
    for j in range(num_steps):
        
        u = core.mpc_solver(x.squeeze())
        
        dxdt = core.f_func_np_noise(x) + core.B_func_np_noise(x) * u
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
        plt.pause(0.001)

        if x[0, 0, 0] > -0.7:
            if stage == 0:
                stage = 1
        if x[0, 0, 0] > 0.7:
            if stage == 1:
                stage = 2

        if stage == 1:
            traj.append(np.reshape(top, (1, -1)))
        print(stage, j)

    traj = np.concatenate(traj, axis=0)
    np.save(open('./mpc_traj.npy', 'wb'), traj)