import numpy as np
import matplotlib.pyplot as plt
import core
import os
import config
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


plt.ion()
plt.close()
fig, ax = plt.subplots(figsize=(7, 7))

num_testing_epochs = 10
num_steps = 10000
for i in range(num_testing_epochs):
    x = np.array([-1, 0, 0, 0], dtype=np.float32).reshape(1, 4, 1)
    for j in range(num_steps):
        
        u = core.mpc_solver(x.squeeze())
        
        dxdt = core.f_func_np_noise(x, 1.0, 1.0) + core.B_func_np_noise(x, 1.0, 1.0) * u
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
