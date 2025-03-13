import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import config
from tqdm import tqdm
import core
import os
import matplotlib.patches as patches


# plt.ion()
# plt.close()
# fig, ax = plt.subplots(figsize=(7, 7))

nominal_controller = core.LQR()
mpc_data = []
for i in tqdm(range(100)):
#     xx = np.array([])
    rho_init = 10 #np.random.uniform(10, 12)
    theta_init = np.pi/4 #np.random.uniform(np.pi/4, np.pi/3)
    v_init = np.random.uniform(-1, 1, size=2)
    s = np.array([[np.cos(theta_init) * rho_init, 
                  np.sin(theta_init) * rho_init,
                  v_init[0], v_init[1]]], dtype=np.float32)

    for j in tqdm(range(config.NUM_STEPS_PER_EPOCH)):
        
        u_nominal = nominal_controller(np.squeeze(s))

        use_nominal = False
        if s[0, 1] < 0 and abs(s[0, 0]) < abs(s[0, 1]) or np.linalg.norm(s[0, :2]) < 4:
            u = u_nominal
            use_nominal = True

        else:
            u = core.mpc_solver(np.squeeze(s))
        u = np.clip(u, -20, 20)
#         print('MPC->u: ',u)
        dsdt = core.dynamics_np(s, u[np.newaxis])

        if use_nominal:
            timestep = config.TIME_STEP * 0.5
        else:
            timestep = config.TIME_STEP * 2.0
        #s = np.clip(s + dsdt * timestep, -7, 7)
        s = s + dsdt * timestep

#         if True: #np.mod(j, 3) == 0:

#             plt.clf()
#             plt.scatter(s[0, 0], s[0, 1], s=100, color='b', alpha=0.5)

#             ax = fig.gca()
#             ax.add_patch(plt.Circle((0, 0), 4, color='g', alpha=0.3, fill=False, linewidth=2, linestyle='--'))
#             ax.plot([0, -2.8], [0, -2.8], color='g', alpha=0.3, linewidth=2, linestyle='--')
#             ax.plot([0, 2.8], [0, -2.8], color='g', alpha=0.3, linewidth=2, linestyle='--')

#             ax.add_patch(patches.Wedge((0, 0), 4, -135, -45, linewidth=2,
#                          edgecolor='g', facecolor='g', fill=True, alpha=0.3))

#             plt.xlim(-12, 12)
#             plt.ylim(-11, 12)

#             plt.axis('off')

#             fig.canvas.draw()
#             plt.pause(0.2)
            
    mpc_data.append(u)
    np.save('mpc_data.npy', np.array(mpc_data)) 
