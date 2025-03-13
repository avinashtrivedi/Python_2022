import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import config
import core
import os


plt.ion()
plt.close()
# fig, ax = plt.subplots(figsize=(7, 7))

nominal_controller = core.LQR()
# data = []
for i in range(1):
    data = []
    rho_init = 10 #np.random.uniform(10, 12)
    theta_init = np.pi/4 #np.random.uniform(np.pi/4, np.pi/3)
    v_init = np.random.uniform(-1, 1, size=2)
    s = np.array([[np.cos(theta_init) * rho_init, 
                  np.sin(theta_init) * rho_init,
                  v_init[0], v_init[1]]], dtype=np.float32)

    s_traj = []

    for j in range(120): #(config.NUM_STEPS_PER_EPOCH):
        u_nominal = nominal_controller(np.squeeze(s))

        if s[0, 1] < 0 and abs(s[0, 0]) < abs(s[0, 1]) or np.linalg.norm(s[0, :2]) < 1:
            u = u_nominal

        else:
            u = core.mpc_solver(np.squeeze(s))
        u = np.clip(u, -20, 20)
        data.append((j,u))
#         print('------------------------------------')
#         print('data: ',data)
#         print('------------------------------------')
        dsdt = core.dynamics_np_noise(s, u[np.newaxis])
        s = s + dsdt * config.TIME_STEP * 2

        s_traj.append(s.reshape(1, -1))

#         if True: #np.mod(j, 3) == 0:

#             plt.clf()
#             plt.scatter(s[0, 0], s[0, 1], s=100)

#             ax = fig.gca()
#             ax.add_patch(plt.Circle((0, 0), 4, color='r', alpha=0.3, fill=False, linewidth=4))
#             ax.plot([0, -2.8], [0, -2.8], color='r', alpha=0.3, linewidth=4)
#             ax.plot([0, 2.8], [0, -2.8], color='r', alpha=0.3, linewidth=4)

#             plt.xlim(-13, 13)
#             plt.ylim(-13, 13)

#             plt.axis('off')

#             fig.canvas.draw()
#             plt.pause(0.01)
    s_traj = np.concatenate(s_traj, axis=0)
    f = open('mpc_traj.npy', 'wb')
    np.save(f, s_traj)
# print('data: ',data[0])

fig, axs = plt.subplots(2)
t = [(i,j[0],j[1]) for i,j in data]
x = np.array(t)[:,0]
y = np.array(t)[:,1]
y1 = np.array(t)[:,2]

axs[0].plot(x, y)
axs[1].plot(x, y1)

axs[0].set_xlabel('Step')
axs[0].set_ylabel('ux')

axs[1].set_xlabel('Step')
axs[1].set_ylabel('uy')

fig.tight_layout()

# shift subplots down:
# st.set_y(0.95)
fig.subplots_adjust(top=0.85)

plt.savefig('foo.png')