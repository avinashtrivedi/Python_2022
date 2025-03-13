import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import config

import core
import os

net = core.CLF_QP_Net(4, 256, 2)
net.load_state_dict(torch.load('./logs/model.pth'))
net.eval()

res = 100 # resultion
num_iter = 30

x, y = np.meshgrid(np.linspace(-12, 12, res), np.linspace(-12, 12, res))
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

z = np.zeros(shape=(res, res))
for j in tqdm(range(num_iter)):
    vx = np.random.uniform(-5, 5, size=x.shape)
    vy = np.random.uniform(-5, 5, size=y.shape)

    s = np.concatenate([x, y, vx, vy], axis=1).astype(np.float32)
    s = torch.from_numpy(s)
    V, _ = net.compute_lyapunov(s)
    z = z + np.reshape(V.detach().numpy(), (res, res))

z = z / num_iter

x = np.reshape(x, (res, res))
y = np.reshape(y, (res, res))

contours = plt.contourf(x, y, z, cmap="twilight_shifted", levels=20)
plt.colorbar(contours, orientation="vertical")
plt.axis('off')

plt.savefig('contour.png')
plt.show()