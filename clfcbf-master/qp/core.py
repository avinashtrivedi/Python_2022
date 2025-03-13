import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import cvxpy as cp

import config

N = config.N
M = config.M

A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [3 * N**2, 0, 0, 2 * N],
              [0, 0, -2 * N, 0]])
B = np.array([[0, 0],
              [0, 0],
              [1/M, 0],
              [0, 1/M]])

def dynamics_np(s, u, n=N, m=M):
    """
    s (4)
    n (2)
    """
    sdot = np.array(np.dot(A, s) + np.dot(B, u), dtype=np.float32)
    return sdot


class LQR(object):

    def __init__(self, n=N, m=M):
        Q = np.diag([1, 1, 0.01, 0.01])
        R = np.diag([0.01, 0.01])
        X = np.matrix(sp.linalg.solve_continuous_are(A, B, Q, R))
        K = np.matrix(sp.linalg.inv(R)*(B.T*X))
        self.K = -K
    
    def __call__(self, s):
        u = np.squeeze(np.array(np.dot(self.K, s), dtype=np.float32))
        return u


class CBFNet(nn.Module):

    def __init__(self, n_input=4, n_hidden=64, n_control=2, n=N, m=M, cbf_alpha=1):
        super(CBFNet, self).__init__()
        self.fc_layer_1 = nn.Linear(n_input, n_hidden)
        self.fc_layer_2 = nn.Linear(n_hidden, n_hidden)
        self.A = A
        self.B = B
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_control = n_control
        self.lqr = LQR(n=n, m=m)
        self.cbf_alpha = cbf_alpha

    def forward(self, s, constraint_weight=2):
        tanh = nn.Tanh()
        fc1_act = tanh(self.fc_layer_1(s))
        fc2_act = tanh(self.fc_layer_2(fc1_act))
        h = fc2_act.mean(0)

        def d_tanh_dx(tanh):
            return torch.diag_embed(1 - tanh**2)

        # Jacobian of first layer wrt input (n_hidden x n_input)
        dfc1_act = torch.matmul(d_tanh_dx(fc1_act), self.fc_layer_1.weight)
        # Jacobian of second layer wrt input (n_hidden x n_input)
        dfc2_act = torch.mm(torch.matmul(d_tanh_dx(fc2_act), self.fc_layer_2.weight), dfc1_act)
        # Gradient of h wrt input (n_input)
        par_h_par_s = dfc2_act.mean(0)

        u_ref = self.lqr(s.detach().numpy())
        u = cp.Variable(self.n_control)

        cbf_third_cond = par_h_par_s.detach().numpy() @ (
            self.A @ s.detach().numpy() + self.B @ u) + config.CBF_ALPHA * h.detach().numpy()
        relaxed_constraint = cp.square(cp.maximum(-cbf_third_cond, 0.001))

        objective = cp.Minimize(cp.sum_squares(u_ref - u) + constraint_weight * relaxed_constraint)
        cp.Problem(objective).solve()
        u = u.value

        return h, par_h_par_s, u
    

class CBFNetTest(nn.Module):

    def __init__(self, n_input=4, n_hidden=64, n_control=2, n=N, m=M, cbf_alpha=1):
        super(CBFNetTest, self).__init__()
        self.fc_layer_1 = nn.Linear(n_input, n_hidden)
        self.fc_layer_2 = nn.Linear(n_hidden, n_hidden)
        self.A = A
        self.B = B
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_control = n_control
        self.lqr = LQR(n=n, m=m)
        self.cbf_alpha = cbf_alpha

    def forward(self, s):
        tanh = nn.Tanh()
        fc1_act = tanh(self.fc_layer_1(s))
        fc2_act = tanh(self.fc_layer_2(fc1_act))
        h = fc2_act.mean(0)

        def d_tanh_dx(tanh):
            return torch.diag_embed(1 - tanh**2)

        # Jacobian of first layer wrt input (n_hidden x n_input)
        dfc1_act = torch.matmul(d_tanh_dx(fc1_act), self.fc_layer_1.weight)
        # Jacobian of second layer wrt input (n_hidden x n_input)
        dfc2_act = torch.mm(torch.matmul(d_tanh_dx(fc2_act), self.fc_layer_2.weight), dfc1_act)
        # Gradient of h wrt input (n_input)
        par_h_par_s = dfc2_act.mean(0)

        u_ref = self.lqr(s.detach().numpy())
        u = cp.Variable(self.n_control)

        cbf_third_cond = par_h_par_s.detach().numpy() @ (
            self.A @ s.detach().numpy() + self.B @ u) + config.CBF_ALPHA * h.detach().numpy()
        constraint = [cbf_third_cond >= 0]

        objective = cp.Minimize(cp.sum_squares(u_ref - u))
        cp.Problem(objective, constraint).solve()
        u = u.value
        if u is None:
            u = u_ref

        return h, par_h_par_s, u

"""
plt.ion()
plt.close()
fig = plt.figure(figsize=(7, 5))

cbf_net = CBFNet()

s = np.array([10, 0, 0, -10], dtype=np.float32)


for i in range(1000):
    u = cbf_net(torch.from_numpy(s))[2]
    dsdt = dynamics_np(s, u)
    s = np.array(s + dsdt * config.TIME_STEP, dtype=np.float32)
    plt.clf()
    plt.scatter(s[0], s[1])
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    fig.canvas.draw()
    plt.pause(0.01)
"""
