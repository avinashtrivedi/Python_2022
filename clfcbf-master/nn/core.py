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
              [0, 0, -2 * N, 0]], dtype=np.float32)
B = np.array([[0, 0],
              [0, 0],
              [1/M, 0],
              [0, 1/M]], dtype=np.float32)

A_torch = torch.from_numpy(A)
B_torch = torch.from_numpy(B)

def dynamics_np(s, u, n=N, m=M):
    """
    s (4)
    n (2)
    """
    sdot = np.array(np.dot(A, s) + np.dot(B, u), dtype=np.float32)
    return sdot


def dynamics_torch(s, u):
    """
    s (4)
    n (2)
    """
    sdot = torch.matmul(A_torch, s) + torch.matmul(B_torch, u)
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


class PnCLQR(object):

    def __init__(self, n=N, m=M):
        Q = np.diag([10, 10, 0.01, 0.01])
        R = np.diag([0.01, 0.01])
        X = np.matrix(sp.linalg.solve_continuous_are(A, B, Q, R))
        K = np.matrix(sp.linalg.inv(R)*(B.T*X))
        self.K = -K
    
    def __call__(self, s):
        nearest_index = np.argmin(np.linalg.norm(self.traj - s[:2], axis=1))
        target_index = min(nearest_index + 1, self.traj.shape[0] - 1)
        target = np.concatenate([self.traj[target_index], np.array([0, 0])])
        u = np.squeeze(np.array(np.dot(self.K, s - target), dtype=np.float32))
        return u

    def plan_trajectory(self, s, safe_dist=4.0):
        point_0 = s[:2]
        point_1 = safe_dist * s[:2] / np.linalg.norm(s[:2])
        point_2 = np.array([0, -safe_dist])
        point_3 = np.array([0, 0])

        point_4 = (point_1 + point_0) * 0.5
        point_5 = safe_dist * self.interp(point_1, point_2)
        point_6 = safe_dist * self.interp(point_1, point_5)
        point_7 = safe_dist * self.interp(point_5, point_2)

        point_8 = (point_2 + point_3) * 0.5

        traj = np.array([point_0, point_4,
                         point_6, point_5, point_7, point_8, point_3])
        self.traj = traj

    def interp(self, v1, v2):
        v1_unit = v1 / np.linalg.norm(v1)
        v2_unit = v2 / np.linalg.norm(v2)
        v = v1_unit + v2_unit
        v = v / (1e-5 + np.linalg.norm(v))
        return v

        


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

        return h, par_h_par_s


class ActionNet(nn.Module):

    def __init__(self, n_input=4, n_hidden=64, n_control=2):
        super(ActionNet, self).__init__()
        self.fc_layer_1 = nn.Linear(n_input, n_hidden)
        self.fc_layer_2 = nn.Linear(n_hidden, n_hidden)
        self.fc_layer_3 = nn.Linear(n_hidden, n_control)

    def forward(self, s):
        tanh = nn.Tanh()
        fc1_act = tanh(self.fc_layer_1(s))
        fc2_act = tanh(self.fc_layer_2(fc1_act))
        fc3_act = self.fc_layer_3(fc2_act)
        h = fc3_act
        return h