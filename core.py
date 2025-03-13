import numpy as np
import scipy as sp
import scipy.linalg

import torch
import torch.nn as nn
import torch.nn.functional as F
import config

import casadi

N = 1
M = 1.5

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


N_noise = 1
M_noise = 1


#A_noise = np.array([[0, 0, 1, 0],
#              [0, 0, 0, 1],
#              [3 * N_noise**2, 0, 0, 2 * N_noise],
#              [0, 0, -2 * N_noise, 0]], dtype=np.float32)
#B_noise = np.array([[0, 0],
#              [0, 0],
#              [1/M_noise, 0],
#              [0, 1/M_noise]], dtype=np.float32)

#A_torch_noise = torch.from_numpy(A_noise)
#B_torch_noise = torch.from_numpy(B_noise)


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


def dynamics_np(s, u, n=N, m=M):
    """
    s (nBatch, 4)
    n (nBatch, 2)
    """
    sdot = np.array(np.dot(s, A.T) + np.dot(u, B.T), dtype=np.float32)
    return sdot


def dynamics_np_noise(s, u, M_noise=M_noise, N_noise=N_noise):
    """
    s (nBatch, 4)
    n (nBatch, 2)
    """
    A_noise = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [3 * N_noise**2, 0, 0, 2 * N_noise],
              [0, 0, -2 * N_noise, 0]], dtype=np.float32)
    B_noise = np.array([[0, 0],
              [0, 0],
              [1/M_noise, 0],
              [0, 1/M_noise]], dtype=np.float32)

    sdot = np.array(np.dot(s, A_noise.T) + np.dot(u, B_noise.T), dtype=np.float32)
    return sdot


def dynamics_torch(s, u):
    """
    s (nBatch, 4)
    n (nBatch, 2)
    """
    sdot = torch.matmul(s, torch.transpose(A_torch, 0, 1)
    ) + torch.matmul(u, torch.transpose(B_torch, 0, 1))
    return sdot


def dynamics_torch_noise(s, u, M_noise=M_noise, N_noise=N_noise):
    """
    s (nBatch, 4)
    n (nBatch, 2)
    """
    A_noise = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [3 * N_noise**2, 0, 0, 2 * N_noise],
              [0, 0, -2 * N_noise, 0]], dtype=np.float32)
    B_noise = np.array([[0, 0],
              [0, 0],
              [1/M_noise, 0],
              [0, 1/M_noise]], dtype=np.float32)
    A_torch_noise = torch.from_numpy(A_noise)
    B_torch_noise = torch.from_numpy(B_noise)
    
    sdot = torch.matmul(s, torch.transpose(A_torch_noise, 0, 1)
    ) + torch.matmul(u, torch.transpose(B_torch_noise, 0, 1))
    return sdot


def control_affine_dynamics(x):
    n_batch = x.shape[0]
    f = torch.matmul(x, torch.transpose(A_torch, 0, 1))
    g = B_torch.unsqueeze(0).expand(n_batch, 4, 2)
    return f, g

def d_tanh_dx(tanh):
    return torch.diag_embed(1 - tanh**2)


class CLF_QP_Net(nn.Module):
    """A neural network for simultaneously computing the Lyapunov function and the
    control input. The neural net makes the Lyapunov function, and the control input
    is computed by solving a QP.
    """

    def __init__(self, n_input, n_hidden, n_controls,
                 control_affine_dynamics=control_affine_dynamics):
        """
        Initialize the network
        args:
            n_input: number of states the system has
            n_hidden: number of hiddent layers to use
            n_controls: number of control outputs to use
            clf_lambda: desired exponential convergence rate for the CLF
            control_affine_dynamics: a function that takes n_batch x n_dims and returns a tuple of:
                f_func: a function n_batch x n_dims -> n_batch x n_dims that returns the
                        state-dependent part of the control-affine dynamics
                g_func: a function n_batch x n_dims -> n_batch x n_dims x n_controls that returns
                        the input coefficient matrix for the control-affine dynamics
        """
        super(CLF_QP_Net, self).__init__()

        # Save the dynamics and nominal controller functions
        self.dynamics = control_affine_dynamics

        # The network will have the following architecture
        #
        # n_input -> VFC1 (n_input x n_hidden) -> VFC2 (n_hidden, n_hidden)
        # -> VFC2 (n_hidden, n_hidden) -> V = x^T x --> QP -> u
        self.Vfc_layer_1 = nn.Linear(n_input, n_hidden)
        self.Vfc_layer_2 = nn.Linear(n_hidden, n_hidden)
        self.Vfc_layer_3 = nn.Linear(n_hidden, n_hidden // 4)

        # We also train a controller to learn the nominal control input
        self.Ufc_layer_1 = nn.Linear(n_input, n_hidden)
        self.Ufc_layer_2 = nn.Linear(n_hidden, n_hidden)
        self.Ufc_layer_3 = nn.Linear(n_hidden, n_controls)
        self.n_controls = n_controls


    def compute_controls(self, x):
        """
        Computes the control input (for use in the QP filter)
        args:
            x: the state at the current timestep [n_batch, n_dims]
        returns:
            u: the value of the barrier at each provided point x [n_batch, n_controls]
        """
        tanh = nn.Tanh()
        Ufc1_act = tanh(self.Ufc_layer_1(x))
        Ufc2_act = tanh(self.Ufc_layer_2(Ufc1_act))
        U = self.Ufc_layer_3(Ufc2_act)
        U = tanh(U) * 3.0

        return U

    def compute_lyapunov(self, x):
        """
        Computes the value and gradient of the Lyapunov function
        args:
            x: the state at the current timestep [n_batch, n_dims]
        returns:
            V: the value of the Lyapunov at each provided point x [n_batch, 1]
            grad_V: the gradient of V [n_batch, n_dims]
        """
        # Use the first two layers to compute the Lyapunov function
        tanh = nn.Tanh()
        Vfc1_act = tanh(self.Vfc_layer_1(x))
        Vfc2_act = tanh(self.Vfc_layer_2(Vfc1_act))
        Vfc3_act = tanh(self.Vfc_layer_3(Vfc2_act))
        # Compute the Lyapunov function as the square norm of the last layer activations
        V = 0.5 * (Vfc3_act * Vfc3_act).sum(1)

        # We also need to calculate the Lie derivative of V along f and g
        #
        # L_f V = \grad V * f
        # L_g V = \grad V * g
        #
        # Since V = tanh(w2 * tanh(w1*x + b1) + b1),
        # grad V = d_tanh_dx(V) * w2 * d_tanh_dx(tanh(w1*x + b1)) * w1

        # Jacobian of first layer wrt input (n_batch x n_hidden x n_input)
        DVfc1_act = torch.matmul(d_tanh_dx(Vfc1_act), self.Vfc_layer_1.weight)
        # Jacobian of second layer wrt input (n_batch x n_hidden x n_input)
        DVfc2_act = torch.bmm(torch.matmul(d_tanh_dx(Vfc2_act), self.Vfc_layer_2.weight), DVfc1_act)
        # Jacobian of third layer wrt input (n_batch x n_hidden // 4 x n_input)
        DVfc3_act = torch.bmm(torch.matmul(d_tanh_dx(Vfc3_act), self.Vfc_layer_3.weight), DVfc2_act)
        # Gradient of V wrt input (n_batch x 1 x n_input)
        grad_V = torch.bmm(Vfc3_act.unsqueeze(1), DVfc3_act)

        return V, grad_V

    def forward(self, x):
        """
        Compute the forward pass of the controller
        args:
            x: the state at the current timestep [n_batch, n_dims]
        returns:
            u: the input at the current state [n_batch, n_controls]
            r: the relaxation required to satisfy the CLF inequality
            V: the value of the Lyapunov function at a given point
            Vdot: the time derivative of the Lyapunov function
        """
        # Compute the Lyapunov and barrier functions
        V, grad_V = self.compute_lyapunov(x)
        u_learned = self.compute_controls(x)

        # Compute lie derivatives for each scenario
        L_f_Vs = []
        L_g_Vs = []

        f, g = self.dynamics(x)
        # Lyapunov Lie derivatives
        L_f_Vs.append(torch.bmm(grad_V, f.unsqueeze(-1)).squeeze(-1))
        L_g_Vs.append(torch.bmm(grad_V, g).squeeze(1))

        u = u_learned

        # Average across scenarios
        Vdot = L_f_Vs[0].unsqueeze(-1) + torch.bmm(L_g_Vs[0].unsqueeze(1), u.unsqueeze(-1))

        return u, V, Vdot


def lyapunov_loss(x,
                  x_goal,
                  safe_mask,
                  unsafe_mask,
                  net,
                  clf_lambda=0.005,
                  safe_level=1.0,
                  timestep=0.001,
                  margin=0.3,
                  print_loss=False):
    """
    Compute a loss to train the Lyapunov function
    args:
        x: the points at which to evaluate the loss
        x_goal: the origin
        safe_mask: the points in x marked safe
        unsafe_mask: the points in x marked unsafe
        net: a CLF_CBF_QP_Net instance
        clf_lambda: the rate parameter in the CLF condition
        safe_level: defines the safe region as the sublevel set of the lyapunov function
        timestep: the timestep used to compute a finite-difference approximation of the
                  Lyapunov function
        print_loss: True to enable printing the values of component terms
    returns:
        loss: the loss for the given Lyapunov function
    """
    # Compute loss based on...
    loss = 0.0
    #   1.) squared value of the Lyapunov function at the goal
    V0, _ = net.compute_lyapunov(x_goal)
    loss += V0.pow(2).squeeze()

    #   3.) term to encourage V <= safe_level in the safe region
    V_safe, _ = net.compute_lyapunov(x[safe_mask])
    safe_region_lyapunov_term = F.relu(V_safe - (safe_level - margin))
    loss += safe_region_lyapunov_term.mean()

    #   4.) term to encourage V >= safe_level in the unsafe region
    V_unsafe, _ = net.compute_lyapunov(x[unsafe_mask])
    unsafe_region_lyapunov_term = F.relu(safe_level + margin - V_unsafe)
    loss += unsafe_region_lyapunov_term.mean() * 1.5

    #   5.) A term to encourage satisfaction of CLF condition
    u, V, _ = net(x)
    # To compute the change in V, simulate x forward in time and check if V decreases in each
    # scenario
    lyap_descent_term = 0.0

    f, g = net.dynamics(x)
    xdot = f + torch.bmm(g, u.unsqueeze(-1)).squeeze()
    x_next = x + timestep * xdot
    V_next, _ = net.compute_lyapunov(x_next)
    Vdot = (V_next.squeeze() - V.squeeze()) / timestep
    lyap_descent_term += F.relu(Vdot + clf_lambda * V.squeeze())
    lyap_descent_term *= 100
    loss += lyap_descent_term.mean()

    if print_loss:
        print(f"                     CLF origin: {V0.pow(2).squeeze().item()}")
        print(f"           CLF safe region term: {safe_region_lyapunov_term.mean().item()}")
        print(f"         CLF unsafe region term: {unsafe_region_lyapunov_term.mean().item()}")
        print(f"               CLF descent term: {lyap_descent_term.mean().item()}")

    return loss


def mpc_solver(s_init, quiet=True):
    x, y, vx, vy = np.squeeze(s_init)
    xy = np.array([x, y])
    xy_goal = np.array([0, -5.5])
    distance = np.linalg.norm(xy_goal - xy)
    direction = (xy_goal - xy) / (1e-6 + distance)
    
    mpc_goal_dist = 0.5
    if distance < mpc_goal_dist:
        mpc_goal = xy_goal
    else:
        mpc_goal = xy + direction * mpc_goal_dist

    T = 10
    dt = config.TIME_STEP * 2

    opti = casadi.Opti()
    s = opti.variable(T + 1, 4)  # state (x, y, vx, vy)
    u = opti.variable(T, 2)

    # distance to the reference goal
    opti.minimize(casadi.sumsqr(s[T, 0] - mpc_goal[0]) + 
        casadi.sumsqr(s[T, 1] - mpc_goal[1]) + casadi.sumsqr(s[T, 2:]))

    # initial conditions
    opti.subject_to(s[0, 0] == x)
    opti.subject_to(s[0, 1] == y)
    opti.subject_to(s[0, 2] == vx)
    opti.subject_to(s[0, 3] == vy)

    for k in range(T):
        x_t, y_t, vx_t, vy_t = s[k, 0], s[k, 1], s[k, 2], s[k, 3]
        Fx_t, Fy_t = u[k, 0], u[k, 1]

        # dynamics
        opti.subject_to(s[k + 1, 0] == x_t + vx_t * dt)
        opti.subject_to(s[k + 1, 1] == y_t + vy_t * dt)

        ax_t = 3 * N**2 * x_t + 2 * N * vy_t + Fx_t / M
        ay_t = -2 * N * vx_t + Fy_t / M

        opti.subject_to(s[k + 1, 2] == vx_t + ax_t * dt)
        opti.subject_to(s[k + 1, 3] == vy_t + ay_t * dt)

        # constraint
        opti.subject_to(s[k + 1, 0]**2 + s[k + 1, 1]**2 >= 5**2)

        opti.subject_to(Fx_t <= 20)
        opti.subject_to(Fx_t >= -20)
        opti.subject_to(Fy_t <= 20)
        opti.subject_to(Fy_t >= -20)


    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000}
    if quiet:
        p_opts["print_time"] = 0
        s_opts["print_level"] = 0
        s_opts["sb"] = "yes"
    opti.solver("ipopt", p_opts, s_opts)
    sol1 = opti.solve()

    try:
        sol1 = opti.solve()
        x_sol, u_sol = sol1.value(x), sol1.value(u)
        return u_sol[0]
    except:
        return 0