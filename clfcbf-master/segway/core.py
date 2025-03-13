import numpy as np
import torch
from torch.autograd import grad
import torch
import torch.nn as nn
import torch.nn.functional as F
import casadi


num_dim_x = 4
num_dim_control = 1
x_min = np.array([-3, -np.pi / 2, -1, -3]).reshape(1, 4, 1)
x_max = np.array([3, np.pi / 2, 1, 3]).reshape(1, 4, 1)

mass_noise = 1
inertia_noise = 1.3


def f_func_np(x):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]

    p, theta, v, omega = [x[:,i,0] for i in range(num_dim_x)]

    f = np.zeros((bs, num_dim_x, 1))
    f[:, 0, 0] = v
    f[:, 1, 0] = omega
    f[:, 2, 0] = (np.cos(theta) * (9.8 * np.sin(theta) + 11.5 * v) + 68.4 * v - 1.2 * (omega ** 2) * np.sin(theta)) / (np.cos(theta) - 24.7)
    f[:, 3, 0] = (-58.8 * v * np.cos(theta) - 243.5 * v - np.sin(theta) * (208.3 + (omega ** 2) * np.cos(theta))) / (np.cos(theta) ** 2 - 24.7)
    return f


def B_func_np(x):
    bs = x.shape[0]

    p, theta, v, omega = [x[:,i,0] for i in range(num_dim_x)]

    B = np.zeros((bs, num_dim_x, num_dim_control))

    B[:, 2, 0] = (-1.8 * np.cos(theta) - 10.9) / (np.cos(theta) - 24.7)
    B[:, 3, 0] = (9.3 * np.cos(theta) + 38.6) / (np.cos(theta) ** 2 - 24.7)
    return B


def f_func_torch(x):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]

    p, theta, v, omega = [x[:,i,0] for i in range(num_dim_x)]

    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    f[:, 0, 0] = v
    f[:, 1, 0] = omega
    f[:, 2, 0] = (torch.cos(theta) * (9.8 * torch.sin(theta) + 11.5 * v) + 68.4 * v - 1.2 * (omega ** 2) * torch.sin(theta)) / (torch.cos(theta) - 24.7)
    f[:, 3, 0] = (-58.8 * v * torch.cos(theta) - 243.5 * v - torch.sin(theta) * (208.3 + (omega ** 2) * torch.cos(theta))) / (torch.cos(theta) ** 2 - 24.7)
    return f


def B_func_torch(x):
    bs = x.shape[0]

    p, theta, v, omega = [x[:,i,0] for i in range(num_dim_x)]

    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())

    B[:, 2, 0] = (-1.8 * torch.cos(theta) - 10.9) / (torch.cos(theta) - 24.7)
    B[:, 3, 0] = (9.3 * torch.cos(theta) + 38.6) / (torch.cos(theta) ** 2 - 24.7)
    return B


def f_func_np_noise(x, mass_noise=mass_noise, inertia_noise=inertia_noise):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]

    p, theta, v, omega = [x[:,i,0] for i in range(num_dim_x)]

    f = np.zeros((bs, num_dim_x, 1))
    f[:, 0, 0] = v
    f[:, 1, 0] = omega
    f[:, 2, 0] = (np.cos(theta) * (9.8 * np.sin(theta) / inertia_noise + 11.5 * v) + 68.4 * v - 1.2 * (omega ** 2) * np.sin(theta)) / (np.cos(theta) / inertia_noise - 24.7 * mass_noise)
    f[:, 3, 0] = (-58.8 * v * np.cos(theta) - 243.5 * v - np.sin(theta) * (208.3 * mass_noise + (omega ** 2) * np.cos(theta))) / (np.cos(theta) ** 2 - 24.7 * mass_noise * inertia_noise)
    return f


def B_func_np_noise(x, mass_noise=mass_noise, inertia_noise=inertia_noise):
    bs = x.shape[0]

    p, theta, v, omega = [x[:,i,0] for i in range(num_dim_x)]

    B = np.zeros((bs, num_dim_x, num_dim_control))

    B[:, 2, 0] = (-1.8 * np.cos(theta) - 10.9) / (np.cos(theta) / inertia_noise - 24.7 * mass_noise)
    B[:, 3, 0] = (9.3 * np.cos(theta) + 38.6) / (np.cos(theta) ** 2 - 24.7 * mass_noise * inertia_noise)
    return B


def f_func_torch_noise(x, mass_noise=mass_noise, inertia_noise=inertia_noise):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]

    p, theta, v, omega = [x[:,i,0] for i in range(num_dim_x)]

    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    f[:, 0, 0] = v
    f[:, 1, 0] = omega
    f[:, 2, 0] = (torch.cos(theta) * (9.8 * torch.sin(theta) / inertia_noise + 11.5 * v) + 68.4 * v - 1.2 * (omega ** 2) * torch.sin(theta)) / (torch.cos(theta) / inertia_noise - 24.7 * mass_noise)
    f[:, 3, 0] = (-58.8 * v * torch.cos(theta) - 243.5 * v - torch.sin(theta) * (208.3 * mass_noise + (omega ** 2) * torch.cos(theta))) / (torch.cos(theta) ** 2 - 24.7 * mass_noise * inertia_noise)
    return f


def B_func_torch_noise(x, mass_noise=mass_noise, inertia_noise=inertia_noise):
    bs = x.shape[0]

    p, theta, v, omega = [x[:,i,0] for i in range(num_dim_x)]

    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())

    B[:, 2, 0] = (-1.8 * torch.cos(theta) - 10.9) / (torch.cos(theta) / inertia_noise - 24.7 * mass_noise)
    B[:, 3, 0] = (9.3 * torch.cos(theta) + 38.6) / (torch.cos(theta) ** 2 - 24.7 * mass_noise * inertia_noise)
    return B


def Jacobian(f, x):
    # NOTE that this function assume that data are independent of each other
    f = f + 0. * x.sum() # to avoid the case that f is independent of x
    # f: B x m x 1
    # x: B x n x 1
    # ret: B x m x n
    bs = x.shape[0]
    m = f.size(1)
    n = x.size(1)
    J = torch.zeros(bs, m, n).type(x.type())
    for i in range(m):
        J[:, i, :] = grad(f[:, i, 0].sum(), x, create_graph=True)[0].squeeze(-1)
    return J


def get_safe_mask(x, bar_center=[0, 1], bar_radius=0.15, 
                  unsafe_tilt_thres=np.pi/4, safe_tilt_thres=np.pi/6,
                  return_meta=False):
    # x: bs x n x 1
    # ret: bs
    p, theta, v, omega = [x[:,i,0] for i in range(num_dim_x)]
    dist = (p + np.sin(theta) - bar_center[0])**2 + (np.cos(theta) - bar_center[1])**2
    inside_bar = dist < bar_radius ** 2
    # tilt_too_much = np.abs(theta) > unsafe_tilt_thres
    # unsafe_mask = np.logical_or(inside_bar, tilt_too_much)
    unsafe_mask = inside_bar

    #straight = np.abs(theta) < safe_tilt_thres
    #safe_mask = np.logical_and(np.logical_not(unsafe_mask), straight)
    #safe_mask = np.logical_and(safe_mask, dist > (bar_radius * 1.3) ** 2)

    safe_mask = dist > (bar_radius * 1.5) ** 2

    #if return_meta:
    #    meta = {'inside_bar': inside_bar, 'tilt_too_much': tilt_too_much}
    #    return safe_mask, unsafe_mask, meta
    
    return safe_mask, unsafe_mask


def d_tanh_dx(tanh):
    return torch.diag_embed(1 - tanh**2)


def segway_dynamics_torch(x):
    bs = x.shape[0]
    x = x.view(bs, num_dim_x, 1)
    f = f_func_torch(x)[:, :, 0]
    B = B_func_torch(x)
    return f, B


class Segway_LQR(object):

    def __init__(self):
        K = np.array([0, -17.6, -18.8, -6.3], dtype=np.float32)
        self.K = torch.from_numpy(K)

    def __call__(self, x):
        bs = x.shape[0]
        x = x.view(bs, -1)
        u = -torch.sum(x * self.K, dim=1, keepdim=True)
        return u


class CLF_QP_Net(nn.Module):
    """A neural network for simultaneously computing the Lyapunov function and the
    control input. The neural net makes the Lyapunov function, and the control input
    is computed by solving a QP.
    """

    def __init__(self, n_input, n_hidden, n_controls,
                 control_affine_dynamics=segway_dynamics_torch):
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
        n_batch = x.shape[0]
        x = x.view(n_batch, -1)
        tanh = nn.Tanh()
        Ufc1_act = tanh(self.Ufc_layer_1(x))
        Ufc2_act = tanh(self.Ufc_layer_2(Ufc1_act))
        U = self.Ufc_layer_3(Ufc2_act)
        U = tanh(U) * 20.0

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
        n_batch = x.shape[0]
        x = x.view(n_batch, -1)
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
        n_batch = x.shape[0]
        x = x.view(n_batch, -1)

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
                  nominal_controller,
                  clf_lambda=0.005,
                  safe_level=1.0,
                  timestep=0.001,
                  margin=0.2,
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
    loss += unsafe_region_lyapunov_term.mean()

    #   5.) A term to encourage satisfaction of CLF condition
    u, V, Vdot = net(x)
    # To compute the change in V, simulate x forward in time and check if V decreases in each
    # scenario
    lyap_descent_term = 0.0

    Vdot = Vdot.squeeze()
    lyap_descent_term += F.relu(Vdot + clf_lambda * (V.squeeze() - safe_level))
    lyap_descent_term *= 100
    loss += lyap_descent_term.mean()

    #   6.) term to use a nominal controller to guide the learned controller
    u_nominal = nominal_controller(x) - 5.0 # u = -5.0 moves the segway right
    u_nominal_term = (u_nominal - u) ** 2 * 1e-3
    loss += u_nominal_term.mean()

    if print_loss:
        print("-----------------------------------------")
        print("                     CLF origin: {:.3f}".format(V0.pow(2).squeeze().item()))
        print("           CLF safe region term: {:.3f}".format(safe_region_lyapunov_term.mean().item()))
        print("         CLF unsafe region term: {:.3f}".format(unsafe_region_lyapunov_term.mean().item()))
        print("               CLF descent term: {:.3f}".format(lyap_descent_term.mean().item()))
        print("           Nominal control term: {:.3f}".format(u_nominal_term.mean().item()))

    return loss


def mpc_solver(x_init, quiet=True):
    p, theta, velocity, omega = np.squeeze(x_init)
    p_ref = p + 0.2

    T = 10
    dt = 0.005

    opti = casadi.Opti()
    x = opti.variable(T + 1, 4)  # state (x, y, psi, v)
    u = opti.variable(T, 1)

    # distance to the reference goal
    opti.minimize(
        casadi.sumsqr(x[T, 0] - p_ref) + casadi.sumsqr(x[T, 1]) * 0.6)

    # initial conditions
    opti.subject_to(x[0, 0] == p)
    opti.subject_to(x[0, 1] == theta)
    opti.subject_to(x[0, 2] == velocity)
    opti.subject_to(x[0, 3] == omega)

    for k in range(T):  # timesteps
        p_t, theta_t, v_t, omega_t = x[k, 0], x[k, 1], x[k, 2], x[k, 3]
        torque = u[k, 0]

        # dynamics of position and angle
        opti.subject_to(x[k + 1, 0] == p_t + v_t * dt)  # p+=v*dt
        opti.subject_to(x[k + 1, 1] == theta_t + omega_t * dt)  # theta+=omega*dt

        # dynamics of velocity and angular velocity
        f_v = (casadi.cos(theta_t) * (9.8 * casadi.sin(theta_t) + 11.5 * v_t) + 68.4 * v_t - 1.2 * (omega_t ** 2) * casadi.sin(theta_t)) / (casadi.cos(theta_t) - 24.7)
        f_omega = (-58.8 * v_t * casadi.cos(theta_t) - 243.5 * v_t - casadi.sin(theta_t) * (208.3 + (omega_t ** 2) * casadi.cos(theta_t))) / (casadi.cos(theta_t) ** 2 - 24.7)
        u_v = (-1.8 * casadi.cos(theta_t) - 10.9) / (casadi.cos(theta_t) - 24.7) * torque
        u_omega = (9.3 * casadi.cos(theta_t) + 38.6) / (casadi.cos(theta_t) ** 2 - 24.7) * torque

        dvdt = f_v + u_v
        domgdt = f_omega + u_omega

        opti.subject_to(x[k + 1, 2] == v_t + dvdt * dt)  
        opti.subject_to(x[k + 1, 3] == omega_t + domgdt * dt)

        # torque limits
        opti.subject_to(torque <= 80)
        opti.subject_to(torque >= -80)

        # obstacle avoidance
        opti.subject_to((x[k + 1, 0] + casadi.sin(x[k + 1, 1]))**2 + (casadi.cos(x[k + 1, 1]) - 1)**2 >= 0.1**2)
        

    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000}
    if quiet:
        p_opts["print_time"] = 0
        s_opts["print_level"] = 0
        s_opts["sb"] = "yes"
    opti.solver("ipopt", p_opts, s_opts)

    try:
        sol1 = opti.solve()
        x_sol, u_sol = sol1.value(x), sol1.value(u)
        return u_sol[0]
    except:
        return 0
