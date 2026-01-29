import casadi as ca
import numpy as np

def dynamics_rigid_simple(dt: float,
                          M_diag: np.ndarray,
                          D_diag: np.ndarray):
    """
    Returns a CasADi function f_disc(x,u) implementing a simple
    joint-space rigid-body surrogate with semi-implicit Euler.

    x = [q (6), dq (6)]   shape: (12,)
    u = tau (6)           shape: (6,)
    """
    na = len(M_diag)
    nx = 2 * na
    x = ca.SX.sym('x', nx)
    u = ca.SX.sym('u', na)

    q  = x[:na]
    dq = x[na:]

    M_inv = ca.DM(np.diag(1.0 / M_diag))
    D     = ca.DM(np.diag(D_diag))

    # ddq = M^{-1}(tau - D dq)
    ddq   = ca.mtimes(M_inv, (u - ca.mtimes(D, dq)))

    # semi-implicit Euler
    dq_next = dq + dt * ddq
    q_next  = q  + dt * dq_next

    x_next  = ca.vertcat(q_next, dq_next)
    return ca.Function('f_disc', [x, u], [x_next])
