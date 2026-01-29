import casadi as ca
import numpy as np

class HalfCheetahMPC:
    def __init__(self, n_joints=6, dt=0.02, horizon=20,
                 w_vel=5.0, w_u=1e-2, w_du=1e-3, w_term=5.0,
                 u_min=None, u_max=None):
        """
        n_joints: number of actuated joints (6 for HalfCheetah)
        dt: control timestep
        horizon: number of steps N
        weights: stage and terminal weights
        u_min/u_max: joint accel bounds (np.array shape [n_joints])
        """
        self.nj = n_joints
        self.dt = dt
        self.N = horizon
        self.w_vel = w_vel
        self.w_u = w_u
        self.w_du = w_du
        self.w_term = w_term

        self.u_min = u_min if u_min is not None else -10.0*np.ones(self.nj)
        self.u_max = u_max if u_max is not None else  10.0*np.ones(self.nj)

        # CasADi symbols
        nx = 2*self.nj  # x = [q, dq]
        nu = self.nj    # u = ddq

        X = ca.SX.sym('X', nx, self.N+1)    # [x0 ... xN]
        U = ca.SX.sym('U', nu, self.N)      # [u0 ... uN-1]

        # Parameters (given each solve)
        x0 = ca.SX.sym('x0', nx)            # measured initial state
        vx_ref = ca.SX.sym('vx_ref', 1)     # reference forward velocity
        q_nom = ca.SX.sym('q_nom', self.nj) # nominal joint posture for terminal cost

        cost = 0
        g = []        # constraints
        lb_g = []
        ub_g = []

        # Initial condition constraint: x0 == X[:,0]
        g.append(X[:,0] - x0)
        lb_g += [0]*nx
        ub_g += [0]*nx

        # Helper to split state
        def split_x(x):
            q  = x[:self.nj]
            dq = x[self.nj:]
            return q, dq

        # Your forward velocity model from state (you may replace this)
        def vx_of_state(x):
            # Simple proxy: weighted sum of hip joint velocity as a placeholder.
            # For better performance, pass the measured vx from env as a parameter
            # and penalize (measured_vx - vx_ref). Or learn a linear regressor.
            _, dq = split_x(x)
            return dq[0]  # placeholder

        # Dynamics: x_{k+1} = f(x_k, u_k)
        def f_discrete(x, u):
            q, dq = split_x(x)
            q_next  = q  + self.dt*dq
            dq_next = dq + self.dt*u
            return ca.vertcat(q_next, dq_next)

        # Build stage costs and dynamics constraints
        prev_u = None
        for k in range(self.N):
            xk = X[:,k]
            uk = U[:,k]

            # Stage cost
            vx = vx_of_state(xk)
            vel_err = vx - vx_ref
            c_vel = self.w_vel * (vel_err**2)
            c_u   = self.w_u   * ca.dot(uk, uk)
            c_du  = 0 if prev_u is None else self.w_du*ca.dot(uk - prev_u, uk - prev_u)

            cost += c_vel + c_u + c_du

            # Dynamics constraint: X[:,k+1] - f(X[:,k], U[:,k]) = 0
            x_next = f_discrete(xk, uk)
            g.append(X[:,k+1] - x_next)
            lb_g += [0]*nx
            ub_g += [0]*nx

            # Input bounds (box)
            # These are easier as variable bounds, but you can keep as inequality constraints if you prefer
            prev_u = uk

        # -------- Terminal cost: this is where the "final state" enters --------
        xT = X[:, self.N]
        vx_T = vx_of_state(xT)
        vel_err_T = vx_T - vx_ref
        qT, _ = split_x(xT)
        cost += self.w_term * (vel_err_T**2) + 1e-2 * ca.dot(qT - q_nom, qT - q_nom)
        # ----------------------------------------------------------------------

        # Vectorize decision vars
        w = [X.reshape((-1,1)), U.reshape((-1,1))]
        w = ca.vertcat(*w)

        # Variable bounds (use wide bounds on states; tighter on inputs)
        lbw = []
        ubw = []
        # X bounds
        big = 1e6
        lbw += [-big]*((self.N+1)*nx)
        ubw += [ big]*((self.N+1)*nx)
        # U bounds
        for _ in range(self.N):
            lbw += list(self.u_min)
            ubw += list(self.u_max)

        # Build NLP
        nlp = {"x": w, "f": cost, "g": ca.vertcat(*g),
               "p": ca.vertcat(x0, vx_ref, q_nom)}

        # Choose a solver (SQP-ish with ipopt)
        opts = {"ipopt.print_level": 0, "print_time": 0}
        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        # Shapes for unpacking
        self.nx = nx
        self.nu = nu

        # Warm-start memory
        self.warm_w = np.zeros(((self.N+1)*nx + self.N*nu, 1))
        self.warm_lam_g = np.zeros((len(lb_g), 1))

        # Store bounds
        self.lbw = np.array(lbw)
        self.ubw = np.array(ubw)
        self.lbg = np.array(lb_g)
        self.ubg = np.array(ub_g)

    def solve(self, x0_val, vx_ref_val, q_nom_val):
        """
        x0_val: np.ndarray (2*nj,)
        vx_ref_val: float
        q_nom_val: np.ndarray (nj,)
        Returns: u0* (first joint acceleration) as np.ndarray (nj,)
        """
        p = np.concatenate([x0_val, np.array([vx_ref_val]), q_nom_val]).reshape((-1,1))

        # Warm-start:
        arg = {
            "x0": self.warm_w,
            "lbx": self.lbw, "ubx": self.ubw,
            "lbg": self.lbg, "ubg": self.ubg,
            "lam_g0": self.warm_lam_g,
            "p": p
        }
        sol = self.solver(**arg)
        w_opt = np.array(sol["x"]).reshape((-1,1))

        # Unpack first control
        # Layout: [X(:); U(:)]
        NX = (self.N+1)*self.nx
        U_flat = w_opt[NX:]             # length N*nu
        u0 = U_flat[:self.nu].flatten()  # first nu

        # Save warm-start (shift previous plan forward)
        # Shift X and U to warm-start next solve
        X_flat = w_opt[:NX].copy()
        U_next = np.zeros_like(U_flat)
        U_next[:-self.nu] = U_flat[self.nu:]
        w_guess = np.vstack([X_flat, U_next])
        self.warm_w = 0.9*w_guess + 0.1*w_opt  # light filter

        # Not updating lam_g here for simplicity

        return u0

    # ---- Torque mapping (call MuJoCo inverse dynamics in your code) ----
    # In your control loop, after solve(), do:
    # tau = compute_torque_from_qdd(mj_model, mj_data, u0, actuated_idx)
