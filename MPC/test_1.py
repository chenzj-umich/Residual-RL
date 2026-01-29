# quick test

# hc_env.py
# Minimal HalfCheetah environment wrapper for MPC-style control
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
import mujoco  # 仅用于 mj_forward，可按需删除
from dataclasses import dataclass
import env_wrapper as env
from dynamics import dynamics_rigid_simple
import casadi as ca


# ---------- Minimal portal adapter to your env ----------
class PortalAdapter:
    def __init__(self, env):
        self.env = env
        self._last_obs = None  # optional cache

    # --- state API ---
    def get_state(self):
        """Return (qpos, qvel) as copies, regardless of wrapper style."""
        st = self.env.get_state()
        if isinstance(st, tuple) and len(st) == 2:
            qpos, qvel = st
            return np.array(qpos, copy=True), np.array(qvel, copy=True)
        # Flat vector case
        x = np.array(st, copy=True).reshape(-1)
        nq = getattr(self.env, "nq", None)
        nv = getattr(self.env, "nv", None)
        if nq is None or nv is None:
            # fallback: split half/half
            nq = x.size // 2
            nv = x.size - nq
        assert x.size == (nq + nv), f"state size {x.size} != nq+nv={nq+nv}"
        return x[:nq].copy(), x[nq:nq+nv].copy()

    def set_state(self, qpos, qvel):
        """Call the wrapper's set_state in whichever signature it supports."""
        # Prefer two-argument signature if available
        try:
            return self.env.set_state(qpos, qvel)
        except TypeError:
            # Wrapper expects a single concatenated vector
            x = np.concatenate([np.asarray(qpos).reshape(-1),
                                np.asarray(qvel).reshape(-1)], axis=0)
            return self.env.set_state(x)
        except AssertionError:
            # The wrapper asserted on length; pass concatenated vector
            x = np.concatenate([np.asarray(qpos).reshape(-1),
                                np.asarray(qvel).reshape(-1)], axis=0)
            return self.env.set_state(x)

    # --- control/step ---
    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self._last_obs = obs
        return obs, rew, term, trunc, info

    # --- misc helpers ---
    def action_low(self):
        low, high = self.env.action_bounds()
        low = np.array(low, dtype=np.float32)
        return low if low.ndim else np.full(self.env.action_dim, float(low), dtype=np.float32)

    def action_high(self):
        low, high = self.env.action_bounds()
        high = np.array(high, dtype=np.float32)
        return high if high.ndim else np.full(self.env.action_dim, float(high), dtype=np.float32)

    def observe(self):
        # prefer last obs; if not available, synthesize from state
        if self._last_obs is not None:
            return self._last_obs
        qpos, qvel = self.get_state()
        # Half-Cheetah obs ~ [8 qpos (no x), 9 qvel] — if you have a custom obs, replace this.
        # Here we just concatenate for a placeholder:
        return np.concatenate([qpos[:8], qvel])  # safe default for logging/vel extract

    def get_dt(self):
        return self.env.get_dt()

    def get_forward_speed(self):
        # your wrapper already has this:
        return float(self.env.get_forward_speed())


@dataclass
class NMPCConfig:
    horizon: int = 20               # prediction steps
    w_vel: float = 5.0              # velocity tracking weight
    w_u: float = 1e-3               # torque magnitude weight
    w_du: float = 1e-3              # torque rate weight
    w_term: float = 5.0             # terminal velocity weight
    vx_ref: float = 1.5             # target forward speed (m/s)
    # Joint-space surrogate (diagonal) dynamics parameters:
    M_diag: np.ndarray = None       # shape (act_dim,)
    D_diag: np.ndarray = None       # shape (act_dim,)
    solver: str = "ipopt"           # CasADi NLP solver
    ipopt_verbosity: int = 0
    verbose: bool = False

def _extract_vx_from_obs(obs: np.ndarray) -> float:
    # Half-Cheetah: last 9 entries are velocities; vx is index 0 of them
    if obs.shape[0] >= 17:
        return float(obs[-9:][0])
    # fallback: if obs unknown, rely on 0
    return 0.0

def _running_cost(obs, act, cfg: MPCConfig, prev_act):
    vx = _extract_vx_from_obs(obs)
    c_vel = cfg.w_vel * (vx - cfg.target_velocity) ** 2
    c_act = cfg.w_act * float((act ** 2).sum())
    c_dact = 0.0 if prev_act is None else cfg.w_dact * float(((act - prev_act) ** 2).sum())
    return c_vel + c_act + c_dact

def _rollout_sequence(portal: PortalAdapter, init_qpos, init_qvel, action_seq, cfg: MPCConfig):
    portal.set_state(init_qpos, init_qvel)
    total = 0.0
    prev = None
    disc = 1.0
    for t in range(action_seq.shape[0]):
        act = np.clip(action_seq[t], portal.action_low(), portal.action_high())
        obs, rew, term, trunc, _ = portal.step(act)
        total += disc * _running_cost(obs, act, cfg, prev)
        disc *= cfg.gamma
        prev = act
        if term or trunc:  # penalize early termination
            total += (cfg.horizon - t - 1) * 10.0
            break
    return total

class CasadiTorqueMPC:
    """
    NMPC that optimizes a torque sequence using a simple joint-space rigid-body surrogate.
    State x_k = [q_act, dq_act] (2*na), control u_k = tau (na), na = env.action_dim.
    """
    def __init__(self, portal, cfg: NMPCConfig):
        self.p = portal
        self.cfg = cfg

        self.na = int(self.p.env.action_dim)
        self.dt = float(self.p.get_dt())

        # action bounds (torque bounds from env)
        self.u_lo = self.p.action_low().astype(np.float64)
        self.u_hi = self.p.action_high().astype(np.float64)

        # Default diagonal masses/dampings if not provided
        if cfg.M_diag is None:
            cfg.M_diag = np.ones(self.na) * 3.0     # effective inertia per joint (tune)
        if cfg.D_diag is None:
            cfg.D_diag = np.ones(self.na) * 0.5     # viscous damping per joint (tune)

        # State extraction mapping: by default, use the LAST na entries in qpos/qvel as actuated joints
        # If your model differs, pass a custom function when you instantiate (see below).
        self._state_extract = lambda qpos, qvel: (qpos[-self.na:], qvel[-self.na:])

        # Build CasADi problem once
        self._build_nlp()

        # Warm starts
        self._w_guess = np.zeros((self._nx*(self.N+1) + self._nu*self.N, 1))
        self._lam_g = None

    # ---------- public helpers ----------

    def set_state_extractor(self, fn):
        """
        fn(qpos: np.ndarray, qvel: np.ndarray) -> (q_act, dq_act)
        Use this if your actuated joints are not the last 'na'.
        """
        self._state_extract = fn

    def set_vx_ref(self, vx_target: float):
        self.cfg.vx_ref = float(vx_target)

    def plan(self):
        """Solve MPC and return the first torque (na,)."""
        qpos, qvel = self.p.get_state()
        q_act, dq_act = self._state_extract(qpos, qvel)
        x0 = np.concatenate([q_act, dq_act])

        # Parameters (constant over horizon here)
        vx_ref = np.array([self.cfg.vx_ref])
        q_nom = np.zeros_like(q_act)  # terminal posture nominal (can set to a crouch)

        u0 = self._solve_once(x0, vx_ref, q_nom)
        # Clip to env torque bounds for safety
        return np.clip(u0, self.u_lo, self.u_hi)

    # ---------- CasADi model & solver ----------

    def _build_nlp(self):
        cfg = self.cfg
        self.N = int(cfg.horizon)
        na = self.na
        dt = self.dt

        # Symbols
        nx = 2*na
        nu = na
        X = ca.SX.sym("X", nx, self.N+1)         # states x_0..x_N
        U = ca.SX.sym("U", nu, self.N)           # torques u_0..u_{N-1}
        x0 = ca.SX.sym("x0", nx)                  # measured initial state (q_act,dq_act)
        vx_ref = ca.SX.sym("vx_ref", 1)           # target forward velocity
        q_nom = ca.SX.sym("q_nom", na)            # nominal terminal posture

        # Split helpers
        def split_x(x):
            q  = x[:na]
            dq = x[na:]
            return q, dq

        # Velocity proxy: vx ≈ C * dq (tune the C vector if needed)
        C = ca.DM.zeros((1, na))
        C[0,0] = 1.0    # start simple: use joint-0 velocity; adjust for your model
        def vx_of_state(x):
            _, dq = split_x(x)
            return ca.mtimes(C, dq)  # (1xna)*(na,) -> (1,1)

        # Joint-space surrogate dynamics:
        M_inv = ca.DM(np.diag(1.0 / self.cfg.M_diag))
        D     = ca.DM(np.diag(self.cfg.D_diag))
        def f_disc(x, u):
            q, dq = split_x(x)
            # dqdot = M^{-1}(u - D dq)
            dqdot = ca.mtimes(M_inv, (u - ca.mtimes(D, dq)))
            q_next  = q  + dt*dq
            dq_next = dq + dt*dqdot
            return ca.vertcat(q_next, dq_next)

        # Dynamics function
        f_disc = dynamics_rigid_simple(dt, self.cfg.M_diag, self.cfg.D_diag)

        # Cost & constraints
        cost = 0
        g = []; lbg = []; ubg = []

        # Initial condition
        g.append(X[:,0] - x0); lbg += [0]*nx; ubg += [0]*nx

        prev_u = None
        for k in range(self.N):
            xk, uk = X[:,k], U[:,k]
            # stage cost
            vx_k = vx_of_state(xk)                 # (1,1)
            c_vel = cfg.w_vel * (vx_k - vx_ref)**2 # scalar
            c_u   = cfg.w_u   * ca.dot(uk, uk)
            c_du  = 0 if prev_u is None else cfg.w_du*ca.dot(uk - prev_u, uk - prev_u)
            cost += c_vel + c_u + c_du

            # dynamics
            x_next = f_disc(xk, uk)
            g.append(X[:,k+1] - x_next); lbg += [0]*nx; ubg += [0]*nx

            prev_u = uk

        # Terminal cost (final state x_N)
        xT = X[:, self.N]
        vx_T = vx_of_state(xT)
        qT, _ = split_x(xT)
        cost += cfg.w_term * (vx_T - vx_ref)**2 + 1e-2*ca.dot(qT - q_nom, qT - q_nom)

        # Pack decision vars
        w = ca.vertcat(X.reshape((-1,1)), U.reshape((-1,1)))

        # Bounds on decision vars
        lbx = []; ubx = []
        BIG = 1e6
        # state bounds (wide)
        lbx += [-BIG]*((self.N+1)*nx); ubx += [ BIG]*((self.N+1)*nx)
        # input bounds (torque)
        for _ in range(self.N):
            lbx += list(self.u_lo); ubx += list(self.u_hi)

        # Build NLP
        nlp = {"x": w, "f": cost, "g": ca.vertcat(*g), "p": ca.vertcat(x0, vx_ref, q_nom)}
        opts = {
            "ipopt.print_level": self.cfg.ipopt_verbosity,
            "print_time": 0,
            "ipopt.max_iter": 200,
        }
        self.solver = ca.nlpsol("solver", self.cfg.solver, nlp, opts)

        # Store shapings
        self._nx, self._nu = nx, nu
        self._nX = (self.N+1)*nx
        self._nU = self.N*nu
        self._lbx = np.array(lbx); self._ubx = np.array(ubx)
        self._lbg = np.array(lbg); self._ubg = np.array(ubg)

    def _solve_once(self, x0_val: np.ndarray, vx_ref_val: np.ndarray, q_nom_val: np.ndarray):
        # Pack parameters
        p = np.concatenate([x0_val, vx_ref_val, q_nom_val]).reshape((-1,1))

        # Warm-start the decision vector (shift previous U forward)
        if self._w_guess is None or self._w_guess.shape[0] != (self._nX + self._nU):
            self._w_guess = np.zeros((self._nX + self._nU, 1))
        else:
            Xg = self._w_guess[:self._nX]
            Ug = self._w_guess[self._nX:]
            Ug_shift = np.zeros_like(Ug)
            Ug_shift[:-self._nu] = Ug[self._nu:]  # shift by 1
            self._w_guess = np.vstack([Xg, Ug_shift])

        arg = {
            "x0": self._w_guess,
            "lbx": self._lbx, "ubx": self._ubx,
            "lbg": self._lbg, "ubg": self._ubg,
            "p": p
        }
        if self._lam_g is not None:
            arg["lam_g0"] = self._lam_g

        sol = self.solver(**arg)
        w_opt = np.array(sol["x"]).reshape((-1,1))
        self._w_guess = w_opt
        try:
            self._lam_g = np.array(sol["lam_g"]).reshape((-1,1))
        except Exception:
            pass

        # Extract first control u0
        U_flat = w_opt[self._nX:]
        u0 = U_flat[:self._nu].flatten()
        return u0




# ---------- Your quick test, now with MPC ----------
if __name__ == "__main__":
    cfg = env.HCEnvConfig(render=True, obs_as_state=False, seed=0)
    env = env.HalfCheetahEnv(cfg)
    portal = PortalAdapter(env)

    # --- Build NMPC ---
    nmpc_cfg = NMPCConfig(
        horizon=20,
        vx_ref=1.5,
        w_vel=6.0, w_u=1e-3, w_du=1e-3, w_term=8.0,
        # Rough joint inertias & damping (tune per joint if needed):
        M_diag=np.array([3.0, 2.5, 2.0, 2.0, 1.8, 1.8]),
        D_diag=np.array([0.5, 0.5, 0.4, 0.4, 0.3, 0.3]),
        ipopt_verbosity=0,
    )
    mpc = CasadiTorqueMPC(portal, nmpc_cfg)

    # OPTIONAL: if your actuated joints are not the last 6, define the extractor:
    # mpc.set_state_extractor(lambda qpos, qvel: (qpos[idxs], qvel[idxs]))

    try:
        x0, _ = env.reset()
        print("info:", env.info())
        print("x0.shape:", np.array(x0).shape, "dt:", env.get_dt(), "action_dim:", env.action_dim)

        ep_ret = 0.0
        for t in range(100):
            # 1) Solve NMPC for current torque
            u = mpc.plan()  # returns torque (na,)

            # 2) Apply to real env
            x1, r, term, trunc, _ = env.step(u)
            ep_ret += r

            if (t+1) % 20 == 0:
                print(f"t={t+1:03d}  r={r:+.3f}  vx={portal.get_forward_speed():.3f}  |u|={np.linalg.norm(u):.2f}")

            if term or trunc:
                print(f"Episode ended at t={t}, return={ep_ret:.2f}")
                break

    finally:
        env.close()

