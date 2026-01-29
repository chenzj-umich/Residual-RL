"""
File: ilqr_mpc.py
A compact, readable iLQR-based MPC controller that plugs into a MuJoCo Gymnasium-like env.
Tested interface assumptions:
- env.get_state() -> np.ndarray of shape (nq+nv,)
- env.set_state(x: np.ndarray, recompute: bool=True)  # uses mujoco.mj_forward under the hood
- env.action_bounds() -> (u_min: np.ndarray, u_max: np.ndarray)
- env.action_dim -> int
- env.get_dt() -> float
- env.step(u: np.ndarray) -> (x_next, reward, terminated, truncated, info)
- Optionally: env.model, env.data for low-level access (not strictly required)

Core classes:
- MuJoCoDynamics: wraps the env for forward rollout and finite-diff linearization (A_t, B_t)
- QuadraticCost (customizable): stage cost and terminal cost + derivatives
- ILQR: solver (backward pass, regularization, forward line-search)
- MPCController: receding-horizon wrapper with warm-start of control sequence

This file favors clarity over absolute performance. It is structured for you to extend
(e.g., JAX autograd, learned models, constraint handling, contact-aware costs, etc.).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Optional

# ----------------------------
# Utilities
# ----------------------------

def clamp(u: np.ndarray, umin: np.ndarray, umax: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(u, umin), umax)


def symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


# ----------------------------
# Dynamics wrapper (finite-diff)
# ----------------------------

@dataclass
class MuJoCoDynamics:
    env: object
    dt: Optional[float] = None
    du: Optional[int] = None
    x_eps: float = 1e-5
    u_eps: float = 1e-5

    def __post_init__(self):
        if self.dt is None:
            self.dt = float(self.env.get_dt())
        if self.du is None:
            self.du = int(self.env.action_dim)
        self.umin, self.umax = self.env.action_bounds()
        self.umin = np.asarray(self.umin, dtype=np.float64)
        self.umax = np.asarray(self.umax, dtype=np.float64)

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """One-step transition using the *real* simulator: x_{t+1} = f(x_t, u_t).
        Uses env.set_state + a single env.step. Deterministic step for planning.
        """
        # Save state
        x_backup = self.env.get_state()
        # Write state and roll one step with action u
        self.env.set_state(x, recompute=True)
        x1, _, _, _, _ = self.env.step(clamp(u, self.umin, self.umax))
        # Restore state (do not advance the real env during planning)
        self.env.set_state(x_backup, recompute=True)
        return x1

    def linearize(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Finite-difference linearization around (x,u):
        A = df/dx (nx×nx), B = df/du (nx×nu).
        """
        nx = x.size
        nu = u.size
        A = np.zeros((nx, nx), dtype=np.float64)
        B = np.zeros((nx, nu), dtype=np.float64)
        f0 = self.f(x, u)

        # df/dx
        for i in range(nx):
            dx = np.zeros_like(x)
            dx[i] = self.x_eps
            f_plus = self.f(x + dx, u)
            f_minus = self.f(x - dx, u)
            A[:, i] = (f_plus - f_minus) / (2.0 * self.x_eps)

        # df/du
        for j in range(nu):
            du = np.zeros_like(u)
            du[j] = self.u_eps
            f_plus = self.f(x, u + du)
            f_minus = self.f(x, u - du)
            B[:, j] = (f_plus - f_minus) / (2.0 * self.u_eps)

        return A, B


# ----------------------------
# Cost (customize as needed)
# ----------------------------

@dataclass
class QuadraticCost:
    Q: np.ndarray  # (nx, nx)
    R: np.ndarray  # (nu, nu)
    Qf: np.ndarray  # (nx, nx)
    x_goal: Optional[np.ndarray] = None

    def stage(self, x: np.ndarray, u: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        """l(x,u) and derivatives up to second order."""
        if self.x_goal is None:
            dx = x
        else:
            dx = x - self.x_goal
        l = 0.5 * (dx.T @ self.Q @ dx) + 0.5 * (u.T @ self.R @ u)
        lx = self.Q @ dx
        lu = self.R @ u
        lxx = self.Q
        luu = self.R
        lux = np.zeros((u.size, x.size))
        return float(l), {"lx": lx, "lu": lu, "lxx": lxx, "luu": luu, "lux": lux}

    def terminal(self, x: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        if self.x_goal is None:
            dx = x
        else:
            dx = x - self.x_goal
        l = 0.5 * (dx.T @ self.Qf @ dx)
        Vx = self.Qf @ dx
        Vxx = self.Qf
        return float(l), {"Vx": Vx, "Vxx": Vxx}


# ----------------------------
# iLQR Solver
# ----------------------------

@dataclass
class ILQR:
    dynamics: MuJoCoDynamics
    cost: QuadraticCost
    N: int  # horizon length (number of *controls*)
    max_iters: int = 30
    reg_min: float = 1e-6
    reg_max: float = 1e6
    reg_init: float = 1e-3
    accept_ratio: float = 1e-4
    alphas: Tuple[float, ...] = (1.0, 0.6, 0.3, 0.1, 0.03, 0.01)

    def solve(self, x0: np.ndarray, U_init: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        nx = x0.size
        nu = self.dynamics.du
        umin, umax = self.dynamics.umin, self.dynamics.umax

        # Warm start
        if U_init is None:
            U = np.zeros((self.N, nu))
        else:
            U = U_init.copy()

        X = np.zeros((self.N + 1, nx))
        K = np.zeros((self.N, nu, nx))  # feedback gains
        k = np.zeros((self.N, nu))      # feedforward terms

        lam = self.reg_init

        def rollout(x0, U, K=None, k=None, alpha: float = 1.0):
            Xr = np.zeros_like(X)
            Ur = np.zeros_like(U)
            Xr[0] = x0
            J = 0.0
            for t in range(self.N):
                if K is not None and k is not None:
                    du = alpha * k[t] + K[t] @ (Xr[t] - X[t])  # note: uses previous X[t]
                    u = clamp(U[t] + du, umin, umax)
                else:
                    u = clamp(U[t], umin, umax)
                Ur[t] = u
                lt, d = self.cost.stage(Xr[t], u)
                J += lt
                Xr[t + 1] = self.dynamics.f(Xr[t], u)
            lT, dT = self.cost.terminal(Xr[-1])
            J += lT
            return J, Xr, Ur

        # Initial rollout
        J, X, U = rollout(x0, U)

        for it in range(self.max_iters):
            # Linearize dynamics and expand cost
            nx = X.shape[1]
            nu = U.shape[1]
            A = [None] * self.N
            B = [None] * self.N
            l = np.zeros(self.N)
            lx = np.zeros((self.N, nx))
            lu = np.zeros((self.N, nu))
            lxx = np.zeros((self.N, nx, nx))
            luu = np.zeros((self.N, nu, nu))
            lux = np.zeros((self.N, nu, nx))

            for t in range(self.N):
                A[t], B[t] = self.dynamics.linearize(X[t], U[t])
                lt, d = self.cost.stage(X[t], U[t])
                l[t] = lt
                lx[t], lu[t], lxx[t], luu[t], lux[t] = d["lx"], d["lu"], d["lxx"], d["luu"], d["lux"]

            lT, dT = self.cost.terminal(X[-1])
            Vx = dT["Vx"].copy()
            Vxx = dT["Vxx"].copy()

            # Backward pass with Levenberg–Marquardt regularization
            diverged = False
            for t in reversed(range(self.N)):
                Qx  = lx[t] + A[t].T @ Vx
                Qu  = lu[t] + B[t].T @ Vx
                Qxx = lxx[t] + A[t].T @ Vxx @ A[t]
                Quu = luu[t] + B[t].T @ Vxx @ B[t]
                Qux = lux[t] + B[t].T @ Vxx @ A[t]

                # regularize (ensure Quu PD)
                Quu_reg = Quu + lam * np.eye(nu)
                try:
                    Quu_inv = np.linalg.inv(symmetrize(Quu_reg))
                except np.linalg.LinAlgError:
                    diverged = True
                    break

                k[t] = - Quu_inv @ Qu
                K[t] = - Quu_inv @ Qux

                Vx  = Qx + K[t].T @ Quu @ k[t] + K[t].T @ Qu + Qux.T @ k[t]
                Vxx = Qxx + K[t].T @ Quu @ K[t] + K[t].T @ Qux + Qux.T @ K[t]
                Vxx = symmetrize(Vxx)

            if diverged:
                lam = min(self.reg_max, lam * 10.0)
                continue

            # Forward line-search
            accepted = False
            for alpha in self.alphas:
                J_new, X_new, U_new = rollout(x0, U, K, k, alpha)
                dJ = J - J_new
                if dJ > self.accept_ratio * (abs(J)):
                    J, X, U = J_new, X_new, U_new
                    lam = max(self.reg_min, lam / 5.0)
                    accepted = True
                    break

            if not accepted:
                lam = min(self.reg_max, lam * 10.0)

            # Convergence (very simple criterion)
            if accepted and dJ < 1e-6:
                break

        return {"X": X, "U": U, "K": K, "k": k, "J": J, "lambda": lam}


# ----------------------------
# MPC wrapper (receding horizon)
# ----------------------------

@dataclass
class MPCController:
    ilqr: ILQR
    warm_start: bool = True

    def __post_init__(self):
        self._U_ws: Optional[np.ndarray] = None

    def act(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        if self.warm_start and self._U_ws is not None:
            U0 = np.vstack([self._U_ws[1:], self._U_ws[-1]])  # shift
        else:
            U0 = None
        sol = self.ilqr.solve(x0=x, U_init=U0)
        u = sol["U"][0]
        if self.warm_start:
            self._U_ws = sol["U"]
        return u, sol


# ----------------------------
# Minimal usage example (pseudo-main)
# ----------------------------
if __name__ == "__main__":
    # Your env should provide the small API described above.
    from env_warpper import HalfCheetahEnv, HCEnvConfig  # replace with your actual paths

    cfg = HCEnvConfig(render=False, obs_as_state=True, seed=0)
    env = HalfCheetahEnv(cfg)

    x0, _ = env.reset()
    nx = x0.size
    nu = env.action_dim

    Q  = np.eye(nx) * 0.0
    R  = np.eye(nu) * 1e-3
    Qf = np.eye(nx) * 1.0

    # Example: track a target forward velocity via a soft state penalty.
    # Suppose env.get_forward_speed() maps from x to scalar v; you can embed this into x_goal.
    x_goal = x0.copy()
    # If you have a selector to penalize only body velocity components, set Q accordingly.

    cost = QuadraticCost(Q=Q, R=R, Qf=Qf, x_goal=x_goal)
    dyn = MuJoCoDynamics(env)
    ilqr = ILQR(dynamics=dyn, cost=cost, N=25, max_iters=30)
    mpc = MPCController(ilqr)

    for t in range(300):
        x = env.get_state()
        u, sol = mpc.act(x)
        env.step(u)
        if (t % 50) == 0:
            print(f"t={t} J={sol['J']:.3f} |u|={np.linalg.norm(u):.3f}")

    env.close()
