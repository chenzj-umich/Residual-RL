

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
import mujoco  # 仅用于 mj_forward，可按需删除
import env_warpper

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

@dataclass
class QuadraticCost:
    """
    l(x,u) = 1/2 (x - x_goal)^T Q (x - x_goal) + 1/2 u^T R u
    l_T(x) = 1/2 (x - x_goal)^T Qf (x - x_goal)

    返回值含必要的导数:
      stage():  l, {lx, lu, lxx, luu, lux}
      terminal(): lT, {Vx, Vxx}
    """
    Q:  np.ndarray        # (nx, nx)
    R:  np.ndarray        # (nu, nu)
    Qf: np.ndarray        # (nx, nx)
    x_goal: Optional[np.ndarray] = None  # (nx,)



    # ---------- 阶段代价 ---------- #
    def stage(self, x: np.ndarray, u: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:

        dx = x - (self.x_goal if self.x_goal is not None else 0.0)

        l   = 0.5 * (dx @ self.Q @ dx) + 0.5 * (u @ self.R @ u)
        lx  = self.Q @ dx
        lu  = self.R @ u
        lxx = self.Q
        luu = self.R
        lux = np.zeros((u.size, x.size), dtype=x.dtype)

        return float(l), {"lx": lx, "lu": lu, "lxx": lxx, "luu": luu, "lux": lux}

 
    def terminal(self, x: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:

        dx  = x - (self.x_goal if self.x_goal is not None else 0.0)

        lT  = 0.5 * (dx @ self.Qf @ dx)
        Vx  = self.Qf @ dx
        Vxx = self.Qf

        return float(lT), {"Vx": Vx, "Vxx": Vxx}



cfg = env_warpper.HCEnvConfig(render=True, obs_as_state=False, seed=0)
env = env_warpper.HalfCheetahEnv(cfg)
x0, _ = env.reset()
vx_idx      = env.nq + 0      # rootx_dot
pitch_idx   = 2           # rooty
pitchdot_idx= env.nq + 2      # rooty_dot

###weight matrix for cost function###
Q  = np.zeros((18, 18)); Qf = np.zeros((18, 18))
Q[vx_idx, vx_idx]           = 5.0
Q[pitch_idx, pitch_idx]     = 2.0
Q[pitchdot_idx, pitchdot_idx]= 0.5
Qf[:] = 10.0 * Q
R = np.eye(6) * 1e-3

x0, _ = env.reset()
x_goal = x0.copy()
x_goal[vx_idx] = 1.0


cost = QuadraticCost(Q=Q, R=R, Qf=Qf, x_goal=x_goal)



#print(env.nq, env.nv, env.nu)
#xpos,zpos, pitch
#print(env.get_state())