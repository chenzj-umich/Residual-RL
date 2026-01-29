

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
import mujoco  # 仅用于 mj_forward，可按需删除
import env_warpper
#x0()
cfg = env_warpper.HCEnvConfig(render=True, obs_as_state=False, seed=0)
env = env_warpper.HalfCheetahEnv(cfg)
x0, _ = env.reset()
print("info:", env.info())
print("x0.shape:", x0.shape, "dt:", env.get_dt(), "action_dim:", env.action_dim)
dt = env.get_dt()
print(dt)


for t in range(1):
    u = np.random.uniform(*env.action_bounds(), size=(env.action_dim,))
    print(u)
    print(x0)
    x1, r, term, trunc, _ = env.step(u)
    print(f"t={t} r={r:.3f} v={env.get_forward_speed():.3f} term={term} trunc={trunc}")
env.close()



####change friction####
# fr = env.model.geom_friction  
# print( "friction:", env.model.geom_friction)      # shape = (ngeom, 3)
# fr[:, 0] = 0.25
# env.model.geom_friction[:] = fr
# print( "friction:", env.model.geom_friction)