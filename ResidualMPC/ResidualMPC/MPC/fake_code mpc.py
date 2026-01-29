from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
import mujoco  # 仅用于 mj_forward，可按需删除
import env_warpper

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple




cfg = env_warpper.HCEnvConfig(render=True, obs_as_state=False, seed=0)
env = env_warpper.HalfCheetahEnv(cfg)

x0, _ = env.reset()


import numpy as np

def compute_com(env) -> np.ndarray:
    masses = env.model.body_mass[1:]           # (nb-1,)
    xipos  = env.data.xipos[1:, :]             # (nb-1, 3) 世界系质心位置
    M = masses.sum()
    com_pos = (masses[:, None] * xipos).sum(axis=0) / M
    return np.array([com_pos[0], com_pos[2]], dtype=float)  # 取 x,z

def compute_theta(env) -> float:
    return float(env.data.qpos[2])              # planar: [x, z, theta, ...]

def compute_com_vel(env) -> np.ndarray:
    masses = env.model.body_mass[1:]            # (nb-1,)

    # cvel: (nb, 6) 每个刚体的局部6D速度 [wx, wy, wz, vx, vy, vz]（在刚体局部坐标系）
    cvel = env.data.cvel                        # (nb, 6)

    # xmat: (nb, 9) 行主序展平的旋转矩阵，把局部量旋到世界系
    xmat = env.data.xmat.reshape(-1, 3, 3)      # (nb, 3, 3)

    # 取线速度的局部部分并旋到世界系
    v_local = cvel[1:, 3:6]                     # (nb-1, 3)
    v_world = np.einsum('bij,bj->bi', xmat[1:], v_local)  # (nb-1, 3)

    M = masses.sum()
    com_vel = (masses[:, None] * v_world).sum(axis=0) / M
    return np.array([com_vel[0], com_vel[2]], dtype=float)  # 取 vx, vz

def compute_theta_dot(env) -> float:
    return float(env.data.qvel[2])              # planar: [xdot, zdot, thetadot, ...]

def get_single_rigidbody_state(env) -> np.ndarray:
    px, pz   = compute_com(env)
    vx, vz   = compute_com_vel(env)
    th       = compute_theta(env)
    w        = compute_theta_dot(env)
    return np.array([px, pz, th, vx, vz, w], dtype=float)


def get_grf_cfrc(env, left_body="bfoot", right_body="ffoot") -> np.ndarray: 
    model, data = env.model, env.data
    lid = model.body(left_body).id
    rid = model.body(right_body).id
    fL_world = data.cfrc_ext[lid, 3:6]
    fR_world = data.cfrc_ext[rid, 3:6]
    FxL, FzL = float(fL_world[0]), float(fL_world[2])
    FxR, FzR = float(fR_world[0]), float(fR_world[2])
    return np.array([FxL, FzL, FxR, FzR], dtype=np.float64)


def debug_list_contacts(env):
    m, d = env.model, env.data
    names = []
    for i in range(d.ncon):
        con = d.contact[i]
        b1 = m.geom_bodyid[con.geom1]
        b2 = m.geom_bodyid[con.geom2]
        name1 = m.body(b1).name
        name2 = m.body(b2).name
        names.append((name1, name2))
    return names



for t in range(2000):
  low, high = env.action_bounds()
  amplitude = 2.0  # 扩大动作振幅
  u = np.random.uniform(low * amplitude, high * amplitude, size=(env.action_dim,))
  u = np.clip(u, low * amplitude, high * amplitude)

  # print(u)
  # masses = env.model.body_mass[1:]          # (nb-1,)
  # xipos  = env.data.xipos[1:, :]            # (nb-1, 3) 各刚体的质心在世界坐标
  # M = masses.sum()
  # com_pos = (masses[:, None] * xipos).sum(axis=0) / M
  x = get_single_rigidbody_state(env)
  y = get_grf_cfrc(env)
  # name = debug_list_contacts(env)
  # print(name)
  # print(x)
  print(y)

  x1, r, term, trunc, _ = env.step(u)


env.close()


#print("com_pos:", com_pos)  #质心位置
#print(env.nq, env.nv, env.nu)
#xpos,zpos, pitch
#print(env.get_state())