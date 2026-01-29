# hc_env.py
# Minimal HalfCheetah environment wrapper for MPC-style control
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
import mujoco  # 仅用于 mj_forward，可按需删除


@dataclass
class HCEnvConfig:
    env_id: str = "HalfCheetah-v5"
    xml_file: Optional[str] = None      # 可选：自定义 MJCF 路径
    seed: Optional[int] = 42
    render: bool = False                # True -> render_mode="human"
    obs_as_state: bool = False          # False: 返回 qpos‖qvel（适合MPC）；True: 返回RL obs


class HalfCheetahEnv:
    """Thin wrapper exposing full MuJoCo state and a clean step/reset API."""

    def __init__(self, cfg: HCEnvConfig = HCEnvConfig()):
        self.cfg = cfg

        make_kwargs: Dict[str, Any] = {}
        if cfg.render:
            make_kwargs["render_mode"] = "human"
        if cfg.xml_file is not None:
            make_kwargs["xml_file"] = cfg.xml_file

        self.env = gym.make(cfg.env_id, **make_kwargs)
        self.model = self.env.unwrapped.model
        self.data = self.env.unwrapped.data

        # seed（Gymnasium 推荐在 reset 时传 seed；同时尽量 seed spaces）
        if cfg.seed is not None:
            try:
                self.env.action_space.seed(cfg.seed)
                self.env.observation_space.seed(cfg.seed)
            except Exception:
                pass

        # dimensions
        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        self.nu = int(getattr(self.model, "nu", self.env.action_space.shape[0]))

        # dt(control time:0.05) = base_ts(least time:0.01) * frame_skip（steps to skip:5）
        base_ts = float(self.model.opt.timestep)
        frame_skip = int(getattr(self.env.unwrapped, "frame_skip", 1))
        self.dt = base_ts * frame_skip 

        # action bounds
        self.u_min = np.asarray(self.env.action_space.low, dtype=np.float64)
        self.u_max = np.asarray(self.env.action_space.high, dtype=np.float64)

        self._last_info: Dict[str, Any] = {}
        self._last_obs: Optional[np.ndarray] = None

    # ---------- core API ----------
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset env and return initial state (qpos‖qvel by default) and info."""
        obs, info = self.env.reset(seed=self.cfg.seed)
        self._last_obs, self._last_info = obs, info
        return (obs if self.cfg.obs_as_state else self.get_state(), info)

    def step(self, u: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply action u (auto-clipped), advance sim one step and return:
        next_state, reward, terminated, truncated, info
        - next_state is qpos‖qvel unless obs_as_state=True
        """
        ###frame_skip steps
        u = np.asarray(u, dtype=np.float64).ravel()
        u = np.clip(u, self.u_min, self.u_max)
        obs, reward, terminated, truncated, info = self.env.step(u)
        self._last_obs, self._last_info = obs, info
        x1 = obs if self.cfg.obs_as_state else self.get_state()
        return x1, float(reward), bool(terminated), bool(truncated), info

    # ---------- state helpers ----------
    def get_state(self) -> np.ndarray:
        """Return full physical state x = [qpos; qvel]."""
        qpos = np.array(self.data.qpos, dtype=np.float64).copy()
        qvel = np.array(self.data.qvel, dtype=np.float64).copy()
        return np.concatenate([qpos, qvel], axis=0)

    def set_state(self, x: np.ndarray, recompute: bool = True) -> None:
        """Write x=[qpos; qvel] back into MuJoCo (useful for MPC rollouts)."""
        x = np.asarray(x, dtype=np.float64).ravel()
        assert x.size == (self.nq + self.nv), f"state size {x.size} != nq+nv={self.nq+self.nv}"
        self.data.qpos[:] = x[: self.nq]
        self.data.qvel[:] = x[self.nq : self.nq + self.nv]
        if recompute:
            mujoco.mj_forward(self.model, self.data)

    # ---------- convenience ----------
    @property
    def state_dim(self) -> int:
        return self.nq + self.nv

    @property
    def action_dim(self) -> int:
        return self.nu

    def get_dt(self) -> float:
        return float(self.dt)

    def action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.u_min.copy(), self.u_max.copy()

    def get_forward_speed(self) -> float:
        """HalfCheetah 的前向速度通常是根关节 x 方向速度 data.qvel[0]."""
        return float(self.data.qvel[0])

    def info(self) -> Dict[str, Any]:
        return dict(nq=self.nq, nv=self.nv, nu=self.nu, dt=self.dt)

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


# quick test
if __name__ == "__main__":
    cfg = HCEnvConfig(render=True, obs_as_state=False, seed=0)
    env = HalfCheetahEnv(cfg)

    x0, _ = env.reset()

    # print("info:", env.info())
    # print("x0.shape:", x0.shape, "dt:", env.get_dt(), "action_dim:", env.action_dim)
    u = np.random.uniform(*env.action_bounds(), size=(env.action_dim,))
    print('action {}',u)
    for t in range(200):
        u = np.random.uniform(*env.action_bounds(), size=(env.action_dim,))
        x1, r, term, trunc, _ = env.step(u)
        print(env.nq,env.nv,env.nu)  #nq=9 nv=9 nu=6
        # print(f"t={t} r={r:.3f} v={env.get_forward_speed():.3f} term={term} trunc={trunc}")
        # print(env.get_state())
    env.close()



    # State-Space (name/joint/parameter):
    #     - rootx     slider      position (m)
    #     - rootz     slider      position (m)
    #     - rooty     hinge       angle (rad)
    #     - bthigh    hinge       angle (rad)
    #     - bshin     hinge       angle (rad)
    #     - bfoot     hinge       angle (rad)
    #     - fthigh    hinge       angle (rad)
    #     - fshin     hinge       angle (rad)
    #     - ffoot     hinge       angle (rad)

    #----------------------------------------------
    #     - rootx     slider      velocity (m/s)
    #     - rootz     slider      velocity (m/s)
    #     - rooty     hinge       angular velocity (rad/s)
    #     - bthigh    hinge       angular velocity (rad/s)
    #     - bshin     hinge       angular velocity (rad/s)
    #     - bfoot     hinge       angular velocity (rad/s)
    #     - fthigh    hinge       angular velocity (rad/s)
    #     - fshin     hinge       angular velocity (rad/s)
    #     - ffoot     hinge       angular velocity (rad/s)
    #----------------------------------------------
    # Actuators (name/actuator/parameter):
    #     - bthigh    hinge       torque (N m)
    #     - bshin     hinge       torque (N m)
    #     - bfoot     hinge       torque (N m)
    #     - fthigh    hinge       torque (N m)
    #     - fshin     hinge       torque (N m)
    #     - ffoot     hinge       torque (N m)
