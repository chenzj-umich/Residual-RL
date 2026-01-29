import argparse
import os
import sys
import gymnasium as gym
import numpy as np
import torch
from ResidualMPC.policies.MLP_policy import MLPPolicySL
from ResidualMPC.infrastructure import pytorch_util as ptu
import mujoco
from mujoco import mjtObj
import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def get_repo_root():
    """Get the absolute path to the repository root."""
    # Current file: ResidualMPC/scripts/IL_Test.py
    # Go up 3 levels: scripts -> ResidualMPC -> ResidualMPC (root)
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def compute_com(env) -> np.ndarray:
    """
    返回 torso 的 COM (世界系下)，输出 [x, z]
    """
    model = env.model
    data  = env.data

    # 用 mujoco.mj_name2id 在 body 名字表里查 "torso"
    torso_id = mujoco.mj_name2id(model, mjtObj.mjOBJ_BODY, "torso")

    if torso_id < 0:
        # 名字没找到，帮你打印一下所有 body 名字方便检查
        print("[compute_com] Error: body name 'torso' not found in model.")
        print("Available body names:")
        # model.body('name').id 也是新接口的一部分，但这里直接从 names 里扫一遍更保险
        for i in range(model.nbody):
            # mujoco 把名字存在一个大 names buffer 里，一般 gym 的 wrapper 会有 body_names 属性
            try:
                name = model.body(i).name
            except Exception:
                name = f"<id {i}>"
            print(f"  id={i}: {name}")
        raise ValueError("Body name 'torso' not found.")

    com_3d = data.xipos[torso_id]

    # 只取 x 和 z
    return np.array([com_3d[0], com_3d[2]], dtype=float)

def compute_theta(env) -> float:
    mj_env = _get_mj_env(env)
    # 对 planar half-cheetah，通常 qpos = [x, z, theta, ...]
    return float(mj_env.data.qpos[2])


def compute_com_vel(env) -> np.ndarray:
    mj_env = _get_mj_env(env)

    masses = mj_env.model.body_mass[1:]      # (nb-1,)

    # cvel: (nb, 6) 每个刚体的局部6D速度 [wx, wy, wz, vx, vy, vz]
    cvel = mj_env.data.cvel                 # (nb, 6)

    # xmat: (nb, 9) 行主序展平的旋转矩阵，把局部量旋到世界系
    xmat = mj_env.data.xmat.reshape(-1, 3, 3)  # (nb, 3, 3)

    # 取线速度的局部部分并旋到世界系
    v_local = cvel[1:, 3:6]                     # (nb-1, 3)
    v_world = np.einsum("bij,bj->bi", xmat[1:], v_local)  # (nb-1, 3)

    M = masses.sum()
    com_vel = (masses[:, None] * v_world).sum(axis=0) / M
    return np.array([com_vel[0], com_vel[2]], dtype=float)  # 取 vx, vz


def compute_theta_dot(env) -> float:
    mj_env = _get_mj_env(env)
    # qvel = [xdot, zdot, thetadot, ...]
    return float(mj_env.data.qvel[2])


def get_single_rigidbody_state(env) -> np.ndarray:
    px, pz   = compute_com(env)
    vx, vz   = compute_com_vel(env)
    th       = compute_theta(env)
    w        = compute_theta_dot(env)
    return np.array([px, pz, th, vx, vz, w], dtype=float)


def load_policy(state_dict_path, ob_dim, ac_dim, n_layers=2, size=64):
    """Re-create MLPPolicySL and load state_dict"""
    policy = MLPPolicySL(
        ac_dim=ac_dim,
        ob_dim=ob_dim,
        n_layers=n_layers,
        size=size,
        learning_rate=1e-3,
    )

    state_dict = torch.load(state_dict_path, map_location="cpu")
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def evaluate_policy(policy_path, env_name="HalfCheetah-v5",
                    n_eval_episodes=100, render=True, seed=1):

    # create env
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)
    env.reset(seed=seed)

    ob_dim = 6
    ac_dim = 6

    print(f"Environment: {env_name}")
    print(f"Observation dim: {ob_dim}, Action dim: {ac_dim}")

    # Load model
    print(f"Loading policy state_dict from: {policy_path}")
    policy = load_policy(policy_path, ob_dim, ac_dim)
    print("Policy loaded successfully.")

    returns = []

    for ep in range(n_eval_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_ret = 0

        while not (done or truncated):
            # MLPPolicySL has method get_action(obs)
            obs = get_single_rigidbody_state(env.unwrapped)
            action = policy.get_action(obs)

            # convert to numpy if needed
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

            if action.ndim > 1:
                action = action.squeeze()
            obs, reward, done, truncated, info = env.step(action)
            ep_ret += reward

        print(f"Episode {ep+1}: return = {ep_ret:.2f}")
        returns.append(ep_ret)

    env.close()

    print("\n===== Evaluation Summary =====")
    print(f"Average Return: {np.mean(returns):.2f}")
    print(f"Std Return: {np.std(returns):.2f}")
    print("===============================")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--policy_file", "-pf", type=str, default="/Users/rui/Desktop/RL_Formujoco/ResidualMPC/bc_policies/bc_policy_HalfCheetah-v5_1.pt")
    parser.add_argument("--env_name", "-env", type=str, default="HalfCheetah-v5")
    parser.add_argument("--n_eval_episodes", "-n", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    evaluate_policy(
        args.policy_file,
        env_name=args.env_name,
        n_eval_episodes=args.n_eval_episodes,
        render=True,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
