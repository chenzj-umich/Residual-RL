import argparse
import os
import sys
import gymnasium as gym
import numpy as np
import torch
import mujoco
from mujoco import mjtObj

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ResidualMPC.policies.MLP_policy import MLPPolicySL
from ResidualMPC.infrastructure import pytorch_util as ptu


def get_repo_root():
    """Get the absolute path to the repository root."""
    # Current file: ResidualMPC/scripts/RL_Test.py
    # Go up 3 levels: scripts -> ResidualMPC -> ResidualMPC (root)
    print(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
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
    mj_env = env.unwrapped
    # 对 planar half-cheetah，通常 qpos = [x, z, theta, ...]
    return float(mj_env.data.qpos[2])


def compute_com_vel(env) -> np.ndarray:
    mj_env = env.unwrapped

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
    mj_env = env.unwrapped
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
    # keep all episode reward trajectories for richer analysis after evaluation
    ep_reward_trajectories = []
    ep_obs_trajectories = []

    for ep in range(n_eval_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_ret = 0
        ep_obs_list = []
        ep_rewards_list = []

        while not (done or truncated):
            # MLPPolicySL has method get_action(obs)
            obs = get_single_rigidbody_state(env.unwrapped)
            ep_obs_list.append(obs.copy())
            action = policy.get_action(obs)

            # convert to numpy if needed
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

            if action.ndim > 1:
                action = action.squeeze()
            obs, reward, done, truncated, info = env.step(action)
            ep_ret += reward
            ep_rewards_list.append(reward)

        print(f"Episode {ep+1}: return = {ep_ret:.2f}, length = {len(ep_rewards_list)}")
        
        # Print trajectory rewards and critic/baseline values if available
        if ep < 3:  # Print details for first 3 episodes
            ep_obs_array = np.array(ep_obs_list)
            print(f"  Rewards (first 100): {np.array(ep_rewards_list[:100])}")
            print(f"  Total steps: {len(ep_rewards_list)}")
            
            # Try to get critic values if policy has a critic attribute
            try:
                if hasattr(policy, 'critic') and policy.critic is not None:
                    with torch.no_grad():
                        critic_vals = policy.critic(torch.from_numpy(ep_obs_array).float()).cpu().numpy().flatten()
                    print(f"  Critic values (first 10): {critic_vals[:10]}")
                    print(f"  Critic value range: [{np.min(critic_vals):.4f}, {np.max(critic_vals):.4f}]")
                    advantages = np.array(ep_rewards_list) - critic_vals
                    print(f"  Advantages (first 10): {advantages[:10]}")
            except Exception as e:
                print(f"  (No critic available: {e})")
        
    returns.append(ep_ret)
    ep_reward_trajectories.append(np.array(ep_rewards_list))
    ep_obs_trajectories.append(np.array(ep_obs_list))

    env.close()

    # Aggregate analysis across all evaluation episodes
    all_rewards = np.concatenate(ep_reward_trajectories) if len(ep_reward_trajectories) > 0 else np.array([])

    print("\n===== Evaluation Summary =====")
    print(f"Num episodes: {len(returns)}")
    print(f"Average Episode Return: {np.mean(returns):.2f}")
    print(f"Std Episode Return: {np.std(returns):.2f}")

    if all_rewards.size > 0:
        print("\n-- Step-level reward statistics (across all episodes) --")
        print(f"  Total steps evaluated: {all_rewards.size}")
        print(f"  Mean reward / step : {np.mean(all_rewards):.4f}")
        print(f"  Median reward / step: {np.median(all_rewards):.4f}")
        print(f"  Std dev: {np.std(all_rewards):.4f}")
        print(f"  Min: {np.min(all_rewards):.4f}, Max: {np.max(all_rewards):.4f}")

        pos = np.sum(all_rewards > 0)
        neg = np.sum(all_rewards < 0)
        zer = np.sum(all_rewards == 0)
        print(f"  Positive steps: {pos} ({100*pos/all_rewards.size:.1f}%)")
        print(f"  Negative steps: {neg} ({100*neg/all_rewards.size:.1f}%)")
        print(f"  Zero steps: {zer} ({100*zer/all_rewards.size:.1f}%)")

        # percentiles
        p10, p25, p50, p75, p90 = np.percentile(all_rewards, [10,25,50,75,90])
        print(f"  Percentiles - 10/25/50/75/90: {p10:.4f}, {p25:.4f}, {p50:.4f}, {p75:.4f}, {p90:.4f}")

        # top/bottom step values (global)
        top_k = 5
        top_idx = np.argsort(all_rewards)[-top_k:][::-1]
        bot_idx = np.argsort(all_rewards)[:top_k]
        print("\n  Top rewards (step global index : value):")
        for idx in top_idx:
            print(f"    {int(idx)} : {all_rewards[int(idx)]:.4f}")
        print("  Bottom rewards (step global index : value):")
        for idx in bot_idx:
            print(f"    {int(idx)} : {all_rewards[int(idx)]:.4f}")

    else:
        print("No step-level rewards were recorded.")

    # If policy includes a critic (baseline), evaluate it on collected observations
    try:
        if hasattr(policy, 'critic') and policy.critic is not None and len(ep_obs_trajectories) > 0:
            all_obs = np.concatenate(ep_obs_trajectories)
            with torch.no_grad():
                critic_vals = policy.critic(torch.from_numpy(all_obs).float()).cpu().numpy().flatten()

            # Compare critic predictions to empirical returns-to-go (quick approx using step rewards)
            # For simplicity, compute cumulative discounted return from each step in each episode (no discount here)
            empirical_vals = []
            for traj in ep_reward_trajectories:
                # compute reward-to-go for each timestep
                rtg = np.array([np.sum(traj[t:]) for t in range(len(traj))])
                empirical_vals.append(rtg)
            empirical_vals = np.concatenate(empirical_vals)

            # quick diagnostics
            diff = empirical_vals - critic_vals
            print("\n-- Critic diagnostics --")
            print(f"  Critic mean: {np.mean(critic_vals):.4f}, std: {np.std(critic_vals):.4f}")
            print(f"  Empirical mean: {np.mean(empirical_vals):.4f}, std: {np.std(empirical_vals):.4f}")
            print(f"  Mean error (empirical - critic): {np.mean(diff):.4f}, RMSE: {np.sqrt(np.mean(diff**2)):.4f}")
        else:
            print('\n(No critic/baseline found on the policy; skipping critic diagnostics)')
    except Exception as e:
        print(f"\nCould not compute critic diagnostics: {e}")

    print("\n===============================\n")


def main():
    parser = argparse.ArgumentParser()

    repo_root = get_repo_root()
    default_policy_path = os.path.join(repo_root, "./ResidualMPC/model_test.pth")
    
    parser.add_argument("--policy_file", "-pf", type=str, default=default_policy_path)
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
