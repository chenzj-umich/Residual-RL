"""
Simple Behavior Cloning script

- Only uses offline expert data (observations, actions)
- No DAgger, no logger, no replay buffer
- Environment: HalfCheetah-v5 (can be changed by flag)
"""

import os
import pickle
import argparse

import gymnasium as gym
import numpy as np
import torch

from ResidualMPC.infrastructure import pytorch_util as ptu
from ResidualMPC.policies.MLP_policy import MLPPolicySL


def get_repo_root():
    """Get the absolute path to the repository root."""
    # Current file: ResidualMPC/scripts/IL_Script.py
    # Go up 3 levels: scripts -> ResidualMPC -> ResidualMPC (root)
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def run_bc_training(params):
    """
    Pure behavior cloning:

    1) Load expert dataset (observations, actions)
    2) Build policy network
    3) Train via supervised learning
    4) Save trained policy
    """

    # ----------------- init device & seeds -----------------
    seed = params["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    ptu.init_gpu(
        use_gpu=not params["no_gpu"],
        gpu_id=params["which_gpu"],
    )

    # ----------------- env (just to get obs_dim / ac_dim) -----------------
    env = gym.make(params["env_name"])
    env.reset(seed=seed)

    assert isinstance(env.action_space, gym.spaces.Box), "Environment must be continuous"

    ob_dim = 6
    ac_dim = 6

    print(f"Environment: {params['env_name']}")
    print(f"ob_dim = {ob_dim}, ac_dim = {ac_dim}")

    # ----------------- load expert data -----------------
    print(f"Loading expert data from: {params['expert_data']}")
    with open(params["expert_data"], "rb") as f:
        expert_data = pickle.load(f)

    # 约定：expert_data 是一个 dict，包含 'observations' 和 'actions'
    observations = expert_data["observations"]  # shape (N, ob_dim)
    actions = expert_data["actions"]            # shape (N, ac_dim)

    observations = observations.astype(np.float32)
    actions = actions.astype(np.float32)

    N = observations.shape[0]
    print(f"Loaded expert dataset with {N} samples")

    # ----------------- build policy -----------------
    actor = MLPPolicySL(
        ac_dim=ac_dim,
        ob_dim=ob_dim,
        n_layers=params["n_layers"],
        size=params["size"],
        learning_rate=params["learning_rate"],
    )

    # ----------------- training loop -----------------
    n_epochs = params["n_epochs"]
    batch_size = params["batch_size"]

    print(f"Start BC training: epochs={n_epochs}, batch_size={batch_size}")

    for epoch in range(n_epochs):
        # 打乱数据索引
        perm = np.random.permutation(N)

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            end = start + batch_size
            idx = perm[start:end]

            ob_batch = observations[idx]
            ac_batch = actions[idx]

            # MLPPolicySL.update 接受 numpy，内部会转到 torch
            log = actor.update(ob_batch, ac_batch)
            # log 里一般会有 'Training Loss'
            if isinstance(log, dict) and "Training Loss" in log:
                epoch_loss += log["Training Loss"]
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"[Epoch {epoch+1}/{n_epochs}] avg BC loss = {avg_loss:.6f}")

    # ----------------- save policy -----------------
    save_path = params["output_policy_file"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    actor.save(save_path)
    print(f"\nBehavior Cloning finished. Policy saved to:\n  {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Simple Behavior Cloning for HalfCheetah-v5")

    # 必要参数：expert_data 路径
    parser.add_argument(
        "--expert_data",
        "-ed",
        type=str,
        default ="/Users/rui/Desktop/RL_Formujoco/ResidualMPC/ResidualMPC/expert_data/expert_data_HalfCheetah_SRB_1.pkl",
        help="Path to expert dataset pkl (must contain 'observations' and 'actions')",
    )

    # 环境（默认 HalfCheetah-v5）
    parser.add_argument(
        "--env_name",
        "-env",
        type=str,
        default="HalfCheetah-v5",
        help="Gymnasium environment name (default: HalfCheetah-v5)",
    )

    # 训练超参数（简单几个就够了）
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--n_layers", type=int, default=2, help="Policy MLP depth")
    parser.add_argument("--size", type=int, default=64, help="Policy MLP width")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)

    # 设备与随机种子
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)

    # 策略保存路径
    repo_root = get_repo_root()
    default_policy_path = os.path.join(repo_root, "bc_policies", "bc_policy_HalfCheetah-v5_1.pt")
    
    parser.add_argument(
        "--output_policy_file",
        "-opf",
        type=str,
        default=default_policy_path,
        help="Where to save the trained policy weights (.pt)",
    )

    args = parser.parse_args()
    params = vars(args)

    run_bc_training(params)


if __name__ == "__main__":
    main()
