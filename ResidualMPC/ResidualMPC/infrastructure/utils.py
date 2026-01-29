"""A
Some miscellaneous utility functions

Functions to edit:
    1. sample_trajectory
"""

from collections import OrderedDict
import cv2
import numpy as np
import time
import mujoco
from mujoco import mjtObj
from ResidualMPC.infrastructure import pytorch_util as ptu

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




def sample_trajectory(env, policy, max_path_length, render=False):
    """Sample a rollout in the environment from a policy."""
    
    # initialize env for the beginning of a new rollout
    

#——————————————————obs要改————————————————————#
    ob,info =  env.reset() # TODO: initial observation after resetting the env
#——————————————————obs要改————————————————————#以下为修改内容
    ob = get_single_rigidbody_state(env.unwrapped)


    expert_policy = policy
    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:

        # render image of the simulated env
        if render:
            if hasattr(env, 'sim'):
                img = env.sim.render(camera_name='track', height=500, width=500)[::-1]
            else:
                img = env.render(mode='single_rgb_array')
            image_obs.append(cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC))
    
        # TODO use the most recent ob to decide what to do
        ac = expert_policy.get_action(ob) # HINT: this is a numpy array
 
       # ac = ac[0]

        # TODO: take that action and get reward and next ob

#——————————————————obs要改————————————————————#
        next_ob, rew, terminated, truncated,_ = env.step(ac)
#——————————————————obs要改————————————————————#  
        next_ob = get_single_rigidbody_state(env.unwrapped)


        done = terminated or truncated
        # TODO rollout can end due to done, or due to max_path_length
        steps += 1
        rollout_done = done  # HINT: this is either 0 or 1
        
        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob # jump to next timestep

        # end the rollout if the rollout ended
        if rollout_done:
            break

    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False):
    """Collect rollouts until we have collected min_timesteps_per_batch steps."""

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:

        #collect rollout
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)

        #count steps
        timesteps_this_batch += get_pathlength(path)

    return paths, timesteps_this_batch


def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False):
    """Collect ntraj rollouts."""

    paths = []
    for i in range(ntraj):
        # collect rollout
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)
    return paths


########################################
########################################


def convert_listofrollouts(paths, concat_rew=True):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    return observations, actions, rewards, next_observations, terminals


########################################
########################################
            

def compute_metrics(paths, eval_paths):
    """Compute metrics for logging."""

    # returns, for logging
    train_returns = [path["reward"].sum() for path in paths]
    eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

    # episode lengths, for logging
    train_ep_lens = [len(path["reward"]) for path in paths]
    eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

    # decide what to log
    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_returns)
    logs["Train_StdReturn"] = np.std(train_returns)
    logs["Train_MaxReturn"] = np.max(train_returns)
    logs["Train_MinReturn"] = np.min(train_returns)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs


############################################
############################################


def get_pathlength(path):
    return len(path["reward"])
