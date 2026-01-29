
# python scripts/RL_Script.py --env_name HalfCheetah-v5 \
#  -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
#  --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline
import pickle
import os
import sys
import time
import gymnasium as gym

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ResidualMPC.agents.pg_agent import PGAgent

import numpy as np
import torch

from ResidualMPC.infrastructure import pytorch_util as ptu
from ResidualMPC.infrastructure import utils
from ResidualMPC.infrastructure.logger import Logger
from ResidualMPC.infrastructure.action_noise_wrapper import ActionNoiseWrapper


MAX_NVIDEO = 2

def run_training_loop(args):
    logger = Logger(args.logdir)

    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = gym.make(args.env_name, render_mode=None)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # add action noise, if needed
    if args.action_noise_std > 0:
        assert not discrete, f"Cannot use --action_noise_std for discrete environment {args.env_name}"
        env = ActionNoiseWrapper(env, args.seed, args.action_noise_std)

    max_ep_len = args.ep_len or env.spec.max_episode_steps
    
#————————————————————————ob_dim要改————————————————————————#
    ob_dim = env.observation_space.shape[0]
#————————————————————————ob_dim要改————————————————————————#
    ob_dim = 6


    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if hasattr(env, "model"):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    # initialize agent
    agent = PGAgent(
        ob_dim,
        ac_dim,
        discrete,
        n_layers=args.n_layers,
        layer_size=args.layer_size,
        gamma=args.discount,
        learning_rate=args.learning_rate,
        use_baseline=args.use_baseline,
        use_reward_to_go=args.use_reward_to_go,
        normalize_advantages=args.normalize_advantages,
        baseline_learning_rate=args.baseline_learning_rate,
        baseline_gradient_steps=args.baseline_gradient_steps,
        gae_lambda=args.gae_lambda,
    )

    # state_dict = torch.load("/Users/rui/Desktop/RL_Formujoco/ResidualMPC/bc_policies/bc_policy_HalfCheetah-v5_1.pt", map_location="cpu")
    # agent.actor.load_state_dict(state_dict)


    total_envsteps = 0
    start_time = time.time()

    for itr in range(args.n_iter):
        print(f"\n********** Iteration {itr} ************")
        # TODO: sample `args.batch_size` transitions using utils.sample_trajectories
        # make sure to use `max_ep_len`


#——————————————————Sample_trajectories要改————————————————————#
        trajs, envsteps_this_batch = utils.sample_trajectories(env,agent.actor,args.batch_size,max_ep_len,render = False)# TODO
#——————————————————Sample_trajectories要改————————————————————#
        

        total_envsteps += envsteps_this_batch

        # trajs should be a list of dictionaries of NumPy arrays, where each dictionary corresponds to a trajectory.
        # this line converts this into a single dictionary of lists of NumPy arrays.
        trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}
        #LIST[traj1_obs, traj2_obs,...]---> {observation: [traj1_obs, traj2_obs,...], action: [traj1_ac, traj2_ac,...],...}
        #T1*dimensions

        # === DEBUG: Print trajectory rewards and baseline/critic values ===
        if itr % args.scalar_log_freq == 0:
            print(f"\n{'='*80}")
            print(f"Iteration {itr}: Trajectory Analysis")
            print(f"{'='*80}")
            for traj_idx in range(min(3, len(trajs))):  # Print first 3 trajectories
                traj = trajs[traj_idx]
                traj_rewards = traj["reward"]
                traj_obs = traj["observation"]
                total_return = np.sum(traj_rewards)
                
                print(f"\n[Trajectory {traj_idx}]")
                print(f"  Length: {len(traj_rewards)}, Total Return: {total_return:.4f}")
                print(f"  Reward range: [{np.min(traj_rewards):.4f}, {np.max(traj_rewards):.4f}]")
                print(f"  Rewards (first 10): {traj_rewards[:10]}")
                
                # Compute baseline/critic values if using baseline
                if agent.critic is not None:
                    critic_values = agent.critic(ptu.from_numpy(traj_obs))
                    critic_values = ptu.to_numpy(critic_values).flatten()
                    print(f"  Critic values (first 10): {critic_values[:10]}")
                    print(f"  Critic value range: [{np.min(critic_values):.4f}, {np.max(critic_values):.4f}]")
                    # Print advantages (reward - critic value)
                    advantages = traj_rewards - critic_values
                    print(f"  Advantages (first 10): {advantages[:10]}")
            print(f"{'='*80}\n")

        # TODO: train the agent using the sampled trajectories and the agent's update function
        train_info: dict = agent.update(
            trajs_dict["observation"], trajs_dict["action"], trajs_dict["reward"], trajs_dict["terminal"])
    

        if itr % args.scalar_log_freq == 0:
            # save eval metrics
            print("\nCollecting data for eval...")
            eval_trajs, eval_envsteps_this_batch = utils.sample_trajectories(
                env, agent.actor, args.eval_batch_size, max_ep_len
            )

            logs = utils.compute_metrics(trajs, eval_trajs)
            # compute additional metrics
            logs.update(train_info)
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs[
                    "Train_AverageReturn"
                ]

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
                logger.log_scalar(value, key, itr)
            print("Done logging...\n\n")

            logger.flush()

        if args.video_log_freq != -1 and itr % args.video_log_freq == 0:
            print("\nCollecting video rollouts...")
            eval_video_trajs = utils.sample_n_trajectories(
                env, agent.actor, MAX_NVIDEO, max_ep_len, render=True
            )

            logger.log_trajs_as_videos(
                eval_video_trajs,
                itr,
                fps=fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="eval_rollouts",
            )
    repo_root = get_repo_root()
    model_path = os.path.join(repo_root, "ResidualMPC/model_test.pth")
    print(f"Saving model to {model_path}")
    torch.save(agent.actor.state_dict(), model_path)


def get_repo_root():
    """Get the absolute path to the repository root."""
    # Current file: ResidualMPC/scripts/RL_Script.py
    # Go up 3 levels: scripts -> ResidualMPC -> ResidualMPC (root)
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--n_iter", "-n", type=int, default=200)

    parser.add_argument("--use_reward_to_go", "-rtg", action="store_true")
    parser.add_argument("--use_baseline", action="store_true")
    parser.add_argument("--baseline_learning_rate", "-blr", type=float, default=5e-3)
    parser.add_argument("--baseline_gradient_steps", "-bgs", type=int, default=5)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--normalize_advantages", "-na", action="store_true")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1000
    )  # steps collected per train iteration
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=400
    )  # steps collected per eval iteration

    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--layer_size", "-s", type=int, default=64)

    parser.add_argument(
        "--ep_len", type=int
    )  # students shouldn't change this away from env's default
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=1)

    parser.add_argument("--action_noise_std", type=float, default=0)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "q2_pg_"  # keep for autograder

    repo_root = get_repo_root()
    data_path = os.path.join(repo_root, "data")

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = (
        logdir_prefix
        + args.exp_name
        + "_"
        + args.env_name
        + "_"
        + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    logdir = os.path.join(data_path, logdir)
    args.logdir = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    
    run_training_loop(args)

    
if __name__ == "__main__":
    main()
