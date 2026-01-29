# import os
# os.environ["MUJOCO_GL"] = "egl"

# !pip install gymnasium stable-baselines3 shimmy mediapy mujoco

import gymnasium as gym
import numpy as np
import mujoco
import mediapy as media
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium import spaces
# from google.colab import files

class StrictCheetahEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode="rgb_array"):
        super(StrictCheetahEnv, self).__init__()

        self.xml = """
        <mujoco model="cheetah">
          <compiler angle="radian" coordinate="local" inertiafromgeom="true" settotalmass="14"/>
          <default>
            <joint armature=".1" damping=".01" limited="true" solimplimit="0 .8 .03" solreflimit=".02 1" stiffness="8"/>
            <geom conaffinity="0" condim="3" contype="1" friction=".4 .1 .1" rgba="0.8 0.6 .4 1" solimp="0.0 0.8 0.01" solref="0.02 1"/>
            <motor ctrllimited="true" ctrlrange="-1 1"/>
          </default>
          <size nstack="300000" nuser_geom="1"/>
          <option gravity="0 0 -9.81" timestep="0.01"/>
          <asset>
            <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
            <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
            <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
            <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
            <material name="geom" texture="texgeom" texuniform="true"/>
          </asset>
          <worldbody>
            <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
            <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="20 20 20" type="plane"/>
            <body name="torso" pos="0 0 .7">
              <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
              <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
              <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" stiffness="0" type="slide"/>
              <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 0" stiffness="0" type="hinge"/>
              <geom fromto="-.5 0 0 .5 0 0" name="torso" size="0.046" type="capsule"/>
              <geom axisangle="0 1 0 .87" name="head" pos=".6 0 .1" size="0.046 .15" type="capsule"/>
              <body name="bthigh" pos="-.5 0 0">
                <joint axis="0 1 0" damping="6" name="bthigh" pos="0 0 0" range="-.52 1.05" stiffness="240" type="hinge"/>
                <geom fromto="0 0 0 0 0 -.145" name="bthigh" size="0.046" type="capsule"/>
                <body name="bshin" pos="0 0 -.145">
                  <joint axis="0 1 0" damping="4.5" name="bshin" pos="0 0 0" range="-.785 .785" stiffness="180" type="hinge"/>
                  <geom fromto="0 0 0 0 0 -.15" name="bshin" size="0.046" type="capsule"/>
                  <body name="bfoot" pos="0 0 -.15">
                    <joint axis="0 1 0" damping="3" name="bfoot" pos="0 0 0" range="-.4 .785" stiffness="120" type="hinge"/>
                    <geom fromto="0 0 0 .145 0 0" name="bfoot" size="0.046" type="capsule"/>
                  </body>
                </body>
              </body>
              <body name="fthigh" pos=".5 0 0">
                <joint axis="0 1 0" damping="4.5" name="fthigh" pos="0 0 0" range="-1 .7" stiffness="180" type="hinge"/>
                <geom fromto="0 0 0 0 0 -.133" name="fthigh" size="0.046" type="capsule"/>
                <body name="fshin" pos="0 0 -.133">
                  <joint axis="0 1 0" damping="3" name="fshin" pos="0 0 0" range="-1.2 .87" stiffness="120" type="hinge"/>
                  <geom fromto="0 0 0 0 0 -.106" name="fshin" size="0.046" type="capsule"/>
                  <body name="ffoot" pos="0 0 -.106">
                    <joint axis="0 1 0" damping="1.5" name="ffoot" pos="0 0 0" range="-.5 .5" stiffness="60" type="hinge"/>
                    <geom fromto="0 0 0 .07 0 0" name="ffoot" size="0.046" type="capsule"/>
                  </body>
                </body>
              </body>
            </body>
          </worldbody>
          <actuator>
            <motor gear="120" joint="bthigh" name="bthigh"/>
            <motor gear="90" joint="bshin" name="bshin"/>
            <motor gear="60" joint="bfoot" name="bfoot"/>
            <motor gear="120" joint="fthigh" name="fthigh"/>
            <motor gear="60" joint="fshin" name="fshin"/>
            <motor gear="30" joint="ffoot" name="ffoot"/>
          </actuator>
        </mujoco>
        """

        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        # 17 dims
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        self.render_mode = render_mode

        # Get IDs for feet to track them
        self.bfoot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bfoot")
        self.ffoot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ffoot")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Reset to a neutral, slightly noisy state
        self.data.qpos[1:] += np.random.uniform(-0.05, 0.05, size=8)
        self.data.qvel[:] += np.random.uniform(-0.1, 0.1, size=9)

        # Important: Ensure it starts UPRIGHT
        self.data.qpos[1] = 0.0 # Vertical (z) displacement from root

        mujoco.mj_step(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = action

        # Frame Skip 5
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        # --- Physics State ---
        x_velocity = self.data.qvel[0]
        z_pos = self.data.qpos[1] # Local Z of torso (0 is initial height)
        # Note: In this XML, qpos[1] is a slider joint.
        # But we need absolute height to check if falling.
        # The Torso geom size is 0.046. The initial pos is 0.7.
        # So absolute height approx = 0.7 + qpos[1].
        abs_height = 0.7 + z_pos

        pitch_angle = self.data.qpos[2]

        # --- Rewards ---

        # 1. VELOCITY (Strong incentive to move)
        # We increase the weight significantly so it cares about moving
        reward_run = -5.0 * (x_velocity - 1.0)**2

        # 2. UPRIGHT POSTURE (The "Correct Track" fix)
        # Penalize pitch heavily to keep back flat
        reward_posture = -1.0 * (pitch_angle ** 2)

        # 3. FEET CLEARANCE (The Anti-Shuffle)
        # We want feet to lift up.
        # Access body positions
        bfoot_z = self.data.xpos[self.bfoot_body_id][2]
        ffoot_z = self.data.xpos[self.ffoot_body_id][2]

        # Small reward for lifting feet (cyclical nature usually results from this + velocity)
        # We reward the MAX height of either foot to encourage stepping
        feet_lift_reward = 2.0 * max(0, max(bfoot_z, ffoot_z) - 0.05)

        # 4. CONTROL COST (Efficiency)
        reward_ctrl = -0.1 * np.sum(np.square(action))

        # 5. STRICT SURVIVAL (The Lava)
        # If absolute height < 0.35m, it is crawling. Kill it.
        is_healthy = abs_height > 0.45 and abs(pitch_angle) < 0.6
        reward_healthy = 2.0 if is_healthy else -50.0 # Big penalty for falling

        reward = reward_run + reward_posture + reward_ctrl + reward_healthy + feet_lift_reward

        terminated = not is_healthy

        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos.flat[1:], self.data.qvel.flat]).astype(np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data, camera="track")
            return self.renderer.render()

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
env = StrictCheetahEnv()

# Larger network to learn the complex balance required for "Slow walking"
policy_kwargs = dict(net_arch=[256, 256])

model = PPO("MlpPolicy", env,
            verbose=1,
            device="auto",
            policy_kwargs=policy_kwargs,
            learning_rate=2e-4, # Slightly lower LR for stability
            ent_coef=0.01) # Entropy coefficient to encourage exploration (try new gaits)

print("Training strict posture agent...")
# We need substantial steps because the "Survivor" constraint is hard to satisfy at first.
model.learn(total_timesteps=400_000)
print("Training Complete!")

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
print("Generating Video...")
obs, _ = env.reset()
frames = []

for _ in range(500):
    frames.append(env.render())
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, _, _ = env.step(action)
    if terminated:
        obs, _ = env.reset()

media.show_video(frames, fps=30)