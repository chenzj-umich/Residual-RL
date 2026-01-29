import casadi as ca
import numpy as np
import env_warpper
import mujoco
from mujoco import mjtObj
import matplotlib.pyplot as plt
import pickle
import os
import time

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
        # model.body('name').id 也是新接口的一部分，但这里直接从 names 里扫一遍更保险å
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

def map_grf_to_u(env, FxL, FzL, FxR, FzR,
                     ):
    """ 
    将左右足端在世界系中的接触力 (Fx, Fz) 映射到各关节力矩（返回顺序与 MuJoCo 中 actuator 顺序一致）。

    Args:
        env: 具有 .model, .data 的 MuJoCo/Gym 环境对象
        FxL, FzL: 左脚在世界系中的水平/竖直力（N）
        FxR, FzR: 右脚在世界系中的水平/竖直力（N）
        left_site_candidates: 左脚 site 的候选名称（按顺序尝试）
        right_site_candidates: 右脚 site 的候选名称（按顺序尝试）

    Returns:
        tau_act (np.ndarray): 形状 (nu,) 的关节力矩（对每个 actuator 对应的关节 DOF）。
                              注意：这是“关节力矩”，如果你的 actuator 是力矩型且带 gear，
                              则 ctrl = tau_act / gear。
    """
    model, data = env.model, env.data
    nv, nu = int(model.nv), int(model.nu)


    # ---------- 1) 先把 GRF 映射到“全自由度的广义力” ----------
    # 尝试解析前后脚（front/back）的 site，失败则退回 body 质心雅可比
    front_candidates = ["ffoot_site"]
    back_candidates  = ["bfoot_site"]

    def _resolve_jac(cands):
        for name in cands:
            sid = mujoco.mj_name2id(model, mjtObj.mjOBJ_SITE, name)
            if sid != -1:
                jacp = np.zeros((3, nv)); jacr = np.zeros((3, nv))
                mujoco.mj_jacSite(model, data, jacp, jacr, sid)
                return jacp, jacr

    jacp_front, _ = _resolve_jac(front_candidates)  #这边返回的是body的雅可比矩阵
    jacp_back,  _  = _resolve_jac(back_candidates)


    f_front = np.array([FxR, 0.0, FzR], dtype=float)
    f_back  = np.array([FxL, 0.0, FzL], dtype=float)
    m_zero  = np.zeros(3)

    tau_front = jacp_front.T @ f_front 

    tau_back  = jacp_back.T  @ f_back 
    tau_total = tau_front + tau_back          # shape: (nv,)

    # 若是标准 half-cheetah：前3个是 base (x,z,theta)，后6个为电机关节
    tau_joints = tau_total[3:3+6].copy()  # 只取受控关节部分


    # ---------- 2) 把“关节力矩 τ_joints”映射成“控制量 ctrl (u)” ----------
    # 思路：遍历 actuator，找到它驱动的 joint（actuator_trnid[i,0]），
    #       取 gear = actuator_gear[i,0]，然后 u_i = τ_desired(joint)/gear。
    ctrl = np.zeros(nu, dtype=float)

    # 用一个小工具把“受控关节的局部索引(6个)”映射到全局 joint id
    # 标准 half-cheetah: 受控关节一般是索引 3..8（共6个），这里做一个通用推断：
    #   找出所有被某个 actuator 驱动的 hinge 关节，按它们的 joint id 升序排列，
    #   然后把 tau_joints 依次对齐（保持 gym 半猎豹的常见顺序也基本成立）。
    driven_joint_ids = []
    for i in range(nu):
        jid = int(model.actuator_trnid[i, 0])
        if 0 <= jid < nv:  # 关节传动
            driven_joint_ids.append(jid)
    driven_joint_ids = sorted(set(driven_joint_ids))

    # 构建：joint_id -> tau 值（依次填入）
    tau_per_joint = {jid: 0.0 for jid in driven_joint_ids}
    for k, jid in enumerate(driven_joint_ids[:len(tau_joints)]):
        tau_per_joint[jid] = float(tau_joints[k])

    # 映射到每个 actuator 的 ctrl（考虑 gear 和 ctrlrange 裁剪）
    for i in range(nu):
        jid = int(model.actuator_trnid[i, 0])
  
        g = float(model.actuator_gear[i, 0])

        if np.isclose(g, 0.0):
            # 无法从力矩反推 ctrl
            ctrl[i] = 0.0
            continue

        tau_des = tau_per_joint.get(jid, 0.0)
        u = tau_des / g

        # 裁剪到 ctrlrange（若有限）
        cr = model.actuator_ctrlrange[i]
        if np.all(np.isfinite(cr)):
            u = float(np.clip(u, cr[0], cr[1]))

        ctrl[i] = u
        
    return ctrl

def detect_foot_contact(env):
    """
    用 data.contact 判断左右脚是否与任何东西发生接触。
    返回: contact_L, contact_R (bool)
    """
    model, data = env.model, env.data

    # 1) 找到左右脚的 body id（名字要和 XML 对上）
    bid_back  = mujoco.mj_name2id(model, mjtObj.mjOBJ_BODY, "bfoot")
    bid_front = mujoco.mj_name2id(model, mjtObj.mjOBJ_BODY, "ffoot")
    if bid_back < 0 or bid_front < 0:
        raise RuntimeError("找不到 bfoot / ffoot body，请检查 XML 命名。")

    # 2) 找出分别属于两个 foot body 的所有 geom id
    back_geoms  = [gid for gid in range(model.ngeom)
                   if model.geom_bodyid[gid] == bid_back]
    front_geoms = [gid for gid in range(model.ngeom)
                   if model.geom_bodyid[gid] == bid_front]

    contact_L = False
    contact_R = False

    # 3) 遍历当前所有接触，看是否涉及这些 geoms
    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = con.geom1, con.geom2

        if g1 in back_geoms or g2 in back_geoms:
            contact_L = True
        if g1 in front_geoms or g2 in front_geoms:
            contact_R = True

        if contact_L and contact_R:
            break

    return contact_L, contact_R



def get_foot_state(env, leg: str = "front"):
    """
    读取脚的世界系位置/速度（只用 x,z）
    leg: "front" 对应 ffoot, "back" 对应 bfoot
    返回: pos (2,), vel (2,)
    """
    model, data = env.model, env.data
    body_name = "ffoot" if leg == "front" else "bfoot"
    bid = mujoco.mj_name2id(model, mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise RuntimeError(f"找不到 body {body_name}，请检查 XML。")

    # 位置：xipos(bid, :) 是世界系 body 质心位置
    pos_world = data.xipos[bid].copy()     # (3,)
    # 线速度：用 cvel + xmat 旋到世界系
    cvel = data.cvel                       # (nb, 6), [wx, wy, wz, vx, vy, vz]
    xmat = data.xmat.reshape(-1, 3, 3)     # (nb, 3, 3)

    v_local = cvel[bid, 3:6]               # (3,)
    v_world = xmat[bid] @ v_local          # (3,)

    # 只用 x,z 分量
    pos_xz = np.array([pos_world[0], pos_world[2]], dtype=float)
    vel_xz = np.array([v_world[0], v_world[2]], dtype=float)
    return pos_xz, vel_xz







def build_contact_from_swing(swing_ctrl, mpc):
    """
    根据 swing_ctrl 当前的 swing 状态 + swing_T
    生成长度 N 的 contact sequence（L/R 各一条）
    """
    N  = mpc.N
    dt = mpc.dt
    swing_T = swing_ctrl.swing_T

    cL = np.ones(N, dtype=float)
    cR = np.ones(N, dtype=float)

    for leg, arr in [("back", cL), ("front", cR)]:
        if swing_ctrl.active[leg]:
            # 剩余 swing 时间
            phi = swing_ctrl.phase[leg]          # 当前 [0,1]
            remaining_T = max(0.0, swing_T * (1.0 - phi))
            remaining_steps = int(np.ceil(remaining_T / dt))

            remaining_steps = min(remaining_steps, N)
            if remaining_steps > 0:
                arr[:remaining_steps] = 0.0      # 未来这几步视为 swing
            # 后面的步 arr[remaining_steps:] 保持 1.0（视为 stance）
        else:
            # 不在 swing，相当于整段 horizon 都可以当 stance 用
            arr[:] = 1.0

    return {"L": cL, "R": cR}




class SwingController:
    """
    简单 swing 控制器：
      - 每条腿有一个 phase \in [0,1]，表示 swing 归一化时间
      - 检测到从接触 -> 离地 时启动 swing，相当于“抬脚”
      - swing_T 秒内按照抛物线轨迹把脚从 p_start 挥到 p_target
      - 提供一个 tau_joint_swing (nv,) 叠加到现有的 stance 力矩上
    """

    def __init__(
        self,
        env,
        swing_T=0.25,          # Swing 相持续时间（秒）
        step_length=0.2,       # 每步向前迈多远（m）
        step_height=0.1,       # 脚尖抬多高（m）
        Kp=800.0,              # 任务空间 PD 的 Kp (对 x,z)
        Kd=80.0,
    ):
        self.env = env
        self.swing_T = float(swing_T)
        self.step_length = float(step_length)
        self.step_height = float(step_height)
        self.Kp = Kp
        self.Kd = Kd

        self.dt = 0.01

        # 记录每条腿的 swing 状态
        self.legs = ["back", "front"]
        self.active = {leg: False for leg in self.legs}
        self.phase = {leg: 0.0 for leg in self.legs}
        self.p_start = {leg: np.zeros(2) for leg in self.legs}
        self.p_target = {leg: np.zeros(2) for leg in self.legs}

        # 上一时刻的接触状态（用来检测 liftoff）
        self.prev_contact = {"back": True, "front": True}



        # ========= 新增：debug 用的缓存（只记录 front 腿） =========
        self.debug = {
            "t": [],          # 时间
            "phi": [],        # 相位
            "p_now": [],      # 当前脚位置 (x,z)
            "p_des": [],      # 期望脚位置 (x,z)
            "v_now": [],      # 当前脚速度 (vx,vz)
            "v_des": [],      # 期望脚速度 (vx,vz)
            "ep": [],         # 位置误差
            "ev": [],         # 速度误差
            "f_cmd": [],      # 任务空间力
        }
        self._step_count = 0

    # ---- 计算某个 phi \in [0,1] 下的 swing 轨迹参考 ----
    def _swing_traj(self, p0, p1, phi):
        """
        p0, p1: (2,) = (x,z)
        phi in [0,1]，表示 t / swing_T
        返回: 
            p_des: (x,z) 位置 [m]
            v_des: (vx,vz) 速度 [m/s]

        - x: 从 x0 到 x1 线性插值，对应常速 vx_des
        - z: 通过 (0,z0), (1,z1), (0.5,z_mid) 的抛物线
        """

        # 起点、终点
        x0, z0 = float(p0[0]), float(p0[1])
        x1, z1 = float(p1[0]), float(p1[1])

        T = float(self.swing_T)      # 整个 swing 持续时间

        # ===== 1) x 方向：线性插值 + 常速 =====
        # 位置：x(t) = x0 + (x1-x0) * phi
        x_des = (1.0 - phi) * x0 + phi * x1
        # 速度：vx = dx/dt = (x1-x0) / T
        vx_des = (x1 - x0) / T

        # ===== 2) z 方向：抛物线轨迹 =====
        # 最高点高度
        z_mid = max(z0, z1) + self.step_height

        # 用 (0,z0), (1,z1), (0.5,z_mid) 解二次多项式 z(phi) = a*phi^2 + b*phi + c
        dz1 = z1 - z0
        dzm = z_mid - z0
        a = 2.0 * dz1 - 4.0 * dzm
        b = -dz1 + 4.0 * dzm
        c = z0

        # 位置：z_des = a*phi^2 + b*phi + c
        z_des = a * phi**2 + b * phi + c

        # 速度：vz = dz/dt = (dz/dphi) * (dphi/dt) = (2a*phi + b) / T
        dz_dphi = 2.0 * a * phi + b
        vz_des = dz_dphi / T

        p_des = np.array([x_des, z_des], dtype=float)
        v_des = np.array([vx_des, vz_des], dtype=float)
        return p_des, 0.5 *v_des


    def _foot_jacobian_xz(self, leg: str):
        """
        求某条腿的脚的雅可比，只要位置雅可比 Jp 的 x,z 两行。
        返回: J_xz: shape (2, nv)
        """
        model, data = self.env.model, self.env.data
        if leg == "front":
            cands = ["ffoot_site"]
        else:
            cands = ["bfoot_site"]

        # 找 site，再退回 body 的逻辑跟你 map_grf_to_u 一样
        for name in cands:
            sid = mujoco.mj_name2id(model, mjtObj.mjOBJ_SITE, name)
            if sid != -1:
                nv = int(model.nv)
                jacp = np.zeros((3, nv))
                jacr = np.zeros((3, nv))
                mujoco.mj_jacSite(model, data, jacp, jacr, sid)
                # 取 x,z 行
                J_xz = np.vstack([jacp[0, :], jacp[2, :]])
                return J_xz


        raise RuntimeError(f"找不到 {leg} 脚的雅可比，请检查 XML 命名。")

    def update_and_compute_tau(self, contact_L_now: bool, contact_R_now: bool):
        """
        外部每个控制步调用一次：
          - 根据当前接触状态更新 swing 相（检测 liftoff / touchdown）
          - 返回 swing 控制产生的 tau_joint_swing (nv,)
        contact_L_now: 后脚是否接触（bfoot）
        contact_R_now: 前脚是否接触（ffoot）
        """
        model = self.env.model
        nv = int(model.nv)
        FxL = 0
        FzL = 0
        FxR = 0
        FzR = 0
        tau_joint = np.zeros(nv, dtype=float)

        # 当前接触状态转成 dict
        contact_now = {"back": bool(contact_L_now), "front": bool(contact_R_now)}

        for leg in self.legs:
            was_contact = self.prev_contact[leg]
            is_contact = contact_now[leg]

            # === 1) 检测 liftoff：接触 -> 离地，启动 swing ===
            if was_contact and (not is_contact):
                self.active[leg] = True
                self.phase[leg] = 0.0

                p0, v0 = get_foot_state(self.env, leg=("front" if leg == "front" else "back"))
                self.p_start[leg] = p0.copy()

                # 落脚点
                com_x, com_z = compute_com(self.env)
                if leg == "front":
                    target_x = com_x + 0.6
                else:
                    target_x = com_x - 0.3

                target_z = p0[1]
                self.p_target[leg] = np.array([target_x, target_z], dtype=float)

            # === 2) 如果正在 swing，就按轨迹算任务空间力，再投影成关节力矩 ===
            if self.active[leg]:
                phi = self.phase[leg]
                phi = max(0.0, min(1.0, phi))   # clamp

                p_des, v_des = self._swing_traj(self.p_start[leg], self.p_target[leg], phi)
                p_now, v_now = get_foot_state(self.env, leg=("front" if leg == "front" else "back"))

                e_p = p_des - p_now
                e_v = v_des - v_now

                # ========== 任务空间 PD ==========
                f_cmd = self.Kp * e_p + self.Kd * e_v   # (2,)

                # （可选）在这里加 saturate / phase-based gain，就不展开了

                # ========== 记录 front 腿的调试数据 ==========
                if leg == "front":
                    t_now = self._step_count * self.dt
                    self.debug["t"].append(t_now)
                    self.debug["phi"].append(phi)
                    self.debug["p_now"].append(p_now.copy())
                    self.debug["p_des"].append(p_des.copy())
                    self.debug["v_now"].append(v_now.copy())
                    self.debug["v_des"].append(v_des.copy())
                    self.debug["ep"].append(e_p.copy())
                    self.debug["ev"].append(e_v.copy())
                    self.debug["f_cmd"].append(f_cmd.copy())

                    # 如果你还想看少量打印，可以留一个低频打印：
                    # if self._step_count % 50 == 0:
                    #     print("front ep:", e_p, "ev:", e_v, "f_cmd:", f_cmd)
                if leg == "back":
                    FxL += f_cmd[0]
                    FzL += f_cmd[1]
                else:
                    FxR += f_cmd[0]
                    FzR += f_cmd[1]
                # 雅可比 J_xz: (2, nv)
                # J_xz = self._foot_jacobian_xz(leg)
                # tau_leg = J_xz.T @ f_cmd                # (nv,)

                # tau_joint += tau_leg
                # print(tau_joint)
                # 更新相位
                self.phase[leg] += self.dt / self.swing_T

                # touchdown 结束 swing
                if is_contact or self.phase[leg] >= 1.0:
                    self.active[leg] = False
                    self.phase[leg] = 0.0

            # === 3) 更新 prev_contact ===
            self.prev_contact[leg] = is_contact

        self._step_count += 1
        FxL, FzL, FxR, FzR = np.clip([FxL, FzL, FxR, FzR], -15.0, 15.0)
        return FxL, FzL, FxR, FzR


    def plot_debug(self):
        """
        把 front 腿的 p_now / p_des / v_now / v_des / ep / ev / f_cmd 画出来。
        在仿真结束后调用：
            swing_ctrl.plot_debug()
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if len(self.debug["t"]) == 0:
            print("[SwingController] No debug data recorded.")
            return

        t   = np.array(self.debug["t"])
        p_n = np.stack(self.debug["p_now"], axis=0)   # (T,2)
        p_d = np.stack(self.debug["p_des"], axis=0)
        v_n = np.stack(self.debug["v_now"], axis=0)
        v_d = np.stack(self.debug["v_des"], axis=0)
        ep  = np.stack(self.debug["ep"], axis=0)
        ev  = np.stack(self.debug["ev"], axis=0)
        f   = np.stack(self.debug["f_cmd"], axis=0)
        phi = np.array(self.debug["phi"])

        fig, axs = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
        axs = axs.reshape(4, 2)

        # 1) 位置 x,z
        axs[0, 0].plot(t, p_n[:, 0], label="x_now")
        axs[0, 0].plot(t, p_d[:, 0], "--", label="x_des")
        axs[0, 0].set_ylabel("foot x [m]")
        axs[0, 0].legend()

        axs[0, 1].plot(t, p_n[:, 1], label="z_now")
        axs[0, 1].plot(t, p_d[:, 1], "--", label="z_des")
        axs[0, 1].set_ylabel("foot z [m]")
        axs[0, 1].legend()

        # 2) 速度 vx,vz
        axs[1, 0].plot(t, v_n[:, 0], label="vx_now")
        axs[1, 0].plot(t, v_d[:, 0], "--", label="vx_des")
        axs[1, 0].set_ylabel("vx [m/s]")
        axs[1, 0].legend()

        axs[1, 1].plot(t, v_n[:, 1], label="vz_now")
        axs[1, 1].plot(t, v_d[:, 1], "--", label="vz_des")
        axs[1, 1].set_ylabel("vz [m/s]")
        axs[1, 1].legend()

        # 3) 误差 ep, ev（z 或 x 都行，这里全画）
        axs[2, 0].plot(t, ep[:, 0], label="ep_x")
        axs[2, 0].plot(t, ep[:, 1], label="ep_z")
        axs[2, 0].set_ylabel("pos error [m]")
        axs[2, 0].legend()

        axs[2, 1].plot(t, ev[:, 0], label="ev_x")
        axs[2, 1].plot(t, ev[:, 1], label="ev_z")
        axs[2, 1].set_ylabel("vel error [m/s]")
        axs[2, 1].legend()

        # 4) 任务空间力 f_cmd
        axs[3, 0].plot(t, f[:, 0], label="fx")
        axs[3, 0].set_ylabel("fx [N]")
        axs[3, 0].legend()

        axs[3, 1].plot(t, f[:, 1], label="fz")
        axs[3, 1].set_ylabel("fz [N]")
        axs[3, 1].legend()

        for i in range(4):
            for j in range(2):
                axs[i, j].grid(True)

        axs[3, 0].set_xlabel("time [s]")
        axs[3, 1].set_xlabel("time [s]")

        fig.suptitle("Front leg swing debug (task-space tracking)")
        plt.tight_layout()
        plt.show()



class SRB_MPC:
    """
    Single-Rigid-Body NMPC for 2D planar COM + pitch:
      state x = [px, pz, th, vx, vz, w]
      input u = [FxL, FzL, FxR, FzR]
    Dynamics: explicit Euler (可按需改成 RK4)
    """

    def __init__(
        self,
        dt=0.05,
        N=10,
        m=14.0,
        I=0.9,
        g=9.81,
        foot_x=0.5,
        ground_z=0.0,
        mu=0.16,
        # 参考（默认 px 不严格跟踪；pz=0.4, th=0, vx=1, vz=0, w=0）
        x_ref=np.array([None, 0.4, 0.0, .5, 0.0, 0.0], dtype=object),
        # 权重（与你原来相近；px=0 不罚）s 
        Q=np.diag([ 0,  100,  10,  
                   10,  10, 1 ]),
        Qf=None,                      # 若 None，使用 Q
        R=np.diag([5e-4, 1e-3, 5e-4, 1e-3]),
        use_nominal_support=False,     # 输入项使用 (u - u_ref)^T R (u - u_ref)
        ipopt_p_opts={"print_time": 0},
        ipopt_s_opts={"print_level": 0, "max_iter": 100, "tol": 1e-3,"hessian_approximation": "limited-memory",},
    ):
        self.dt, self.N = float(dt), int(N)
        self.m, self.I, self.g = float(m), float(I), float(g)
        self.foot_x, self.ground_z = float(foot_x), float(ground_z)
        self.mu = float(mu)

        self.Q = np.array(Q, dtype=float)

        if Qf is None:
            Qf_eff = self.Q.copy()
            # 索引：px=0, pz=1, th=2, vx=3, vz=4, w=5
            scales = {1: 1.5, 2: 2.0, 3: 4.0}  # pz×1.5, th×2, vx×4
            for idx, s in scales.items():
                Qf_eff[idx, idx] *= s
            self.Qf = Qf_eff
        else:
            self.Qf = np.array(Qf, dtype=float)

        self.R = np.array(R, dtype=float)
        self.use_nominal_support = bool(use_nominal_support)

        # 参考状态：px 可设为 None 表示“不跟踪 px”（Q 对应维度=0 也可）
        self.x_ref = np.array(x_ref, dtype=object)

        self.ipopt_p_opts = dict(ipopt_p_opts or {})
        self.ipopt_s_opts = dict(ipopt_s_opts or {})

        self._build()

        # 缓存上一次解用于 warm-start shift
        self._last_X = None
        self._last_U = None

    # ---------------- dynamics ----------------
    def _f_cont(self, x, u):
        """SRB continuous dynamics."""
        m, I, g, foot_x, ground_z = self.m, self.I, self.g, self.foot_x, self.ground_z
        px, pz, th, vx, vz, w = x[0], x[1], x[2], x[3], x[4], x[5]
        FxL, FzL, FxR, FzR = u[0], u[1], u[2], u[3]

        rxL, rzL = -foot_x, ground_z - pz
        rxR, rzR = +foot_x, ground_z - pz

        dpx, dpz, dth = vx, vz, w
        dvx = (FxL + FxR) / m
        dvz = (FzL + FzR) / m - g
        tau = (rxL * FzL - rzL * FxL) + (rxR * FzR - rzR * FxR)
        dw = -tau / I   # 与前倾正向一致

        return ca.vertcat(dpx, dpz, dth, dvx, dvz, dw)

    def _f_disc(self, x, u):
        """RK4 discrete step."""
        f = self._f_cont
        dt = self.dt

        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)

        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # ---------------- build opti ----------------
    def _build(self):
        N = self.N

        # 符号
        x = ca.SX.sym("x", 6)
        u = ca.SX.sym("u", 4)
        x_next = self._f_disc(x, u)
        self.f = ca.Function("f", [x, u], [x_next])

        opti = ca.Opti()
        X = opti.variable(6, N + 1)
        U = opti.variable(4, N)
        X0 = opti.parameter(6)

        # 参数化参考（每步相同；若需轨迹跟踪可改为矩阵参数）
        xref = opti.parameter(6)

        # ⭐ 新增：每步的接触模式（由外部传入的参数）
        contact_L = opti.parameter(N)   # 左脚（bfoot）
        contact_R = opti.parameter(N)   # 右脚（ffoot）

        # 代价矩阵
        Q = ca.DM(self.Q)
        Qf = ca.DM(self.Qf)
        R = ca.DM(self.R)

        cost = 0
        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]

            # 动力学约束
            opti.subject_to(X[:, k + 1] == self.f(xk, uk))

            # 状态误差
            x_err = xk - xref
            x_err = ca.vertcat(
                0.0 if self.x_ref[0] is None else x_err[0],
                x_err[1],
                x_err[2],
                x_err[3],
                x_err[4],
                x_err[5],
            )

            # 输入误差：是否围绕名义支撑
            if self.use_nominal_support:
                u_ref = ca.DM([0.0, self.m * self.g / 2.0, 0.0, self.m * self.g / 2.0])
                u_err = uk - u_ref
                stage_cost = ca.mtimes([x_err.T, Q, x_err]) + ca.mtimes([u_err.T, R, u_err])
            else:
                stage_cost = ca.mtimes([x_err.T, Q, x_err]) + ca.mtimes([uk.T, R, uk])

            cost += stage_cost

            # ⭐ 当前步的接触标志（标量）
            cL = contact_L[k]   # 0 或 1，或介于中间（soft）
            cR = contact_R[k]

            # ---- 左脚（bfoot）约束 ----
            # U[1,k] = FzL
            opti.subject_to(U[1, k] >= 0)             # FzL >= 0
            opti.subject_to(U[1, k] <= 200 * cL)      # FzL <= Fz_max * cL
            # 摩擦锥：|FxL| <= mu * FzL
            opti.subject_to( U[0, k] <=  self.mu * U[1, k])
            opti.subject_to(-U[0, k] <=  self.mu * U[1, k])

            # ---- 右脚（ffoot）约束 ----
            # U[3,k] = FzR
            opti.subject_to(U[3, k] >= 0)             # FzR >= 0
            opti.subject_to(U[3, k] <= 200 * cR)      # FzR <= Fz_max * cR
            opti.subject_to( U[2, k] <=  self.mu * U[3, k])
            opti.subject_to(-U[2, k] <=  self.mu * U[3, k])


        # 终端代价
        xN = X[:, N]
        xN_err = xN - xref
        xN_err = ca.vertcat(
            0.0 if self.x_ref[0] is None else xN_err[0],
            xN_err[1], xN_err[2], xN_err[3], xN_err[4], xN_err[5]
        )
        cost += ca.mtimes([xN_err.T, Qf, xN_err])

        # 初值约束
        opti.subject_to(X[:, 0] == X0)

        # 求解器
        opti.minimize(cost)
        opti.solver("ipopt", self.ipopt_p_opts, self.ipopt_s_opts)

        # 缓存句柄
        self.opti = opti
        self.X, self.U = X, U
        self.X0, self.xref = X0, xref
        self.cost_expr = cost

  
        self.contact_L_param = contact_L
        self.contact_R_param = contact_R

    # ---------------- API: set refs ----------------
    def set_reference(
        self,
        px=None, pz=0.4, th=0.0, vx=1.0, vz=0.0, w=0.0
    ):
        """
        设定“定常参考”。px=None 表示不跟踪 px（对应误差置零/或 Q[0,0]=0）。
        """
        self.x_ref = np.array([px, pz, th, vx, vz, w], dtype=object)
        # 把 None 用当前 X0 的 px 或 0 代替到参数里（误差仍会被置 0，不影响）
        xref_val = np.zeros(6, dtype=float)
        xref_val[0] = 0.0 if px is None else float(px)
        xref_val[1:] = [float(pz), float(th), float(vx), float(vz), float(w)]
        self.opti.set_value(self.xref, xref_val)

    # ---------------- API: warm start ----------------
    def _default_warm_start(self):
        """
        第一次求解的简单 warm-start：
        px 以 vx_ref 线性前进；pz,th,vx,vz,w 线性趋向参考；
        力初值：Fx=0，Fz=mg/2 对称。
        """
        N, dt = self.N, self.dt
        x0 = np.array(self.opti.value(self.X0), dtype=float).reshape(6)
        # 参考（把 None -> 用 x0[0] 占位，只影响 warm-start）
        ref = np.array([x0[0] if self.x_ref[0] is None else float(self.x_ref[0]),
                        float(self.x_ref[1]), float(self.x_ref[2]),
                        float(self.x_ref[3]), float(self.x_ref[4]), float(self.x_ref[5])])

        X_init = np.zeros((6, N + 1))
        tgrid = np.arange(N + 1) * dt
        # px：按参考速度推进
        X_init[0, :] = x0[0] + ref[3] * tgrid
        # 其它维线性插到参考
        for i in [1, 2, 3, 4, 5]:
            X_init[i, :] = np.linspace(x0[i], ref[i], N + 1)

        U_init = np.zeros((4, N))
        U_init[1, :] = 0.5 * self.m * self.g
        U_init[3, :] = 0.5 * self.m * self.g

        self.opti.set_initial(self.X, X_init)
        self.opti.set_initial(self.U, U_init)

    def shift_warm_start(self):
        """把上次的最优解右移一格作为新的初值。"""
        if self._last_X is None or self._last_U is None:
            return
        Xs = np.hstack([self._last_X[:, 1:], self._last_X[:, -1:]])
        Us = np.hstack([self._last_U[:, 1:], self._last_U[:, -1:]])
        self.opti.set_initial(self.X, Xs)
        self.opti.set_initial(self.U, Us)

    # ---------------- API: solve / step ----------------
    def solve(self, x0, refs=None, verbose=False, contact=None):
        """
        一次求解：
        x0: 当前状态 (6,)
        refs: 可选 dict 覆盖参考，如 {'pz':0.42, 'vx':1.2}
        contact: 可选 dict，形如：
                {
                    'L': 标量 或 长度 N 的 array（左脚接触 0/1 或 [0/1,...]）,
                    'R': 同上，右脚
                }
                若给标量，则会在 horizon 内广播；给长度 N，则逐步使用。
        返回: u0(4,), X_opt(6,N+1), U_opt(4,N), J_opt(float)
        """
        x0 = np.asarray(x0, dtype=float).reshape(6)
        self.opti.set_value(self.X0, x0)

        # 设置参考
        if refs:
            px = refs.get("px", self.x_ref[0])
            self.set_reference(
                px=px,
                pz=refs.get("pz", float(self.x_ref[1])),
                th=refs.get("th", float(self.x_ref[2])),
                vx=refs.get("vx", float(self.x_ref[3])),
                vz=refs.get("vz", float(self.x_ref[4])),
                w=refs.get("w",  float(self.x_ref[5])),
            )
        else:
            self.set_reference(
                px=self.x_ref[0],
                pz=float(self.x_ref[1]),
                th=float(self.x_ref[2]),
                vx=float(self.x_ref[3]),
                vz=float(self.x_ref[4]),
                w=float(self.x_ref[5]),
            )

        # ⭐ 设置接触 schedule
        N = self.N
        if contact is not None:
            # 取出左右脚的接触标志
            cL = contact.get("L")
            cR = contact.get("R")

            cL = np.asarray(cL, dtype=float).reshape(-1)
            cR = np.asarray(cR, dtype=float).reshape(-1)

            # 若只给了一个标量，则在 horizon 内广播
            if cL.size == 1:
                cL = np.full(N, cL.item(), dtype=float)
            if cR.size == 1:
                cR = np.full(N, cR.item(), dtype=float)

            assert cL.size == N and cR.size == N, "contact['L'], contact['R'] 长度必须为 N 或标量"

            self.opti.set_value(self.contact_L_param, cL)
            self.opti.set_value(self.contact_R_param, cR)

        else:
            # 默认：两脚全程允许接触（都可以出力）
            self.opti.set_value(self.contact_L_param, np.ones(N))
            self.opti.set_value(self.contact_R_param, np.ones(N))

        # warm-start：第一次自动生成，之后自动 shift
        if self._last_X is None or self._last_U is None:
            self._default_warm_start()

        try:
            sol = self.opti.solve()
        except RuntimeError as e:
            if verbose:
                print("Solve failed, retry with fresh warm-start. Error:", str(e))
            self._default_warm_start()
            sol = self.opti.solve()

        X_opt = np.array(sol.value(self.X), dtype=float)
        U_opt = np.array(sol.value(self.U), dtype=float)
        J_opt = float(sol.value(self.cost_expr))
        u0 = U_opt[:, 0].copy()

        # 缓存用于下一次 shift
        self._last_X, self._last_U = X_opt.copy(), U_opt.copy()

        if verbose:
            np.set_printoptions(precision=4, suppress=True)
            print("Optimal cost J* =", J_opt)
            print("u0 (FxL,FzL,FxR,FzR) =", u0)

        return u0, X_opt, U_opt, J_opt

    def step(self, x0, refs=None, verbose=False, contact=None):
        """滚动一步：先右移 warm-start，再 solve。"""
        self.shift_warm_start()
        return self.solve(x0, refs=refs, verbose=verbose, contact=contact)



class GaitScheduler:
    """
    MIT 风格的“步态调度器”（简化版，两条腿交替）：
      - 给出一个全局相位 phase ∈ [0,1)
      - back（后脚）和 front（前脚）相位相差 180°
      - 每条腿各自 cycle_T = stance_T + swing_T
      - 在一个周期内：前 duty 比例是 stance，相当于 contact=1；其余是 swing，相当于 contact=0
    MPC 只看 contact sequence（0/1），SwingController 用自己那套 liftoff/touchdown 逻辑，
    但你可以以后让 SwingController 也读 scheduler 的相位。
    """

    def __init__(self, swing_T=0.2, stance_T=0.25, phase0=0.0):
        self.swing_T = float(swing_T)
        self.stance_T = float(stance_T)
        self.T_cycle = self.swing_T + self.stance_T  # 单腿完整周期时长
        self.phase = float(phase0) % 1.0             # 全局相位 φ ∈ [0,1)

    def reset(self, phase0=0.0):
        self.phase = float(phase0) % 1.0

    def update(self, dt):
        """每个控制周期调用一次，更新全局相位"""
        self.phase = (self.phase + dt / self.T_cycle) % 1.0

    # ------- 单步判定：当前时刻这一条腿是 stance 还是 swing -------

    def _leg_phase(self, leg: str, base_phase: float = None) -> float:
        """
        返回给定腿的相位：
          back: 使用全局 phase
          front: 相位 + 0.5 （半周期错开，交替步态）
        """
        phi = self.phase if base_phase is None else base_phase
        if leg == "back":
            return phi
        elif leg == "front":
            return (phi + 0.5) % 1.0
        else:
            raise ValueError(f"Unknown leg name: {leg}")

    @property
    def duty_factor(self) -> float:
        """duty factor = stance_T / cycle_T"""
        return self.stance_T / self.T_cycle

    def leg_is_stance(self, leg: str) -> bool:
        """当前时刻，这条腿是否在 stance（可出地面力）"""
        phi_leg = self._leg_phase(leg)
        return phi_leg < self.duty_factor

    def leg_is_swing(self, leg: str) -> bool:
        return not self.leg_is_stance(leg)

    # ------- 生成未来 N 步的 contact sequence -------

    def build_contact_sequence(self, N: int, dt: float):
        """
        构造未来 N 步的 contact sequence：
          - 对 back（后脚，L）：返回 cL[k] ∈ {0,1}
          - 对 front（前脚，R）：返回 cR[k] ∈ {0,1}
        这里不做软过渡，先用干净的 0/1 交替步态，方便你看清步态结构。
        """
        N = int(N)
        dt = float(dt)
        cL = np.zeros(N, dtype=float)
        cR = np.zeros(N, dtype=float)
        duty = self.duty_factor

        for k in range(N):
            # 预测时刻的“全局相位”
            phi_k = (self.phase + k * dt / self.T_cycle) % 1.0

            # 后脚（back）相位
            phi_back = phi_k
            # 前脚（front）相位：相位 + 0.5 实现交替
            phi_front = (phi_k + 0.5) % 1.0

            cL[k] = 1.0 if phi_back  < duty else 0.0
            cR[k] = 1.0 if phi_front < duty else 0.0

        return cL, cR


# ---------------------- quick usage ----------------------
if __name__ == "__main__":
    mpc = SRB_MPC(
        dt=0.05, N=20, m=14.0, I=0.90, g=9.81, foot_x=0.5, mu=0.4,
        use_nominal_support=False,
        ipopt_p_opts={"print_time": 0},
        ipopt_s_opts={"print_level": 0, "max_iter": 500, "tol": 1e-6},
    )
    cfg = env_warpper.HCEnvConfig(render=True, obs_as_state=False, seed=0)
    env = env_warpper.HalfCheetahEnv(cfg)
    obs, _ = env.reset() 

    # 参考（vx=0.5 m/s, 其余默认）
    mpc.set_reference(px=None, pz=0.4, th=0.0, vx=0.3, vz=0.0, w=0.0)
 
    # 新建一个 swing 控制器  
    swing_ctrl = SwingController(
        env,
        swing_T=0.25,
        step_length=0.5,
        step_height=0.4,
        Kp=[150,210],
        Kd=[5,3],
    )

    gait = GaitScheduler(
        swing_T=0.25,
        stance_T=0.25,
        phase0=0.0,   # 初始相位，可以随便设
    )

    T = 100
    x_history = []

    # # ===== 新增：记录 stance / swing 的控制量 =====
    stance_actions = []   # 每步的 ctrl_stance (nu,)
    swing_actions  = []   # 每步的 ctrl_swing  (nu,)


    # # ===== 正弦参考的参数 =====
    # z_center = 0.3       # 中心高度
    # z_amp    = 0.15      # 振幅
    # z_freq   = 0.5       # 频率 (Hz)，2 秒一个周期，你可以改

    try:
        for t in range(T):
            ### TIMER: 当前迭代开始的时间戳



            # # ===== 在这里根据当前时间更新 MPC 的参考 =====
            # t_sec = t * mpc.dt   # 当前物理时间（秒）
            # z_ref = z_center + z_amp * np.sin(2.0 * np.pi * z_freq * t_sec)

            # mpc.set_reference(
            #     px=None,
            #     pz=z_ref,
            #     th=0.0,
            #     vx=0.0,
            #     vz=0.0,
            #     w=0.0,
            # )
            # ===============================================



            x_t = get_single_rigidbody_state(env)
            x_history.append(np.copy(x_t))

            # === 1) 更新 gait scheduler 的相位 ===
            gait.update(mpc.dt)   # 假设控制周期 = mpc.dt
            
            # === 2) 真实接触（只给 SwingController 用）===
            contact_L_now, contact_R_now = detect_foot_contact(env)
            
            # === 3) Swing 控制器：根据真实接触更新内部 swing 状态，生成 tau_swing ===
            # tau_swing = swing_ctrl.update_and_compute_tau(
            #     contact_L_now=contact_L_now,
            #     contact_R_now=contact_R_now,
            # )
            FxL, FzL, FxR, FzR = swing_ctrl.update_and_compute_tau(
                contact_L_now=contact_L_now,
                contact_R_now=contact_R_now,
            )

            # === 7) swing 部分：joint tau -> actuator ctrl ===
            # ctrl_swing = map_joint_tau_to_u(env, tau_swing) 
            ctrl_swing = map_grf_to_u(env, FxL, FzL, FxR, FzR)
            print(FxR,FzR)

            # ===================MPC=============
            if t % 1 == 0 or t == 0 :
                contact_Seq = gait.build_contact_sequence(20,0.05)
                contact_Seq[0][0] = 1.0 if contact_L_now else 0
                contact_Seq[1][0] = 1.0 if contact_R_now else 0
                contact_dict = {"L": contact_Seq[0], "R": contact_Seq[1]}
                # print(contact_dict)
                # === 5) MPC：在给定 contact sequence 下规划支撑脚 GRF ===
                u0, X_opt, U_opt, J_opt = mpc.step(
                    x_t,
                    refs=None,
                    verbose=False,
                    contact=contact_dict,
                )

                FxL, FzL, FxR, FzR = float(u0[0]), float(u0[1]), float(u0[2]), float(u0[3])
                ctrl_stance = map_grf_to_u(env, FxL, FzL, FxR, FzR)



            # print(ctrl_swing)
            # ====== 新增：把两部分都存起来 ======
            stance_actions.append(np.copy(-ctrl_stance))
            swing_actions.append(np.copy(ctrl_swing))

            # === 8) 合成动作并与环境交互 ===
            action = -ctrl_stance + ctrl_swing
            # print("ctrl_swing", ctrl_swing)
            if hasattr(env, "action_space"):
                low, high = env.action_space.low, env.action_space.high
                action = np.clip(np.asarray(action, dtype=np.float32), low, high)
            loop_start = time.time()
            obs, rew, terminated, truncated, info = env.step(action)
 
            # debug 打印
            # if t % 10 == 0: 
            #     print(f"[t={t}] real contact: L={contact_L_now}, R={contact_R_now}")
            #     # print(f"         sched[0]: L={contact_L_hor[0]}, R={contact_R_hor[0]}")
            #     print(f"         u0: FxL={FxL:.2f}, FzL={FzL:.2f}, FxR={FxR:.2f}, FzR={FzR:.2f}")

            # if (t % 50) == 0:
            #     print(f"[t={t}] J={J_opt:.2f}  reward={rew:.3f}")

            # if terminated or truncated:
            #     print(f"Episode ended at t={t} (terminated={terminated}, truncated={truncated}). Resetting.")
            #     obs, _ = env.reset()
        ## TIMER: 补足 0.05 秒控制频率


    finally:
        env.close()
    swing_ctrl.plot_debug()

    # ================== 画图：stance vs swing 的 action ==================

    stance_actions = np.array(stance_actions)   # (T_eff, nu)
    swing_actions  = np.array(swing_actions)    # (T_eff, nu)
    T_eff, nu = stance_actions.shape

    time = np.arange(T_eff) * mpc.dt

    plt.figure(figsize=(10, 2 * nu))

    for i in range(nu):
        plt.subplot(nu, 1, i + 1)
        plt.plot(time, stance_actions[:, i], label="stance (MPC)", linewidth=1.0)
        plt.plot(time, swing_actions[:, i],  label="swing (PD)",  linewidth=1.0, linestyle="--")
        plt.ylabel(f"u[{i}]")
        plt.grid(True)
        if i == 0:
            plt.title("Actuator commands: stance vs swing")
        if i == nu - 1:
            plt.xlabel("Time [s]")
        plt.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


    # 画轨迹（保持你原来的画图代码）
    x_history = np.array(x_history)
    time_arr = np.arange(len(x_history)) * env.dt

    labels = ["px", "pz", "theta", "vx", "vz", "omega"]
    plt.figure(figsize=(10, 8))
    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.plot(time_arr, x_history[:, i])
        plt.xlabel("Time [s]")
        plt.ylabel(labels[i])
        plt.grid(True)
    plt.tight_layout()
    plt.show()

# # ===== 转 numpy =====
# expert_obs = np.asarray(expert_obs, dtype=np.float32)
# expert_actions = np.asarray(expert_actions, dtype=np.float32)

# expert_data = {
#     "observations": expert_obs,
#     "actions": expert_actions,
# }

# # ===== 获取当前 Python 文件所在目录 =====
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # ===== expert_data 文件夹路径 =====
# save_dir = os.path.join(current_dir, "expert_data")

# # ===== 自动创建 expert_data 文件夹 =====
# os.makedirs(save_dir, exist_ok=True)

# # ===== 最终文件完整路径 =====
# save_path = os.path.join(save_dir, "expert_data_HalfCheetah_SRB_1.pkl")

# # ===== 保存 pkl =====
# with open(save_path, "wb") as f:
#     pickle.dump(expert_data, f)

# print("Expert data saved to:", save_path)