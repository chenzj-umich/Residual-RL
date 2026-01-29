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
    masses = env.model.body_mass[1:]           # (nb-1,)
    xipos  = env.data.xipos[1:, :]             # (nb-1, 3) 世界系质心位置
    M = masses.sum()
    com_pos = (masses[:, None] * xipos).sum(axis=0) / M
    return np.array([com_pos[0], com_pos[2]], dtype=float)   # COM Px Pz


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


# -------------------- 足端 + Jacobian 工具 -------------------- #

def get_foot_pos_and_jac(env, leg="back"):
    """
    获取给定腿（back/front）的足端世界系位置 p (3,) 和位置雅可比 jacp (3, nv)
    leg: "back" or "front"
    """
    model, data = env.model, env.data
    nv = int(model.nv)

    if leg == "back":
        cands = ["bfoot", "bfoot_site", "left_foot", "l_foot", "rear_foot_site"]
    else:
        cands = ["ffoot", "ffoot_site", "right_foot", "r_foot", "front_foot_site"]

    # 先找 site，没有再退回 body
    sid = -1
    for name in cands:
        sid = mujoco.mj_name2id(model, mjtObj.mjOBJ_SITE, name)
        if sid != -1:
            break

    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    if sid != -1:
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        p = data.site_xpos[sid].copy()
    else:
        # 用 body 质心
        bid = -1
        for name in cands:
            bid = mujoco.mj_name2id(model, mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                break
        if bid == -1:
            raise RuntimeError(f"找不到 {leg} 足端的 site/body，请检查 XML 命名。")

        mujoco.mj_jacBody(model, data, jacp, jacr, bid)
        p = data.xipos[bid].copy()   # body COM 位置近似足端

    return p, jacp


def swing_leg_task_space_torque(
    env,
    leg,
    p_des_world,
    Kp=np.diag([500.0, 0.0, 500.0]),
    Kd=np.diag([60.0, 0.0, 60.0])
):
    """
    对 back/front 某一条腿做足端空间 PD：
    leg: "back" 或 "front"
    p_des_world: 期望足端世界系位置 (3,)
    这里只控制 x,z（即 index 0,2），y 方向忽略。
    返回：tau_joints_swing (6,) —— 只对受控 6 个关节的力矩
    """
    model, data = env.model, env.data
    nv = int(model.nv)

    # 1) 当前足端位置 & jacobian
    p, jacp = get_foot_pos_and_jac(env, leg)      # p (3,), jacp (3,nv)
    # 当前 joint 速度
    qvel = data.qvel.copy()
    v_foot = jacp @ qvel                          # (3,)

    # 2) 只管 x,z 两个方向
    idx = [0, 2]   # x,z
    p_err = p_des_world[idx] - p[idx]
    v_err = -v_foot[idx]   # v_ref = 0 ⇒ v_err = -v

    # 3) 足端空间力
    Kp2 = Kp[np.ix_(idx, idx)]
    Kd2 = Kd[np.ix_(idx, idx)]
    F = Kp2 @ p_err + Kd2 @ v_err                 # (2,)

    # 4) 映射成全自由度力矩
    J2 = jacp[idx, :]                             # (2, nv)
    tau_full = J2.T @ F                           # (nv,)

    # 5) 只取受控 6 个关节（去掉 base 的 3 个）
    tau_joints = tau_full[3:3+6].copy()

    return tau_joints


# -------------------- GRF 映射 + 允许叠加 swing 扭矩 -------------------- #

def map_grf_to_u(env, FxL, FzL, FxR, FzR, tau_extra_joints=None):
    """
    将左右足端在世界系中的接触力 (Fx, Fz) 映射到各关节力矩（返回顺序与 MuJoCo 中 actuator 顺序一致）。

    额外支持 tau_extra_joints (6,) —— 例如 swing leg 的足端空间 PD 扭矩，会叠加到受控 6 个关节上。
    """
    model, data = env.model, env.data
    nv, nu = int(model.nv), int(model.nu)

    # ---------- 1) 先把 GRF 映射到“全自由度的广义力” ----------
    # 尝试解析前后脚（front/back）的 site，失败则退回 body 质心雅可比
    front_candidates = ["ffoot", "ffoot_site", "right_foot", "r_foot", "front_foot_site"]
    back_candidates  = ["bfoot",  "bfoot_site", "left_foot",  "l_foot", "rear_foot_site"]

    def _resolve_jac(cands):
        for name in cands:
            sid = mujoco.mj_name2id(model, mjtObj.mjOBJ_SITE, name)
            if sid != -1:
                jacp = np.zeros((3, nv)); jacr = np.zeros((3, nv))
                mujoco.mj_jacSite(model, data, jacp, jacr, sid)
                return jacp, jacr
        for name in cands:
            bid = mujoco.mj_name2id(model, mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                jacp = np.zeros((3, nv)); jacr = np.zeros((3, nv))
                mujoco.mj_jacBody(model, data, jacp, jacr, bid)
                return jacp, jacr
        raise ValueError(f"找不到候选中的 site/body：{cands}。请检查 XML 命名。")

    jacp_front, jacr_front = _resolve_jac(front_candidates)  # 前脚雅可比
    jacp_back,  jacr_back  = _resolve_jac(back_candidates)   # 后脚雅可比

    f_front = np.array([FxR, 0.0, FzR], dtype=float)
    f_back  = np.array([FxL, 0.0, FzL], dtype=float)
    m_zero  = np.zeros(3)

    tau_front = jacp_front.T @ f_front + jacr_front.T @ m_zero
    tau_back  = jacp_back.T  @ f_back  + jacr_back.T  @ m_zero
    tau_total = tau_front + tau_back          # shape: (nv,)

    # 若是标准 half-cheetah：前3个是 base (x,z,theta)，后6个为电机关节
    tau_joints = tau_total[3:3+6].copy()  # 只取受控关节部分

    # 叠加 swing leg 的关节力矩
    if tau_extra_joints is not None:
        tau_joints += np.asarray(tau_extra_joints, dtype=float).reshape(6,)

    # ---------- 2) 把“关节力矩 τ_joints”映射成“控制量 ctrl (u)” ----------
    ctrl = np.zeros(nu, dtype=float)

    # 找出所有被 actuator 驱动的 hinge 关节
    driven_joint_ids = []
    for i in range(nu):
        jid = int(model.actuator_trnid[i, 0])
        if 0 <= jid < nv:  # 关节传动
            driven_joint_ids.append(jid)
    driven_joint_ids = sorted(set(driven_joint_ids))

    # joint_id -> tau 值
    tau_per_joint = {jid: 0.0 for jid in driven_joint_ids}
    for k, jid in enumerate(driven_joint_ids[:len(tau_joints)]):
        tau_per_joint[jid] = float(tau_joints[k])

    # 映射到每个 actuator 的 ctrl（考虑 gear 和 ctrlrange 裁剪）
    for i in range(nu):
        jid = int(model.actuator_trnid[i, 0])
        g = float(model.actuator_gear[i, 0])

        if np.isclose(g, 0.0):
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


def get_contact_schedule_aligned(env, t, N, dt, step_time=0.25, ds_time=0.05):
    """
    交替步态 + 允许双支撑 + 和 MuJoCo 实际接触大致对齐

    step_time: 单脚支撑相总时长（包含单支撑 + 双支撑），比如 0.25s
    ds_time:   每个半周期里的双支撑时长，比如 0.05s
    """
    # ===== 1) 基于相位生成“名义 pattern”，包含双支撑 =====
    # 每只脚支撑 step_time 秒，其中 ds_time 秒是双支撑
    step_len = max(2, int(round(step_time / dt)))   # 半周期总步数（左支撑 or 右支撑）
    ds_len   = max(1, int(round(ds_time   / dt)))   # 双支撑步数
    single_len = max(1, step_len - ds_len)          # 单支撑步数

    period_steps = step_len * 2                     # 左支撑相 + 右支撑相

    contact_L = np.zeros(N, dtype=float)
    contact_R = np.zeros(N, dtype=float)

    for k in range(N):
        phase = (t + k) % period_steps

        if phase < step_len:
            # ------ 左脚支撑半周期 ------
            if phase < single_len:
                # 前半段：左单支撑
                contact_L[k] = 1.0
                contact_R[k] = 0.0
            else:
                # 后半段：双支撑（左 + 右）
                contact_L[k] = 1.0
                contact_R[k] = 1.0
        else:
            # ------ 右脚支撑半周期 ------
            phase2 = phase - step_len
            if phase2 < ds_len:
                # 前半段：双支撑（左 + 右）
                contact_L[k] = 1.0
                contact_R[k] = 1.0
            else:
                # 后半段：右单支撑
                contact_L[k] = 0.0
                contact_R[k] = 1.0

    # ===== 2) 读取当前真实接触 =====
    contact_L_now, contact_R_now = detect_foot_contact(env)

    # ===== 3) 对齐当前 step：如果现在是单脚支撑，但名义支撑脚反了，就整体翻转 L/R =====
    L_nom = contact_L[0] > 0.5
    R_nom = contact_R[0] > 0.5

    # 只有在“确实单脚支撑”的情况下才翻转；双支撑就不翻
    if (L_nom and not R_nom) and (contact_R_now and not contact_L_now):
        # 名义: 左支撑，实际: 右支撑  -> 翻转
        contact_L, contact_R = contact_R.copy(), contact_L.copy()
    elif (R_nom and not L_nom) and (contact_L_now and not contact_R_now):
        # 名义: 右支撑，实际: 左支撑 -> 翻转
        contact_L, contact_R = contact_R.copy(), contact_L.copy()

    # ===== 4) 如果当前真实是双支撑，就强制 horizon 开头几步双支撑 =====
    if contact_L_now and contact_R_now:
        ds0 = min(ds_len, N)
        contact_L[:ds0] = 1.0
        contact_R[:ds0] = 1.0

    return contact_L, contact_R


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
        I=3.5,
        g=9.81,
        foot_x=0.57,
        ground_z=0.0,
        mu=0.16,
        # 参考（默认 px 不严格跟踪；pz=0.4, th=0, vx=0.5, vz=0, w=0）
        x_ref=np.array([None, 0.4, 0.0, .5, 0.0, 0.0], dtype=object),
        # 权重
        Q=np.diag([0, 80, 5, 100, 40, 10]),
        Qf=None,                      # 若 None，使用放大 Q
        R=np.diag([5e-4, 1e-3, 5e-4, 1e-3]),
        use_nominal_support=False,
        ipopt_p_opts={"print_time": 0},
        ipopt_s_opts={"print_level": 0, "max_iter": 100, "tol": 1e-3,
                      "hessian_approximation": "limited-memory"},
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

        # 参数化参考
        xref = opti.parameter(6)

        # 每步的接触模式（由外部传入的参数）
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

            # 当前步的接触标志
            cL = contact_L[k]   # 0 或 1
            cR = contact_R[k]

            # 左脚（bfoot）约束——U[1,k] = FzL
            opti.subject_to(U[1, k] >= 0)             # FzL >= 0
            opti.subject_to(U[1, k] <= 200 * cL)      # FzL <= Fz_max * cL
            opti.subject_to( U[0, k] <=  self.mu * U[1, k])
            opti.subject_to(-U[0, k] <=  self.mu * U[1, k])

            # 右脚（ffoot）约束——U[3,k] = FzR
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

        # 接触参数
        self.contact_L_param = contact_L
        self.contact_R_param = contact_R

    # ---------------- API: set refs ----------------
    def set_reference(
        self,
        px=None, pz=0.4, th=0.0, vx=1.0, vz=0.0, w=0.0
    ):
        """
        设定“定常参考”。px=None 表示不跟踪 px。
        """
        self.x_ref = np.array([px, pz, th, vx, vz, w], dtype=object)
        xref_val = np.zeros(6, dtype=float)
        xref_val[0] = 0.0 if px is None else float(px)
        xref_val[1:] = [float(pz), float(th), float(vx), float(vz), float(w)]
        self.opti.set_value(self.xref, xref_val)

    # ---------------- API: warm start ----------------
    def _default_warm_start(self):
        """
        第一次求解的简单 warm-start。
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

        # 设置接触 schedule
        N = self.N
        if contact is not None:
            cL = contact.get("L", 1.0)
            cR = contact.get("R", 1.0)

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


# ---------------------- main: 跑 SRB+MPC + swing leg ---------------------- #

if __name__ == "__main__":
    mpc = SRB_MPC(
        dt=0.05,
        N=20,
        m=14.0,
        I=3.5,
        g=9.81,
        foot_x=0.57,
        mu=0.5,
        use_nominal_support=False,
        ipopt_p_opts={"print_time": 0},
        ipopt_s_opts={"print_level": 0, "max_iter": 500, "tol": 1e-6},
    )

    cfg = env_warpper.HCEnvConfig(render=True, obs_as_state=False, seed=0)
    env = env_warpper.HalfCheetahEnv(cfg)
    obs, _ = env.reset()
    x0 = get_single_rigidbody_state(env)

    # 参考（vx=0.5m/s）
    mpc.set_reference(px=None, pz=0.37, th=0.0, vx=0.5, vz=0.0, w=0.0)

    T = 500

    x_history = []
    expert_obs = []
    expert_actions = []

    # swing 足端落点的参数
    vx_ref = float(mpc.x_ref[3]) if mpc.x_ref[3] is not None else 0.5
    step_time = 0.25   # 和 contact_schedule_aligned 中一致
    z_clear = 0.2      # 摆动时抬脚高度

    try:
        for t in range(T):
            # 当前 SRB 状态
            x_t = get_single_rigidbody_state(env)  # (6,)
            x_history.append(np.copy(x_t))
            expert_obs.append(np.copy(x_t))

            # 预测用接触 schedule
            contact_L_hor, contact_R_hor = get_contact_schedule_aligned(
                env,
                t,
                mpc.N,
                mpc.dt,
                step_time=step_time
            )
            contact_dict = {"L": contact_L_hor, "R": contact_R_hor}

            # 一步 SRB-MPC
            u0, X_opt, U_opt, J_opt = mpc.step(
                x_t,
                refs=None,
                verbose=False,
                contact=contact_dict,
            )
            FxL, FzL, FxR, FzR = float(u0[0]), float(u0[1]), float(u0[2]), float(u0[3])

            # 当前真实接触（决定 stance / swing）
            contact_L_now, contact_R_now = detect_foot_contact(env)

            # 默认没有 swing torque
            tau_swing_joints = np.zeros(6, dtype=float)

            # 当前 COM，用于生成 swing 足端目标
            com_x, com_z = compute_com(env)
            step_offset = vx_ref * step_time * 0.5

            if contact_L_now and not contact_R_now:
                # 左脚支撑，右脚摆动：只让左脚出力
                FxR, FzR = 0.0, 0.0

                # 右脚（front）预期落点：COM 前方 step_offset + foot_x
                p_des_front = np.array([
                    com_x + step_offset + mpc.foot_x,
                    0.0,
                    mpc.ground_z + z_clear
                ])
                tau_swing_joints = swing_leg_task_space_torque(
                    env, leg="front", p_des_world=p_des_front
                )

            elif contact_R_now and not contact_L_now:
                # 右脚支撑，左脚摆动：只让右脚出力
                FxL, FzL = 0.0, 0.0

                # 左脚（back）预期落点：COM 前方 step_offset - foot_x
                p_des_back = np.array([
                    com_x + step_offset - mpc.foot_x,
                    0.0,
                    mpc.ground_z + z_clear
                ])
                tau_swing_joints = swing_leg_task_space_torque(
                    env, leg="back", p_des_world=p_des_back
                )
            else:
                # 双支撑 or 都没接触：两条腿都作为 stance，不加 swing PD
                pass

            # GRF + swing torque 综合映射成动作
            action = map_grf_to_u(env, FxL, FzL, FxR, FzR,
                                  tau_extra_joints=tau_swing_joints)

            # 裁剪到 action space
            if hasattr(env, "action_space"):
                low, high = env.action_space.low, env.action_space.high
                action = np.clip(np.asarray(action, dtype=np.float32), low, high)

            expert_actions.append(np.copy(action))

            # 环境 step
            obs, rew, terminated, truncated, info = env.step(action)

            if t % 10 == 0:
                print(f"[t={t}] real contact: L={contact_L_now}, R={contact_R_now}")
                print(f"         sched[0]: L={contact_L_hor[0]}, R={contact_R_hor[0]}")
                print(f"         u0: FxL={FxL:.2f}, FzL={FzL:.2f}, FxR={FxR:.2f}, FzR={FzR:.2f}")

            if (t % 50) == 0:
                print(f"[t={t}] u0=[{FxL:.2f},{FzL:.2f},{FxR:.2f},{FzR:.2f}]  "
                      f"J={J_opt:.2f}  reward={rew:.3f}")

            if terminated or truncated:
                print(f"Episode ended at t={t} (terminated={terminated}, truncated={truncated}). Resetting.")
                obs, _ = env.reset()
    finally:
        env.close()

    # 画 SRB 状态轨迹
    x_history = np.array(x_history)  # shape (T, 6)
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

    # 如需保存 expert data，解开下面注释
    # expert_obs = np.asarray(expert_obs, dtype=np.float32)
    # expert_actions = np.asarray(expert_actions, dtype=np.float32)
    # expert_data = {
    #     "observations": expert_obs,
    #     "actions": expert_actions,
    # }
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # save_dir = os.path.join(current_dir, "expert_data")
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, "expert_data_HalfCheetah_SRB_1.pkl")
    # with open(save_path, "wb") as f:
    #     pickle.dump(expert_data, f)
    # print("Expert data saved to:", save_path)
