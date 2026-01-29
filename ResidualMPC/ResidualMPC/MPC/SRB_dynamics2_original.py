import casadi as ca
import numpy as np
import env_warpper
import mujoco
from mujoco import mjtObj
import matplotlib.pyplot as plt
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
    front_candidates = ["ffoot", "ffoot_site", "right_foot", "r_foot", "front_foot_site"]
    back_candidates  = ["bfoot",  "bfoot_site", "left_foot",  "l_foot", "rear_foot_site"]

    def _resolve_jac(cands):
        for name in cands:
            sid = mujoco.mj_name2id(model, mjtObj.mjOBJ_SITE, name)
            if sid != -1:
                jacp = np.zeros((3, nv)); jacr = np.zeros((3, nv))
                mujoco.mj_jacSite(model, data, jacp, jacr, sid)
                print("found")
                return jacp, jacr
        for name in cands:
            bid = mujoco.mj_name2id(model, mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                jacp = np.zeros((3, nv)); jacr = np.zeros((3, nv))
                mujoco.mj_jacBody(model, data, jacp, jacr, bid)
                return jacp, jacr
        raise ValueError(f"找不到候选中的 site/body：{cands}。请检查 XML 命名。")

    jacp_front, jacr_front = _resolve_jac(front_candidates)  #这边返回的是body的雅可比矩阵
    jacp_back,  jacr_back  = _resolve_jac(back_candidates)


    f_front = np.array([FxR, 0.0, FzR], dtype=float)
    f_back  = np.array([FxL, 0.0, FzL], dtype=float)
    m_zero  = np.zeros(3)

    tau_front = jacp_front.T @ f_front + jacr_front.T @ m_zero
    tau_back  = jacp_back.T  @ f_back  + jacr_back.T  @ m_zero
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
        N=20,
        m=14.0,
        I=3.5,
        g=9.81,
        foot_x=0.57,
        ground_z=0.0,
        mu=0.16,
        # 参考（默认 px 不严格跟踪；pz=0.4, th=0, vx=1, vz=0, w=0）
        x_ref=np.array([None, 0.4, 0.0, 1.0, 0.0, 0.0], dtype=object),
        # 权重（与你原来相近；px=0 不罚）s
        Q=np.diag([ 0,  80,  5,  100,  40, 10 ]),
        Qf=None,                      # 若 None，使用 Q
        R=np.diag([5e-4, 1e-3, 5e-4, 1e-3]),
        use_nominal_support=False,     # 输入项使用 (u - u_ref)^T R (u - u_ref)
        ipopt_p_opts={"print_time": 0},
        ipopt_s_opts={"print_level": 0, "max_iter": 500, "tol": 1e-6},
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

        # 代价累积
        Q = ca.DM(self.Q)
        Qf = ca.DM(self.Qf)
        R = ca.DM(self.R)

        cost = 0
        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]

            # 动力学约束
            opti.subject_to(X[:, k + 1] == self.f(xk, uk))

            # 状态误差：px 若 x_ref 为 None，误差置 0（或保持 Q[0,0]=0）
            x_err = xk - xref
            # 处理 None：把对应误差设为 0（避免 NaN）
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
                cost += ca.mtimes([x_err.T, Q, x_err]) + ca.mtimes([u_err.T, R, u_err])
            else:
                cost += ca.mtimes([x_err.T, Q, x_err]) + ca.mtimes([uk.T, R, uk])

            # 摩擦与法向力非负（逐时刻）
            # 左脚
            opti.subject_to(U[1, k] >= 0)                        # FzL >= 0
            opti.subject_to( U[0, k] <=  self.mu * U[1, k])      #  FxL <=  mu*FzL
            opti.subject_to(-U[0, k] <=  self.mu * U[1, k])      # -FxL <=  mu*FzL
            # 右脚
            opti.subject_to(U[3, k] >= 0)                        # FzR >= 0
            opti.subject_to( U[2, k] <=  self.mu * U[3, k])      #  FxR <=  mu*FzR
            opti.subject_to(-U[2, k] <=  self.mu * U[3, k])      # -FxR <=  mu*FzR
            opti.subject_to(U[1,k] <= 160)
            opti.subject_to(U[3,k] <= 160) 

            opti.subject_to(U[0,k] >= U[2,k])                     # FxL >= FxR
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
    def solve(self, x0, refs=None, verbose=False):
        """
        一次求解：
          x0: 当前状态 (6,)
          refs: 可选 dict 覆盖参考，如 {'pz':0.42, 'vx':1.2}
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
            # 没传就用现有的 self.x_ref 写入参数
            self.set_reference(
                px=self.x_ref[0],
                pz=float(self.x_ref[1]),
                th=float(self.x_ref[2]),
                vx=float(self.x_ref[3]),
                vz=float(self.x_ref[4]),
                w=float(self.x_ref[5]),
            )

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

    def step(self, x0, refs=None, verbose=False):
        """滚动一步：先右移 warm-start，再 solve。"""
        self.shift_warm_start()
        return self.solve(x0, refs=refs, verbose=verbose)


# ---------------------- quick usage ----------------------
if __name__ == "__main__":
    mpc = SRB_MPC(
        dt=0.05, N=20, m=14.0, I=3.5, g=9.81, foot_x=0.57, mu=0.5,
        use_nominal_support=False,
        ipopt_p_opts={"print_time": 1},
        ipopt_s_opts={"print_level": 0, "max_iter": 500, "tol": 1e-6},
    )

    cfg = env_warpper.HCEnvConfig(render=True, obs_as_state=False, seed=0)
    env = env_warpper.HalfCheetahEnv(cfg)
    obs, _ = env.reset()
    x0 = get_single_rigidbody_state(env)

    # x0 = np.array([0.0, 0.4, 0.0, 0.0, 0.0, 0.0])

    # 参考（vx=1m/s, 其余默认）
    mpc.set_reference(px=None, pz=0.37, th=0.0, vx=1, vz=0.0, w=0.0)

    #这里给我写一段代码，让MPC跑1000步
    # 一次求解
    T = 500

    x_history = []
    try:
        for t in range(T):
            # 1) 从环境读取当前 SRB 状态 x_t
            x_t = get_single_rigidbody_state(env)  # 期望 shape (6,)

            x_history.append(np.copy(x_t))
            # 2) MPC：滚动一步（内部会做 warm-start shift）
            u0, X_opt, U_opt, J_opt = mpc.step(x_t, refs=None, verbose=False)
            print("u0:",u0)

            # 3) 将足端力映射到关节力矩/动作空间
            FxL, FzL, FxR, FzR = float(u0[0]), float(u0[1]), float(u0[2]), float(u0[3])
            action = map_grf_to_u(env, FxL, FzL, FxR, FzR)
            print("action:",action)
            # 4) 安全裁剪到动作空间（如果环境定义了 action_space）
            if hasattr(env, "action_space"):
                low, high = env.action_space.low, env.action_space.high
                action = np.clip(np.asarray(action, dtype=np.float32), low, high)

            # 5) 与环境交互一步
            obs, rew, terminated, truncated, info = env.step(action)


            # 6) 可视化（如果 render=True）和周期性打印

            if (t % 50) == 0:
                print(f"[t={t}] u0=[{FxL:.2f},{FzL:.2f},{FxR:.2f},{FzR:.2f}]  "
                      f"J={J_opt:.2f}  reward={rew:.3f}")

            if terminated or truncated:
                print(f"Episode ended at t={t} (terminated={terminated}, truncated={truncated}). Resetting.")
                obs, _ = env.reset()
                # 可选：重置 MPC 的 warm-start（通常不需要，继续滚动即可）


    finally:
        env.close()

    x_history = np.array(x_history)  # shape (T, 6)
    time = np.arange(len(x_history)) * env.dt

    labels = ["px", "pz", "theta", "vx", "vz", "omega"]

    plt.figure(figsize=(10, 8))
    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.plot(time, x_history[:, i])
        plt.xlabel("Time [s]")
        plt.ylabel(labels[i])
        plt.grid(True)
    plt.tight_layout()
    plt.show()