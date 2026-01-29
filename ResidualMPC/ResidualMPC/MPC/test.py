import numpy as np

class FixedGaitScheduler:
    """
    完全固定步态（交替支撑，支持可选双支撑），不依赖真实接触。
    只在时间轴上按节拍切换 Swing / Stance。
    """

    def __init__(self, dt_ctrl: float, step_time: float = 0.25, ds_time: float = 0.05):
        """
        参数:
            dt_ctrl : 控制周期（你的主循环 dt），比如 0.05
            step_time : 半周期时长（左支撑 or 右支撑一段），单位 s
            ds_time   : 每个半周期中的双支撑总时长 (0 表示没有双支撑)
        """
        self.dt_ctrl = float(dt_ctrl)
        self.step_time = float(step_time)
        self.ds_time = float(ds_time)

        # 一个完整 gait 周期：左支撑半周期 + 右支撑半周期
        self.cycle = 2.0 * self.step_time

        # 当前在周期中的时间（0 ~ cycle）
        self.phase_time = 0.0

    # ==========================================================
    # 内部工具：给定某个“相对时间”算出 L/R 是否接触
    # ==========================================================
    def _contacts_at_time(self, t: float):
        """
        输入:
            t : 从当前时刻起，向前看的一个时间偏移（可以 > cycle，会自动 mod）
        返回:
            contact_L, contact_R : bool
        """
        # 把时间折回一个 gait 周期内
        t_cycle = (self.phase_time + t) % self.cycle

        # 判断是哪个半周期：0~step_time -> 左腿主支撑；step_time~2*step_time -> 右腿主支撑
        if t_cycle < self.step_time:
            stance_leg = "L"
            t_local = t_cycle
        else:
            stance_leg = "R"
            t_local = t_cycle - self.step_time  # 映射到该半周期内部 [0, step_time)

        # ===== 决定双支撑 / 单支撑 =====
        # ds_time = 0 的话就只有单支撑
        if self.ds_time <= 0.0 or self.ds_time >= self.step_time:
            # 简单版：整个半周期都是“单支撑”
            if stance_leg == "L":
                return True, False
            else:
                return False, True

        # 有双支撑的情况：
        ds_half = 0.5 * self.ds_time
        s = self.step_time

        # [0, ds_half)  和  (s - ds_half, s] : 双支撑
        if t_local < ds_half or t_local >= s - ds_half:
            # 双支撑，两条腿都着地
            return True, True
        else:
            # 中间单支撑，只让当前半周期那条腿支撑
            if stance_leg == "L":
                return True, False
            else:
                return False, True

    # ==========================================================
    # 对外接口
    # ==========================================================
    def update(self):
        """
        每跑一次控制循环就调用一次，推进一下全局相位。
        （你可以在主循环里：gait.update()）
        """ 
        self.phase_time = (self.phase_time + self.dt_ctrl) % self.cycle

    def get_contact_now(self):
        """
        当前时刻的接触标志（给 Swing / Stance 控制器用）
        返回:
            contact_L_now, contact_R_now : bool
        """
        return self._contacts_at_time(t=0.0)

    def build_contact_sequence(self, N: int, dt: float):
        """
        给 MPC 用的 horizon 内接触序列（完全固定步态，不考虑真实接触）

        参数:
            N  : MPC 预测步数（你现在 N=20）
            dt : MPC 步长（你现在 dt=0.05, 一般和控制周期一致）

        返回:
            contact_L_hor, contact_R_hor : dict/list/np.array 皆可
                长度 N 的 0/1 浮点数组，1 表示 stance，0 表示 swing
        """
        contact_L = np.zeros(N, dtype=float)
        contact_R = np.zeros(N, dtype=float)

        for k in range(N):
            t_k = k * dt
            cL, cR = self._contacts_at_time(t_k)
            contact_L[k] = 1.0 if cL else 0.0
            contact_R[k] = 1.0 if cR else 0.0

        return contact_L, contact_R


gait = FixedGaitScheduler(
    dt_ctrl=0.05,
    step_time=0.25,   # 半周期 0.25s → 整个周期 0.5s
    ds_time=0.05      # 每个半周期 0.05s 双支撑 (可改成 0.0 变成纯单支撑)
)

t=0
while t <= 20:
    # === 1) 更新步态相位（完全按时间推进） ===
    t += 1
    gait.update()

    # === 2) 当前时刻的接触（给 Swing / Stance 控制器）===
    contact_L_now, contact_R_now = gait.get_contact_now()
    print(contact_L_now, contact_R_now)
    # 现在如果你想完全固定步态，就不要再用真实接触覆盖，比如：
    # contact_L_now, contact_R_now = detect_foot_contact(env)  # <-- 把这句注释掉

    # === 3) 给 Swing 控制器：用 contact_L_now / contact_R_now 决定 Swing or Stance ===


    # === 4) 给 MPC：构造 horizon 内的固定 contact sequence ===
    contact_L_hor, contact_R_hor = gait.build_contact_sequence(
        N=20,
        dt=0.05)
    print(contact_L_hor, contact_R_hor) 