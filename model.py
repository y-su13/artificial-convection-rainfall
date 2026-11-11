"""
model.py — r-z 轴对称变密度（anelastic）大气对流求解器

数值方案
--------
网格      Arakawa C 交错；半径 12 km（主算例），高 16 km；
          近场均匀 4 m（r < 650 m，z < 600 m），远场几何拉伸，
          单元边长封顶 140 m，末单元裁至边界（故 dr 末值 < 140）
动量对流  QUICK（三阶迎风偏置，二阶精度）
标量对流  MUSCL + minmod 限制器，通量形式（守恒）
时间推进  前向 Euler + 每步一次压力投影；dt 由全波数稳定扫描给出
压力      anelastic 泊松方程；算子只依赖网格与参考密度，
          稀疏 LU 分解一次、每步仅回代
湍流      Smagorinsky (0.23Δ)²·S + Richardson 数修正 1/(1+8Ri)；
          旋翼尾流区附加涡粘
微物理    六类水物质 bulk：qv / qc / qi / qs / qr / qg（+ AgI 被动示踪）。
          含饱和调整（液面 Lv、冰面 Ls）、云水蒸发与冰晶升华、自动转化、
          碰并、云水冻结、冰晶转雪、雨滴冻结成霰、雪与霰融化、雨滴蒸发、
          雪的升华，以及各类的沉降输运
边界      底面 w=0 不可入 + 感热/潜热通量；轴线对称；
          顶部与外侧开口 + Rayleigh 海绵层

控制方程
--------
  du/dt + u du/dr + w du/dz =  v²/r - dP/dr + D_u
  dw/dt + u dw/dr + w dw/dz = -dP/dz + B - e|w|w + D_w + F_rotor
  dv/dt + u dv/dr + w dv/dz = -u v/r + D_v + F_swirl
  (1/r) d(r rho u)/dr + d(rho w)/dz = 0
  B = g (theta_v - theta_v,env) / theta_v,env
  theta_v = theta (1 + 0.608 qv - qc - qi - qr - qs - qg)

重要（2026-11-02）：水凝物负载 -qc - qi - qr - qs - qg 已经通过虚温浮力
进入垂直动量方程。不得再另加独立的 -g(qr+qs+qg) 拖曳项，那是同一物理
贡献的重复计入。详见 hydro_drag 处的作废说明。

e|w|w 为显式夹卷动量汇，e = k_e / R_eff，默认关闭；配套的标量卷吸在
scalars() 里，需与 entrain 同时开启才生效。
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

Rd, Rv, cp, g, Lv = 287.04, 461.5, 1004.5, 9.80665, 2.5e6
Lf = 3.34e5
# 升华潜热。冰面饱和对应的潜热是 Ls 而不是 Lv——第 16 轮之前这里一律用
# Lv，冰相潜热被低估 11.8%（2.5e6 vs 2.834e6）。深对流高层几乎全是冰相，
# 这个误差直接压低云顶与上升气流。
Ls = Lv + Lf
p0 = 100000.0
eps = Rd / Rv


def esat(T):
    Tc = np.clip(T - 273.15, -80.0, 60.0)
    return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))


def qsat(T, p):
    es = esat(T)
    return eps * es / np.maximum(p - es, 1.0)


def esat_ice(T):
    """冰面饱和水汽压（Magnus 型），用于 T<0 的饱和调整"""
    Tc = np.clip(T - 273.15, -80.0, 0.0)
    return 611.15 * np.exp(22.452 * Tc / (Tc + 272.55))


def qsat_ice(T, p):
    es = esat_ice(T)
    return eps * es / np.maximum(p - es, 1.0)


def exner(p):
    return (p / p0) ** (Rd / cp)


# ------------------------------------------------------------------ 网格
def build_grid(rmax, zmax, dr0, dz0, r_fine, z_fine, gr=1.032, gz=1.030,
               dr_max=None, dz_max=None):
    """近场均匀 + 远场几何拉伸，拉伸后的间距可用 dr_max/dz_max 封顶。

    封顶的必要性：不封顶时 1.03 的拉伸因子到 10 km 处会到 300 m 以上，
    远场格点被浪费；而近场（旋翼盘面与叶尖剪切层）又不够密。封顶后
    同样的格点数可以把近场加密到 4 m，远场保持在 100–150 m。
    """
    def axis(L, d0, fine, gfac, dmax):
        # 近场均匀段：格距取 fine/nf，使该段最后一个单元不被截短
        nf = max(int(np.ceil(fine / d0)), 1)
        df = fine / nf
        pos = list(np.arange(0.0, fine + 1e-9, df))
        d = df
        while pos[-1] < L - 1e-9:
            d = d * gfac
            if dmax is not None:
                d = min(d, dmax)
            rem = L - pos[-1]
            if rem <= d:
                # 收尾不能留薄片单元：显式格式的 dt 由最小格距决定，
                # 一个 0.3 m 的残片会把全局步长压到原来的几分之一。
                ns = max(int(np.ceil(rem / d)), 1)
                for _ in range(ns):
                    pos.append(pos[-1] + rem / ns)
                break
            pos.append(pos[-1] + d)
        return np.array(pos)

    rf = axis(rmax, dr0, r_fine, gr, dr_max)
    zf = axis(zmax, dz0, z_fine, gz, dz_max)
    rc = 0.5 * (rf[1:] + rf[:-1])
    zc = 0.5 * (zf[1:] + zf[:-1])
    return dict(rf=rf, zf=zf, rc=rc, zc=zc, dr=np.diff(rf), dz=np.diff(zf),
                nr=rc.size, nz=zc.size, rmax=rmax, zmax=zmax)


class Atmos:
    def __init__(self, grid, env, cfg):
        self.g = grid
        self.cfg = cfg
        self._env = env          # 供复制状态用（step_budget.py 的 clone）
        self.nr, self.nz = grid['nr'], grid['nz']
        zc, zf, rc = grid['zc'], grid['zf'], grid['rc']

        self.rho_c = np.interp(zc, env['z'], env['rho'])
        self.rho_w = np.interp(zf, env['z'], env['rho'])
        self.p_c = np.interp(zc, env['z'], env['p'])
        self.th0 = np.interp(zc, env['z'], env['theta'])
        self.qv0 = np.interp(zc, env['z'], env['qv'])
        self.thv0 = self.th0 * (1.0 + 0.608 * self.qv0)
        self.ex_c = exner(self.p_c)

        self.u = np.zeros((self.nr + 1, self.nz))
        self.w = np.zeros((self.nr, self.nz + 1))
        # 方位角速度分量：轴对称假定下各方位相同，存于单元中心。
        # 真机尾流旋流数受整流组件控制，方案指标为切向/轴向 ≤ 0.35。
        self.v = np.zeros((self.nr, self.nz))
        self.th = np.tile(self.th0, (self.nr, 1))
        self.qv = np.tile(self.qv0, (self.nr, 1))
        self.qc = np.zeros((self.nr, self.nz))
        # 雾工况：初始就有液态水。雾滴在模式里与云水同等对待（都是 qc），
        # 蒸发靠饱和调整自动处理，不需要单独的雾方案。
        if env.get('qc0') is not None and np.any(env['qc0'] > 0):
            self.qc[:] = np.interp(zc, env['z'], env['qc0'])[None, :]
        # 雾的水平边界（2026-11-05 按实际场景修）：
        # 真实辐射雾是有限范围的雾团，不是铺满整个计算域。若全域有雾，
        # 风机的低层抽吸永远有饱和空气可补，测出来的是"搅动"而不是"除雾"。
        # fog_r 给雾的外缘，edge 是过渡宽度（真实雾边有 1 km 量级的渐变，
        # 硬边会激起虚假的重力波）。
        fog_r = env.get('fog_r', self.cfg.get('fog_r'))
        if fog_r:
            edge = env.get('fog_edge', self.cfg.get('fog_edge', 1000.0))
            # rc 是 (nr,)，取列向量即可。真正的坑在下面 qsat 那一行：
            # th 已经是二维 (nr,nz)，再加 [None,:] 会让 qv 被广播成三维。
            rr = np.asarray(self.g['rc']).ravel()[:, None]
            m = np.clip((fog_r - rr) / edge, 0.0, 1.0)
            m = m * m * (3.0 - 2.0 * m)
            self.qc *= m
            # 雾区外把水汽压到 95% 饱和，否则饱和调整会立刻把 qc 长回来
            # 这里 th 已是二维 (nr,nz)，不能再加 [None,:]，否则 qv 变三维
            qs_c = qsat(self.th * self.ex_c, self.p_c)
            self.qv = self.qv * m + (1.0 - m) * np.minimum(self.qv, 0.95 * qs_c)
        self.qi = np.zeros((self.nr, self.nz))
        self.qr = np.zeros((self.nr, self.nz))
        self.qs = np.zeros((self.nr, self.nz))
        # 霰。深对流中霰是主要降水物，落速 2–5 m/s，能把冰区的水快速清下来。
        # 没有它，被上升气流带上 0°C 层以上的雨滴无处可去——见微物理里的说明。
        self.qg = np.zeros((self.nr, self.nz))
        # 碘化银示踪（被动标量，不落）。烟炉在盘面中心释放，随流场上升，
        # 到达过冷区后按浓度加速 qc->qi 的冻结（人工冰核）。
        self.qagi = np.zeros((self.nr, self.nz))

        if cfg.get('bubble_amp', 0.0) > 0:
            R, zC, amp = cfg['bubble_R'], cfg['bubble_z'], cfg['bubble_amp']
            r2 = rc[:, None] ** 2 + (zc[None, :] - zC) ** 2
            self.th += np.where(r2 < R * R,
                                amp * np.cos(0.5 * np.pi * np.sqrt(r2) / R) ** 2, 0.0)

        self._precompute()
        self._build_poisson()

        self.rain_acc = 0.0
        self.rain_inst = 0.0
        self.t = 0.0
        self.umax_hist = []
        # 逐时刻收支累加器。单位：kg（水）、J（能量）。
        # 约定：所有分量都是"进入域内"为正。闭合时应有
        #   ΔM = q_surf + q_side + q_top + q_bot
        #   ΔE = e_surf + e_rotor + e_extra + e_latent + e_side + e_top + e_bot
        # 因为本格式的扩散算子在所有边界上通量为零（梯度置零），所以不需要
        # 额外的扩散项；相变是域内过程，只搬能量不改总量，单独记 e_latent
        # 是为了让"位温方程是否闭合"这件事可查。
        self.bud = dict(q_surf=0.0, q_side=0.0, q_top=0.0, q_bot=0.0,
                        e_surf=0.0, e_rotor=0.0, e_extra=0.0, e_latent=0.0,
                        e_side=0.0, e_top=0.0, e_bot=0.0,
                        q_clip=0.0, e_clip=0.0,
                        q0=0.0, e0=0.0, nstep=0)
        self.bud['q0'] = self.water_storage()
        self.bud['e0'] = self.energy_storage()
        self.rad_state = {}          # radiation() 的诊断快照

        # 卷吸用的环境剖面（v18）。标量卷吸 −ε|w|(q − q_env) 需要一个"环境值"，
        # 取 t=0 的水平均匀剖面。局限：真实环境中途会被补偿下沉、砧状外流改变，
        # 用固定初始剖面会低估高空的干燥效应。这一点在方案第七节风险 R4 里记着。
        # 只存 1D（nz,），使用时广播。
        self.env_prof = dict(
            qv=self.qv0.copy(), qc=np.zeros(self.nz), qi=np.zeros(self.nz),
            qs=np.zeros(self.nz), qr=np.zeros(self.nz), qg=np.zeros(self.nz),
            th=self.th0.copy())


    # -------------------------------------------------------------- 预计算
    def _precompute(self):
        G = self.g
        nr, nz = self.nr, self.nz
        rc, zc, rf, zf = G['rc'], G['zc'], G['rf'], G['zf']
        dr, dz = G['dr'], G['dz']

        self.rho_uf = np.repeat(self.rho_c[None, :], nr + 1, axis=0)
        self.rho_wf = np.tile(self.rho_w[:, None], (1, nr)).T          # (nr, nz+1)

        # 四种间距必须分清（混用是这一族错误的根源）：
        #   dr     = rf[i+1]-rf[i]      面到面，长度 nr    → u 的径向间距
        #   rc_gap = rc[i+1]-rc[i]      中心到中心，长度 nr-1 → w 的径向二阶导间距
        #   zc_gap = zc[j+1]-zc[j]      长度 nz-1          → u / 标量的垂直二阶导间距
        # 这三组间距只在**二阶导**里需要区分（见 step() 的 d2r/d2z）。
        # 一阶差商不需要另建数组：self.drf/self.dzf 已存中心距且索引对齐。
        self.rc_gap = rc[1:] - rc[:-1]
        self.zc_gap = zc[1:] - zc[:-1]

        drf = np.empty(nr + 1)
        drf[1:-1] = rc[1:] - rc[:-1]
        drf[0] = dr[0]
        drf[-1] = dr[-1]
        self.drf = drf
        dzf = np.empty(nz + 1)
        dzf[1:-1] = zc[1:] - zc[:-1]
        dzf[0] = dz[0]
        dzf[-1] = dz[-1]
        self.dzf = dzf

        # 说明：self.drf / self.dzf 存的正是"相邻单元中心距"，且索引天然对齐：
        #   drf[i] = rc[i]-rc[i-1] → 面 rf[i] 两侧的中心距（_diffuse 的 g_r 用）
        #   drf[i+1] = rc[i+1]-rc[i] → 相邻压力中心距（泊松 cE 用）
        # 两处原本就正确。曾在 metric_fix 里误改成面距 dr，已撤销。

        self.area_cell = 2 * np.pi * rc * dr
        self.area_tot = np.pi * G['rmax'] ** 2

        # 单步能量账用的对偶体积（与 step_budget.py 同一套定义）。
        # u 在 (rf, zc)、w 在 (rc, zf)，对偶宽度取相邻间距的一半。
        wr = np.zeros(nr + 1)
        wr[1:-1] = 0.5 * (rf[2:] - rf[:-2])
        wr[0] = 0.5 * (rf[1] - rf[0])
        wr[-1] = 0.5 * (rf[-1] - rf[-2])
        self._V_u = 2 * np.pi * rf[:, None] * wr[:, None] * dz[None, :]
        hz = np.zeros(nz + 1)
        hz[1:-1] = 0.5 * (dz[:-1] + dz[1:])
        hz[0], hz[-1] = dz[0], dz[-1]
        self._V_w = self.area_cell[:, None] * hz[None, :]

        sz = self.cfg.get('sponge_z', 0.28)
        sr = self.cfg.get('sponge_r', 0.14)
        z0 = G['zmax'] * (1 - sz)
        r0 = G['rmax'] * (1 - sr)
        cz = (np.clip((zc - z0) / max(G['zmax'] - z0, 1e-9), 0, 1)) ** 2
        cr = (np.clip((rc - r0) / max(G['rmax'] - r0, 1e-9), 0, 1)) ** 2
        self.sponge = np.maximum(cz[None, :], cr[:, None])
        self.tau_sp = self.cfg.get('tau_sponge', 150.0)
        su = np.empty((nr + 1, nz))
        su[1:-1, :] = 0.5 * (self.sponge[:-1, :] + self.sponge[1:, :])
        su[0, :] = self.sponge[0, :]; su[-1, :] = self.sponge[-1, :]
        self.sponge_u = su
        sw = np.empty((nr, nz + 1))
        sw[:, 1:-1] = 0.5 * (self.sponge[:, :-1] + self.sponge[:, 1:])
        sw[:, 0] = self.sponge[:, 0]; sw[:, -1] = self.sponge[:, -1]
        self.sponge_w = sw

    # -------------------------------------------------------------- 泊松
    def _build_poisson(self):
        G = self.g
        nr, nz = self.nr, self.nz
        rc, dr, dz, rf = G['rc'], G['dr'], G['dz'], G['rf']
        R, Rw = self.rho_uf, self.rho_wf

        # 泊松：压力位于单元中心，相邻压力差 (P[i+1]-P[i]) 的距离是**中心距**
        # rc[i+1]-rc[i] = self.drf[i+1]（因为 drf[1:-1]=rc[1:]-rc[:-1]）。
        # 原实现用的正是这个，本来就对。曾误改成面距 dr[i]，那是错的
        # （第五轮审计意见按变量位置核对后指出）。
        cE = R[1:, :] * rf[1:, None] / (rc[:, None] * dr[:, None] * self.drf[1:, None])
        cW = R[:-1, :] * rf[:-1, None] / (rc[:, None] * dr[:, None] * self.drf[:-1, None])
        cN = Rw[:, 1:] / (dz[None, :] * self.dzf[1:][None, :])
        cS = Rw[:, :-1] / (dz[None, :] * self.dzf[:-1][None, :])
        cW[0, :] = 0.0
        cS[:, 0] = 0.0

        N = nr * nz
        cEf = cE.ravel(); cWf = cW.ravel(); cNf = cN.ravel(); cSf = cS.ravel()
        diag = (-(cE + cW + cN + cS)).ravel()
        r0 = np.arange(N)
        kk = r0 % nz
        jj = r0 // nz

        rows = [r0]; cols = [r0]; data = [diag]
        m = kk != nz - 1
        rows.append(r0[m]); cols.append(r0[m] + 1); data.append(cNf[m])
        m = kk != 0
        rows.append(r0[m]); cols.append(r0[m] - 1); data.append(cSf[m])
        m = jj != nr - 1
        rows.append(r0[m]); cols.append(r0[m] + nz); data.append(cEf[m])
        m = jj != 0
        rows.append(r0[m]); cols.append(r0[m] - nz); data.append(cWf[m])
        rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)

        is_dir = (jj == nr - 1) | (kk == nz - 1)
        keep = ~is_dir[rows]
        rows = np.concatenate([rows[keep], r0[is_dir]])
        cols = np.concatenate([cols[keep], r0[is_dir]])
        data = np.concatenate([data[keep], np.ones(is_dir.sum())])
        A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsc()
        self.lu = spla.splu(A, permc_spec='COLAMD')

    # -------------------------------------------------------------- 工具
    def _pad(self, phi):
        p = np.empty((phi.shape[0] + 2, phi.shape[1] + 2))
        p[1:-1, 1:-1] = phi
        p[0, 1:-1] = phi[0, :]
        p[-1, 1:-1] = phi[-1, :]
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        return p

    def _pos_pad(self, x, axis_len):
        return np.concatenate(([x[0] - (x[1] - x[0])], x,
                               [x[-1] + (x[-1] - x[-2])]))

    def _slopes(self, pp, xp, axis):
        s = np.zeros_like(pp)
        if axis == 0:
            dm = (pp[1:-1, :] - pp[:-2, :]) / (xp[1:-1] - xp[:-2])[:, None]
            dp = (pp[2:, :] - pp[1:-1, :]) / (xp[2:] - xp[1:-1])[:, None]
            sl = 0.5 * (np.sign(dm) + np.sign(dp)) * np.minimum(np.abs(dm), np.abs(dp))
            s[1:-1, :] = sl
            s[0, :] = s[1, :]
            s[-1, :] = s[-2, :]
        else:
            dm = (pp[:, 1:-1] - pp[:, :-2]) / (xp[1:-1] - xp[:-2])[None, :]
            dp = (pp[:, 2:] - pp[:, 1:-1]) / (xp[2:] - xp[1:-1])[None, :]
            sl = 0.5 * (np.sign(dm) + np.sign(dp)) * np.minimum(np.abs(dm), np.abs(dp))
            s[:, 1:-1] = sl
            s[:, 0] = s[:, 1]
            s[:, -1] = s[:, -2]
        return s

    def _flux_r(self, phi, uface):
        """r 方向面通量 u*phi（不含 rf 权重）。uface (nr+1,nz)，phi (nr,nz)"""
        G = self.g
        pp = self._pad(phi)
        xp = self._pos_pad(G['rc'], self.nr)
        s = self._slopes(pp, xp, 0)
        fL = pp[0:-1, 1:-1] + s[0:-1, 1:-1] * (G['rf'] - xp[0:-1])[:, None]
        fR = pp[1:, 1:-1] + s[1:, 1:-1] * (G['rf'] - xp[1:])[:, None]
        return np.where(uface > 0, fL, fR) * uface

    def _flux_z(self, phi, wface):
        """z 方向面通量 w*phi。wface (nr,nz+1)"""
        G = self.g
        pp = self._pad(phi)
        xp = self._pos_pad(G['zc'], self.nz)
        s = self._slopes(pp, xp, 1)
        fL = pp[1:-1, 0:-1] + s[1:-1, 0:-1] * (G['zf'] - xp[0:-1])[None, :]
        fR = pp[1:-1, 1:] + s[1:-1, 1:] * (G['zf'] - xp[1:])[None, :]
        return np.where(wface > 0, fL, fR) * wface

    def _diffuse(self, phi, nu):
        """柱坐标守恒形式的扩散算子，质量加权

            (1/rho)·[(1/r)·∂(r·rho·nu·∂phi/∂r)/∂r + ∂(rho·nu·∂phi/∂z)/∂z]

        两处都容易写错，两个都错过：
        1) 用笛卡尔 Laplacian 在柱坐标下会在轴线附近产生虚假源（实测 30 s
           可增 13 % 的热量/水量），必须用通量形式（旧版已修）。
        2) 只写 ∇·(nu∇phi) 而不写 ρ：变密度大气里这不守恒。旧版就是这样，
           60 s 内可凭空造出水/热，量级 ~1e-8 相对值，但 24 min 累积可达 1 K。
           本版改为与 _adv_center 相同的 rho_uf / rho_wf / rho_c 三重加权。
        """
        G = self.g
        nr, nz = self.nr, self.nz
        rc, dr, dz, rf = G['rc'], G['dr'], G['dz'], G['rf']
        nu_r = np.empty((nr + 1, nz))
        nu_r[1:-1, :] = 0.5 * (nu[:-1, :] + nu[1:, :])
        nu_r[0, :] = nu[0, :]; nu_r[-1, :] = nu[-1, :]
        g_r = np.zeros((nr + 1, nz))
        # 面 rf[i] 两侧是 rc[i-1] 与 rc[i]，距离 = rc[i]-rc[i-1] = self.drf[i]。
        # 注意 self.drf[i] 存的就是这个中心距（drf[1:-1] = rc[1:]-rc[:-1]），
        # 索引恰好对齐，原实现本来就对。曾误判为"偏一格"并改成面距 dr，
        # 那是引入新错误——第五轮审计意见指出并纠正了这一点。
        g_r[1:-1, :] = (phi[1:, :] - phi[:-1, :]) / self.drf[1:-1, None]
        F = nu_r * g_r * self.rho_uf
        out = (F[1:, :] * rf[1:, None] - F[:-1, :] * rf[:-1, None]) / (rc[:, None] * dr[:, None])
        nu_z = np.empty((nr, nz + 1))
        nu_z[:, 1:-1] = 0.5 * (nu[:, :-1] + nu[:, 1:])
        nu_z[:, 0] = nu[:, 0]; nu_z[:, -1] = nu[:, -1]
        g_z = np.zeros((nr, nz + 1))
        # 同理，zf[j] 两侧是 zc[j-1] 与 zc[j]，距离 = self.dzf[j]，原实现正确。
        g_z[:, 1:-1] = (phi[:, 1:] - phi[:, :-1]) / self.dzf[1:-1][None, :]
        Fz = nu_z * g_z * self.rho_wf
        out += (Fz[:, 1:] - Fz[:, :-1]) / dz[None, :]
        return out / self.rho_c[None, :]

    def _adv_center(self, phi, u, w, weff=None):
        """单元中心标量的变密度平流项（-∇·(rho V phi)/rho）"""
        G = self.g
        rc, dr, dz = G['rc'], G['dr'], G['dz']
        pu = self._flux_r(phi, u) * self.rho_uf
        d = -(pu[1:, :] * G['rf'][1:, None] - pu[:-1, :] * G['rf'][:-1, None]) \
            / (rc[:, None] * dr[:, None])
        wf = w if weff is None else weff
        pz = self._flux_z(phi, wf) * self.rho_wf
        d += -(pz[:, 1:] - pz[:, :-1]) / dz[None, :]
        return d / self.rho_c[None, :]

    def _quick_grad(self, phi, xc, axis, v):
        """一阶导数，phi 与 v 同形，位于 xc 中心（axis 方向）。

        adv_scheme='quick'（默认）用 QUICK 迎风偏置三点式；
        adv_scheme='upwind1' 用一阶迎风。加这个开关是为了第十一轮审计要求的
        格式敏感性对照——一阶迎风耗散强，若它不爆只说明"更耗散抑制了事件"，
        不能直接证明 QUICK 是唯一根因。
        """
        if self.cfg.get('adv_scheme', 'quick') == 'upwind1':
            return self._upwind1_grad(phi, xc, axis, v)
        if axis == 0:
            p = np.concatenate([np.repeat(phi[:1], 2, axis=0), phi,
                                np.repeat(phi[-1:], 2, axis=0)], axis=0)
        else:
            p = np.concatenate([np.repeat(phi[:, :1], 2, axis=1), phi,
                                np.repeat(phi[:, -1:], 2, axis=1)], axis=1)
        if axis == 0:
            a = p[2:-2]; am1 = p[1:-3]; am2 = p[0:-4]; ap1 = p[3:-1]; ap2 = p[4:]
            dx = np.gradient(xc)[:, None]
        else:
            a = p[:, 2:-2]; am1 = p[:, 1:-3]; am2 = p[:, 0:-4]
            ap1 = p[:, 3:-1]; ap2 = p[:, 4:]
            dx = np.gradient(xc)[None, :]
        gq = np.where(v > 0, (3 * a - 4 * am1 + am2) / (2 * dx),
                      (-3 * a + 4 * ap1 - ap2) / (2 * dx))
        return gq

    def _upwind1_grad(self, phi, xc, axis, v):
        """一阶迎风差分（对照用）。"""
        if axis == 0:
            p = np.concatenate([phi[:1], phi, phi[-1:]], axis=0)
            a = p[1:-1]; am1 = p[:-2]; ap1 = p[2:]
            dx = np.gradient(xc)[:, None]
        else:
            p = np.concatenate([phi[:, :1], phi, phi[:, -1:]], axis=1)
            a = p[:, 1:-1]; am1 = p[:, :-2]; ap1 = p[:, 2:]
            dx = np.gradient(xc)[None, :]
        return np.where(v > 0, (a - am1) / dx, (ap1 - a) / dx)

    # -------------------------------------------------------------- 诊断
    def divergence(self, u, w):
        G = self.g
        Fu = self.rho_uf * G['rf'][:, None] * u
        Fw = self.rho_wf * w
        d = (Fu[1:, :] - Fu[:-1, :]) / (G['rc'][:, None] * G['dr'][:, None])
        d += (Fw[:, 1:] - Fw[:, :-1]) / G['dz'][None, :]
        return d

    def theta_v(self):
        return self.th * (1.0 + 0.608 * self.qv - self.qc - self.qi
                          - self.qr - self.qs - self.qg)

    def buoyancy(self):
        thv = self.theta_v()
        return g * (thv - self.thv0[None, :]) / self.thv0[None, :]

    # ---------------------------------------------------- v18 微物理辅助量
    def _particle_diameter(self, q, N_kg, rho_p):
        """由混合比与固定数浓度反算平均直径 (m)。

        bulk 方案没有粒径分布，用 D = (6 m_p / (π ρ_p))^(1/3)、m_p = q/N 反算。
        这意味着"质量增长 = 粒径增长"，是 v18 让落速能随质量走的唯一途径。
        假设的局限：真实云里 N 本身也在变（淞附分裂、聚并减少），N 固定会
        让粒径增长得比实际快。见方案 R1。
        """
        m_p = np.maximum(q, 1e-12) / max(N_kg, 1e-6)
        return np.maximum((6.0 * m_p / (np.pi * rho_p)) ** (1.0 / 3.0), 1.0e-4)

    def _graupel_diameter(self):
        """霰的平均直径，ρ_g = 500 kg/m³，默认数浓度 1e4 /kg。"""
        return self._particle_diameter(self.qg, self.cfg.get('N_graupel', 1e4), 500.0)

    def _snow_diameter(self):
        """雪的平均直径，ρ_s = 100 kg/m³。"""
        return self._particle_diameter(self.qs, self.cfg.get('N_snow', 1e4), 100.0)

    @staticmethod
    def _fallspeed_locatelli(D):
        """Locatelli & Hobbs (1974) 量级关系：V = a·D^0.5，D 单位 mm，V 单位 m/s。
        D=1 mm 给 3.0 m/s，D=5 mm 给 6.7 m/s，与实测霰的落速相当。"""
        Dmm = D * 1.0e3
        return np.maximum(3.0 * np.sqrt(np.maximum(Dmm, 1e-3)), 0.5)

    def _entrain_eps(self):
        """夹卷率 ε = k_e / R_eff，R_eff 由含水量质量加权半径给出 (1/m)。"""
        hyd = self.qc + self.qi + self.qs + self.qg
        mcol = hyd * self.rho_c[None, :]
        tot = np.sum(mcol, axis=0)
        wgt = np.sum(mcol * self.g['rc'][:, None] ** 2, axis=0)
        ok = tot > 1e-12
        R2 = np.where(ok, wgt / np.where(ok, tot, 1.0), 1.0)
        R = np.sqrt(np.maximum(R2, 1.0))
        # R_eff 下限（2026-11-09 加）。原来只有 1 m 的数值下限，云刚形成时凝结物
        # 只占轴心一两个格点，质量加权半径算出来就是 1–5 m，于是 ε = ke/R 被放大
        # 三个数量级：ke=0.2、R=1 m 给出 ε=0.2 /m，w=5 m/s 时稀释时间 1 秒。
        # 这解释了 G1/G5/G6/R2 为什么全部停在 1.2 km——不是夹卷"太强"，是分母病态。
        # 物理上羽流不可能比强迫源更窄，取盘面尺度作下限：默认 0（保持既有算例
        # 逐位不变），新算例显式给 entrain_rmin=75（半个盘面半径）。
        R = np.maximum(R, self.cfg.get('entrain_rmin', 0.0))
        return np.where(ok, self.cfg.get('entrain_ke', 0.2) / R, 0.0)


    def eddy_visc(self):
        G = self.g
        rc, zc, dr, dz = G['rc'], G['zc'], G['dr'], G['dz']
        nr, nz = self.nr, self.nz
        u, w = self.u, self.w
        u_c = 0.5 * (u[1:, :] + u[:-1, :])
        w_c = 0.5 * (w[:, 1:] + w[:, :-1])
        du_dr = (u[1:, :] - u[:-1, :]) / dr[:, None]
        dw_dz = (w[:, 1:] - w[:, :-1]) / dz[None, :]
        du_dz = np.zeros((nr, nz)); dw_dr = np.zeros((nr, nz))
        du_dz[:, 1:-1] = (u_c[:, 2:] - u_c[:, :-2]) / (zc[2:] - zc[:-2])[None, :]
        du_dz[:, 0] = du_dz[:, 1]; du_dz[:, -1] = du_dz[:, -2]
        dw_dr[1:-1, :] = (w_c[2:, :] - w_c[:-2, :]) / (rc[2:] - rc[:-2])[:, None]
        dw_dr[0, :] = 0.0; dw_dr[-1, :] = dw_dr[-2, :]
        v = self.v
        dv_dr = np.zeros((nr, nz)); dv_dz = np.zeros((nr, nz))
        dv_dr[1:-1, :] = (v[2:, :] - v[:-2, :]) / (rc[2:] - rc[:-2])[:, None]
        dv_dr[0, :] = 0.0; dv_dr[-1, :] = dv_dr[-2, :]
        dv_dz[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (zc[2:] - zc[:-2])[None, :]
        dv_dz[:, 0] = dv_dz[:, 1]; dv_dz[:, -1] = dv_dz[:, -2]
        # 切向剪切的柱坐标形式：(dv/dr - v/r) 与 dv/dz
        S2 = 2 * (du_dr ** 2 + dw_dz ** 2 + (u_c / np.maximum(rc[:, None], 1.0)) ** 2) \
             + (du_dz + dw_dr) ** 2 \
             + (dv_dr - v / np.maximum(rc[:, None], 1.0)) ** 2 + dv_dz ** 2
        Sm = np.sqrt(np.maximum(S2, 1e-12))
        Delta = np.sqrt(dr[:, None] * dz[None, :])
        nu = (0.23 * Delta) ** 2 * Sm
        thv = self.theta_v()
        dth = np.zeros((nr, nz))
        dth[:, 1:-1] = (thv[:, 2:] - thv[:, :-2]) / (zc[2:] - zc[:-2])[None, :]
        N2 = np.maximum(g / self.thv0[None, :] * dth, 0.0)
        Ri = N2 / np.maximum(Sm ** 2, 1e-8)
        nu = nu / (1.0 + 8.0 * Ri)

        # 旋翼尾流附加涡粘：叶片尺度湍流（叶尖涡、叶片尾迹）在 4 m 网格上
        # 完全落在亚网格，Smagorinsky 只用可解应变估不出来。显式给一段
        # 随高度衰减的涡粘，代表这部分混合。
        nu_w = self.cfg.get('nu_wake', 0.0)
        if nu_w > 0.0:
            zw = self.cfg.get('rotor_z', 140.0)
            Rw = self.cfg.get('rotor_R', 150.0)
            dzp = np.maximum(zc[None, :] - zw, 0.0)
            rr = rc[:, None]
            fz = np.exp(-dzp / (1.5 * Rw))
            fr = np.exp(-((np.maximum(rr - Rw, 0.0)) / (0.8 * Rw)) ** 2)
            nu = nu + nu_w * fz * fr
        return np.clip(nu, self.cfg.get('nu_min', 2.0), self.cfg.get('nu_max', 120.0))

    # -------------------------------------------------------------- 微物理
    def microphysics(self, dt):
        """Kessler 暖雨 + 简化冰相（云冰/雪）+ 雨滴蒸发

        冰相的必要性：纯暖雨方案在深对流中自动转化过慢（k1~1e-3 s^-1），
        凝结物无法及时清除，柱内潜热持续累积会把位温抬升 30 K 量级，
        产生 80 m/s 的非物理上升速度。真实深对流靠冰晶—雪—霰的快速增长
        降水，这里用"云冰快速冻结 + 云冰转雪（快）+ 雪下落融化"代表该过程。
        """
        cfg = self.cfg
        ex = self.ex_c[None, :]
        p = self.p_c[None, :]
        th_in = self.th.copy()          # 收支用：记录相变前后的位温差

        # --- 液/冰两条饱和调整 ---
        T = self.th * ex
        warm = T > 273.15
        qs_w = qsat(T, p)
        qs_i = qsat_ice(T, p)
        qs_ = np.where(warm, qs_w, qs_i)
        # 液面凝结用 Lv，冰面凝华用 Ls
        L_c = np.where(warm, Lv, Ls)
        beta = 1.0 + L_c ** 2 * qs_ / (cp * Rv * T ** 2)

        sup = self.qv - qs_
        cond = np.where(sup > 0, sup / beta, 0.0)
        self.qv -= cond
        self.qc += np.where(warm, cond, 0.0)
        self.qi += np.where(warm, 0.0, cond)
        self.th += (L_c / (cp * ex)) * cond

        # --- 云水/云冰的蒸发与升华 ---
        T = self.th * ex
        qs_w = qsat(T, p)
        qs_i = qsat_ice(T, p)
        qs_ = np.where(T > 273.15, qs_w, qs_i)
        beta = 1.0 + Lv ** 2 * qs_ / (cp * Rv * T ** 2)
        need = np.where(self.qv < qs_, (qs_ - self.qv) / beta, 0.0)
        ev_c = np.minimum(self.qc, need)
        self.qv += ev_c; self.qc -= ev_c
        self.th -= (Lv / (cp * ex)) * ev_c
        T = self.th * ex
        qs_i = qsat_ice(T, p)
        need_i = np.where((self.qv < qs_i) & (self.qi > 0),
                          (qs_i - self.qv) / (1.0 + Ls ** 2 * qs_i / (cp * Rv * T ** 2)), 0.0)
        sub_i = np.minimum(self.qi, need_i)
        self.qv += sub_i; self.qi -= sub_i
        self.th -= (Ls / (cp * ex)) * sub_i

        # --- 云水自动转化 + 碰并 -> 雨 ---
        # M1（accr_exp）：Kessler (1969) 原式是 k2·qc·qr^0.875，v17 及之前误写成
        # k2·qc·qr，漏了 0.875 指数（见方案 A4）。强降水段 qr 大，会低估碰并率；
        # 弱降水段略微高估。开关默认关，保证与 v17 逐位兼容。
        qc0 = cfg.get('qc0', 0.7e-3)
        auto = cfg.get('k_auto', 1.0e-3) * np.maximum(self.qc - qc0, 0.0)
        k2 = cfg.get('k_accr', 2.2)
        if cfg.get('accr_exp', False):
            accr = k2 * self.qc * np.maximum(self.qr, 0.0) ** 0.875
        else:
            accr = k2 * self.qc * self.qr
        conv = np.minimum((auto + accr) * dt, self.qc)
        self.qc -= conv; self.qr += conv

        # --- 云水冻结 -> 云冰（T<0）---
        # M2（ice_nuc）：v17 用固定时间常数 20 s，等价于冰核无限多，过冷水一到
        # 0 °C 以下就被冻干。实测 4–7 km 的 qc 只剩 0.01–0.12 g/kg，导致即使补上
        # 淞附也没有原料（方案 A3）。
        # 开启后改用温度依赖的冻结速率：τ(T) 从 0 °C 的 20 s 增加到 −20 °C 的
        # 约 400 s，−35 °C 以下触发均质冻结（τ=1 s），对应 Bigg (1953) 与
        # Rutledge & Hobbs (1983) 的量级。
        T = self.th * ex
        if cfg.get('ice_nuc', False):
            # 冻结时间常数随温度呈钟形：−5 °C 约 86 s、−15 °C 约 400 s（过冷水
            # 最稳定）、−25 °C 回落到 86 s、−35 °C 以下均质冻结 1 s。
            # 物理依据：冰核活性在 −10 ~ −20 °C 最弱，过冷水能维持；−35 °C 以下
            # 均质核化，冻结核不再需要（Rutledge & Hobbs 1983；Pruppacher & Klett）。
            Tp = cfg.get('T_frz_peak', -15.0)
            wid = cfg.get('w_frz', 8.0)
            tau_pk = cfg.get('tau_frz_peak', 400.0)
            tau_mn = cfg.get('tau_frz_min', 2.0)
            Tc_ = np.minimum(T, 273.15) - 273.15
            tau_ = tau_mn + (tau_pk - tau_mn) * np.exp(-((Tc_ - Tp) / wid) ** 2)
            # 均质冻结不再用 T<−35 °C 的硬切换（那会让 τ 从 2.76 s 跳到 1 s，
            # 温度在阈值附近来回时会造出人为的冻结率突变）。改成在 −33 ~ −38 °C
            # 之间线性增强的附加速率，与原速率相加，处处连续。
            rate_ = 1.0 / tau_ + np.clip((-33.0 - Tc_) / 5.0, 0.0, 1.0)
            frz = np.where(T < 273.15,
                           np.minimum(self.qc, self.qc * rate_ * dt), 0.0)
        else:
            frz = np.where(T < 273.15,
                           np.minimum(self.qc, self.qc * dt / cfg.get('tau_frz', 20.0)), 0.0)
        self.qc -= frz; self.qi += frz
        self.th += (Lf / (cp * ex)) * frz

        # --- M3/M4 淞附：雪/霰扫过过冷水长大（riming）---
        # v17 及之前完全没有这一项，降水粒子的质量只能靠"冻结"增加，粒径与落速
        # 不变，于是 12 g/kg 的 qg 被 48 m/s 的上升气流钉在 6–10 km（方案 A2）。
        # 单分散连续碰并：dm/dt = N·(π/4 D²)·E·|V−w|·qc，N 由混合比与固定数浓度反算。
        # 淞附同时把云水冻成冰（释放 Lf），按 Lin et al. (1983) 的 PSACW/PGACW
        # 归入 qg。收集效率 E=1（Dudhia 1989）。
        if cfg.get('riming', False):
            rho_a = self.rho_c[None, :]
            qcp = np.maximum(self.qc, 0.0)
            E_col = cfg.get('E_collect', 1.0)
            # 碰撞相对速度（2026-11-10 修正）。
            # 云滴随空气运动，地面系速度 = w；霰是 w − Vg。两者之差 = Vg，
            # 与 w 无关。原写成 |Vg − w|，在强上升气流里把空气的绝对速度混进了
            # 粒子间相对速度——w=30、Vg=3 时算出 27，淞附率被放大 9 倍。
            # 手算测试取 w=0，|Vg − 0| = Vg，所以测试永远通过；这是审计指出的
            # 测试盲区。现改为直接用 Vg（云滴自身落速 ~0.01 m/s，可忽略）。
            # 霰收集
            Ng = cfg.get('N_graupel', 1e4)
            Dg = self._graupel_diameter()
            Vg = self._fallspeed_locatelli(Dg)
            relg = np.maximum(Vg, 1e-3)
            dmg = Ng * rho_a * (0.25 * np.pi * Dg ** 2) * E_col * relg * qcp
            rim_g = np.minimum(qcp, np.maximum(dmg, 0.0) * dt)
            self.qc -= rim_g; self.qg += rim_g
            self.th += (Lf / (cp * ex)) * rim_g
            # 雪收集（雪淞附后转霰）
            qcp = np.maximum(self.qc, 0.0)
            Ns = cfg.get('N_snow', 1e4)
            Ds = self._snow_diameter()
            Vs = self._fallspeed_locatelli(Ds)
            rels = np.maximum(Vs, 1e-3)
            dms = Ns * rho_a * (0.25 * np.pi * Ds ** 2) * E_col * rels * qcp
            rim_s = np.minimum(qcp, np.maximum(dms, 0.0) * dt)
            self.qc -= rim_s; self.qg += rim_s
            self.th += (Lf / (cp * ex)) * rim_s

        # --- 碘化银烟炉：释放 + 过冷区催化 ---
        # 物理链：烟炉在盘面中心释放 AgI 气溶胶 -> 随上升气流进入云中 ->
        # 在 -5 ~ -15 °C 的过冷水区充当人工冰核 -> 冰晶经贝吉龙过程夺取
        # 过冷水长大 -> 形成雪/霰 -> 下落融化。
        # 注意 AgI 是冰核，不是凝结核。暖云催化用的是吸湿性粒子，会减小
        # 滴径反而抑制降水，两者不能混。
        # `agi_t0` 控制点火时刻：真实作业讲时机，早期点火会让烟羽随上升气流
        # 一路被带到 13 km（实测），真正在 -5~-15 °C 层停留的时间很短。
        if cfg.get('agi_on', False) and self.t >= cfg.get('agi_t0', 0.0):
            G = self.g
            rc_, zc_ = G['rc'], G['zc']
            m_rel = ((rc_ <= cfg.get('agi_R', 50.))[:, None]
                     & (np.abs(zc_ - cfg.get('agi_z', 140.))
                        <= cfg.get('agi_dz', 25.))[None, :])
            self.qagi[m_rel] += cfg.get('agi_rate', 2e-7) * dt

            T = self.th * ex
            win = (T < 268.15) & (T > 258.15) & (self.qc > 0.0)
            conc = np.minimum(self.qagi / max(cfg.get('agi_ref', 1.0e-6), 1e-30), 1.0)
            k_agi = cfg.get('k_agi', 0.05)
            frz_a = np.where(win, np.minimum(self.qc, self.qc * k_agi * conc * dt), 0.0)
            self.qc -= frz_a
            self.qi += frz_a
            self.th += (Lf / (cp * ex)) * frz_a

        # --- 云冰 -> 雪（快速自动转化，代表冰晶聚并增长）---
        qi0 = cfg.get('qi0', 1.0e-4)
        aci = np.minimum(cfg.get('k_ice', 1.0e-2) * np.maximum(self.qi - qi0, 0.0) * dt, self.qi)
        self.qi -= aci; self.qs += aci

        # --- 雪融化 -> 雨（T>0）---
        T = self.th * ex
        mlt = np.where(T > 273.15,
                       np.minimum(self.qs, self.qs * dt / cfg.get('tau_mlt', 120.0)), 0.0)
        self.qs -= mlt; self.qr += mlt
        self.th -= (Lf / (cp * ex)) * mlt

        # --- 雨冻结 -> 霰（T<0）---
        # 这一段是 v16 之后补的。此前只有云水冻结（qc->qi），雨滴不冻结，于是
        # 被 w~50 m/s 的上升气流带上 0°C 层以上仍是雨（落速 3–12 m/s 托不住），
        # 在 8940 m 堆到 9.12 g/kg（第十六轮 79 min 末态）。真实大气里雨滴在
        # 该层会冻成霰，落速 2–5 m/s 能落回来。
        T = self.th * ex
        frzr = np.where(T < 273.15,
                        np.minimum(self.qr, self.qr * dt / cfg.get('tau_frzr', 10.0)), 0.0)
        self.qr -= frzr; self.qg += frzr
        self.th += (Lf / (cp * ex)) * frzr

        # --- 霰融化 -> 雨（T>0）---
        T = self.th * ex
        mltg = np.where(T > 273.15,
                        np.minimum(self.qg, self.qg * dt / cfg.get('tau_mltg', 60.0)), 0.0)
        self.qg -= mltg; self.qr += mltg
        self.th -= (Lf / (cp * ex)) * mltg

        # --- 雨滴蒸发（云下不饱和层）---
        T = self.th * ex
        qs_w = qsat(T, p)
        RH = np.clip(self.qv / qs_w, 0.0, 1.0)
        evr = np.minimum(self.qr, self.qr * (1.0 - RH) * dt / cfg.get('tau_evr', 400.0))
        self.qr -= evr; self.qv += evr
        self.th -= (Lv / (cp * ex)) * evr

        # --- 雪的升华 ---
        need_s = np.where((self.qv < qs_i) & (self.qs > 0),
                          (qs_i - self.qv) / (1.0 + Ls ** 2 * qs_i / (cp * Rv * T ** 2)), 0.0)
        sub_s = np.minimum(self.qs, need_s * dt / cfg.get('tau_sub', 200.0) * 1.0)
        sub_s = np.minimum(sub_s, self.qs)
        self.qs -= sub_s; self.qv += sub_s
        self.th -= (Ls / (cp * ex)) * sub_s

        for nm in ('qc', 'qi', 'qr', 'qs', 'qg'):
            np.maximum(getattr(self, nm), 0.0, out=getattr(self, nm))

        # 相变潜热总量：所有凝结/蒸发/冻结/融化项之和，统一由位温变化反算，
        # 避免逐项记账时漏项。乘 rho*cp*ex 即能量（J），进入域内为正。
        self.bud['e_latent'] += float(np.sum((self.th - th_in) * self._cell_mass()))

    # -------------------------------------------------------------- 压力投影
    def project(self, u, w, dt, tol=None):
        """压力投影：把 u,w 投影到离散无散度空间。

        独立成方法，是为了让**测试与生产走同一份代码**。此前所有"散度消干净了"
        的说法都缺少直接证据（依据是收支残差和场看起来正常），而自己重写一份
        等价投影去测又会引入测试实现的错误——第六轮就在这上面栽过。

        返回投影前后的真实离散散度，供验收用。

        三个区域必须分开看：
          - 内部：D_h(ρu) 应达到泊松求解容差允许的水平
          - 边界：泊松中设为 Dirichlet 的单元（jj==nr-1 或 kk==nz-1），
                  不能套用内单元的机器零标准，要按边界闭合式验收
          - 全域：体积加权范数 + 净质量通量，防止局部误差被正负抵消掩盖
        """
        G = self.g
        nr, nz = self.nr, self.nz
        rc, dr, dz, rf = G['rc'], G['dr'], G['dz'], G['rf']

        div0 = self.divergence(u, w)

        rhs = div0 / dt
        rhs[-1, :] = 0.0
        rhs[:, -1] = 0.0
        P = self.lu.solve(rhs.ravel()).reshape(nr, nz)
        u[1:-1, :] -= dt * (P[1:, :] - P[:-1, :]) / self.drf[1:-1, None]
        w[:, 1:-1] -= dt * (P[:, 1:] - P[:, :-1]) / self.dzf[1:-1][None, :]

        # 反解前的"朴素零梯度"参考值。反解修正量 = 反解值 − 该值，
        # 用来回答"边界条件到底在替内部吸收多少质量"，以及
        # "若换成朴素零梯度会凭空造出多少质量"。
        # 只做快照，不写回 u/w：这两处赋值在反解里是死代码（反解会整列覆盖），
        # 但 w[:, -1] = w[:, -2] 会污染角落单元垂直散度用到的 Fw_b[-1]，
        # 曾使角落出流偏 1.9%（bc_verify.py S1 定位）。
        u_og, w_og = u[-2, :].copy(), w[:, -2].copy()
        # 反解真正的"前值"：审计意见 Q2 要求把侧边/顶面/角落的动能变化分开，
        # 所以不能只用零梯度参考值。
        u_pre, w_pre = u[-1, :].copy(), w[:, -1].copy()

        # 边界单元（泊松中设为 Dirichlet、未参与求解）强制满足连续方程，
        # 否则该处质量堆积会产生虚假上升运动与伪凝结
        rf_ = rf
        # 外侧：由 u[-1] 吸收 z 向净通量
        Fw_b = self.rho_wf[-1, :] * w[-1, :]
        dz_term = (Fw_b[1:] - Fw_b[:-1]) / dz
        # 角落单元 (nr-1, nz-1) 的两个边界面都在未知之列，而连续方程只有一条，
        # 所以严格说欠定。这里的闭合规则是：外侧分支用**上一步留下的**顶面通量
        # 算角落的垂直散度（step 里已不再预置零梯度，故这个值是上个时间步投影
        # 后的自洽值），随后顶面分支再整行覆盖它。
        # 若在读取前把顶面置成零梯度，该垂直散度恒为 0，角落出流偏 1.86%/0.81%
        # （bc_verify.py S1/S4 实测，是这两项测试最初 FAIL 的唯一原因）。
        Fu_prev = self.rho_uf[-1, :] * rf_[-2] * u[-2, :]
        Fu_last = Fu_prev - (rc[-1] * dr[-1]) * dz_term
        u[-1, :] = Fu_last / (self.rho_uf[-1, :] * rf_[-1])
        # 顶部：由 w[-1] 吸收 r 向净通量，使顶单元也满足连续方程
        Fu_t = self.rho_uf[:, -1] * rf_ * u[:, -1]
        r_term = (Fu_t[1:] - Fu_t[:-1]) / (rc * dr)
        Fw_prev = self.rho_wf[:, -2] * w[:, -2]
        w[:, -1] = (Fw_prev - dz[-1] * r_term) / self.rho_wf[:, -1]

        self._proj = dict(div_before=div0.copy(),
                          div_after=self.divergence(u, w).copy(),
                          P=P.copy(),
                          bc=dict(u_out=u[-1, :].copy(), w_top=w[:, -1].copy(),
                                  u_og=u_og, w_og=w_og, u_pre=u_pre, w_pre=w_pre))
        return self._proj

    def divergence_norms(self, div):
        """把散度场分成内部/边界/全域三组范数。

        内部 = 泊松中参与求解的单元（jj<nr-1 且 kk<nz-1）
        边界 = 泊松中设 Dirichlet 的单元
        全域 = 体积加权，含边界
        另报净质量通量积分（全域），用于检查内部残差与边界通量是否相容。
        """
        G = self.g
        nr, nz = self.nr, self.nz
        V = self._cell_mass() / np.maximum(self.rho_c[None, :], 1e-30)   # 体积
        is_dir = np.zeros((nr, nz), bool)
        is_dir[-1, :] = True
        is_dir[:, -1] = True
        inner = ~is_dir

        def nrm(mask):
            if not mask.any():
                return 0.0, 0.0
            d = div[mask]
            linf = float(np.abs(d).max())
            l2v = float(np.sqrt(np.sum(V[mask] * d ** 2) / np.sum(V[mask])))
            return linf, l2v

        out = {}
        out['inner_linf'], out['inner_l2V'] = nrm(inner)
        out['bound_linf'], out['bound_l2V'] = nrm(is_dir)
        out['all_linf'], out['all_l2V'] = nrm(np.ones_like(is_dir))
        # 全域净质量通量：∫ D_h dV = 净边界通量（应≈0）
        out['net_flux'] = float(np.sum(div * V))
        out['scale'] = float(np.abs(div).max()) if np.abs(div).max() > 0 else 1.0
        return out

    def _markF(self, Fu, Fw, tag):
        """记录"从上一个标记到现在"这一段力对流体做的功。

        审计意见 Q3-A2 要求把力做功拆开（对流/浮力/离心/扩散/转子/海绵）。
        这里不重推任何项：只对 Fu/Fw 做差，用的就是 step() 真正累加的数组。
        """
        if not self.cfg.get('budget', False):
            return
        if getattr(self, '_Fprev', None) is None:
            self._Fprev = (np.zeros_like(Fu), np.zeros_like(Fw))
            self._Wterm = {}
        Fu0, Fw0 = self._Fprev
        dFu, dFw = Fu - Fu0, Fw - Fw0
        self._Wterm[tag] = float(
            np.sum(self.rho_uf * self._V_u * self.u * dFu)
            + np.sum(self.rho_wf * self._V_w * self.w * dFw))
        # 定位爆发时需要逐项力场（不只是积分），代价高，单独开关
        if self.cfg.get('burst_probe', False):
            self._Fterms = getattr(self, '_Fterms', {})
            self._Fterms[tag] = (dFu, dFw)
        self._Fprev = (Fu.copy(), Fw.copy())

    def _filt_extent(self, G):
        """2Δ 滤波的作用区：近场均匀段。返回 (R, K) 两个 slice。"""
        rf, zf = G['rf'], G['zf']
        r_lim = float(self.cfg.get('filt_rmax', 650.0))
        z_lim = float(self.cfg.get('filt_zmax', 600.0))
        i1 = int(np.searchsorted(rf, r_lim))
        k1 = int(np.searchsorted(zf, z_lim))
        return slice(1, max(i1, 2)), slice(1, max(k1, 2))

    # -------------------------------------------------------------- 单步
    def step(self, dt):
        if self.cfg.get('freeze', False):
            self.u[:] = 0.0; self.w[:] = 0.0; self.v[:] = 0.0
            self.scalars(dt, self.eddy_visc())
            self.t += dt
            return
        G = self.g
        nr, nz = self.nr, self.nz
        rc, zc, dr, dz = G['rc'], G['zc'], G['dr'], G['dz']
        u, w = self.u, self.w

        # 力项记账的复位必须在**力装配之前**：复位放在步末会把刚存下的
        # 逐项力场清空（第十一轮踩过，导致 _Fterms 永远是空的）。
        if self.cfg.get('budget', False):
            self._Fprev = None
            if self.cfg.get('burst_probe', False):
                self._Fterms = {}

        u_c = 0.5 * (u[1:, :] + u[:-1, :])          # (nr,nz) 单元中心 u
        w_c = 0.5 * (w[:, 1:] + w[:, :-1])          # (nr,nz) 单元中心 w

        # ---------------- u 方程（位于 r 面）
        Fu = np.zeros_like(u)
        w_c = 0.5 * (w[:, 1:] + w[:, :-1])              # (nr,nz) 单元中心 w
        w_at_f = np.zeros((nr + 1, nz))
        w_at_f[1:-1, :] = 0.5 * (w_c[:-1, :] + w_c[1:, :])
        w_at_f[0, :] = w_c[0, :]
        w_at_f[-1, :] = w_c[-1, :]
        if not self.cfg.get('no_adv', False):
            Fu += -u * self._quick_grad(u, G['rf'], 0, u)
            Fu += -w_at_f * self._quick_grad(u, zc, 1, w_at_f)
        # 极轴几何项 u*w/r 在交错网格上不出现（v=0）

        # ---------------- w 方程（位于 z 面）
        Fw = np.zeros_like(w)
        u_at_w = np.zeros((nr, nz + 1))
        u_at_w[:, 1:-1] = 0.5 * (u_c[:, :-1] + u_c[:, 1:])
        u_at_w[:, 0] = u_c[:, 0]
        u_at_w[:, -1] = u_c[:, -1]
        if not self.cfg.get('no_adv', False):
            Fw += -u_at_w[:, :] * self._quick_grad(w[:, :], rc, 0, u_at_w)
            Fw += -w * self._quick_grad(w, G['zf'], 1, w)
        self._markF(Fu, Fw, 'adv')

        B = self.buoyancy()
        Fw[:, 1:-1] += 0.5 * (B[:, :-1] + B[:, 1:])
        self._markF(Fu, Fw, 'buoy')

        # 显式夹卷（entrain=True）。积云的动量夹卷率经验式 ε ≈ k_e/R_eff，
        # 在 w 方程加动量汇 -ε|w|w。R_eff 由当地 (qc+qi+qs+qg) 的质量加权半径
        # 自洽给出：羽流宽则 ε 小、w 掉得慢；窄则 ε 大。
        #
        # 为什么必须有：现在云宽约 300 m、w 到 48 m/s，而 w 方程里没有任何
        # 夹卷项，柱子升到 9 km 不减速。按 ε=0.2/R，R=150 m 时 ε=1.3e-3 /m，
        # 从 140 m 升到 9 km 应衰减 e^-11.8≈7e-6——真实大气里这根柱子根本
        # 不存在。这也是 v16 里「云顶越高雨越小」这个反常响应的来源之一。
        if self.cfg.get('entrain', False):
            hyd = self.qc + self.qi + self.qs + self.qg
            mcol = hyd * self.rho_c[None, :]
            tot = np.sum(mcol, axis=0)
            wgt = np.sum(mcol * rc[:, None] ** 2, axis=0)
            ok = tot > 1e-12
            R2 = np.where(ok, wgt / np.where(ok, tot, 1.0), 1.0)
            R_eff = np.sqrt(np.maximum(R2, 1.0))
            eps = np.where(ok, self.cfg.get('entrain_ke', 0.2) / R_eff, 0.0)
            sink = -eps[None, :] * np.abs(w_c) * w_c
            Fw[:, 1:-1] += 0.5 * (sink[:, :-1] + sink[:, 1:])
            self._markF(Fu, Fw, 'entrain')

        # ⚠️ 已作废，不要开启（2026-11-02 查实）。
        #
        # 本项与 theta_v 里的水凝物负载重复计入。theta_v 已经含 -qr-qs-qg，
        # 这里再加 -g(qr+qs+qg)，等于把负载扣两遍。v17 末态 5 km 实测：
        # 热力贡献 0.323、负载 -0.220、净浮力 0.096（theta_v 已扣过一次）；
        # 开启本项后净变 -0.124。从 v17 末态起跑 4 min 的对照因此把降水
        # 从 250.9 虚增到 605.3 m³（2.4 倍），那是把云压得更狠挤出来的水，
        # 不是输送改善。
        #
        # 正确做法二选一：
        #   (a) 保持 theta_v 含水凝物，不加这一项——当前生产配置即此，推荐；
        #   (b) 从 theta_v 移出沉降类 qr/qs/qg（只留 qc/qi），再加这一项。
        # 注意 (b) 不是严格代数等价，2026-11-02 复审指出：
        #   原负载项  B_old = -g·(θ/θv_env)·q_d
        #   独立项    B_new = -g·q_d
        #   差        B_new − B_old = -g·q_d·(1 − θ/θv_env) ≈ 1%
        # （因为 θ/θv_env ≈ 1/(1+0.608qv)）。所以走 (b) 时系数不能直接取 1，
        # 必须按所用动量变量与密度归一化重推。本模型不采用 (b)。
        if self.cfg.get('hydro_drag', False):
            qd = self.qr + self.qs + self.qg
            Fw[:, 1:-1] += -g * 0.5 * (qd[:, :-1] + qd[:, 1:])
            self._markF(Fu, Fw, 'drag')

        # 离心力：r 动量方程中的 -v^2/r 移到右端为 +v^2/r。
        # 有旋流时它把空气甩向外侧，是尾流扩张（喇叭口）的主要来源之一。
        if np.any(self.v):
            v = self.v
            v_f = np.empty((nr + 1, nz))
            v_f[1:-1, :] = 0.5 * (v[:-1, :] + v[1:, :])
            v_f[0, :] = v[0, :]; v_f[-1, :] = v[-1, :]
            Fu[1:-1, :] += v_f[1:-1, :] ** 2 / G['rf'][1:-1, None]
        self._markF(Fu, Fw, 'cent')

        # 湍流扩散（非保守 Laplacian，粘性系数随格点平滑变化）
        nu = self.eddy_visc()
        self._nu_last = float(nu.max())
        # 存整个场：limiter 与独立诊断必须用同一份涡粘，否则会出现
        # "限制器算的 S"与"实际状态的 S"不是同一个量（第十四轮审计表述 3）。
        self._nu_field = nu
        # stamp 记"这次 nu 对应的时刻"。step 末尾才把 t 加上 dt，故用 t+dt。
        self._nu_stamp = self.t + dt
        nu_uf = np.empty((nr + 1, nz))
        nu_uf[1:-1, :] = 0.5 * (nu[:-1, :] + nu[1:, :])
        nu_uf[0, :] = nu[0, :]
        nu_uf[-1, :] = nu[-1, :]
        nu_wf = np.empty((nr, nz + 1))
        nu_wf[:, 1:-1] = 0.5 * (nu[:, :-1] + nu[:, 1:])
        nu_wf[:, 0] = nu[:, 0]
        nu_wf[:, -1] = nu[:, -1]
        # 湍流扩散必须用柱坐标的**矢量**拉普拉斯（符号推导见 cyl_check.py E）：
        #   (∇²u)_r = ∂²u/∂r² + (1/r)∂u/∂r − u/r² + ∂²u/∂z²
        #   (∇²u)_z = ∂²w/∂r² + (1/r)∂w/∂r      + ∂²w/∂z²     ← 无 −w/r²
        # 两个易错点，都错过：
        #   1) 漏 (1/r)∂u/∂r —— 近轴 ∂u/∂r 大而 r 小，与 ∂²u/∂r² 同量级；
        #   2) 漏 −u/r²     —— 这一项必须与 (1/r)∂u/∂r 配对。射流核心 u∝r
        #      （实测 u/r 在 r=4–36 m 恒为 0.0085），两项本身各是 O(c/r)、
        #      互抵后为零；只补前者会在轴心造成 6.7–692 倍的虚假扩散源。
        # 变涡粘：ν 空间变化时 ∇·τ = ν∇²u + (∇ν)·∇u（∇·u≈0 时），
        # 后者不可省 —— 实测 r=199–299 m 处 ∂ν/∂r·∂u/∂r 是 ν∂²u/∂r² 的 1.7–2.2 倍。
        dr2 = self.drf[1:-1][:, None] ** 2
        dz2u = dz[1:-1][None, :] ** 2
        dz2w = dz[:-1][None, :] ** 2
        rf_i = G['rf'][1:-1, None]                     # u 所在半径
        drf_i = self.drf[1:-1, None]
        cyl = self.cfg.get('cyl_fix', True)            # False = 旧版（笛卡尔 Laplacian）
        var_nu = self.cfg.get('var_nu_fix', True)      # 是否含 (∇ν)·∇u
        mfix = self.cfg.get('metric_fix', True)        # 非均匀网格的二阶导

        # 二阶导的非均匀形式：
        #   d²φ/dx² = 2[(φ₊−φ)/h₊ − (φ−φ₋)/h₋] / (h₊+h₋)
        # 均匀网格退化为 (φ₊−2φ+φ₋)/h²。旧版一律用后者，拉伸段误差 100%–405%。
        # 传坐标数组而非间距，由函数内部取差，避免切片错位。
        def d2r(f, x):
            dx = np.diff(x)
            hr = dx[1:][:, None]
            hl = dx[:-1][:, None]
            if mfix:
                return 2.0 * ((f[2:] - f[1:-1]) / hr - (f[1:-1] - f[:-2]) / hl) / (hr + hl)
            return (f[2:] - 2 * f[1:-1] + f[:-2]) / ((0.5 * (hr + hl)) ** 2)

        def d2z(f, x):
            dx = np.diff(x)
            hr = dx[1:][None, :]
            hl = dx[:-1][None, :]
            if mfix:
                return 2.0 * ((f[:, 2:] - f[:, 1:-1]) / hr - (f[:, 1:-1] - f[:, :-2]) / hl) / (hr + hl)
            return (f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]) / ((0.5 * (hr + hl)) ** 2)

        # u 的矢量拉普拉斯（u 在 rf × zc 上）：两方向各自算全形再切片
        lap_u = (d2r(u, G['rf'])[:, 1:-1] + d2z(u, zc)[1:-1, :])
        if cyl:
            lap_u = lap_u + ((u[2:, 1:-1] - u[:-2, 1:-1]) / (dr[:-1] + dr[1:])[:, None] / rf_i
                             - u[1:-1, 1:-1] / rf_i ** 2)
        Fu[1:-1, 1:-1] += nu_uf[1:-1, 1:-1] * lap_u

        # w 的矢量拉普拉斯（w 在 rc × zf 上）
        lap_w = (d2r(w, rc)[:, 1:-1] + d2z(w, G['zf'])[1:-1, :])
        if cyl:
            lap_w = lap_w + ((w[2:, 1:-1] - w[:-2, 1:-1])
                             / (self.rc_gap[:-1] + self.rc_gap[1:])[:, None]
                             / np.maximum(rc[1:-1, None], 1.0))
        Fw[1:-1, 1:-1] += nu_wf[1:-1, 1:-1] * lap_w

        if var_nu:
            # (∇ν)·∇u：ν 从 nu_wake 的 exp 衰减与海绵层出来，梯度很大。
            # 注意位置：u 与 nu_uf 都在 rf 上，径向梯度用 rf 的间距；
            # w 与 nu_wf 的径向梯度用 rc 的间距。
            # u：∂ν/∂r 与 ∂u/∂r（都在 rf 上）
            dnu_dr_u = np.zeros((nr + 1, nz))
            dnu_dr_u[1:-1] = (nu_uf[2:, :] - nu_uf[:-2, :]) / (G['rf'][2:] - G['rf'][:-2])[:, None]
            du_dr_u = np.zeros((nr + 1, nz))
            du_dr_u[1:-1] = (u[2:, :] - u[:-2, :]) / (G['rf'][2:] - G['rf'][:-2])[:, None]
            du_dz = np.zeros((nr + 1, nz))
            du_dz[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (zc[2:] - zc[:-2])[None, :]
            dnu_dz_u = np.zeros((nr + 1, nz))
            dnu_dz_u[:, 1:-1] = (nu_uf[:, 2:] - nu_uf[:, :-2]) / (zc[2:] - zc[:-2])[None, :]
            Fu[1:-1, 1:-1] += (dnu_dr_u[1:-1, 1:-1] * du_dr_u[1:-1, 1:-1]
                               + dnu_dz_u[1:-1, 1:-1] * du_dz[1:-1, 1:-1])
            # w：径向梯度用 rc，垂直梯度用 zc（w 位于 rc × zf，故径向只有 nr-2 个内部点）
            dnu_dr_w = np.zeros((nr, nz + 1))
            dnu_dr_w[1:-1, 1:-1] = (nu_wf[2:, 1:-1] - nu_wf[:-2, 1:-1]) / (rc[2:] - rc[:-2])[:, None]
            dw_dr_w = np.zeros((nr, nz + 1))
            dw_dr_w[1:-1, 1:-1] = (w[2:, 1:-1] - w[:-2, 1:-1]) / (rc[2:] - rc[:-2])[:, None]
            dnu_dz_w = np.zeros((nr, nz + 1))
            dnu_dz_w[:, 1:-1] = (nu_wf[:, 2:] - nu_wf[:, :-2]) / (G['zf'][2:] - G['zf'][:-2])[None, :]
            dw_dz_w = np.zeros((nr, nz + 1))
            dw_dz_w[:, 1:-1] = (w[:, 2:] - w[:, :-2]) / (G['zf'][2:] - G['zf'][:-2])[None, :]
            Fw[1:-1, 1:-1] += (dnu_dr_w[1:-1, 1:-1] * dw_dr_w[1:-1, 1:-1]
                               + dnu_dz_w[1:-1, 1:-1] * dw_dz_w[1:-1, 1:-1])

        self._markF(Fu, Fw, 'diff')

        # 风机致动盘
        cfg = self.cfg
        if cfg.get('rotor_on', False):
            # 记录**实际施加**到动量方程上的增量，供能量审计使用。
            # 不能在诊断脚本里重推一遍 acc——第八轮审计要求"通量、推力、功率
            # 从同一次真实运行里取"，重推就等于又写了一份实现。
            Fu_rot0, Fw_rot0 = Fu.copy(), Fw.copy()
            Rw, zw = cfg['rotor_R'], cfg['rotor_z']
            hw = cfg.get('rotor_thick', 24.0)
            rh = cfg.get('rotor_hub', 0.0)          # 轮毂/传动舱遮挡半径
            tipf = cfg.get('rotor_tipfrac', 0.10)   # 叶尖损失过渡带占半径比例
            # 径向载荷分布：叶尖最后一段平滑衰减到 0（叶尖损失），
            # 轮毂以内不吹风（实体遮挡）。旧版是 r<=R 的平顶，盘缘是速度阶跃，
            # 剪切层靠数值耗散抹平，近场形态不可信。
            rr_ = rc[:, None]
            ftip = np.clip((Rw - rr_) / max(tipf * Rw, 1e-9), 0.0, 1.0)
            ftip = 0.5 * (1.0 - np.cos(np.pi * ftip))          # 平滑过渡
            m = (rr_ >= rh) & (rr_ <= Rw) & (np.abs(zc[None, :] - zw) <= hw / 2)
            target, tau = cfg['w_target'], cfg.get('rotor_tau', 4.0)
            tgt = target * ftip
            acc = np.where(m & (w_c < tgt), (tgt - w_c) / tau, 0.0)
            Fw[:, 1:-1] += 0.5 * (acc[:, :-1] + acc[:, 1:])
            self._disk_F_acc = np.zeros_like(Fw)
            self._disk_F_acc[:, 1:-1] = 0.5 * (acc[:, :-1] + acc[:, 1:])

            # 轮毂是实体（传动舱），不能让空气穿过去。仅把 r<rh 从受力区挖掉，
            # 等于在盘心开了一个孔，气流会从中心倒灌（实测 −5 m/s，与射流同量级），
            # 那是孔洞效应而非真机行为。这里给实体区加强阻尼，代表绕流阻塞。
            if cfg.get('rotor_hub_block', True) and rh > 0:
                hb = cfg.get('rotor_hub_h', 2.0 * rh + hw)
                mb = (rr_ < rh) & (np.abs(zc[None, :] - zw) <= hb / 2)
                tb = cfg.get('rotor_hub_tau', 0.5)
                bu = np.zeros((nr + 1, nz))
                bu[1:-1, :] = 0.5 * (mb[:-1, :] + mb[1:, :])
                Fw[:, 1:-1] += np.where(mb[:, :-1] | mb[:, 1:],
                                        -w[:, 1:-1] / tb, 0.0)
                Fu[1:-1, :] += -bu[1:-1, :] * u[1:-1, :] / tb

            # 旋流：整流组件之后的残余切向速度，取切向/轴向比 swirl_ratio
            sw = cfg.get('swirl_ratio', 0.0)
            if sw > 0.0:
                vt = sw * tgt
                av = np.where(m, (vt - self.v) / cfg.get('rotor_tau_v', 4.0), 0.0)
                self.v += av * dt

            # 审计快照：实际施加的力增量（含轮毂阻塞）+ 载荷结构
            self._disk_force = (Fu - Fu_rot0, Fw - Fw_rot0)
            self._disk_mask = m.copy()
            self._disk_acc = acc.copy()
            self._disk_tgt = tgt.copy()

            # 装置产热：轴功率中未变成气流动能的部分，在叶片与尾流内耗散成热。
            #   有用动能 P_kin = 1/2 * m_dot * w^2
            #   耗散成热 P_heat = P_shaft - P_kin
            #   盘面温升 Δθ = P_heat / (m_dot * cp)
            # 流量必须用**实际**盘面速度，不能用目标值。目标值抬高后实际速度只
            # 达到六七成，按目标算会把有效动能估高一倍以上，P_heat 被清零，
            # 产热反而消失——这是一个只在提速时才暴露的错误。
            if cfg.get('rotor_heat', False):
                A_ring = (np.pi * (rc + dr / 2) ** 2
                          - np.pi * np.maximum(rc - dr / 2, 0) ** 2)
                sel = m.any(axis=1)
                A_disc = max(float(np.sum(A_ring[sel])), 1.0)
                k_d = int(np.argmin(np.abs(zc - zw)))
                rho_d = float(self.rho_c[k_d])
                # 盘面实际轴向速度：按环形面积加权
                w_disc = float(np.sum(w_c[sel, k_d] * A_ring[sel]) / A_disc)
                m_dot = rho_d * A_disc * max(w_disc, 0.05)
                P_shaft = cfg.get('rotor_P', 106e6)
                if 'rotor_Pheat' in cfg:
                    # 第八轮审计后的口径：把**产热**当输入、轴功率当输出。
                    # 理由是机械功率不是 ½ṁw̄²（实测偏低 4.8 倍），
                    # 而是致动盘力对流体做功的实测量 P_work（energy_audit.py）。
                    # P_shaft = P_work + P_heat，因此 P_shaft 不该是独立旋钮。
                    P_heat = float(cfg['rotor_Pheat'])
                    P_shaft = float('nan')
                else:
                    P_heat = max(P_shaft - 0.5 * m_dot * w_disc ** 2, 0.0)
                V_disc = max(A_disc * hw, 1.0)
                dth_dt = P_heat / (rho_d * cp * V_disc)
                self.th += np.where(m, dth_dt * dt, 0.0)
                self.bud['e_rotor'] += float(np.sum(np.where(m, dth_dt * dt, 0.0)
                                                   * self._cell_mass()))
                self._heat_dth = dth_dt
                self._m_dot = m_dot
                self._w_disc = w_disc
                self._P_heat = P_heat

            # 主动加热：电加热器直接给盘面气流加热，不是轴功率的副产品。
            # 物理上等价于在盘面上方装一组电热元件。热量全部进入通过盘面的空气。
            #   Δθ/dt = P_extra / (rho cp V_disc)
            # 核算见 heat_budget2.py：这 1 K 需要 859 MW，效率只有旋翼的 1/9。
            P_extra = cfg.get('rotor_extra_heat', 0.0)
            if P_extra > 0.0:
                A_ring2 = (np.pi * (rc + dr / 2) ** 2
                           - np.pi * np.maximum(rc - dr / 2, 0) ** 2)
                sel2 = m.any(axis=1)
                A_heat = max(float(np.sum(A_ring2[sel2])), 1.0)
                k_h = int(np.argmin(np.abs(zc - zw)))
                rho_h = float(self.rho_c[k_h])
                V_heat = max(A_heat * hw, 1.0)
                dth2 = P_extra / (rho_h * cp * V_heat)
                self.th += np.where(m, dth2 * dt, 0.0)
                self.bud['e_extra'] += float(np.sum(np.where(m, dth2 * dt, 0.0)
                                                   * self._cell_mass()))
                self._heat_extra_dth = dth2
                self._P_extra = P_extra

        self._markF(Fu, Fw, 'rotor')

        # 海绵层
        Fu[1:-1, :] += -self.sponge_u[1:-1, :] * u[1:-1, :] / self.tau_sp
        Fw[:, 1:-1] += -self.sponge_w[:, 1:-1] * w[:, 1:-1] / self.tau_sp
        self._markF(Fu, Fw, 'sponge')

        # ---------------- 推进 + 投影
        # ---------------- 方位角动量（轴对称：只有 v）
        v = self.v
        Fv = np.zeros_like(v)
        if not self.cfg.get('no_adv', False):
            Fv += self._adv_center(v, u, w)
        # 曲率项 Dv/Dt + u v / r = 0 -> 右端 -u v / r
        Fv += -u_c * v / np.maximum(rc[:, None], 1.0)
        Fv += -self.sponge * v / self.tau_sp          # 海绵层
        v += dt * Fv
        # 切向速度是矢量分量，柱坐标下的矢量 Laplacian 比标量多一项 -nu*v/r²：
        #   (∇²v)_φ = (1/r)∂(r ∂v/∂r)/∂r + ∂²v/∂z² - v/r²
        # _diffuse 给的是标量形式，必须补 -v/r²。实测在近轴 v 有效区，该项与
        # 标量拉普拉斯同量级（中位比 1.7，90 分位 8.1），漏掉会让旋流衰减慢数倍，
        # 进而通过离心力 v²/r 影响尾流扩张。
        v += dt * self._diffuse(v, nu)
        if self.cfg.get('cyl_fix', True):
            v += dt * (-nu * v / np.maximum(rc[:, None], 1.0) ** 2)
        v[0, :] = 0.0
        v[:, 0] = v[:, 1]
        v[:, -1] = v[:, -2]

        # 单步能量账需要把动量段切成"受力前/受力后/投影后"三段。
        # 只在 cfg['budget'] 打开时复制，生产路径零开销（第六轮教训：
        # 诊断不得重推公式，所以这里存的是 step 真正用的那几个数组）。
        if self.cfg.get('budget', False):
            self._snap = dict(u0=u.copy(), w0=w.copy())

        u[1:-1, :] += dt * Fu[1:-1, :]
        u[0, :] = 0.0
        # 底边 u 的边界：bc_bottom='grad' 用零法向梯度（现状），
        # 'free' 交给投影决定（见第九轮 Q1：现写法在 project() 之外执行，
        # 上一步投影对 u[:,0] 的修正被整行丢弃，实测每步约 −55 MW）。
        if self.cfg.get('bc_bottom', 'grad') == 'grad':
            u[:, 0] = u[:, 1]
        w[:, 1:-1] += dt * Fw[:, 1:-1]
        w[:, 0] = 0.0
        # 顶面 w[:, -1] 不在这里赋值：project() 会用反解整行覆盖它，
        # 而反解的外侧分支要读它算角落单元的垂直散度。先置成零梯度
        # 会把那个垂直散度算成 0，使角落出流偏 1.86%（bc_verify.py S1/S4）。
        # u[-1, :] 同理，反解会整列覆盖，无需预置。

        if self.cfg.get('budget', False):
            self._snap.update(Fu=Fu.copy(), Fw=Fw.copy(),
                              u1=u.copy(), w1=w.copy())

        # ---------------- 2Δ 滤波（第十二轮审计 Q1）
        # 模板 <φ>_i = (φ_{i-1} + 2φ_i + φ_{i+1}) / 4，符号 H(θ) = (1+cosθ)/2，
        # 故 H(Nyquist) = 0，Nyquist 波数的单步幅值因子恰为 1-η。
        # η = γ·dt，代码里按 η 存（dt 会随状态变，存 γ 等于每步改 η）。
        # 施加位置：力推进之后、project() 之前 —— 这样投影会重新满足离散
        # 连续性，不会像投影后再滤波那样破坏 D_h(ρu)=0。
        # 作用区域：近场均匀区（rf<650 且 zf<600），非均匀拉伸段不滤，避免
        # 在变尺度上做无意义的平均。边界不滤（轴/地面/外侧/顶面由赋值或反解定）。
        eta = self.cfg.get('filt_eta', 0.0)
        if eta > 0.0:
            self._filt_R, self._filt_K = self._filt_extent(G)
            u_before = u.copy(); w_before = w.copy()

            def smooth2(phi, i0, i1, k0, k1, along_r, along_z):
                if along_r and i1 - i0 >= 2:
                    core = phi[i0:i1, k0:k1]
                    avg = 0.25 * (phi[i0 - 1:i1 - 1, k0:k1]
                                  + 2 * core + phi[i0 + 1:i1 + 1, k0:k1])
                    phi[i0:i1, k0:k1] = core - eta * (core - avg)
                if along_z and k1 - k0 >= 2:
                    core = phi[i0:i1, k0:k1]
                    avg = 0.25 * (phi[i0:i1, k0 - 1:k1 - 1]
                                  + 2 * core + phi[i0:i1, k0 + 1:k1 + 1])
                    phi[i0:i1, k0:k1] = core - eta * (core - avg)

            R, K = self._filt_R, self._filt_K
            smooth2(u, R.start, R.stop, 1, nz - 1, True, True)
            smooth2(w, 1, nr - 1, K.start, K.stop, True, True)
            # 滤波做功与动能变化（审计要求单列，不能并入力做功）
            self._filt_dKE = float(
                0.5 * np.sum(self.rho_uf * self._V_u * (u ** 2 - u_before ** 2))
                + 0.5 * np.sum(self.rho_wf * self._V_w * (w ** 2 - w_before ** 2)))
            self._filt_W = float(
                np.sum(self.rho_uf * self._V_u * u_before * (u - u_before))
                + np.sum(self.rho_wf * self._V_w * w_before * (w - w_before)))

        self.project(u, w, dt)

        if self.cfg.get('budget', False):
            self._snap.update(u2=u.copy(), w2=w.copy(), P=self._proj['P'].copy())

        # ---------------- 标量
        self.scalars(dt, nu)
        self.surface_flux(dt)
        self.budget_step(dt)
        self.t += dt
        self._dt_prev = dt            # 供下一步的增长限制用

    def scalars(self, dt, nu):
        G = self.g
        nr, nz = self.nr, self.nz
        rc, dr, dz = G['rc'], G['dr'], G['dz']
        u, w = self.u, self.w

        # 注意：本模式满足的是变密度连续方程 (1/r)d(rho r u)/dr + d(rho w)/dz = 0，
        # 而不是体积散度为零。标量平流必须按 rho 加权（通量形式），否则上升区会
        # 出现 phi * div(V) = -phi*w*dln(rho)/dz 的虚假源汇（约 0.03 K/s，足以在
        # 1 分钟内抹掉一个 2 K 的气块）。
        def adv(phi, weff=None):
            pu = self._flux_r(phi, u)                       # (nr+1,nz)
            pu = pu * self.rho_uf
            d = -(pu[1:, :] * G['rf'][1:, None] - pu[:-1, :] * G['rf'][:-1, None]) \
                / (rc[:, None] * dr[:, None])
            wf = w if weff is None else weff
            pz = self._flux_z(phi, wf)                      # (nr,nz+1)
            pz = pz * self.rho_wf
            d += -(pz[:, 1:] - pz[:, :-1]) / dz[None, :]
            return d / self.rho_c[None, :]

        # 雨滴落速随雨水含量增长（Kessler 型）；常数落速会把强对流中的雨水
        # 留在空中，使 qr 偏高
        Vt_c = np.clip(self.cfg.get('Vt0', 3.0) + self.cfg.get('Vtk', 80.0)
                       * np.sqrt(np.maximum(self.qr, 0.0)), 3.0, 12.0)
        Vt_f = np.empty((nr, nz + 1))
        Vt_f[:, 1:-1] = 0.5 * (Vt_c[:, :-1] + Vt_c[:, 1:])
        Vt_f[:, 0] = Vt_c[:, 0]
        Vt_f[:, -1] = Vt_c[:, -1]
        w_eff = w - Vt_f
        # 雪的落速（约 1.2 m/s）
        Vt_s = self.cfg.get('Vt_snow', 1.2)
        w_eff_s = w - Vt_s
        # 霰的落速。常数 3.0 m/s 是 v17 及之前的设定，但它让 qg 涨到 12 g/kg
        # 落速也不变，在 48 m/s 的上升气流里永远落不下来（见 L1 诊断）。
        # vt_qg_dep=True 时按 Kessler 型随含水量增长：qg 从 0.1 长到 12 g/kg，
        # 落速从 3.0 到 5.2 m/s。系数 20 取自霰的质量-落速关系 Vt ∝ qg^(1/6)
        # 在 1–12 g/kg 段的线性化，封顶 10 m/s（3 mm 霰的上限，不含雹）。
        if self.cfg.get('vt_qg_dep', False):
            Vt_gc = np.clip(3.0 + self.cfg.get('Vtk_g', 20.0)
                            * np.sqrt(np.maximum(self.qg, 0.0)), 3.0, 10.0)
            Vt_gf = np.empty((nr, nz + 1))
            Vt_gf[:, 1:-1] = 0.5 * (Vt_gc[:, :-1] + Vt_gc[:, 1:])
            Vt_gf[:, 0] = Vt_gc[:, 0]
            Vt_gf[:, -1] = Vt_gc[:, -1]
            w_eff_g = w - Vt_gf
        else:
            Vt_g = self.cfg.get('Vt_graupel', 3.0)
            w_eff_g = w - Vt_g
        if self.cfg.get('no_scalar_adv', False):
            th_new, qv_new = self.th.copy(), self.qv.copy()
            qc_new, qr_new = self.qc.copy(), self.qr.copy()
            qi_new, qs_new = self.qi.copy(), self.qs.copy()
            qg_new = self.qg.copy()
            qagi_new = self.qagi.copy()
        else:
            th_new = self.th + dt * adv(self.th)
            qv_new = self.qv + dt * adv(self.qv)
            qc_new = self.qc + dt * adv(self.qc)
            qi_new = self.qi + dt * adv(self.qi)
            qr_new = self.qr + dt * adv(self.qr, w_eff)
            qs_new = self.qs + dt * adv(self.qs, w_eff_s)
            qg_new = self.qg + dt * adv(self.qg, w_eff_g)
            # AgI 是被动示踪，不落，按 w 输运
            qagi_new = self.qagi + dt * adv(self.qagi, self.w)

        # 标量扩散（显式 Laplacian）
        def diff(phi):
            return dt * self._diffuse(phi, nu)

        # 裁剪会在局部凭空产生质量/能量（负值截到 0）。这里把它记账，
        # 否则收支看起来"漏"了，却不知道漏在哪里。
        cm = self._cell_mass()
        for nm, new in (('qv', qv_new), ('qc', qc_new), ('qi', qi_new),
                        ('qr', qr_new), ('qs', qs_new), ('qg', qg_new),
                        ('qagi', qagi_new)):
            val = new + diff(getattr(self, nm))
            neg = np.maximum(-val, 0.0)
            self.bud['q_clip'] += float(np.sum(neg * cm))
        dth_neg = np.maximum(-(th_new + diff(self.th)), 0.0)
        self.bud['e_clip'] += float(np.sum(dth_neg * cm))

        self.th = np.maximum(th_new + diff(self.th), 150.0)
        self.qv = np.maximum(qv_new + diff(self.qv), 0.0)
        self.qc = np.maximum(qc_new + diff(self.qc), 0.0)
        self.qi = np.maximum(qi_new + diff(self.qi), 0.0)
        self.qr = np.maximum(qr_new + diff(self.qr), 0.0)
        self.qg = np.maximum(qg_new + diff(self.qg), 0.0)
        self.qs = np.maximum(qs_new + diff(self.qs), 0.0)
        self.qagi = np.maximum(qagi_new + diff(self.qagi), 0.0)

        # --- M6 标量卷吸（entrain + entrain_scalar 同时开才生效）---
        # 半套（只有动量汇）会把浮力和动量一起打掉，实测云顶从 4171 掉到 1443 m。
        # 配套的标量卷吸让卷入的空气把云内水汽/位温拉向环境值，这才是完整的夹卷。
        # 形式 −ε|w|(q − q_env)，ε 与动量汇共用 _entrain_eps()。
        # 放在微物理之前：卷吸造成的过饱和由随后的饱和调整处理。
        if self.cfg.get('entrain', False) and self.cfg.get('entrain_scalar', False):
            eps_e = self._entrain_eps()
            wc_e = 0.5 * (self.w[:, 1:] + self.w[:, :-1])
            fac = eps_e[None, :] * np.abs(wc_e) * dt
            for nm in ('qv', 'qc', 'qi', 'qs', 'qr', 'qg'):
                qq = getattr(self, nm)
                qq -= fac * (qq - self.env_prof[nm][None, :])
            self.th -= fac * (self.th - self.env_prof['th'][None, :])

        # 辐射（仅雾工况）：短波吸收 + 雾顶长波冷却，改 th
        if self.cfg.get('radiation', False):
            self.radiation(dt)

        self.microphysics(dt)

    def radiation(self, dt):
        """辐射雾工况的短波吸收与雾顶长波冷却（2026-11-03 新增）。

        为什么需要：辐射雾的消散主导机制是辐射。此前的除雾算例没有辐射，
        测的是"纯机械清雾"；v16 用固定 120 W/m² 地面感热代替，也不对
        ——那是常数，不随太阳高度变化，且方向相反（真实雾的消散从雾顶开始，
        不是从地面烤）。

        两个过程：
          1. 短波：太阳短波从雾顶进入，按 Beer 定律逐层衰减，被吸收的部分
             就地加热。浓雾的光学厚度约 20，地面几乎收不到短波。
          2. 长波：雾顶向外太空辐射，净通量约 96 W/m²，在雾顶 20 m 内
             释放，产生 1–3 K/h 的冷却。冷却使雾顶失稳，触发混合，这是
             雾从顶部开始崩解的原因。

        局限（见 辐射方案设计.md 第七节）：长波用调节因子 f_lw 把 14 K/h
        拉到观测的 3 K/h 量级，是参数化不是辐射传输解；无气溶胶活化；
        无土壤热通量；无水平平流。
        """
        cfg = self.cfg
        G = self.g
        rc, zc, dz = G['rc'], G['zc'], G['dz']
        nr, nz = self.nr, self.nz
        ex = self.ex_c[None, :]
        sigma = 5.67e-8

        # ---- 太阳高度角
        t_local = cfg.get('t_local0', 8.0) + self.t / 3600.0
        phi = np.radians(cfg.get('lat', 30.0))
        dec = np.radians(cfg.get('decl', -8.0))
        Hh = np.radians(15.0 * (t_local - 12.0))
        sinh = float(np.sin(phi) * np.sin(dec)
                     + np.cos(phi) * np.cos(dec) * np.cos(Hh))
        sinh = max(sinh, 0.0)
        if sinh <= 1e-4:
            return

        # ---- 雾顶短波（大气透过率随高度角变化）
        # 用 Kasten-Young 大气质量，不再用 1/sinh 加 0.4 的下限。
        # 旧写法在太阳高度角 2–10° 时把 m 截到 20，透过率被托到 0.4，
        # 6:30 的入射短波因此被高估约 4 倍（23 对实际 ~6 W/m²）。
        # 6:30 起报时这一段正好是主时段，必须按真实路径算。
        am = 1.0 / (sinh + 0.50572 * (sinh + 6.07995) ** (-1.6364))
        am = min(am, 38.0)                                    # Kasten-Young，天顶附近封顶
        T_atm = float(np.exp(-cfg.get('a_atm', 0.15) * am))
        S_top = 1361.0 * sinh * T_atm

        # ---- 逐层消光与吸收
        # cum_above[k] = 第 k 层**以上**的累积光学厚度；本层顶的通量是
        # F_top[k] = S_top·exp(−cum_above[k])，本层吸收 = 该通量 ×(1−e^{−βΔz})
        lwc = np.maximum(self.qc, 0.0) * self.rho_c[None, :] * 1e3      # g/m3
        beta = cfg.get('k_ext', 144.7) * lwc ** 0.88 / 1000.0           # 1/m

        # ---- 反照率（2026-11-03 补）
        # 上一版把入射短波全部送进雾层，等于假设反照率 0。真实雾的反照率
        # 0.3–0.5（浓雾取高值），少算的这部分让消散速度快了近一倍：
        # 322 W/m² 全吸收时 105 g/m² 的雾水 13 分钟就能蒸干。
        # 用 LWP 相关的经验式：雾越厚反射越强，LWP→0 时反射趋于 0。
        LWP = np.sum(lwc * dz[None, :], axis=1)                         # g/m2
        a_max = cfg.get('albedo_max', 0.50)
        lwp_h = cfg.get('albedo_lwp', 30.0)                             # g/m2
        alpha = a_max * (1.0 - np.exp(-LWP / lwp_h))
        S_eff = S_top * (1.0 - alpha)                                   # 进入雾层的部分

        cum_above = np.zeros((nr, nz))
        acc = np.zeros(nr)
        for k in range(nz - 1, -1, -1):
            cum_above[:, k] = acc
            acc = acc + beta[:, k] * dz[k]
        F_top = S_eff[:, None] * np.exp(-cum_above)                     # W/m2
        absorb = F_top * (1.0 - np.exp(-beta * dz[None, :]))            # W/m2
        # 层厚必须进分母：否则 4 m 厚的雾顶层会把 300 W/m² 算成上千 K/h
        dth_sw = absorb / (self.rho_c[None, :] * cp * ex * dz[None, :])
        dth_sw = np.clip(dth_sw, 0.0, cfg.get('dT_max', 0.02))          # K/s 上限

        # ---- 雾顶长波冷却
        lwc_th = 1e-3
        has = lwc > lwc_th
        Ta = self.th * ex
        # 云顶以上的有效辐射温度。
        # 默认用 Idso & Jackson (1969) 晴空经验式，以地面温度推算天空温度——
        # 这是为辐射雾标定的（雾顶 285 K、天空 270 K，给出 1–3 K/h）。
        # 但对深对流顶不成立：10 km 的云顶 235 K，而该式按地面温度算出 270 K，
        # 比云顶还暖，净通量变成负的，冷却被整段跳过。
        # lw_bulk=True 时改用"云顶以上各层的质量加权辐射温度"
        #     T_eff = (∫ σT⁴ dm / ∫ dm)^(1/4)
        # 雾顶（350 m）得到 268 K 量级，深对流顶（10 km）得到 220 K 量级，
        # 两种情形都能给出观测量级的云顶冷却。默认关闭，保证雾工况结果不变。
        if cfg.get('lw_bulk', False):
            Ta1 = Ta[0]
            dmm = self.rho_c * dz
            c4 = np.cumsum((dmm * Ta1 ** 4)[::-1])[::-1]
            cw = np.cumsum(dmm[::-1])[::-1]
            a4 = np.empty(nz); aw = np.empty(nz)
            a4[:-1] = c4[1:]; a4[-1] = 0.0
            aw[:-1] = cw[1:]; aw[-1] = 0.0
            T_eff_prof = np.where(aw > 0.0, (a4 / np.maximum(aw, 1e-30)) ** 0.25, Ta1)
        else:
            T_eff_prof = None
        T_sky = 0.0552 * Ta[:, 0] ** 1.5
        # LWP 已在短波段算过，这里复用
        eps = 1.0 - np.exp(-cfg.get('k_eps', 0.1) * LWP)
        f_lw = cfg.get('f_lw', 0.2)
        # 长波在光学厚度约 1 的层内被吸收/发射，不是整个 4 m 格点。
        # 用 20 m 的有效厚度，等价于把冷却摊到雾顶的辐射层里。
        dz_eff = cfg.get('dz_eff_lw', 20.0)
        dth_lw = np.zeros((nr, nz))
        ktop_all = np.where(has.any(axis=1),
                            nz - 1 - np.argmax(has[:, ::-1], axis=1), -1)
        for i in range(nr):
            k = int(ktop_all[i])
            if k < 0:
                continue
            T_down = T_eff_prof[k] if T_eff_prof is not None else T_sky[i]
            Fnet = eps[i] * sigma * Ta[i, k] ** 4 - sigma * T_down ** 4
            if Fnet <= 0:
                continue
            dth_lw[i, k] = -f_lw * Fnet / (self.rho_c[k] * cp * dz_eff)
        dth_lw = np.clip(dth_lw, -cfg.get('dT_max', 0.02), 0.0)

        self.th += (dth_sw + dth_lw) * dt
        self.rad_state = dict(sinh=sinh, S_top=S_top, T_atm=T_atm,
                              alpha=float(alpha.max()), S_eff=float(S_eff.min()),
                              LWP=float(LWP.max()), eps=float(eps.max()),
                              dT_sw_max=float(dth_sw.max() * 3600.),
                              dT_lw_max=float(dth_lw.min() * 3600.),
                              t_local=t_local)

    def surface_flux(self, dt):
        """地面感热/潜热通量。

        10 m 的首层无法承受 250 W/m2 的净通量（会让首层升温 20 K 以上，
        属于未分辨的地表层效应）。这里按"通量在混合层内均匀分配"处理，
        等价于保持观测到的边界层加热情形，使前期环境维持准定常。
        深度取 flux_depth（默认 700 m，即预设混合层厚度）。"""
        cfg = self.cfg
        # 优先读 env 的设定。雾工况 build_fog 明确返回 SHF=LHF=0、
        # flux_depth=100（雾顶辐射冷却，地面通量近于零）；增雨工况 build_env
        # 不设这三项，仍走 cfg 里 BASE 的 120 / 100 / 700。
        # 2026-11-02 核对发现：此前只从 cfg 取，导致除雾算例全程带着
        # 120 W/m² 感热与 100 W/m² 潜热加热加湿，与 build_fog 的设计相反，
        # v16 的"蒸发消散"结论因此来源不明。
        env_ = self._env
        shf = env_.get('SHF', cfg.get('SHF', 250.0))
        lhf = env_.get('LHF', cfg.get('LHF', 200.0))
        if shf == 0 and lhf == 0:
            return
        depth = env_.get('flux_depth', cfg.get('flux_depth', 700.0))
        zc = self.g['zc']
        kmax = max(int(np.searchsorted(zc, depth)), 1)
        kmax = min(kmax, self.nz)
        dcol = float(np.sum(self.g['dz'][:kmax]))
        qv_in = self.qv[:, :kmax].copy()
        self.th[:, :kmax] += shf * dt / (self.rho_c[None, :kmax] * cp * dcol)
        Tc = self.th[:, :kmax] * self.ex_c[None, :kmax]
        qmax = qsat(Tc, self.p_c[None, :kmax])
        self.qv[:, :kmax] = np.minimum(
            self.qv[:, :kmax] + lhf * dt / (self.rho_c[None, :kmax] * Lv * dcol), qmax)
        # 收支：记录实际加入的量（潜热项可能被 qmax 截断）
        sub_m = self.rho_c[None, :kmax] * (2 * np.pi * self.g['rc'] * self.g['dr'])[:, None] \
            * self.g['dz'][None, :kmax]
        self.bud['q_surf'] += float(np.sum((self.qv[:, :kmax] - qv_in) * sub_m))
        self.bud['e_surf'] += float(np.sum(shf * dt / (self.rho_c[None, :kmax] * cp * dcol)
                                           * sub_m))

    # -------------------------------------------------------------- 雨强
    def rain_rate(self):
        """返回 (面平均雨强, 局地最大雨强, 降水面积) —— 单位 mm/h, mm/h, km2"""
        Vt = np.clip(self.cfg.get('Vt0', 3.0) + self.cfg.get('Vtk', 80.0)
                     * np.sqrt(np.maximum(self.qr[:, 0], 0.0)), 3.0, 12.0)
        loc = self.rho_c[0] * self.qr[:, 0] * Vt * 3600.0     # mm/h（局地）
        mean = float(np.sum(loc * self.area_cell) / self.area_tot)
        amax = float(loc.max())
        area = float(np.sum(self.area_cell[loc > 0.1]) / 1e6)
        return mean, amax, area

    def accumulate(self, dt):
        mean, amax, area = self.rain_rate()
        self.rain_inst = mean
        self.rain_local = amax
        self.rain_area = area
        self.rain_acc += mean * dt / 3600.0

    def max_dt(self, cfl=None):
        """时间步。

        `dt_fixed` 直接给定 dt，绕过自动选步。这是做**独立分离试验**的唯一办法：
        自动选步时 dt = min(CFL 限, 粘性限)，两个数被绑在一起（cfl=0.30 时两者
        都算出 0.063），任何单参数调节都同时改动 CFL 数与扩散数，无法归因。
        固定 dt 后，可以单独扫 CFL 数或单独扫扩散数。

        返回同时记录当前两个约束值与主导项，供诊断用（自查报告第六、七节要用）。
        """
        vmax = max(np.abs(self.w).max(), np.abs(self.u).max(),
                   np.abs(self.v).max(), 1.0)
        dmin = min(self.g['dz'].min(), self.g['dr'].min())
        cfl_eff = cfl if cfl is not None else self.cfg.get('cfl', 0.30)
        dtc = cfl_eff * dmin / vmax

        # 涡粘的时效检查（第十三轮审计③）。`_nu_last` 只在动量段末尾更新，
        # 载入/构造初值后它还是旧值（甚至默认 2.0），第一步的 dt 因此偏大。
        # 2 m 档实测首步 dt=0.0426 是后续的 2.7 倍，S 冲到 5.4。
        if getattr(self, '_nu_stamp', None) != self.t:
            self.refresh_nu()
        nu_now = max(getattr(self, '_nu_last', 2.0), 2.0)
        dtv = 0.20 * dmin ** 2 / nu_now

        # 二维 QUICK 的 Nyquist 稳定限（第十三轮）：
        #   S = 4·dt·(|u|/Δr + |w|/Δz) + 4·ν·dt·(1/Δr² + 1/Δz²) ≤ S_target
        # 注意 Δr 与 Δz 分开：两个方向的对流速度不同（u 方程的垂直平流由 w
        # 提供），扩散也是两个方向各一份，不能合成一个 Δx（审计 Q1 第一条限定）。
        dts = np.inf
        method = self.cfg.get('stab_method', 'scan')
        if self.cfg.get('stab_limiter', False):
            # 'scan' = 全波数扫描（默认，第十四轮修正）；'nyquist' = 旧的闭式
            dts = self._stab_dt_scan() if method == 'scan' else self._stab_dt_quick()
        self._lim = dict(vmax=vmax, dt_cfl=dtc, dt_visc=dtv, dt_stab=dts,
                         dt_raw=dts, cfl_num=0.0, diff_num=0.0,
                         method=method)
        fixed = self.cfg.get('dt_fixed', None)
        if fixed:
            dt = float(fixed)
            dt = min(dt, dtc, dtv, dts)   # 不允许超过任一显式稳定界
            why = 'fixed'
        else:
            cand = {'cfl': dtc, 'visc': dtv, 'stab': dts}
            why = min(cand, key=cand.get)
            dt = min(dtc, dtv, dts, 3.0)
            # 单向增长限制（第十四轮审计 Q3）：允许立即减小，不许突然增大。
            prev = getattr(self, '_dt_prev', None)
            grow = float(self.cfg.get('dt_grow_max', 1.25))
            if prev is not None and grow > 0 and dt > grow * prev:
                dt = grow * prev
                why = why + '+grow'
        self._lim.update(dt=dt, why=why,
                         cfl_num=dt * vmax / dmin,
                         diff_num=dt * nu_now / dmin ** 2)
        return float(dt)

    def refresh_nu(self):
        """用**当前场**重算涡粘并记录时刻。载入初值后必须调用。

        同时清空 `_dt_prev`：审计 Q3 要求"载入或修改初值后旧的 dt_prev 失效"，
        否则增长限制会拿上一个算例的末步去卡新算例的首步。
        """
        nu = self.eddy_visc()
        self._nu_field = nu
        self._nu_last = float(nu.max())
        self._nu_stamp = float(self.t)
        self._dt_prev = None
        return self._nu_last

    def _stab_dt_scan(self, ntheta=40):
        """满足**全波数** |G| ≤ G_target 的最大 dt（第十四轮审计 Q2 的修正版）。

        旧版只看 Nyquist 角点（`S=4dt(u+w)/Δ+8νdt/Δ²<2`），审计给出了一个
        反例：QUICK 在 θ=π/2 处有 |G|²=1−2C+5C²，稳定要求 C≤0.4，而 Nyquist
        只要 C≤0.5。所以 0.4<C<0.5 区间内"Nyquist 稳、π/2 不稳"。

        本函数对 (θ_r, θ_z) 全域扫描。|G|² = 1 + 2dt·Reλ + dt²|λ|²，
        |G| ≤ 1 等价于 dt·(dt|λ|² + 2Reλ) ≤ 0，即
            dt ≤ dt_crit(θ) = −2·Reλ / |λ|²
        取全域最小，再乘裕量 `stab_margin`（默认 0.75，等价于旧的 S_target=1.5，
        因为 S_target = 2·margin）。

        注意不能把目标设成 |G| ≤ 0.8 —— dt→0 时 |G|→1，那个条件无解。裕量只能
        乘在 dt_crit 上。

        面位置按审计要求分开：u 方程用 rf 上的 u 与插值到 rf 的 w_at_f；
        w 方程用插值到 zf 的 u_at_w 与 zf 上的 w。ν 取全场最大（保守）。
        Δ 取最小间距（拉伸段的严格上界）。
        """
        G = self.g
        nr, nz = self.nr, self.nz
        th = np.linspace(0.02 * np.pi, np.pi, ntheta)
        Qr = (3 - 4 * np.exp(-1j * th) + np.exp(-2j * th)) / 2
        Dr = np.exp(1j * th) - 2 + np.exp(-1j * th)

        # 面位置的速度（生产算子真正读取的量）
        w_c = 0.5 * (self.w[:, 1:] + self.w[:, :-1])
        w_at_f = np.empty((nr + 1, nz))
        w_at_f[1:-1, :] = 0.5 * (w_c[:-1, :] + w_c[1:, :])
        w_at_f[0, :] = w_c[0, :]
        w_at_f[-1, :] = w_c[-1, :]
        u_c = 0.5 * (self.u[1:, :] + self.u[:-1, :])
        u_at_w = np.empty((nr, nz + 1))
        u_at_w[:, 1:-1] = 0.5 * (u_c[:, :-1] + u_c[:, 1:])
        u_at_w[:, 0] = u_c[:, 0]
        u_at_w[:, -1] = u_c[:, -1]

        nu_max = max(float(getattr(self, '_nu_last', 2.0)), 0.0)
        drmin = float(G['dr'].min())
        dzmin = float(G['dz'].min())

        margin = float(self.cfg.get('stab_margin', 0.75))
        best = np.inf
        for um, wm in ((np.abs(self.u).max(), np.abs(w_at_f).max()),
                       (np.abs(u_at_w).max(), np.abs(self.w).max())):
            lam = (-um * Qr[:, None] / drmin - wm * Qr[None, :] / dzmin
                   + nu_max * (Dr[:, None] / drmin ** 2
                               + Dr[None, :] / dzmin ** 2))
            Re = lam.real
            A2 = np.abs(lam) ** 2
            ok = (Re < 0) & (A2 > 1e-30)
            if not np.any(ok):
                continue
            dtq = np.full_like(Re, np.inf)
            dtq[ok] = -2.0 * Re[ok] / A2[ok]
            best = min(best, float(dtq.min()))
        return margin * best

    def _stab_dt_quick(self, target=None):
        """满足 S_max ≤ target 的最大 dt（二维 QUICK + 显式 Euler）。

        A_ij = 4(|u_ij|/Δr_i + |w_ij|/Δz_j)
        B_ij = 4ν_ij(1/Δr_i² + 1/Δz_j²)
        dt ≤ target / max(A+B)

        用单元中心的 u、w 与单元尺度的 ν——与 S 判据的定义一致。取全场最大
        而不是分区，是因为显式格式的稳定界由最坏格点决定（审计 Q1）。
        """
        if target is None:
            target = float(self.cfg.get('S_target', 1.5))
        G = self.g
        u_c = 0.5 * (self.u[1:, :] + self.u[:-1, :])
        w_c = 0.5 * (self.w[:, 1:] + self.w[:, :-1])
        dr = G['dr'][:, None]
        dz = G['dz'][None, :]
        A = 4.0 * (np.abs(u_c) / dr + np.abs(w_c) / dz)
        B = 4.0 * max(getattr(self, '_nu_last', 2.0), 0.0) * (1.0 / dr ** 2 + 1.0 / dz ** 2)
        S1 = A + B
        smax = float(S1.max())
        self._lim_S1 = smax
        if smax <= 0:
            return np.inf
        return target / smax

    # -------------------------------------------------------------- 收支
    def _cell_mass(self):
        """单元内空气质量 (nr,nz)"""
        G = self.g
        return self.rho_c[None, :] * (2 * np.pi * G['rc'] * G['dr'])[:, None] * G['dz'][None, :]

    def water_storage(self):
        """域内总水量 kg"""
        qt = self.qv + self.qc + self.qi + self.qr + self.qs + self.qg
        return float(np.sum(qt * self._cell_mass()))

    def energy_storage(self):
        """域内位温积分 ∫ρθ dV，单位 K·kg。

        为什么不用 ∫ρ cp T dV：位温方程在通量形式下守恒的是 θ，不是 cp·ex·θ。
        ex 随高度变，若按能量记账，平流项本身就不守恒，闭合残差里会混进这一项，
        看起来像物理错误。用 θ 记账，闭合残差才是真正的数值误差。"""
        return float(np.sum(self.th * self._cell_mass()))

    def budget_step(self, dt):
        """累加一个时间步的边界通量。与 adv 用同一套离散，保证可比。

        三个易错点，都错过：
          1) 径向面的面积权重要带 2π（漏了会差 2π 倍）；
          2) 雨、雪、霰各按自己的落速输运，与其余物种分开算，不能并进 qt 的 w 项；
          3) 扩散算子必须做密度加权，否则本身就不守恒。
        """
        G = self.g
        rf, dz, rc, dr = G['rf'], G['dz'], G['rc'], G['dr']
        area = 2 * np.pi * rc * dr
        bud = self.bud

        # ---- 水：径向（所有物种同乘 u）
        qt = self.qv + self.qc + self.qi + self.qr + self.qs + self.qg
        pu = self._flux_r(qt, self.u) * self.rho_uf        # (nr+1,nz)
        bud['q_side'] += -float(np.sum(pu[-1, :] * rf[-1] * dz)) * 2 * np.pi * dt

        # ---- 水：垂直。雨/雪各按自己的落速，其余按 w
        Vt_c = np.clip(self.cfg.get('Vt0', 3.0) + self.cfg.get('Vtk', 80.0)
                       * np.sqrt(np.maximum(self.qr, 0.0)), 3.0, 12.0)
        Vt_f = np.empty((self.nr, self.nz + 1))
        Vt_f[:, 1:-1] = 0.5 * (Vt_c[:, :-1] + Vt_c[:, 1:])
        Vt_f[:, 0] = Vt_c[:, 0]; Vt_f[:, -1] = Vt_c[:, -1]
        Vt_s = self.cfg.get('Vt_snow', 1.2)
        if self.cfg.get('vt_qg_dep', False):
            Vt_gc = np.clip(3.0 + self.cfg.get('Vtk_g', 20.0)
                            * np.sqrt(np.maximum(self.qg, 0.0)), 3.0, 10.0)
            Vt_gf = np.empty((self.nr, self.nz + 1))
            Vt_gf[:, 1:-1] = 0.5 * (Vt_gc[:, :-1] + Vt_gc[:, 1:])
            Vt_gf[:, 0] = Vt_gc[:, 0]
            Vt_gf[:, -1] = Vt_gc[:, -1]
            w_g = self.w - Vt_gf
        else:
            w_g = self.w - self.cfg.get('Vt_graupel', 3.0)
        pz = (self._flux_z(qt - self.qr - self.qs - self.qg, self.w)
              + self._flux_z(self.qr, self.w - Vt_f)
              + self._flux_z(self.qs, self.w - Vt_s)
              + self._flux_z(self.qg, w_g)) * self.rho_wf
        bud['q_top'] += -float(np.sum(pz[:, -1] * area)) * dt
        bud['q_bot'] += -float(np.sum(pz[:, 0] * area)) * dt

        # ---- 热力学：θ 的径向与垂直输运
        eth = self.th
        pu_e = self._flux_r(eth, self.u) * self.rho_uf
        bud['e_side'] += -float(np.sum(pu_e[-1, :] * rf[-1] * dz)) * 2 * np.pi * dt
        pz_e = self._flux_z(eth, self.w) * self.rho_wf
        bud['e_top'] += -float(np.sum(pz_e[:, -1] * area)) * dt
        bud['e_bot'] += -float(np.sum(pz_e[:, 0] * area)) * dt

        bud['nstep'] += 1

    def budget_report(self):
        """收支闭合诊断"""
        b = self.bud
        dM = self.water_storage() - b['q0']
        srcM = b['q_surf'] + b['q_side'] + b['q_top'] + b['q_bot'] + b['q_clip']
        dE = self.energy_storage() - b['e0']
        srcE = (b['e_surf'] + b['e_rotor'] + b['e_extra'] + b['e_latent']
                + b['e_side'] + b['e_top'] + b['e_bot'] + b['e_clip'])
        return dict(
            q0=b['q0'], q1=self.water_storage(), dM=dM, srcM=srcM,
            q_res=dM - srcM, q_rel=(dM - srcM) / max(abs(srcM), 1e-30),
            q_surf=b['q_surf'], q_side=b['q_side'], q_top=b['q_top'], q_bot=b['q_bot'],
            q_clip=b['q_clip'], e_clip=b['e_clip'],
            e0=b['e0'], e1=self.energy_storage(), dE=dE, srcE=srcE,
            e_res=dE - srcE, e_rel=(dE - srcE) / max(abs(srcE), 1e-30),
            e_surf=b['e_surf'], e_rotor=b['e_rotor'], e_extra=b['e_extra'],
            e_latent=b['e_latent'], e_side=b['e_side'], e_top=b['e_top'],
            e_bot=b['e_bot'], nstep=b['nstep'])

    # -------------------------------------------------------------- 诊断输出
    def diag(self):
        G = self.g
        zc = G['zc']
        qc_max = float(self.qc.max())
        wmax = float(self.w.max())
        cloud = self.qc > 1e-5
        if cloud.any():
            ktop = int(np.where(cloud.any(axis=0))[0].max())
            ztop = float(zc[ktop])
        else:
            ztop = 0.0
        # 羽流顶：w > 0.5 m/s 的最高高度
        up = self.w[:, 1:] > 0.5
        zplume = float(zc[np.where(up.any(axis=0))[0].max()]) if up.any() else 0.0
        mean, amax, area = self.rain_rate()
        return dict(t=self.t, wmax=wmax, vmax=float(np.abs(self.v).max()),
                    qc_max=qc_max, qr_max=float(self.qr.max()),
                    qi_max=float(self.qi.max()), qs_max=float(self.qs.max()),
                    z_cloud_top=ztop, z_plume=zplume, rain=mean,
                    rain_local=amax, rain_area=area, rain_acc=self.rain_acc)
