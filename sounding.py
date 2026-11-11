"""
sounding.py  —  理想平原夏季上午探空设计 + 气块法验证

设计意图（承接研究记录 05 的判据体系）：
  地面 27 degC / 露点 22 degC / 1000 hPa，混合层约 900 m，
  其上有一个适度逆温盖（提供 CIN），盖顶以上条件不稳定（提供 CAPE）。
  目标：CAPE 2000-2600 J/kg，CIN 10-30 J/kg，LFC 1.0-1.3 km。

方法：
  - 环境：给定 theta(z) 与 qv(z)，静力积分得到 p、T、rho
  - 气块：干绝热抬升到 LCL，之后沿湿绝热线积分 dT/dp（RK4），
    不做假相当位温反解，避免 Bolton 公式的数值问题
"""
import numpy as np

Rd, Rv, cp, g, Lv = 287.04, 461.5, 1004.5, 9.80665, 2.5e6
p0 = 100000.0
eps = Rd / Rv


def esat(T):
    """Bolton 1980 饱和水汽压 (Pa)：e_s = 611.2 exp(17.67 Tc/(Tc+243.5))"""
    Tc = np.clip(T - 273.15, -80.0, 60.0)
    return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))


def qsat(T, p):
    es = esat(T)
    return eps * es / np.maximum(p - es, 1.0)


def exner(p):
    return (p / p0) ** (Rd / cp)


def T_from_theta(th, p):
    return th * exner(p)


def dewpoint_from_q(q, p):
    e = np.maximum(q * p / (eps + q), 1e-6)
    ln = np.log(e / 611.2)
    return 273.15 + 243.5 * ln / (17.67 - ln)


# ----------------------------------------------------------------------------
# 1. 环境廓线
# ----------------------------------------------------------------------------
Z_TOP = 16000.0
dz = 5.0
z = np.arange(0.0, Z_TOP + dz, dz)
nz = z.size


def build_env():
    """位温分段：混合层 -> 逆温盖 -> 条件不稳定 -> 平流层
    参数经 tune_sounding.py 扫描确定，命中 CAPE 3000 / CIN 12 / LFC 1.3 km"""
    CAP_JUMP, CAP_TOP, TH_END, Q_MID_TOP = 5.5, 1100.0, 339.0, 4.5e-3
    th = np.zeros_like(z)
    for i, zz in enumerate(z):
        if zz <= 700.0:
            th[i] = 300.0                                          # 混合层
        elif zz <= CAP_TOP:
            th[i] = 300.0 + CAP_JUMP * (zz - 700.0) / (CAP_TOP - 700.0)   # 逆温盖
        elif zz <= 2200.0:
            th[i] = 300.0 + CAP_JUMP + 1.2 * (zz - CAP_TOP) / (2200.0 - CAP_TOP)
        elif zz <= 11000.0:
            th[i] = (300.0 + CAP_JUMP + 1.2) + (TH_END - 300.0 - CAP_JUMP - 1.2) * \
                    (zz - 2200.0) / 8800.0
        else:
            th[i] = TH_END + 42.0 * (zz - 11000.0) / 5000.0

    p = np.zeros_like(z)
    p[0] = 100000.0
    for i in range(1, nz):
        Tm = 0.5 * (T_from_theta(th[i - 1], p[i - 1]) + T_from_theta(th[i], p[i - 1]))
        p[i] = p[i - 1] * np.exp(-g * dz / (Rd * Tm))
    T = T_from_theta(th, p)

    # 水汽：混合层 17 g/kg，逆温盖内快速减湿（干盖），其上按 RH 给
    qv = np.zeros_like(z)
    for i, zz in enumerate(z):
        if zz <= 700.0:
            qv[i] = 17.0e-3
        elif zz <= 1300.0:
            qv[i] = 17.0e-3 * (1.0 - 0.62 * (zz - 700.0) / 600.0)
        elif zz <= 3000.0:
            qv[i] = 6.46e-3 + (Q_MID_TOP - 6.46e-3) * (zz - 1300.0) / 1700.0
        else:
            qv[i] = 0.45 * qsat(T[i], p[i])

    # 初始场不得过饱和（混合层顶 q=17 g/kg 会超过该高度饱和值）
    qv = np.minimum(qv, 0.95 * qsat(T, p))
    rho = p / (Rd * T * (1.0 + 0.608 * qv))
    RH = qv / qsat(T, p)
    return dict(z=z, p=p, T=T, theta=th, qv=qv, rho=rho, RH=RH)


# ----------------------------------------------------------------------------
# 2. 湿绝热线积分
# ----------------------------------------------------------------------------
def dTdp_moist(T, p, q):
    """湿绝热 dT/dp（T,p 为标量或数组）"""
    qs = qsat(T, p)
    num = (Rd * T / p) * (1.0 + Lv * qs / (Rd * T))
    den = cp + (Lv * Lv * qs * eps) / (Rd * T * T)
    return num / den


def parcel(env, z_src=0.0):
    zz, pp, Te, qe = env['z'], env['p'], env['T'], env['qv']
    i0 = int(round(z_src / dz))
    Ts, ps, qs0 = Te[i0], pp[i0], qe[i0]

    # LCL：Bolton 1980 露点温度公式 + 干绝热抬升
    Td = dewpoint_from_q(qs0, ps)
    T_lcl = 1.0 / (1.0 / (Td - 56.0) + np.log(Ts / Td) / 800.0) + 56.0
    z_lcl = z_src + (Ts - T_lcl) / (g / cp)

    Tp = np.full_like(zz, np.nan)
    qp = np.full_like(zz, np.nan)

    # 干绝热段
    for i in range(i0, nz):
        if zz[i] <= z_lcl:
            Tp[i] = Ts * (pp[i] / ps) ** (Rd / cp)
            qp[i] = qs0
        else:
            break
    i_lcl = min(np.searchsorted(zz, z_lcl), nz - 1)
    Tp[i_lcl - 1] = T_lcl
    qp[i_lcl - 1] = qsat(T_lcl, pp[i_lcl - 1])

    # 湿绝热段：RK4 积分 dT/dp，向上（dp < 0）
    T = T_lcl
    p_prev = pp[i_lcl - 1]
    for i in range(i_lcl, nz):
        p_now = pp[i]
        nsub = 2
        dp = (p_now - p_prev) / nsub
        for _ in range(nsub):
            k1 = dTdp_moist(T, p_prev, None)
            k2 = dTdp_moist(T + 0.5 * dp * k1, p_prev + 0.5 * dp, None)
            k3 = dTdp_moist(T + 0.5 * dp * k2, p_prev + 0.5 * dp, None)
            k4 = dTdp_moist(T + dp * k3, p_prev + dp, None)
            T += dp * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            p_prev += dp
        Tp[i] = T
        qp[i] = qsat(T, p_now)

    # 虚温（直接用 T，不再过一遍 Exner）
    thv_e = Te * (1 + 0.608 * qe)
    thv_p = Tp * (1 + 0.608 * qp)
    B = g * (thv_p - thv_e) / thv_e
    B[:i0] = 0.0

    # CAPE / CIN 判定：
    #   LFC = 与主正浮力层（含最大浮力点）连通的底部高度，避免把 LCL 处的微小
    #         正浮力尖峰误判为 LFC
    #   EL  = 主正浮力层顶部之上第一个负浮力高度
    cape = 0.0
    cin = 0.0
    lfc = np.nan
    el = np.nan
    above = B[i0:]
    if above.max() > 0:
        i_max = int(np.argmax(above))
        i_base = i_max
        while i_base > 0 and above[i_base - 1] > 0:
            i_base -= 1
        i_top = i_max
        while i_top < above.size - 1 and above[i_top + 1] > 0:
            i_top += 1
        i_base += i0
        i_top += i0
        lfc = zz[i_base]
        el = zz[i_top] if i_top < nz - 1 else np.nan
        cin = float(np.trapz(np.clip(B[i0:i_base + 1], None, 0), zz[i0:i_base + 1]))
        cape = float(np.trapz(np.clip(B[i_base:i_top + 1], 0, None), zz[i_base:i_top + 1]))
    return dict(z=zz, Tp=Tp, qp=qp, B=B, CAPE=cape, CIN=cin, LCL=z_lcl, LFC=lfc, EL=el,
                CAPE_dilute=False)


if __name__ == '__main__':
    env = build_env()
    pr = parcel(env)
    print('=== 地面 ===')
    print(f"  p = {env['p'][0]/100:.1f} hPa, T = {env['T'][0]-273.15:.2f} C, "
          f"q = {env['qv'][0]*1000:.2f} g/kg, RH = {env['RH'][0]*100:.0f}%, "
          f"Td = {dewpoint_from_q(env['qv'][0], env['p'][0])-273.15:.2f} C")
    print('=== 气块法 ===')
    print(f"  LCL = {pr['LCL']:.0f} m   LFC = {pr['LFC']:.0f} m   EL = {pr['EL']:.0f} m")
    print(f"  CAPE = {pr['CAPE']:.0f} J/kg   CIN = {pr['CIN']:.0f} J/kg")
    print('=== 廓线与浮力 ===')
    print(f"{'z(m)':>7}{'p(hPa)':>9}{'T(C)':>8}{'theta':>8}{'q(g/kg)':>9}{'rho':>8}{'B(m/s2)':>10}")
    for h in [0, 140, 600, 900, 1000, 1200, 1300, 1500, 2000, 3000, 4000, 6000, 8000, 10000]:
        i = int(h / dz)
        print(f"{h:7.0f}{env['p'][i]/100:9.1f}{env['T'][i]-273.15:8.2f}{env['theta'][i]:8.2f}"
              f"{env['qv'][i]*1000:9.2f}{env['rho'][i]:8.4f}{pr['B'][i]:10.4f}")
    np.savez('sounding.npz', **env)
    print('\n  sounding.npz written')
