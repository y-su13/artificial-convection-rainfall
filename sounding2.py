"""
sounding2.py — 两个新工况的探空

A) 增雨工况：华北平原夏季闷热中午
   地面 34.5 ℃ / 露点 26.5 ℃ / 998 hPa，混合层已发展至约 1.4 km，
   其上弱盖，再往上条件不稳定。目标 CAPE 3500–4500、CIN 20–40、
   LCL 约 1.0 km。比原算例（27 ℃/22 ℃、CAPE 3000、LCL 0.6 km）更热更湿。

B) 除雾工况：四川盆地秋季浓雾早晨
   地面 13 ℃、饱和、盆地气压 960 hPa，雾层 0–350 m，
   雾顶强逆温（+7 K / 250 m），其上为下沉干层。这是完全不同的物理：
   没有对流，要回答的是"能不能把雾层清出一个洞、多大、多快"。

两个工况共用同一套 z 网格（dz = 5 m），便于直接用 model.py 的 Atmos。
"""
import numpy as np

Rd, Rv, cp, g, Lv = 287.04, 461.5, 1004.5, 9.80665, 2.5e6
p0 = 100000.0
eps = Rd / Rv
Z_TOP, dz5 = 16000.0, 5.0
z = np.arange(0.0, Z_TOP + dz5, dz5)


def esat(T):
    Tc = np.clip(T - 273.15, -80.0, 60.0)
    return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))


def qsat(T, p):
    es = esat(T)
    return eps * es / np.maximum(p - es, 1.0)


def exner(p):
    return (p / p0) ** (Rd / cp)


def hydrostatic(th, p_sfc):
    """由位温廓线静力积分得到 p、T"""
    p = np.zeros_like(z)
    p[0] = p_sfc
    for i in range(1, z.size):
        T1 = th[i - 1] * exner(p[i - 1])
        T2 = th[i] * exner(p[i - 1])
        Tm = 0.5 * (T1 + T2)
        p[i] = p[i - 1] * np.exp(-g * dz5 / (Rd * Tm))
    T = th * exner(p)
    return p, T


# ------------------------------------------------------------------ A 华北平原
def build_nc(cap_jump=5.0, cap_top=1650.0, lapse_mid=7.0, z_trop=11000.0,
             T_sfc_C=34.5, Td_sfc_C=26.5, ml_top=900.0):
    """华北平原夏季闷热中午（有盖，尚未触发）

    混合层必须按**虚位温**混合，不能按位温混合。真实对流边界层由浮力湍流
    搅拌，被抹平的是 theta_v。若按 theta 抹平，地面气块的 theta_v 会比环境高
    约 0.6 K，一抬就正浮力，LFC 落到 5 m、CIN 归零——那等于对流已经发生。

    盖以上按**环境温度递减率**给，不按位温线性插值。原因是位温线性插值在
    对流层中上层会无意中接近湿绝热线，使气块一路正浮力到模式顶（实测
    CAPE 超 7000、EL 顶到 16 km），那是假的。
    """
    T_SFC = T_sfc_C + 273.15
    TD_SFC = Td_sfc_C + 273.15
    P_SFC = 99800.0
    ML_TOP = ml_top
    Q_SFC = qsat(TD_SFC, P_SFC)
    Q_TOP = 0.86 * Q_SFC

    th_sfc = T_SFC * (p0 / P_SFC) ** (Rd / cp)
    thv_ml = th_sfc * (1.0 + 0.608 * Q_SFC)
    q_ml = Q_SFC - (Q_SFC - Q_TOP) * np.clip(z / ML_TOP, 0, 1)

    nz = z.size
    th = np.zeros(nz)
    p = np.zeros(nz)
    T = np.zeros(nz)
    i_ml = int(ML_TOP / dz5)
    i_cap = int(cap_top / dz5)

    # --- 段 1：混合层，theta_v 恒定，静力积分
    for i in range(0, i_ml + 1):
        th[i] = thv_ml / (1.0 + 0.608 * q_ml[i])
    p[0] = P_SFC
    for i in range(1, i_ml + 1):
        Tm = 0.5 * (th[i - 1] + th[i]) * exner(p[i - 1])
        p[i] = p[i - 1] * np.exp(-g * dz5 / (Rd * Tm))
    T[:i_ml + 1] = th[:i_ml + 1] * exner(p[:i_ml + 1])

    # --- 段 2：盖上逆温，位温线性加 cap_jump
    cap_base = th[i_ml]
    for i in range(i_ml + 1, i_cap + 1):
        th[i] = cap_base + cap_jump * (z[i] - ML_TOP) / (cap_top - ML_TOP)
    for i in range(i_ml + 1, i_cap + 1):
        Tm = 0.5 * (T[i - 1] + th[i] * exner(p[i - 1]))
        p[i] = p[i - 1] * np.exp(-g * dz5 / (Rd * Tm))
        T[i] = th[i] * exner(p[i])

    # --- 段 3：以上按指定环境温度递减率，直到对流层顶
    i_tr = int(z_trop / dz5)
    for i in range(i_cap + 1, nz):
        lapse = lapse_mid if z[i] <= z_trop else 1.0
        T[i] = T[i - 1] - lapse * dz5 / 1000.0
        Tm = 0.5 * (T[i - 1] + T[i])
        p[i] = p[i - 1] * np.exp(-g * dz5 / (Rd * Tm))
        th[i] = T[i] / exner(p[i])

    qv = np.zeros(nz)
    for i in range(nz):
        zz = z[i]
        if zz <= ML_TOP:
            qv[i] = q_ml[i]
        elif zz <= 2800.0:
            qv[i] = Q_TOP * (1.0 - 0.74 * (zz - ML_TOP) / (2800.0 - ML_TOP))
        else:
            qv[i] = 0.40 * qsat(T[i], p[i])
    qv = np.minimum(qv, 0.97 * qsat(T, p))

    rho = p / (Rd * T * (1.0 + 0.608 * qv))
    env = dict(z=z, p=p, T=T, theta=th, qv=qv, rho=rho, RH=qv / qsat(T, p))
    env['name'] = 'nc'
    env['SHF'], env['LHF'] = 180.0, 260.0
    env['flux_depth'] = ml_top
    env['qc0'] = np.zeros_like(z)
    return env


# ------------------------------------------------------------------ B 四川盆地
def build_fog():
    """四川盆地秋季浓雾早晨

    位温分段
      0–50 m     近地面薄层
      50–350     雾层，弱稳定
      350–600    雾顶强逆温（辐射冷却 + 下沉增温），+7 K
      600–1500   逆温层以上的干层
      1500 以上  稳定层结

    雾含水量：0–350 m，峰值在 180 m 附近，最大 0.28 g/kg
    （对应液态水 0.35 g/m³，能见度约 80–120 m，属浓雾）
    """
    T_SFC = 13.0 + 273.15
    P_SFC = 96000.0              # 盆地底约 500 m 海拔
    FOG_TOP = 350.0
    INV_TOP = 600.0

    th_sfc = T_SFC * (p0 / P_SFC) ** (Rd / cp)
    th = np.zeros_like(z)
    for i, zz in enumerate(z):
        if zz <= 50.0:
            th[i] = th_sfc
        elif zz <= FOG_TOP:
            th[i] = th_sfc + 0.8 * (zz - 50.0) / (FOG_TOP - 50.0)
        elif zz <= INV_TOP:
            th[i] = th_sfc + 0.8 + 7.0 * (zz - FOG_TOP) / (INV_TOP - FOG_TOP)
        elif zz <= 1500.0:
            th[i] = th_sfc + 7.8 + 3.5 * (zz - INV_TOP) / (1500.0 - INV_TOP)
        elif zz <= 11000.0:
            th[i] = (th_sfc + 7.8 + 3.5) + (345.0 - th_sfc - 11.3) * \
                    (zz - 1500.0) / 9500.0
        else:
            th[i] = 345.0 + 40.0 * (zz - 11000.0) / 5000.0

    p, T = hydrostatic(th, P_SFC)

    # 雾层内水汽饱和；雾顶以上快速变干（下沉干层）
    qv = np.zeros_like(z)
    for i, zz in enumerate(z):
        if zz <= FOG_TOP:
            qv[i] = qsat(T[i], p[i])
        elif zz <= INV_TOP:
            qv[i] = 0.50 * qsat(T[i], p[i])
        elif zz <= 1500.0:
            qv[i] = 0.34 * qsat(T[i], p[i])
        else:
            qv[i] = 0.28 * qsat(T[i], p[i])

    # 雾含水量（液态）：峰值 0.28 g/kg 在 180 m
    qc = np.zeros_like(z)
    m = z <= FOG_TOP
    qc[m] = 0.28e-3 * np.sqrt(np.clip(1.0 - ((z[m] - 180.0) / 320.0) ** 2, 0.0, 1.0))
    qc[z < 15.0] *= 0.5          # 贴地层含雾量略低（地面附近有薄干层）

    rho = p / (Rd * T * (1.0 + 0.608 * qv - qc))
    env = dict(z=z, p=p, T=T, theta=th, qv=qv, rho=rho, RH=qv / qsat(T, p))
    env['name'] = 'fog'
    env['SHF'], env['LHF'] = 0.0, 0.0    # 雾顶辐射冷却，地面通量近于零
    env['flux_depth'] = 100.0
    env['qc0'] = qc
    # 雾团是有限范围的（2026-11-05 修）。此前全域铺满雾，风机的低层抽吸
    # 永远有饱和空气补充，测出来的不是"除雾"。取外缘 10 km、过渡 1 km
    # （9–10 km 内雾量平滑降到 0），10 km 以外 qv 压到 95% 饱和、无液水。
    env['fog_r'] = 10000.0
    env['fog_edge'] = 1000.0
    env['stop_sens'] = 0.0
    return env


BUILDERS = dict(nc=build_nc, fog=build_fog)


if __name__ == '__main__':
    from sounding import parcel as parcel_ref
    from sounding import dewpoint_from_q

    for name in ('nc', 'fog'):
        env = BUILDERS[name]()
        print('=' * 66)
        print('工况 %s' % name)
        print('  地面: p=%.0f hPa, T=%.1f C, Td=%.1f C, q=%.2f g/kg, RH=%.0f%%' % (
            env['p'][0] / 100, env['T'][0] - 273.15,
            dewpoint_from_q(env['qv'][0], env['p'][0]) - 273.15,
            env['qv'][0] * 1000, env['RH'][0] * 100))
        if name == 'fog':
            print('  雾含水量峰值 %.3f g/kg（约 %.2f g/m³）' % (
                env['qc0'].max() * 1000, env['qc0'].max() * env['rho'][0] * 1000))
        pr = parcel_ref(env)
        print('  LCL=%.0f m  LFC=%s  EL=%s  CAPE=%.0f  CIN=%.1f' % (
            pr['LCL'],
            '%.0f' % pr['LFC'] if np.isfinite(pr['LFC']) else '无',
            '%.0f' % pr['EL'] if np.isfinite(pr['EL']) else '无',
            pr['CAPE'], -pr['CIN']))
        print('  %6s %9s %8s %9s %9s' % ('z(m)', 'p(hPa)', 'T(C)', 'theta', 'q(g/kg)'))
        for h in [0, 100, 180, 350, 500, 600, 900, 1000, 1400, 1700, 2000, 3000, 4000]:
            i = int(h / dz5)
            print('  %6d %9.1f %8.2f %9.2f %9.2f' % (
                h, env['p'][i] / 100, env['T'][i] - 273.15, env['theta'][i],
                env['qv'][i] * 1000))
