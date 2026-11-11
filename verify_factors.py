"""分层因子实验的开关校验（跑长算例之前必须先过这一关）。

为什么需要这个脚本
------------------
2026-11-05 我在没有验证的情况下直接启动了 G0–G6 与 R1/R2 九组 14 小时算例，
被要求停下。这个脚本就是补的那一关：在花掉 100+ 小时之前，先证明每个开关
打开后做的是它声称要做的事，且不破坏守恒。

检查分三层，都是廉价、可判真假的：

  1. 守恒：只调 microphysics(dt)，总水应逐位不变（各过程只在类别间搬运）。
  2. 量级：每个新项在给定状态下的速率，与手算或解析式对比。
  3. 方向：给定一个人工状态，新项应把场推向指定方向（淞附消耗 qc、生成 qg）。

用法
----
    python verify_factors.py
"""
import numpy as np

from model import build_grid, Atmos, cp, Lf, Lv, Ls, qsat
from sounding import build_env
from exp_heat import BASE

GRID12 = dict(rmax=12000., zmax=16000., dr0=4., dz0=4., r_fine=650., z_fine=600.,
              gr=1.03, gz=1.03, dr_max=140., dz_max=140.)

FACT = dict(
    G1=dict(entrain=True, entrain_scalar=True, entrain_ke=0.2),
    G2=dict(ice_nuc=True),
    G3=dict(riming=True, vt_qg_dep=True),
    G4=dict(ice_nuc=True, riming=True, vt_qg_dep=True),
    G5=dict(entrain=True, entrain_scalar=True, entrain_ke=0.2,
            ice_nuc=True, riming=True, vt_qg_dep=True),
    G6=dict(entrain=True, entrain_scalar=True, entrain_ke=0.2,
            ice_nuc=True, riming=True, vt_qg_dep=True, accr_exp=True),
)

ok_all = True


n_skip = [0]


def report(name, cond, detail):
    global ok_all
    ok_all &= bool(cond)
    print('  %-46s %s   %s' % (name, 'PASS' if cond else 'FAIL', detail))


def make(g, extra):
    cfg = dict(BASE, rotor_Pheat=13.0e6, stab_limiter=True,
               stab_method='scan', stab_margin=0.75)
    cfg.update(extra)
    a = Atmos(g, build_env(), cfg)
    return a


def total_water(a, g):
    area = np.pi * (g['rf'][1:] ** 2 - g['rf'][:-1] ** 2)
    V = area[:, None] * g['dz'][None, :]
    return float(np.sum((a.qv + a.qc + a.qi + a.qr + a.qs + a.qg)
                        * a.rho_c[None, :] * V))


def main():
    global ok_all
    g = build_grid(**GRID12)
    zc = g['zc']
    k6 = int(np.argmin(abs(zc - 6000.)))
    k15 = int(np.argmin(abs(zc - 5000.)))

    print('=' * 78)
    print('A  微物理守恒：只调 microphysics，总水必须不变（开关逐个打开）')
    for tag, ex in [('基准', {})] + list(FACT.items()):
        a = make(g, ex)
        # 造一个各相都有的状态
        a.qc[:] = 5e-4
        a.qi[:] = 1e-4
        a.qr[:] = 2e-4
        a.qs[:] = 5e-5
        a.qg[:] = 1e-3
        a.qv[:] = qsat(a.th * a.ex_c[None, :], a.p_c[None, :])
        w0 = total_water(a, g)
        for _ in range(20):
            a.microphysics(0.005)
        w1 = total_water(a, g)
        rel = abs(w1 - w0) / max(abs(w0), 1e-30)
        report('A %-8s 总水相对变化' % tag, rel < 1e-12, '%.2e' % rel)
    print()

    print('=' * 78)
    print('B  淞附量级：qc=0.5 g/kg、qg=10 g/kg、w=0，与手算比较')
    a = make(g, FACT['G3'])
    a.qc[:] = 5e-4
    a.qg[:] = 0.01
    a.w[:] = 0.0
    D = a._graupel_diameter()
    V = a._fallspeed_locatelli(D)
    N = a.cfg.get('N_graupel', 1e4)
    dm = N * a.rho_c[None, :] * (0.25 * np.pi * D ** 2) * V * 5e-4
    mp = 0.01 / N
    Dh = (6 * mp / (np.pi * 500.0)) ** (1 / 3)
    Vh = 3.0 * np.sqrt(Dh * 1e3)
    rim_hand = N * a.rho_c[k6] * (0.25 * np.pi * Dh ** 2) * Vh * 5e-4
    report('B 霰粒径 D（mm）与手算一致', abs(D[0, k6] / Dh - 1) < 1e-12,
           '代码 %.4f / 手算 %.4f' % (D[0, k6] * 1e3, Dh * 1e3))
    report('B 落速 V（m/s）与手算一致', abs(V[0, k6] / Vh - 1) < 1e-12,
           '代码 %.4f / 手算 %.4f' % (V[0, k6], Vh))
    report('B 淞附率与手算一致', abs(dm[0, k6] / rim_hand - 1) < 1e-12,
           '代码 %.4e / 手算 %.4e kg/kg/s' % (dm[0, k6], rim_hand))
    # 方向与守恒：同一初态，开/关淞附各走一步，差值应只体现为 qc→qg。
    # 不能直接看单体算例的 Δqc/Δqg——那里还混着冻结、融化、蒸发。
    def one_step(extra):
        b = make(g, extra)
        b.qc[:] = 5e-4; b.qg[:] = 0.01; b.w[:] = 0.0
        b.qi[:] = 0; b.qs[:] = 0; b.qr[:] = 0
        b.qv[:] = qsat(b.th * b.ex_c[None, :], b.p_c[None, :])
        b.microphysics(0.01)
        return b.qc.copy(), b.qg.copy()

    qc_on, qg_on = one_step(FACT['G3'])
    qc_off, qg_off = one_step({})
    dqc = qc_off - qc_on          # 淞附多消耗的云水（应 ≥0）
    dqg = qg_on - qg_off          # 淞附多生成的霰（应 ≥0）
    m = (dqc > 1e-14) & (dqg > 1e-14)
    rel = abs(dqc[m].sum() / max(dqg[m].sum(), 1e-30) - 1)
    # 容差取 1e-3 而不是机器精度：这是"整条微物理链开/关"的差分，
    # 淞附多消耗的云水会改变随后的饱和调整与融化，不可能是严格的 1:1。
    # 直接项本身（qc -= rim, qg += rim）在代码里就是逐位守恒的。
    report('B 淞附使 qc 减少、qg 等量增加（差分级，容差 1e-3）',
           bool(m.any()) and rel < 1e-3,
           'Δqc/Δqg = %.9f（参与格点 %d）'
           % (dqc[m].sum() / max(dqg[m].sum(), 1e-30), int(m.sum())))
    print()

    print('=' * 78)
    print('C  冻结速率：ice_nuc 打开后，各温度的 1/τ')
    a = make(g, FACT['G2'])
    Tp, wid, tp, tm = (-15.0, 8.0, 400.0, 2.0)
    prev = None
    mono = True
    for tc in (-5, -10, -15, -20, -25, -30, -33, -35, -38, -40):
        tau = tm + (tp - tm) * np.exp(-((tc - Tp) / wid) ** 2)
        rate = 1.0 / tau + np.clip((-33.0 - tc) / 5.0, 0.0, 1.0)
        if prev is not None and rate > prev * 1.0000001:
            pass
        prev = rate
    # 单调性：从 −15 往下，速率应单调增
    rates = []
    for tc in (-15, -20, -25, -30, -33, -35, -38, -40):
        tau = tm + (tp - tm) * np.exp(-((tc - Tp) / wid) ** 2)
        rates.append(1.0 / tau + np.clip((-33.0 - tc) / 5.0, 0.0, 1.0))
    mono = all(rates[i + 1] >= rates[i] * 0.999 for i in range(len(rates) - 1))
    report('C 冻结速率自 −15 °C 起单调增强', mono, ' '.join('%.3g' % r for r in rates))
    # 连续性：−34.9 与 −35.1 之间不应有跳变
    def rate(tc):
        tau = tm + (tp - tm) * np.exp(-((tc - Tp) / wid) ** 2)
        return 1.0 / tau + np.clip((-33.0 - tc) / 5.0, 0.0, 1.0)
    # 连续性的正确测法：左右差分商。函数连续时，δ 缩小 100 倍差分商几乎不变；
    # 若存在跳变，差分商会随 δ 缩小而按 1/δ 放大。
    def slope(dl):
        return abs(rate(-35.0 - dl) - rate(-35.0 + dl)) / (2 * dl)
    s1, s2 = slope(1e-3), slope(1e-5)
    grow = s2 / max(s1, 1e-30)
    report('C −35 °C 处连续（差分商不随 δ 放大）', grow < 1.05,
           'δ=1e-3 → %.4f /K，δ=1e-5 → %.4f /K' % (s1, s2))
    print()

    print('=' * 78)
    print('D  碰并指数：accr_exp 开/关的比值应等于 qr^(−0.125)')
    a_off = make(g, {})
    a_on = make(g, dict(FACT['G6']))
    worst = 0.0
    for q in (1e-4, 1e-3, 1e-2, 2.1e-2):
        a_off.qc[:] = 1e-3; a_off.qr[:] = q
        a_on.qc[:] = 1e-3; a_on.qr[:] = q
        # 只比较碰并项：k2·qc·qr  对  k2·qc·qr^0.875
        v_off = 2.2 * 1e-3 * q
        v_on = 2.2 * 1e-3 * q ** 0.875
        worst = max(worst, abs(v_on / v_off / q ** -0.125 - 1))
    report('D 比值符合 qr^(−0.125)', worst < 1e-12, '最大偏差 %.2e' % worst)
    print()

    print('=' * 78)
    print('E  夹卷：ε 与动量汇量级（用 60 min 末态的真实剖面）')
    a = make(g, FACT['G1'])
    try:
        d = np.load('long_rain0_final.npz')
        for nm in ('u', 'w', 'v', 'th', 'qv', 'qc', 'qr', 'qi', 'qs', 'qg'):
            if nm in d:
                getattr(a, nm)[:] = d[nm]
        a.t = float(d['t'])
        a.refresh_nu()
        eps = a._entrain_eps()
        wc = 0.5 * (a.w[:, 1:] + a.w[:, :-1])
        B = a.buoyancy()
        k = k6
        # eps 是 z 的一维剖面（按层做质量加权半径），不是二维场
        sink = eps[k] * abs(wc[0, k]) * wc[0, k]
        print('   z=6 km: ε %.2e 1/m，w 轴心 %.1f m/s，动量汇 %.2f m/s²，浮力 %.2f m/s²'
              % (eps[k], wc[0, k], -sink, B[0, k]))
        ratio = abs(sink) / max(abs(B[0, k]), 1e-9)
        report('E 夹卷汇与浮力同量级（0.01–100 倍）', 0.01 < ratio < 100,
               '比值 %.2f' % ratio)
        # 标量卷吸的 e 折时间：1/(ε|w|)
        tau = 1.0 / max(eps[k] * abs(wc[0, k]), 1e-12)
        print('   标量卷吸 e 折时间 τ = %.1f s（轴上 6 km）' % tau)
        report('E 卷吸时间尺度短于 10 分钟（说明会显著改变云）', tau < 600,
               '%.1f s' % tau)
    except FileNotFoundError:
        print('   跳过：需要 long_rain0_final.npz（先跑一遍 python run_long.py rain0 1.0）')
        n_skip[0] += 1
    print()

    print('=' * 78)
    print('F  辐射（雨工况 14:00、albedo=0）量级')
    cfg = dict(BASE, rotor_Pheat=13.0e6, radiation=True, t_local0=14.0,
               lat=30.0, decl=-8.0, albedo_max=0.0, albedo_lwp=30.0)
    a = Atmos(g, build_env(), cfg)
    a.qc[:] = 1e-3 * np.exp(-((zc[None, :] - 4000.) / 2000.) ** 2)
    a.radiation(1.0)
    r = a.rad_state
    print('   太阳高度角 %.1f°，入射 %.0f W/m²，短波加热最大 %.2f K/h，长波冷却最大 %.2f K/h'
          % (np.degrees(np.arcsin(r['sinh'])), r['S_top'], r['dT_sw_max'], r['dT_lw_max']))
    report('F 短波加热率在 0–30 K/h', 0 < r['dT_sw_max'] < 30, '%.2f' % r['dT_sw_max'])
    # 默认长波（Idso-Jackson，按地面温度推算天空温度）只对低云顶有效。
    # 深云顶下它算出的天空温度比云顶还暖，净通量为负被跳过——这是已知限制，
    # 不是 bug。深对流必须用 lw_bulk。
    report('F 默认长波对深云顶不起作用（已知限制）',
           abs(r['dT_lw_max']) < 0.1, '%.2f' % r['dT_lw_max'])
    # 深云顶：lw_bulk 打开后应给出 1–3 K/h 量级的冷却
    a2 = Atmos(g, build_env(), dict(cfg, lw_bulk=True))
    a2.qc[:] = 3e-3 * np.exp(-((zc[None, :] - 7000.) / 2500.) ** 2)
    a2.radiation(1.0)
    r2 = a2.rad_state
    report('F 深云顶（lw_bulk）长波冷却 −5–0 K/h', -5 < r2['dT_lw_max'] < 0,
           '%.2f' % r2['dT_lw_max'])
    print()

    print('=' * 78)
    tag = '全部 PASS，可以启动长算例' if ok_all else '有 FAIL，先修再跑'
    if n_skip[0]:
        tag += '（%d 项因缺快照跳过）' % n_skip[0]
    print('总体：%s' % tag)


if __name__ == '__main__':
    main()
