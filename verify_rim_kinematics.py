"""验证淞附收集核的运动学定义：两条支路的相对速度都不应显式依赖 w。

背景（2026-11-10 审计指出、已修正）
------------------------------------
原代码两条支路都写错：

    relg = np.abs(Vg - wc_)      # 霰
    rels = np.abs(Vs - wc_)      # 雪

云滴随空气运动，地面系速度 = w；霰/雪是 w − V，两者之差 = V，与 w 无关。
把 w 减进去，等于把空气的绝对上升速度混进了粒子间相对速度。
w=30 m/s、V=3 m/s 时算成 27，淞附率被放大 9 倍；w=70 时放大 23 倍。

为什么原有验证没抓到
--------------------
手算对比取 w=0，此时 |V − 0| = V，公式退化为正确形式，测试必然通过。
**任何只在一个 w 值上做的验证，都测不出这个错。**

本脚本的做法
------------
同初态、同 dt，riming 开/关各走一步，差值就是该支路的淞附贡献。
两条支路分别测：测霰时把 qs 置零，测雪时把 qg 置零，互不干扰。
每条支路在 w = 0 / +10 / +30 / −10 四个值下各算一次，比值必须全为 1。

运行
----
    python verify_rim_kinematics.py

期望输出：两组各四行比值全为 1.0000。
若任一行不为 1，说明该支路的相对速度里又混进了 w。
"""
import numpy as np
from model import build_grid, Atmos, qsat
from sounding import build_env
from exp_heat import BASE

GRID12 = dict(rmax=12000., zmax=16000., dr0=4., dz0=4., r_fine=650.,
              z_fine=600., gr=1.03, gz=1.03, dr_max=140., dz_max=140.)

WSETS = (0.0, 10.0, 30.0, -10.0)


def measure(g, k, kind, wset):
    """返回该支路的淞附贡献 Δqg（kg/kg），单位步长。"""
    def run(rim):
        a = Atmos(g, build_env(), dict(BASE, rotor_Pheat=13.0e6, riming=rim))
        a.qc[:] = 3e-3              # 足够大，不被 min(qcp, ...) 截断
        a.qi[:] = 0.0
        a.qr[:] = 0.0
        if kind == 'graupel':
            a.qg[:] = 0.01
            a.qs[:] = 0.0           # 关掉雪支路
        else:
            a.qg[:] = 0.0           # 关掉霰支路
            a.qs[:] = 0.01
        a.qv[:] = qsat(a.th * a.ex_c, a.p_c) * 1.02   # 略过饱和，抑制蒸发
        a.w[:] = wset
        qg0 = a.qg.copy()
        a.microphysics(2e-4)
        return float((a.qg - qg0)[0, k])

    return run(True) - run(False)


def main():
    g = build_grid(**GRID12)
    a0 = Atmos(g, build_env(), dict(BASE, rotor_Pheat=13.0e6))
    T0 = a0.th * a0.ex_c[None, :]
    k = int(np.argmin(np.abs(T0[0, :] - 250.0)))
    print('测试层 z = %.0f m，T = %.1f °C' % (g['zc'][k], T0[0, k] - 273.15))
    print()

    all_ok = True
    for kind, label in (('graupel', '霰支路（qg 收集）'), ('snow', '雪支路（qs 收集）')):
        print('--- %s ---' % label)
        print('%-8s %22s %10s' % ('w (m/s)', '淞附贡献 Δqg (kg/kg)', '相对 w=0'))
        base = None
        ratios = []
        for wset in WSETS:
            d = measure(g, k, kind, wset)
            if base is None:
                base = d
            r = d / base if base else float('nan')
            ratios.append(r)
            print('%-8.1f %22.4e %10.4f' % (wset, d, r))
        ok = all(abs(r - 1.0) < 1e-12 for r in ratios)
        all_ok &= ok
        print('  判定：%s（最大偏差 %.2e）'
              % ('PASS' if ok else 'FAIL', max(abs(r - 1.0) for r in ratios)))
        print()

    print('=' * 58)
    print('总判定：%s' % ('PASS（两条支路的相对速度都不显式依赖 w）' if all_ok else 'FAIL'))
    print('修正前：w=30 时霰支路会放大约 7–9 倍、雪支路同理（Vs 量级更小，倍数更大）。')
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
