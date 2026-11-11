"""产热档敏感性：P_heat 到底影不影响结论

第八轮能量审计的结论是：致动盘实际交付的机械功率 P_work ≈ 117 MW（对实际施加
的力积分），而代码里 P_heat = P_shaft − ½ṁw̄² 用了错的动能代理（½ṁw̄² = 24 MW，
实际扭矩做功是 117 MW），所以 41.7 MW 这个产热值是错的。

错多少会改变结论？本脚本把 P_heat 当**输入**扫三档（这一段已经由 model.py 的
rotor_Pheat 支持），其余全同：

  h0     P_heat =   0 MW   纯动量，无废热（下界）
  h13    P_heat =  13 MW   与 P_work 自洽的预算（轴功率 130 MW、损失 10%）
  h42    P_heat =41.7 MW   现有算例的值（对照，应与 m_on 一致）

用法
----
python exp_heat.py run h0 / h13 / h42
python exp_heat.py cmp h0 h13 h42
"""
import json
import sys
import time

import numpy as np

from model import build_grid, Atmos
from sounding import build_env
from exp_cyl import metrics, GRID, compare

BASE = dict(nu_min=0.5, nu_max=160., sponge_z=0.40, sponge_r=0.15, tau_sponge=30.,
            SHF=120., LHF=100., flux_depth=700., qc0=.7e-3, k_auto=1e-3, k_accr=2.2,
            rotor_on=True, rotor_R=150., rotor_z=140., rotor_thick=24., rotor_hub=15.,
            rotor_tipfrac=.10, w_target=17.0, rotor_tau=1.0, swirl_ratio=.15,
            rotor_tau_v=4.0, rotor_heat=True, rotor_P=66e6, nu_wake=50.,
            cfl=0.30, cyl_fix=True, var_nu_fix=True, metric_fix=True)

CASES = {
    'h0':  (0.0, '无废热（下界）'),
    'h13': (13.0e6, '与 P_work 自洽的预算（轴功率 130 MW，损失 10%）'),
    'h42': (41.73e6, '现有算例（P_shaft−½ṁw̄²）'),
}
MIN = 8.0


def fingerprint():
    """代码指纹：把 model.py 的哈希写进结果，避免再搞混"哪组用了哪版代码"。

    第五、六轮的教训是同一个：改了代码但没标记，事后无法判断两组算例
    是否可比。哈希比手写版本号可靠。
    """
    import hashlib
    h = hashlib.md5(open('model.py', 'rb').read()).hexdigest()[:12]
    return h


def run(name):
    ph, desc = CASES[name]
    g = build_grid(**GRID)
    cfg = dict(BASE, rotor_Pheat=ph)
    a = Atmos(g, build_env(), cfg)
    t0 = time.time()
    target = MIN * 60.
    n = 0
    while a.t < target:
        a.step(min(a.max_dt(), target - a.t))
        n += 1
    m = metrics(a, g)
    m.update(name=name, desc=desc, minutes=MIN, steps=n, wall_s=time.time() - t0,
             P_heat=float(a._P_heat), dth=float(a._heat_dth),
             P_shaft='nan(输出量)', model_md5=fingerprint())
    json.dump(m, open('heat_%s.json' % name, 'w'), ensure_ascii=False, indent=1)
    np.savez('heat_%s.npz' % name, qc=a.qc, w=a.w, u=a.u, v=a.v, th=a.th,
             qv=a.qv, qr=a.qr, rc=g['rc'], zc=g['zc'], dr=g['dr'], dz=g['dz'])
    print('%s  %.0fs  %d 步  P_heat=%.1f MW  Δθ/dt=%.4f K/s  %s' %
          (name, time.time() - t0, n, m['P_heat'] / 1e6, m['dth'], desc))
    print('  云顶 %.0f  qc_max %.2f  w_max %.2f  盘面w %.2f  雨强 %.3f' %
          (m['ctop'], m['qc_max'], m['w_max'], m['w_disc'], m['rain_local']))
    print('  通量 m³/s: ' + '  '.join('%d:%.2e' % (k, v) for k, v in m['flux_prof'].items()))


def cmp():
    D = {}
    for n in CASES:
        p = 'heat_%s.json' % n
        try:
            D[n] = json.load(open(p, encoding='utf-8'))
        except FileNotFoundError:
            print('缺 %s' % n)
    if len(D) < 2:
        return
    keys = [('ctop', '云顶'), ('qc_max', 'qc_max'), ('w_max', 'w_max'),
            ('w_disc', '盘面w'), ('rain_local', '雨强')]
    print('%-8s' % '组' + ''.join('%10s' % t for _, t in keys))
    for n, d in D.items():
        print('%-8s' % n + ''.join('%10.3f' % d[k] for k, _ in keys))
    print()
    hs = ['200', '400', '600', '900', '1200', '1500']
    print('%-8s' % '组' + ''.join('%12s m' % h for h in hs))
    for n, d in D.items():
        print('%-8s' % n + ''.join('%12.2e' % d['flux_prof'].get(h, 0) for h in hs))
    ks = list(D)
    for i in range(len(ks) - 1):
        x, y = D[ks[i]], D[ks[i + 1]]
        print('\n%s → %s' % (ks[i], ks[i + 1]))
        for k, t in keys:
            if abs(x[k]) > 1e-12:
                print('   %-10s %+7.1f%%' % (t, 100 * (y[k] - x[k]) / x[k]))
        for h in hs:
            a_, b_ = x['flux_prof'].get(h, 0), y['flux_prof'].get(h, 0)
            if abs(a_) > 1e-12:
                print('   通量@%-5sm %+7.1f%%' % (h, 100 * (b_ - a_) / a_))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
    elif sys.argv[1] == 'run':
        run(sys.argv[2])
    elif sys.argv[1] == 'cmp':
        cmp()
