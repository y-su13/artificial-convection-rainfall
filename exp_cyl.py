"""柱坐标修正 & 时间步/涡粘分离试验

背景（第三轮审计意见指出的问题）
--------------------------------
原 A/B/C 三组用 `--cfl` 与 `--nuw` 调参，但 dt 是由 min(CFL 限, 粘性限) 自动选出的，
cfl=0.30 时两个限都算出 0.063，于是任何单参数调节都同时改动 CFL 数与扩散数，
无法把差异归因到某一个。

更根本的一点：同一网格上

    CFL 数 / 扩散数  =  v_max·Δx / ν

这个比值不含 dt。所以「固定网格 + 只改 dt」永远让两个数同比缩放，
只有「只改 ν」才能固定 CFL 数、单独改扩散数。

因此本脚本设两组轴：
  dt 轴   —— 两个数同比变（总时间步效应）
  nu 轴   —— CFL 数不变，只有扩散数变（数值阻尼 + 物理涡粘效应）

用法
----
python exp_cyl.py list
python exp_cyl.py run g_old        # 旧版几何
python exp_cyl.py run g_half       # 只补 (1/r)∂u/∂r（已知错误，留作证据）
python exp_cyl.py run g_full       # 全部修正
python exp_cyl.py run dt063 / dt040 / nu25
"""
import json
import os
import sys
import time

import numpy as np

from model import build_grid, Atmos
from sounding import build_env

MIN = 8.0               # 物理时长（分钟）
GRID = dict(rmax=8000., zmax=16000., dr0=4., dz0=4., r_fine=650., z_fine=600.,
            gr=1.03, gz=1.03, dr_max=140., dz_max=140.)

BASE = dict(nu_min=0.5, nu_max=160., sponge_z=0.40, sponge_r=0.15, tau_sponge=30.,
            SHF=120., LHF=100., flux_depth=700., qc0=.7e-3, k_auto=1e-3, k_accr=2.2,
            rotor_on=True, rotor_R=150., rotor_z=140., rotor_thick=24., rotor_hub=15.,
            rotor_tipfrac=.10, w_target=17.0, rotor_tau=1.0, swirl_ratio=.15,
            rotor_tau_v=4.0, rotor_heat=True, rotor_P=66e6, nu_wake=50.)

# 名字 -> (cfg 覆盖, 说明)
# 注意：第四轮跑的四组（g_old/g_full/nu25/dt040）是在 metric_fix 加入**之前**
# 启动的，用的是"拉伸段二阶导仍用均匀公式"的代码。它们的 json 里没有 metric_fix
# 字段，可据此区分。第五轮的 m_* 两组用新代码，用于量化网格度量的影响。
CASES = {
    # --- 几何修正三态（其余全同）---
    'g_old':  (dict(cfl=0.30, cyl_fix=False, var_nu_fix=False),
               '旧版：笛卡尔 Laplacian，无几何项'),
    'g_half': (dict(cfl=0.30, cyl_fix=True,  var_nu_fix=False),
               '只补 (1/r)∂u/∂r 与 −νv/r²，缺 −u/r² 与 (∇ν)·∇u'),
    'g_full': (dict(cfl=0.30, cyl_fix=True,  var_nu_fix=True),
               '完整：矢量拉普拉斯（含 −u/r²）+ (∇ν)·∇u'),

    # --- 网格度量轴（第五轮新增）：拉伸段二阶导是否用非均匀公式 ---
    'm_off':  (dict(cfl=0.30, cyl_fix=True, var_nu_fix=True, metric_fix=False),
               '网格度量未修：拉伸段二阶导用均匀公式（近似旧版，非逐位一致）'),
    'm_on':   (dict(cfl=0.30, cyl_fix=True, var_nu_fix=True, metric_fix=True),
               '网格度量已修：非均匀二阶导 + 正确面间距'),

    # --- dt 轴：两个数同比缩放（耦合轴）---
    'dt063':  (dict(dt_fixed=0.063, cyl_fix=True, var_nu_fix=True),
               '固定 dt=0.063（与自动步相当）'),
    'dt040':  (dict(dt_fixed=0.040, cyl_fix=True, var_nu_fix=True),
               '固定 dt=0.040，CFL 数与扩散数同比减 37%'),
    'dt025':  (dict(dt_fixed=0.025, cyl_fix=True, var_nu_fix=True),
               '固定 dt=0.025，两数同比减 60%'),

    # --- nu 轴：CFL 数不变，只有扩散数变（干净轴）---
    'nu25':   (dict(dt_fixed=0.063, nu_wake=25., cyl_fix=True, var_nu_fix=True),
               'dt 固定，ν_wake 减半 → 只有扩散数减半'),
    'nu100':  (dict(dt_fixed=0.063, nu_wake=100., cyl_fix=True, var_nu_fix=True),
               'dt 固定，ν_wake 加倍 → 只有扩散数加倍'),
}


def metrics(a, g):
    zc, rc = g['zc'], g['rc']
    q = a.qc > 1e-5
    ctop = float(zc[np.where(q.any(axis=0))[0].max()]) if q.any() else 0.0
    wc = 0.5 * (a.w[:, 1:] + a.w[:, :-1])
    k = int(np.argmin(np.abs(zc - 140)))
    sel = (rc >= 15) & (rc <= 150)
    w_disc = float(wc[sel, k].mean())
    # 积分质量通量的整条垂直分布（替代单点 w_max）
    area = 2 * np.pi * rc * g['dr']
    prof = {}
    for h in (200, 400, 600, 900, 1200, 1500):
        j = int(np.argmin(np.abs(zc - h)))
        prof[h] = float(np.sum(np.maximum(wc[:, j], 0) * area))
    qr = a.qr
    Vt = np.clip(3. + 80. * np.sqrt(np.maximum(qr[:, 0], 0.)), 3., 12.)
    loc = a.rho_c[0] * qr[:, 0] * Vt * 3600.
    bud = a.budget_report()
    return dict(ctop=ctop, qc_max=float(a.qc.max()) * 1e3,
                ice_max=float((a.qi + a.qs + getattr(a, 'qg', np.zeros_like(a.qc))).max()) * 1e3,
                w_max=float(a.w.max()), v_max=float(np.abs(a.v).max()),
                w_disc=w_disc, rain_local=float(loc.max()),
                rain_area=float(np.sum(area[loc > 0.1]) / 1e6),
                flux_prof=prof,
                q_rel=abs(float(bud['q_rel'])), e_rel=abs(float(bud['e_rel'])),
                lim={k2: (v if isinstance(v, str) else float(v))
                     for k2, v in a._lim.items()})


def run(name):
    cfg_over, desc = CASES[name]
    g = build_grid(**GRID)
    cfg = dict(BASE, **cfg_over)
    a = Atmos(g, build_env(), cfg)
    t0 = time.time()
    target = MIN * 60.
    n = 0
    while a.t < target:
        a.step(min(a.max_dt(), target - a.t))
        n += 1
    m = metrics(a, g)
    m['name'] = name
    m['desc'] = desc
    m['minutes'] = MIN
    m['steps'] = n
    m['wall_s'] = time.time() - t0
    m['cfg'] = {k: v for k, v in cfg.items() if isinstance(v, (int, float, bool, str))}
    # 代码指纹：确认两个开关在 cfg 里，避免再次搞混"哪组用哪版代码"
    m['switches'] = {k: cfg.get(k, 'absent')
                     for k in ('metric_fix', 'cyl_fix', 'var_nu_fix')}
    json.dump(m, open('exp_%s.json' % name, 'w'), ensure_ascii=False, indent=1)
    np.savez('exp_%s.npz' % name, qc=a.qc, w=a.w, u=a.u, v=a.v, th=a.th,
             qv=a.qv, qr=a.qr, rc=g['rc'], zc=g['zc'], dr=g['dr'], dz=g['dz'])

    L = m['lim']
    print('%s  墙钟 %.0fs  %d 步  %s' % (name, m['wall_s'], n, desc))
    print('  dt=%.4f  主导=%s  CFL数=%.3f  扩散数=%.3f' %
          (L['dt'], L.get('why', '?'), L['cfl_num'], L['diff_num']))
    print('  云顶 %.0f  qc_max %.2f  冰 %.2f  w_max %.2f  v_max %.3f  盘面w %.2f'
          % (m['ctop'], m['qc_max'], m['ice_max'], m['w_max'], m['v_max'], m['w_disc']))
    print('  雨强 %.2f mm/h  落区 %.2f km²  收支残差 水 %.1e 位温 %.1e'
          % (m['rain_local'], m['rain_area'], m['q_rel'], m['e_rel']))
    print('  积分通量 m³/s: ' + '  '.join('%d:%.2e' % (k, v) for k, v in m['flux_prof'].items()))


def compare(names):
    D = {}
    for n in names:
        p = 'exp_%s.json' % n
        if os.path.exists(p):
            D[n] = json.load(open(p, encoding='utf-8'))
        else:
            print('缺 %s' % n)
    if len(D) < 2:
        return
    keys = ['ctop', 'qc_max', 'w_max', 'w_disc', 'rain_local']
    print('%-10s %8s %8s %9s %8s %8s %9s' %
          ('组', '云顶', 'qc_max', 'w_max', '盘面w', '雨强', '收支水'))
    for n, d in D.items():
        print('%-10s %8.0f %8.2f %9.2f %8.2f %8.3f %9.1e' %
              (n, d['ctop'], d['qc_max'], d['w_max'], d['w_disc'],
               d['rain_local'], d['q_rel']))
    print()
    print('%-10s %9s %9s %9s' % ('组', 'dt', 'CFL数', '扩散数'))
    for n, d in D.items():
        L = d.get('lim', {})
        print('%-10s %9.4f %9.3f %9.3f' % (n, L.get('dt', 0), L.get('cfl_num', 0),
                                           L.get('diff_num', 0)))
    print()
    print('积分质量通量垂直分布 m³/s（ChatGPT 要求报整条分布，不报单点）')
    hs = ['200', '400', '600', '900', '1200', '1500']
    print('%-10s' % '组' + ''.join('%12s' % h for h in hs))
    for n, d in D.items():
        fp = d.get('flux_prof', {})
        print('%-10s' % n + ''.join('%12.2e' % fp.get(h, 0) for h in hs))
    print()
    ks = list(D)
    for i in range(len(ks) - 1):
        a, b = ks[i], ks[i + 1]
        print('%s → %s' % (a, b))
        for k in keys:
            x, y = D[a][k], D[b][k]
            if abs(x) > 1e-12:
                print('   %-10s %+7.1f%%' % (k, 100 * (y - x) / x))
        fa, fb = D[a].get('flux_prof', {}), D[b].get('flux_prof', {})
        for h in hs:
            x, y = fa.get(h, 0), fb.get(h, 0)
            if abs(x) > 1e-12:
                print('   通量@%-5sm %+7.1f%%' % (h, 100 * (y - x) / x))


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] == 'list':
        for k, (_, d) in CASES.items():
            print('  %-8s %s' % (k, d))
    elif sys.argv[1] == 'run':
        run(sys.argv[2])
    elif sys.argv[1] == 'cmp':
        compare(sys.argv[2:])
