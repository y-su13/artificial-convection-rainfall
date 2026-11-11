"""长时运行：从已有快照续跑到指定物理时长，回答「下不下雨 / 除不除雾」。

两个算例：
    rain  ckpt_hi_ma07.npz   t=1146 s，增雨配置（对流云，云顶 1781 m）
    fog   ckpt_fog_ma07.npz  t=1025 s，除雾配置（近地雾层 2–686 m）

数值配置：4 m / 2 m 近场原网格，显式 Euler + 全波数扫描 limiter
（`stab_method='scan'`、`stab_margin=0.75`，等价旧 S_target=1.5 但更严格）。
每 30 s 物理时间记一行诊断，每 300 s 存一个快照（断点可续）。

用法
----
python run_long.py rain 1.0      # 跑到 t=3600 s（1 小时）
python run_long.py rain 0.05     # 只跑 3 分钟，验证续跑正常
"""
import json
import os
import sys
import time

import numpy as np

from model import build_grid, Atmos
from exp_cyl import GRID
from exp_heat import BASE

# 2026-11-02 起，两个工况统一到 12 km × 16 km / 60 min。
# 8 km 域的旧结果保留在 long_rain0.json 等文件里作历史参照，不再新增算例。
GRID12 = dict(rmax=12000., zmax=16000., dr0=4., dz0=4., r_fine=650.,
              z_fine=600., gr=1.03, gz=1.03, dr_max=140., dz_max=140.)
# 2026-11-05：雾团外缘取 10 km，域必须比它大得多，否则开口边界就在雾边上。
# 12 km 域只剩 2 km 的清洁环，风机 0.4 m/s 的低层抽吸一小时能把边界空气
# 直接拉到雾缘。16 km 给 6 km 缓冲，网格只多 8.4%（124 920 点）。
GRID16 = dict(rmax=16000., zmax=16000., dr0=4., dz0=4., r_fine=650.,
              z_fine=600., gr=1.03, gz=1.03, dr_max=140., dz_max=140.)

CASES = {
    'rain': dict(ckpt='ckpt_hi_ma07.npz', grid=GRID,
                 note='增雨 v17（含霰）：对流云，盘面 17 m/s，nu_wake=50'),
    # 对照：同一 t=19.1 min 快照，**撤掉转子**（rotor_on=False，盘面强迫与
    # 废热全部取消）。用途是判断增雨的归因：探空 CAPE=2997、CIN=12.2 J/kg，
    # 这样的大气本来就接近自由对流，装置可能只是扳机。若撤掉后羽流仍自持
    # 发展，则 60 min 的降雨不能归给装置；若迅速衰减，装置才是必要条件。
    # 注意这不是「完全不装装置从 t=0 起跑」的严格对照——初值里已含转子
    # 造出的上升流，测的是「撤掉后能否自持」。
    'noRot': dict(ckpt='ckpt_hi_ma07.npz', grid=GRID, rotor=False,
                  note='对照：同快照撤掉转子，测羽流能否自持'),
    # v17 起，重跑从 t=0 开始：旧快照带着 v16 微物理（无霰）的状态，
    # 续跑不干净。ckpt=None 表示用 Atmos 自己的初值。
    'rain0': dict(ckpt=None, grid=GRID12,
                  note='增雨 B0：12 km，从 t=0 跑 60 min（v17 微物理）'),
    # 2026-11-04：从 B0 的 60 min 末态续跑，看第二个小时还在不在产水。
    # 60 min 时雨强仍在爬（55 min 26.6 → 60 min 44.3 mm/h），云顶已从
    # 10620 回落到 8240，属发展后期的衰减段还是准稳态，只能靠续跑分辨。
    'rain0b': dict(ckpt='long_rain0_final.npz', grid=GRID12,
                   note='增雨 B0 续跑：12 km，60→120 min'),
    # 2026-11-04 03:54 机器死机（三个进程同时跑，内存耗尽），三档全部中断。
    # 下面三条都是从最后一个完好快照接续，不从零重跑。
    'rain0c': dict(ckpt='long_rain0b_04200.npz', grid=GRID12,
                   note='增雨 B0 续跑二段：12 km，70→120 min'),
    'noRot0': dict(ckpt=None, grid=GRID12, rotor=False,
                   note='对照 B0：12 km，从 t=0 撤掉转子跑 60 min'),
    # 除雾：2026-11-02 起改为 12 km 域、从 t=0 起跑 60 min。
    # 旧做法是从 17.1 min 的 6 km 快照续跑，跨域跨版本都不干净，已废弃。
    # build_fog 设 SHF=LHF=0、flux_depth=100（雾顶辐射冷却，地面通量近零）；
    # surface_flux 现已优先读 env，这三项才会生效。
    # 除雾两组（2026-11-03）：都开辐射，只差风机。
    # 判据不是"能不能散"，而是有风机相对无风机在各半径处提前多少。
    'fog':  dict(ckpt=None, grid=GRID12, radiation=True,
                 note='除雾 F2：12 km，有风机 + 辐射，地方时 6:30-7:30'),
    'fogNoRot': dict(ckpt=None, grid=GRID12, rotor=False, radiation=True,
                     note='除雾 F1：12 km，无风机 + 辐射，地方时 6:30-7:30（基线）'),
    'fogb': dict(ckpt='long_fog_01200.npz', grid=GRID12, radiation=True,
                 note='除雾 F2 续：20→60 min，有风机'),
    # 雾团版：10 km 内有雾、之外无雾，16 km 域。此前的全域版作废（见 archive）。
    'fogp': dict(ckpt=None, grid=GRID16, radiation=True,
                 note='除雾 P2：雾团 r<10 km，16 km 域，有风机，6:30-7:30'),
    'fogpNoRot': dict(ckpt=None, grid=GRID16, rotor=False, radiation=True,
                      note='除雾 P1：雾团 r<10 km，16 km 域，无风机，6:30-7:30'),
    'fogNoRotb': dict(ckpt='long_fogNoRot_01800.npz', grid=GRID12, rotor=False,
                      radiation=True, note='除雾 F1 续：30→60 min，无风机'),
}

# ---------------------------------------------------------------------------
# 分层因子实验（2026-11-05，按第三稿审计第十节）
#
# 为什么不是"一次改一个变量、逐步叠加"：A1 改上升速度与云水分布，A3 改过冷水
# 供应，A2 改粒径与沉降，A4 改雨水生成与相态分配，四者互相喂养。A2 在无夹卷
# 基准下有效，不代表加上 A1 后仍有效。所以改成因子矩阵，**每组从同一 0 时刻
# 初态重新积分**，不拿上一组末态续跑。
#
# 规格统一：12 km × 16 km，60 min，其余开关与生产配置一致。
# ---------------------------------------------------------------------------
_FACT = dict(
    g1=dict(entrain=True, entrain_scalar=True, entrain_ke=0.2),
    g2=dict(ice_nuc=True),
    g3=dict(riming=True, vt_qg_dep=True),
    g4=dict(ice_nuc=True, riming=True, vt_qg_dep=True),
    g5=dict(entrain=True, entrain_scalar=True, entrain_ke=0.2,
            ice_nuc=True, riming=True, vt_qg_dep=True),
    g6=dict(entrain=True, entrain_scalar=True, entrain_ke=0.2,
            ice_nuc=True, riming=True, vt_qg_dep=True, accr_exp=True),
)
CASES.update({
    'g0': dict(ckpt=None, grid=GRID12, extra={},
               note='G0 基准：A1/A2/A3/A4 全关（应复现 rain0）'),
    'g1': dict(ckpt=None, grid=GRID12, extra=_FACT['g1'],
               note='G1：只开夹卷 A1（含标量卷吸，ke=0.2）'),
    'g2': dict(ckpt=None, grid=GRID12, extra=_FACT['g2'],
               note='G2：只开温度依赖冻结 A3'),
    'g3': dict(ckpt=None, grid=GRID12, extra=_FACT['g3'],
               note='G3：只开淞附+落速 A2（单参数筛查）'),
    'g4': dict(ckpt=None, grid=GRID12, extra=_FACT['g4'],
               note='G4：A3+A2（检查淞附是否有可用过冷水）'),
    'g5': dict(ckpt=None, grid=GRID12, extra=_FACT['g5'],
               note='G5：A1+A3+A2（完整过程组合）'),
    'g6': dict(ckpt=None, grid=GRID12, extra=_FACT['g6'],
               note='G6：A1+A3+A2+A4（最后加碰并指数）'),
    # 辐射 × 夹卷 2×2 的另外两格（另两格即 G0 与 G1）。雨工况取 14:00 地方时。
    'r1': dict(ckpt=None, grid=GRID12, radiation=True, rain_rad=True,
               extra={}, note='R1：辐射开、夹卷关（D1 主效应）'),
    'r2': dict(ckpt=None, grid=GRID12, radiation=True, rain_rad=True,
               extra=_FACT['g1'], note='R2：辐射开、夹卷开（D1×A1 交互）'),
    # 夹卷系数敏感性：G1 用 ke=0.2 把对流完全压死（云顶 1192 m、零降水），
    # 这是 ke 的必然结果而不是实现错误。要分清"系数太强"与"本该如此"，
    # 必须补 0.05 与 0.1 两档。与 G1 同样只开夹卷，只改 ke。
    'k05': dict(ckpt=None, grid=GRID12,
                extra=dict(entrain=True, entrain_scalar=True, entrain_ke=0.05),
                note='K05：只开夹卷，ke=0.05'),
    'k10': dict(ckpt=None, grid=GRID12,
                extra=dict(entrain=True, entrain_scalar=True, entrain_ke=0.10),
                note='K10：只开夹卷，ke=0.10 —— 已停，见下面 k02 的说明'),
    # 2026-11-09：ke=0.10/0.20 判定不合理，已停 K10。原因不是 ke 大，而是
    # ε=ke/R_eff 的分母在云诞生时只有 1 m（质量加权半径），ε 被放大 100–200 倍。
    # 正确做法是给 R_eff 设下限（entrain_rmin=75，半个盘面半径），再取文献值：
    #   Tiedtke 1989 深对流 ε=0.1/km、浅对流 0.3/km；R=75 m 反推 ke=0.008–0.023。
    # 取中间值 ke=0.02，对应 ε ≤ 2.7e-4 /m = 0.27 /km。
    'k02': dict(ckpt=None, grid=GRID12,
                extra=dict(entrain=True, entrain_scalar=True,
                           entrain_ke=0.02, entrain_rmin=75.0),
                note='K02：夹卷 ke=0.02，R_eff 下限 75 m（文献区间内）'),
    # ke=0.01：对应 ε=0.13 /km（R=75 m），紧贴 Tiedtke 深对流值 0.1 /km 的上方。
    # 与 k02 构成文献区间两端：0.13 与 0.27 /km。
    'k01': dict(ckpt=None, grid=GRID12,
                extra=dict(entrain=True, entrain_scalar=True,
                           entrain_ke=0.01, entrain_rmin=75.0),
                note='K01：夹卷 ke=0.01，R_eff 下限 75 m（深对流端）'),
    # 2026-11-09 17:00 前后机器死机（内核池泄漏，非首个病例），两组中断。
    # 从最后一个完好快照续跑，不从零重来。k01 到 36.0 min、快照 35 min；
    # k02 到 42.0 min、快照 40 min。开关与原算例完全一致。
    'k01b': dict(ckpt='long_k01_02100.npz', grid=GRID12,
                 extra=dict(entrain=True, entrain_scalar=True,
                            entrain_ke=0.01, entrain_rmin=75.0),
                 note='K01 续：35→60 min（死机后续跑）'),
    'k02b': dict(ckpt='long_k02_02400.npz', grid=GRID12,
                 extra=dict(entrain=True, entrain_scalar=True,
                            entrain_ke=0.02, entrain_rmin=75.0),
                 note='K02 续：40→60 min（死机后续跑）'),
    # 2026-11-10：第一个"框架 + 微物理都修正过"的算例。
    # 夹卷取 K01 的 ke=0.01（ε=0.13 /km，深对流端），叠加 A2（淞附 + 落速随质量）。
    # 对照：K01 无微物理 914 m³ / 云顶 7 540；G3（无夹卷 + A2）3 466 m³。
    # A3（冻结速率）不开——已三次证明对产水无效；A4（碰并指数）不开——留给下一步。
    'k1a2': dict(ckpt=None, grid=GRID12,
                 extra=dict(entrain=True, entrain_scalar=True,
                            entrain_ke=0.01, entrain_rmin=75.0,
                            riming=True, vt_qg_dep=True),
                 note='K1A2：夹卷 ke=0.01 + 淞附 + 落速随质量'),
    # 2026-11-11 用户要求：K1A2 再跑一小时，看第二小时的产水。
    # 开关与 k1a2 完全一致（夹卷 ke=0.01 + rmin=75 + 淞附 + 落速随质量）。
    # 参考：无微物理的 rain0 第二小时产水是第一小时的 7 倍（70→120 min），
    # 但那组无夹卷、无 A2，不能直接类比。
    'k1a2b': dict(ckpt='long_k1a2_final.npz', grid=GRID12,
                  extra=dict(entrain=True, entrain_scalar=True,
                             entrain_ke=0.01, entrain_rmin=75.0,
                             riming=True, vt_qg_dep=True),
                  note='K1A2 续：60→120 min'),
})


def diag(a, g, rain_area_only=True):
    zc, rc, rf = g['zc'], g['rc'], g['rf']
    area = 2 * np.pi * rc * g['dr']

    qc = a.qc
    prof = np.max(qc, axis=0)
    kk = np.where(prof > 1e-5)[0]
    ctop = float(zc[kk[-1]]) if len(kk) else 0.0
    cbase = float(zc[kk[0]]) if len(kk) else 0.0

    # 地面雨强 mm/h：qr × 落速 × ρ，与 exp_cyl.metrics 同一定义
    qr0 = a.qr[:, 0]
    Vt = np.clip(3. + 80. * np.sqrt(np.maximum(qr0, 0.)), 3., 12.)
    rain = a.rho_c[0] * qr0 * Vt * 3600.
    # 面积加权平均（域平均降水率），mm/h
    rain_mean = float(np.sum(rain * area) / np.sum(area))
    rain_max = float(rain.max())
    rain_cov = float(np.sum(area[rain > 1.0]) / 1e6)   # >1 mm/h 的覆盖面积 km²

    # 域内总量
    V = area[:, None] * g['dz'][None, :]
    qc_tot = float(np.sum(a.qc * a.rho_c[None, :] * V))
    qv_tot = float(np.sum(a.qv * a.rho_c[None, :] * V))
    qr_tot = float(np.sum(a.qr * a.rho_c[None, :] * V))
    # 分相库存（2026-11-05）：A2/D2 的判据要看霰和雪各自的质量，不能只看总量
    qi_tot = float(np.sum(a.qi * a.rho_c[None, :] * V))
    qs_tot = float(np.sum(a.qs * a.rho_c[None, :] * V))
    qg_tot = float(np.sum(a.qg * a.rho_c[None, :] * V))
    # 分相地面通量（同 rain 的折算方式：ρ·q·Vt·3600，单位 mm/h）
    area_tot = float(np.sum(area))
    fl_qs = float(np.sum(a.rho_c[0] * np.maximum(a.qs[:, 0], 0.) * 1.2 * 3600. * area) / area_tot)
    fl_qg = float(np.sum(a.rho_c[0] * np.maximum(a.qg[:, 0], 0.) * 3.0 * 3600. * area) / area_tot)
    # 霰的粒径与落速（A2 筛查用）：按固定数浓度反算，与 model 内部同一公式
    try:
        Dg = a._graupel_diameter()
        Vg = a._fallspeed_locatelli(Dg)
        qg_mask = a.qg > 1e-6
        Dg_max = float((Dg[qg_mask].max() * 1e3) if qg_mask.any() else 0.0)
        Vg_max = float(Vg[qg_mask].max() if qg_mask.any() else 0.0)
        Vt_qg_mass = float(np.sum(Vg * a.qg) / max(np.sum(a.qg), 1e-30))
    except Exception:
        Dg_max = Vg_max = Vt_qg_mass = 0.0

    # 近地面 0-50 m 的面积平均液态水（除雾判据）。
    # 不能用 qc_max：17 分钟快照的 qc_max=0.64 g/kg 是局部孤立点，
    # 面积平均只有 2.5e−4，两者差 2500 倍。
    k50 = int(np.argmin(np.abs(zc - 50.)))
    lwc_col = a.qc[:, :k50 + 1].mean(axis=1) * a.rho_c[None, :k50 + 1].mean() * 1000.
    # 2026-11-05：雾改成有限范围的雾团（r<=fog_r），域里有 40% 是清洁空气。
    # 全域平均会把能见度稀释（16 km 域、10 km 雾团时 91 m 会被平均成 236 m），
    # 所以判据改成"雾团内的面积平均"，另给雾的覆盖面积。
    env_ = getattr(a, '_env', {}) or {}
    fog_r = env_.get('fog_r') or a.cfg.get('fog_r')
    rcv = np.asarray(rc).ravel()
    msel = (rcv <= fog_r) if fog_r else np.ones(rcv.shape, bool)
    lwc_low = float(np.sum(lwc_col[msel] * area[msel]) / np.sum(area[msel]))   # g/m³
    beta = 144.7 * max(lwc_low, 1e-9) ** 0.88                # 1/km, Gultepe 2006
    vis_low = float(min(3.912 / beta * 1000., 99999.))        # m, Koschmieder
    # LWC→0 时上式发散（曾打出 2.2e9 m），封顶 99.999 km 表示"无雾"
    vis_col = np.minimum(3.912 / np.maximum(
        144.7 * np.maximum(lwc_col, 1e-12) ** 0.88, 1e-30) * 1000., 99999.)
    fog_cov = float(np.sum(area[(vis_col < 500.) & msel]) / 1e6)   # km², 能见度<500 m

    # 半径剖面诊断（2026-11-03）：除雾不能只看风机正下方。
    # 每个半径给出 0-350 m 的柱液水（g/m²）与近地 50 m 的能见度（m）。
    k350 = int(np.argmin(np.abs(zc - 350.)))
    prof_rad = {}
    for rr in (0, 1000, 2000, 5000):
        i = int(np.argmin(np.abs(rc - rr)))
        lwp = float(np.sum(a.qc[i, :k350 + 1] * a.rho_c[:k350 + 1] * g['dz'][:k350 + 1]) * 1e3)
        lw = float(a.qc[i, :k50 + 1].mean() * a.rho_c[:k50 + 1].mean() * 1e3)
        b_ = 144.7 * max(lw, 1e-9) ** 0.88
        prof_rad['r%d' % rr] = dict(lwp=lwp, vis=float(min(3.912 / b_ * 1000., 99999.)))

    D = a.divergence(a.u, a.w)
    return dict(t=float(a.t), ctop=ctop, cbase=cbase,
                lwc_low=lwc_low, vis_low=vis_low, fog_cov=fog_cov, prof=prof_rad,
                qi_tot=qi_tot, qs_tot=qs_tot, qg_tot=qg_tot,
                fl_snow=fl_qs, fl_graupel=fl_qg,
                Dg_max=Dg_max, Vg_max=Vg_max, Vt_qg=Vt_qg_mass,
                qc_max=float(qc.max()) * 1e3, qr_max=float(a.qr.max()) * 1e3,
                qg_max=float(a.qg.max()) * 1e3, qs_max=float(a.qs.max()) * 1e3,
                rain_max=rain_max, rain_mean=rain_mean, rain_cov=rain_cov,
                qc_tot=qc_tot, qv_tot=qv_tot, qr_tot=qr_tot,
                u_max=float(np.abs(a.u).max()), w_max=float(np.abs(a.w).max()),
                div_max=float(np.abs(D).max()),
                dt=float(getattr(a, '_lim', {}).get('dt', np.nan)),
                S_max=float(getattr(a, '_lim', {}).get('dt_stab', np.nan)),
                why=getattr(a, '_lim', {}).get('why', ''))


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else 'rain'
    hours = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    c = CASES[case]

    g = build_grid(**c['grid'])
    d = np.load(c['ckpt']) if c.get('ckpt') else None
    # 注意：必须按前缀判断。曾写成 case == 'fog'，导致 fogNoRot 落到
    # build_env（增雨探空），除雾基线里根本没有雾，白跑 9.3 小时。
    if case.startswith('fog'):
        from sounding2 import build_fog as make_env
    else:
        from sounding import build_env as make_env
    cfgp = dict(BASE, rotor_Pheat=13.0e6,
                stab_limiter=True, stab_method='scan', stab_margin=0.75)
    if c.get('radiation'):
        # 雾工况 6:30 起报；雨工况取 14:00（午后对流）。albedo 只对雾有意义，
        # 深对流的反射由云本身决定，这里沿用同一套短波吸收，不外推反照率。
        t0 = 14.0 if c.get('rain_rad') else 6.5
        cfgp.update(radiation=True, t_local0=t0, lat=30.0, decl=-8.0,
                    albedo_max=0.50 if not c.get('rain_rad') else 0.0,
                    albedo_lwp=30.0)
        if c.get('rain_rad'):
            # 深对流顶用体辐射温度算长波；雾工况沿用 Idso-Jackson 经验式，
            # 保证已完成的除雾两组不被改口径。
            cfgp['lw_bulk'] = True
    if c.get('extra'):
        # 分层因子实验用：每个算例只声明自己额外打开的开关
        cfgp.update(c['extra'])
    if not c.get('rotor', True):
        # 撤掉装置：盘面强迫、旋流、废热全在 rotor 块内，一并取消
        cfgp['rotor_on'] = False
    a = Atmos(g, make_env(), cfgp)
    if d is not None:
        for nm in ('u', 'w', 'v', 'th', 'qv', 'qc', 'qr'):
            getattr(a, nm)[:] = d[nm]
        for nm in ('qi', 'qs', 'qg'):
            if hasattr(a, nm) and nm in d:
                getattr(a, nm)[:] = d[nm]
        a.t = float(d['t'])
    a.refresh_nu()
    # 有快照时参数是「续跑多久」；从 t=0 起跑时参数就是「跑多久」
    tend = (a.t if d is not None else 0.0) + hours * 3600.

    print('=== %s ===' % c['note'])
    print('起点 t=%.0f s (%.1f min)，目标 t=%.0f s (%.0f min)' %
          (a.t, a.t / 60, tend, tend / 60))
    print('网格 %d x %d = %d，初值 %s' %
          (g['nr'], g['nz'], g['nr'] * g['nz'], c['ckpt'] or 'Atmos 默认（t=0）'))
    import hashlib
    print('model.py md5 %s' % hashlib.md5(open('model.py', 'rb').read()).hexdigest()[:12])
    print()
    # 列定义写在这里，别再靠记忆：t dt ctop rain_max rain_mean cov qc_max
    # qr_max qg_max lwc vis。v16 的日志没有 qc_max，曾被误当成 cov 抄进材料。
    print('%-9s %7s %8s %9s %9s %9s %8s %8s %8s %8s %8s %8s' %
          ('t(min)', 'dt', '云顶', '雨强max', '雨强均值', '雨覆盖km²',
           'qc_max', 'qr_max', 'qg_max', '地面LWC', '能见度', '雾区km²'))

    rows = []
    t0 = time.time()
    next_rec = a.t
    next_save = a.t + 300.
    while a.t < tend - 1e-9:
        dt = min(float(a.max_dt()), tend - a.t)
        a.step(dt)
        if a.t >= next_rec:
            r = diag(a, g)
            rows.append(r)
            if len(rows) % 4 == 1:
                print('%-9.2f %7.4f %8.0f %9.3f %9.4f %9.2f %8.3f %8.3f %8.3f %8.4f %8.0f %8.2f' %
                      (a.t / 60, r['dt'], r['ctop'], r['rain_max'],
                       r['rain_mean'], r['rain_cov'], r['qc_max'], r['qr_max'],
                       r['qg_max'], r['lwc_low'], r['vis_low'], r['fog_cov']))
            next_rec = a.t + 30.
        if a.t >= next_save:
            np.savez('long_%s_%05.0f.npz' % (case, a.t),
                     u=a.u, w=a.w, v=a.v, th=a.th, qv=a.qv, qc=a.qc,
                     qr=a.qr, qi=getattr(a, 'qi', np.zeros_like(a.qc)),
                     qs=getattr(a, 'qs', np.zeros_like(a.qc)),
                     qg=getattr(a, 'qg', np.zeros_like(a.qc)),
                     t=a.t, rc=g['rc'], zc=g['zc'])
            json.dump(rows, open('long_%s.json' % case, 'w'),
                      ensure_ascii=False, indent=1)
            next_save = a.t + 300.

    json.dump(rows, open('long_%s.json' % case, 'w'), ensure_ascii=False, indent=1)
    np.savez('long_%s_final.npz' % case, u=a.u, w=a.w, v=a.v, th=a.th,
             qv=a.qv, qc=a.qc, qr=a.qr,
             qi=getattr(a, 'qi', np.zeros_like(a.qc)),
             qs=getattr(a, 'qs', np.zeros_like(a.qc)),
             qg=getattr(a, 'qg', np.zeros_like(a.qc)),
             t=a.t, rc=g['rc'], zc=g['zc'])
    r = diag(a, g)
    print()
    print('完成 t=%.0f s，墙钟 %.0f s (%.1f h)' % (a.t, time.time() - t0,
                                                (time.time() - t0) / 3600))
    print('末态：云顶 %.0f m、雨强 max %.3f mm/h、均值 %.4f mm/h、覆盖 %.2f km²' %
          (r['ctop'], r['rain_max'], r['rain_mean'], r['rain_cov']))
    print('      地面 LWC %.4f g/m³、能见度 %.0f m（%s）' %
          (r['lwc_low'], r['vis_low'],
           '雾散' if r['vis_low'] > 1000 else '仍有雾'))


if __name__ == '__main__':
    main()
