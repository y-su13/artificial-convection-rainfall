"""fig_rain.py — 降水强度时间序列 + 两小时累计分布（报告插图）"""
import re
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.edgecolor'] = '#8d9aa7'
plt.rcParams['axes.linewidth'] = 0.7

INK, MUTED, ACC = '#22303d', '#6b7b8c', '#1f4e79'
ACC2 = '#b5651d'


# ---------------------------------------------------------------- 读日志
def read_log(path):
    rows = []
    for ln in open(path, encoding='utf-8', errors='ignore'):
        p = ln.split()
        if len(p) != 12:
            continue
        try:
            rows.append([float(v) for v in p])
        except ValueError:
            continue
    a = np.array(rows)
    return a[:, 0], a[:, 3], a[:, 4], a[:, 5]      # t, rain_max, rain_mean, cover


t1, rmax1, rmean1, cov1 = read_log('k1a2.log')
t2, rmax2, rmean2, cov2 = read_log('k1a2b.log')
t = np.concatenate([t1, t2])
rmax = np.concatenate([rmax1, rmax2])
rmean = np.concatenate([rmean1, rmean2])
cov = np.concatenate([cov1, cov2])
o = np.argsort(t)
t, rmax, rmean, cov = t[o], rmax[o], rmean[o], cov[o]


# ---------------------------------------------------------------- 图 1：雨强时间序列
fig, ax = plt.subplots(figsize=(7.4, 3.9), dpi=230)
fig.patch.set_facecolor('white')

ax.fill_between(t, rmax, color=ACC, alpha=0.10, lw=0)
ax.plot(t, rmax, color=ACC, lw=1.4, label='地面单点最大雨强')
ax.set_xlim(0, 120)
ax.set_ylim(0, 88)
ax.set_xlabel('时间 (min)', fontsize=9, color=INK)
ax.set_ylabel('雨强 (mm/h)', fontsize=9, color=INK)
ax.xaxis.set_major_locator(MultipleLocator(20))
ax.yaxis.set_major_locator(MultipleLocator(20))
ax.grid(alpha=0.18, lw=0.5, color='#8d9aa7')
for s in ['top']:
    ax.spines[s].set_visible(False)
ax.tick_params(labelsize=8, colors='#4a5a6a', length=3)

ax2 = ax.twinx()
ax2.plot(t, cov, color=ACC2, lw=1.2, ls='--', label='雨区面积')
ax2.set_ylim(0, 1.2)
ax2.set_ylabel('雨区面积 (km²)', fontsize=9, color=ACC2)
ax2.tick_params(labelsize=8, colors=ACC2, length=3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_color(ACC2)

# 峰值标注
for tm, txt in [(30, 44.6), (96, 76.6)]:
    ax.annotate(f'{txt:.0f} mm/h', xy=(tm, txt), xytext=(tm - 4, txt + 8),
                fontsize=7.5, color=ACC, ha='right' if tm < 60 else 'left',
                arrowprops=dict(arrowstyle='-', color=ACC, lw=0.6, alpha=0.7))
ax.axvline(60, color='#b0bcc7', lw=0.7, ls=':')
ax.text(61, 80, '第一小时 / 第二小时', fontsize=7, color=MUTED)

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=7.5, frameon=False,
          handlelength=1.8, labelcolor=INK)

ax.set_title('降水强度与雨区面积的时间变化', fontsize=11, color=INK, loc='left', pad=10)
plt.tight_layout()
plt.savefig('fig_rain_series.png', dpi=230, facecolor='white')
print('saved fig_rain_series.png')


# ---------------------------------------------------------------- 图 2：两小时累计分布
files = []
for pat in ['long_k1a2_0*.npz', 'long_k1a2_final.npz',
            'long_k1a2b_0*.npz', 'long_k1a2b_final.npz']:
    for f in glob.glob(pat):
        m = re.search(r'_(\d+)\.npz$', f)
        tt = int(m.group(1)) if m else (3600 if 'k1a2_final' in f else 7200)
        files.append((tt, f))
files = sorted(set(files))

d0 = np.load(files[0][1])
rc = d0['rc']
dr = rc[1] - rc[0]
acc = np.zeros(len(rc))
prev, prev_t = None, 0
for tt, f in files:
    d = np.load(f)
    qr = d['qr'][:, 0]
    Vt = np.clip(3.0 + 80.0 * np.sqrt(np.maximum(qr, 0)), 3, 12)
    rain = 1.14 * qr * Vt * 3600.0
    if prev is not None:
        acc += 0.5 * (rain + prev) / 3600.0 * (tt - prev_t)
    prev, prev_t = rain, tt

# 快照 5 min 采样会漏掉脉冲，按日志梯形积分的总量定标
SCALE = 8265.0 / ((acc / 1000 * 2 * np.pi * rc * dr).sum())
acc_s = acc * SCALE
vol = (acc_s / 1000 * 2 * np.pi * rc * dr)
share = np.cumsum(vol) / vol.sum() * 100

fig, ax = plt.subplots(figsize=(7.4, 3.9), dpi=230)
fig.patch.set_facecolor('white')
ax.fill_between(rc, 0, acc_s, color=ACC, alpha=0.13, lw=0)
ax.plot(rc, acc_s, color=ACC, lw=1.6, label='累计降水量')
ax.set_xlim(0, 800)
ax.set_ylim(0, 68)
ax.set_xlabel('距轴线距离 (m)', fontsize=9, color=INK)
ax.set_ylabel('两小时累计降水 (mm)', fontsize=9, color=INK)
ax.grid(alpha=0.18, lw=0.5, color='#8d9aa7')
ax.spines['top'].set_visible(False)
ax.tick_params(labelsize=8, colors='#4a5a6a', length=3)

ax2 = ax.twinx()
ax2.plot(rc, share, color=ACC2, lw=1.2, ls='--', label='累计水量份额')
ax2.set_ylim(0, 110)
ax2.set_ylabel('累计水量份额 (%)', fontsize=9, color=ACC2)
ax2.tick_params(labelsize=8, colors=ACC2, length=3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_color(ACC2)

R50 = np.interp(50, share, rc)
ax2.axhline(50, color='#c9b8a8', lw=0.6, ls=':')
ax2.annotate(f'半径 {R50:.0f} m 以内占一半', xy=(R50, 50), xytext=(R50 + 110, 26),
             fontsize=7.5, color=ACC2,
             arrowprops=dict(arrowstyle='-', color=ACC2, lw=0.6, alpha=0.7))
ax.annotate(f'峰值 {acc_s.max():.0f} mm', xy=(0, acc_s.max()), xytext=(105, acc_s.max() - 4),
            fontsize=7.5, color=ACC, ha='left')
ax.axvline(642, color='#b0bcc7', lw=0.7, ls=':')
ax.text(655, 52, '落区等效半径 642 m\n（瞬时雨区口径）', fontsize=7, color=MUTED, va='bottom')

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.text(300, 6, '累计降水量', fontsize=7.5, color=ACC)
ax2.text(415, 78, '累计水量份额', fontsize=7.5, color=ACC2)

ax.set_title('两小时累计降水的径向分布', fontsize=11, color=INK, loc='left', pad=10)
plt.tight_layout()
plt.savefig('fig_rain_accum.png', dpi=230, facecolor='white')
print('saved fig_rain_accum.png', f'R50={R50:.0f}')
