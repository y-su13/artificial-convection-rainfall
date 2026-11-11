"""fig_profile.py — 云柱 r-z 剖面（报告插图，商务风）"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

d = np.load('long_k1a2b_final.npz')
rc, zc = d['rc'], d['zc']
qc = d['qc'] * 1000.0
tot = (d['qc'] + d['qi'] + d['qs'] + d['qg']) * 1000.0
rain = d['qr'] * 1000.0
w = d['w'][:, :len(zc)]   # w 在垂直面网格上，多一层

cmap = LinearSegmentedColormap.from_list('cloud', [
    (0.00, '#ffffff'), (0.18, '#eaf1f8'), (0.40, '#bfd4e8'),
    (0.65, '#8fb3d4'), (0.85, '#5d86ae'), (1.00, '#3a5f85')])

fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=220)
fig.patch.set_facecolor('white')

m = ax.pcolormesh(rc, zc, tot.T, cmap=cmap, vmin=0.15, vmax=3.2,
                  shading='auto', rasterized=True)
# 雨区轮廓（云下淡淡的雨）
ax.contour(rc, zc, rain.T, levels=[0.15], colors='#7d93aa', linewidths=0.6,
           linestyles='--')
# 上升气流核心
cs = ax.contour(rc, zc, w.T, levels=[5, 15, 30], colors='#7a8a99',
                linewidths=0.6, alpha=0.9)
ax.clabel(cs, fmt='%d', fontsize=5.5, inline=True, colors='#5d6b78')

ax.set_xlim(0, 900)
ax.set_ylim(0, 7000)
ax.set_xlabel('距轴线距离 (m)', fontsize=9, color='#33414f')
ax.set_ylabel('高度 (m)', fontsize=9, color='#33414f')
ax.tick_params(labelsize=8, colors='#4a5a6a', length=3)

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
for s in ['left', 'bottom']:
    ax.spines[s].set_color('#8d9aa7')
    ax.spines[s].set_linewidth(0.7)
ax.grid(alpha=0.18, linewidth=0.5, color='#6b7b8c')

# 标注
ax.axhspan(0, 140, color='#5d6b78', alpha=0.35, lw=0)
ax.annotate('风机盘面  直径 300 m / 高 140 m', xy=(150, 160), xytext=(360, 480),
            fontsize=7.5, color='#2f3d4a',
            arrowprops=dict(arrowstyle='-', color='#6b7b8c', lw=0.7))
ax.axhline(1300, color='#4a6a8a', lw=0.7, ls=':')
ax.text(880, 1360, '自由对流高度 1 300 m', fontsize=7, color='#4a6a8a', ha='right')
ax.axhline(6280, color='#4a6a8a', lw=0.7, ls=':')
ax.text(880, 6340, '云顶 6 280 m', fontsize=7, color='#4a6a8a', ha='right')

cb = fig.colorbar(m, ax=ax, fraction=0.030, pad=0.02)
cb.set_label('云水与冰相含水量 (g/kg)', fontsize=8, color='#33414f')
cb.ax.tick_params(labelsize=7, colors='#4a5a6a')
cb.outline.set_edgecolor('#8d9aa7')
cb.outline.set_linewidth(0.6)

ax.set_title('两小时末态的云柱剖面', fontsize=11, color='#22303d', pad=12,
             loc='left')
fig.text(0.012, 0.005, '实线：垂直速度等值线 5 / 15 / 30 m/s　虚线：雨水含量 0.15 g/kg 边界',
         fontsize=6.5, color='#6b7b8c')
plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig('fig_cloud_profile.png', dpi=220, facecolor='white')
print('saved fig_cloud_profile.png')
