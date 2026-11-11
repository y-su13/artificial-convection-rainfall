"""render_cloud.py — 轴对称云场的体积渲染（ray marching）

给报告出"像真云"的示意图。做法：把 (r,z) 场绕轴旋成体素，ray marching +
单次散射 + 阴影透射 + 大气透视写成 HDR，再色调映射。

用装有 numpy / scipy / Pillow 的 Python 环境跑。

    python render_cloud.py long_k1a2b_final.npz out.png [preset]

preset: wide（全景竖幅）| mid（中景）| low（仰视近景）
"""
import sys
import time
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from PIL import Image, ImageFilter


# ---------------------------------------------------------------- 场与体素
def load_field(path):
    d = np.load(path)
    rc = d['rc'].astype(np.float64)
    zc = d['zc'].astype(np.float64)
    # g/kg；冰相散射更强，给不同权重让云砧有体积
    cloud = (d['qc'] * 1000.0 * 1.00 + d['qi'] * 1000.0 * 1.25
             + d['qs'] * 1000.0 * 1.55 + d['qg'] * 1000.0 * 1.95)
    rain = (d['qr'] * 1000.0)
    return rc, zc, cloud.astype(np.float32), rain.astype(np.float32)


def build_voxel(rc, zc, cloud, rain=None, R=2000.0, Z=7800.0, nxy=260, nz=340, thr=0.42):
    """旋成体素。rc/zc 是非均匀网格，索引用 interp 映射。"""
    x = np.linspace(-R, R, nxy, dtype=np.float32)
    z = np.linspace(0.0, Z, nz, dtype=np.float32)
    X, Y, Zg = np.meshgrid(x, x, z, indexing='ij')
    r = np.sqrt(X * X + Y * Y)
    ri = np.interp(r.ravel(), rc, np.arange(len(rc))).astype(np.float32)
    zi = np.interp(Zg.ravel(), zc, np.arange(len(zc))).astype(np.float32)
    dens = map_coordinates(cloud, np.stack([ri, zi]), order=1,
                           mode='nearest').reshape(X.shape)
    # 方位角调制：真实云团不是回转体，让迎风侧厚、背侧薄
    th = np.arctan2(Y, X)
    mod = (1.0 + 0.45 * np.sin(2 * th + 0.7) + 0.22 * np.sin(3 * th + 2.1)
           + 0.12 * np.sin(5 * th + 4.0))
    dens = dens * np.clip(mod, 0.10, 1.7)
    # 侵蚀：抬阈值 + 幂律，把软塌塌的雾状边际变成有轮廓的云
    dens = np.maximum(dens - thr, 0.0) ** 1.15
    if rain is None:
        return x, z, dens.astype(np.float32)
    rainv = map_coordinates(rain, np.stack([ri, zi]), order=1,
                            mode='nearest').reshape(X.shape)
    rainv = np.maximum(rainv - 0.05, 0.0)
    return x, z, dens.astype(np.float32), rainv.astype(np.float32)


def fbm3(shape, seed=7):
    """多尺度随机场，波长从 ~600 m 到 ~80 m，用来打碎轴对称的完美圆环。"""
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, np.float32)
    for s, w in [(8, 0.26), (16, 0.24), (30, 0.22), (56, 0.18), (96, 0.14)]:
        g = rng.random((s, s, max(4, int(s * shape[2] / shape[0])))).astype(np.float32)
        zi = np.linspace(0, g.shape[2] - 1, shape[2])
        yi = np.linspace(0, g.shape[1] - 1, shape[1])
        xi = np.linspace(0, g.shape[0] - 1, shape[0])
        XI, YI, ZI = np.meshgrid(xi, yi, zi, indexing='ij')
        up = map_coordinates(g, np.stack([XI.ravel(), YI.ravel(), ZI.ravel()]),
                             order=1).reshape(shape)
        out += w * gaussian_filter(up, 1.2, mode='nearest')
    out -= out.mean()
    out /= (out.std() + 1e-6)
    return out


def warp(dens, amp_xy=16.0, amp_z=7.0, seed=11):
    """域扭曲：用低频随机位移场把回转体掰歪，破坏完美轴对称。"""
    rng = np.random.default_rng(seed)
    sh = dens.shape
    nx, ny, nz = sh
    X, Y, Z = np.meshgrid(np.arange(nx, dtype=np.float32),
                          np.arange(ny, dtype=np.float32),
                          np.arange(nz, dtype=np.float32), indexing='ij')
    ch = []
    for k in range(3):
        g = rng.random((5, 5, 7)).astype(np.float32)
        zi = np.linspace(0, g.shape[2] - 1, nz)
        yi = np.linspace(0, g.shape[1] - 1, ny)
        xi = np.linspace(0, g.shape[0] - 1, nx)
        XI, YI, ZI = np.meshgrid(xi, yi, zi, indexing='ij')
        up = map_coordinates(g, np.stack([XI.ravel(), YI.ravel(), ZI.ravel()]),
                             order=1).reshape(sh)
        up = gaussian_filter(up, 4.0, mode='nearest')
        up -= up.mean()
        up /= (up.std() + 1e-6)
        ch.append(up)
    amp = [amp_xy, amp_xy, amp_z]
    co = np.stack([np.clip(X + ch[k] * amp[k], 0, sh[k] - 1.001) for k in range(3)])
    return map_coordinates(dens, co.reshape(3, -1), order=1).reshape(sh)


# ---------------------------------------------------------------- 光照
def sun_transmission(dens, dx, dz, sun_el=48.0, sun_az=135.0, n=26, step=110.0, k=1.0):
    el, az = np.radians(sun_el), np.radians(sun_az)
    dv = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)],
                  dtype=np.float32)
    tau = np.zeros_like(dens)
    nx, ny, nz = dens.shape
    xs = np.arange(nx, dtype=np.float32)
    ys = np.arange(ny, dtype=np.float32)
    zs = np.arange(nz, dtype=np.float32)
    X0, Y0, Z0 = np.meshgrid(xs, ys, zs, indexing='ij')
    for m in range(1, n + 1):
        d = m * step
        co = np.stack([
            np.clip(X0 + dv[0] * d / dx, 0, nx - 1.001),
            np.clip(Y0 + dv[1] * d / dx, 0, ny - 1.001),
            np.clip(Z0 + dv[2] * d / dz, 0, nz - 1.001),
        ]).reshape(3, -1)
        tau += map_coordinates(dens, co, order=1).reshape(dens.shape) * step
    od = k * 0.010 * tau
    # 多次散射近似：光学厚度开方压缩，云内不会一片死黑
    return 0.28 + 0.72 * np.exp(-0.52 * np.sqrt(od)), dv


# ---------------------------------------------------------------- 渲染
def render(dens, light, sunv, x, z, W, H, cam, target, fov, aniso=1.0, rain=None,
           exposure=1.10, sigma=0.014,
           sky_top=(0.30, 0.49, 0.72), sky_hor=(0.84, 0.89, 0.94)):
    dx = float(x[1] - x[0])
    dz = float(z[1] - z[0])
    nx, ny, nz = dens.shape

    cam = np.array(cam, np.float64)
    tgt = np.array(target, np.float64)
    fwd = tgt - cam
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    f = 1.0 / np.tan(np.radians(fov) / 2)
    px = (np.arange(W) + 0.5) / W * 2 - 1
    py = 1 - (np.arange(H) + 0.5) / H * 2
    PX, PY = np.meshgrid(px, py)
    rd = (fwd[None, None, :]
          + right[None, None, :] * (PX[..., None] * (W / H) / f * aniso)
          + up[None, None, :] * (PY[..., None] / f))
    rd /= np.linalg.norm(rd, axis=-1, keepdims=True)
    rd = rd.reshape(-1, 3)

    nrays = rd.shape[0]
    out = np.zeros((nrays, 3), np.float32)
    T = np.ones(nrays, np.float32)
    ts = np.linspace(0.0, 16000.0, 340)

    chunk = 40000
    for c0 in range(0, nrays, chunk):
        c1 = min(c0 + chunk, nrays)
        rdc = rd[c0:c1]
        col = np.zeros((c1 - c0, 3), np.float32)
        tr = np.ones(c1 - c0, np.float32)
        prev = 0.0
        for it in range(1, ts.size):
            t = ts[it]
            st = t - prev
            prev = t
            pos = cam[None, :] + rdc * t
            xi = (pos[:, 0] - x[0]) / dx
            yi = (pos[:, 1] - x[0]) / dx
            zi = pos[:, 2] / dz
            ins = (zi > 0) & (zi < nz - 1) & (xi > 0) & (xi < nx - 1) & (yi > 0) & (yi < ny - 1)
            if not ins.any():
                continue
            ii = np.where(ins)[0]
            co = np.stack([xi[ii], yi[ii], zi[ii]])
            dd = map_coordinates(dens, co, order=1, mode='nearest')
            ll = map_coordinates(light, co, order=1, mode='nearest')
            rv = (map_coordinates(rain, co, order=1, mode='nearest')
                  if rain is not None else np.zeros_like(dd))

            sig = sigma * dd
            sig_r = 0.0016 * rv
            a = 1.0 - np.exp(-(sig + sig_r) * st)
            cosT = rdc[ii] @ sunv
            phase = 0.55 + 0.45 * np.clip(cosT, 0, 1) ** 4          # 前向散射
            amb = 0.42 + 0.24 * np.clip(zi[ii] / nz, 0, 1)          # 天光，越高越亮
            shade = np.clip(0.26 * amb + 1.30 * ll * phase, 0, 1.55)

            base = np.stack([shade * 1.00, shade * 1.00, shade * 1.02], 1)
            # 云底偏冷、云顶偏暖（真实积云低层受地物反射，略灰蓝）
            tw = np.clip((zi[ii] / nz) * 2.4, 0, 1)[:, None]
            base *= (np.array([0.88, 0.92, 1.00])[None, :] * (1 - tw)
                     + np.array([1.00, 0.99, 0.97])[None, :] * tw)

            # 雨幡：灰蓝色帘幕，压在云下
            rw = np.clip(sig_r / (sig + sig_r + 1e-6), 0, 1)[:, None]
            base = base * (1 - rw) + np.array([0.66, 0.72, 0.80])[None, :] * rw
            col[ii] += (a * tr[ii])[:, None] * base
            tr[ii] *= (1.0 - a * 0.99)
        out[c0:c1] = col
        T[c0:c1] = tr

    # ---- 背景天空 ----
    rd2 = rd.reshape(H, W, 3)
    v = np.clip(rd2[..., 2], 0, 1)
    top = np.array(sky_top)[None, None, :]
    hor = np.array(sky_hor)[None, None, :]
    g = (v ** 0.55)[..., None]
    bg = hor * (1 - g) + top * g
    hz = np.exp(-((rd2[..., 2] / 0.045) ** 2))[..., None]
    bg = bg * (1 - 0.45 * hz) + np.array([0.86, 0.89, 0.93])[None, None, :] * (0.45 * hz)
    tg = np.where(rd2[..., 2] < -1e-4, -cam[2] / rd2[..., 2], 1e9)
    hit = tg < 0
    gnd = np.stack([0.74 + 0 * v, 0.77 + 0 * v, 0.70 + 0 * v], -1)
    ff = np.clip(tg / 9000.0, 0, 1)[..., None] ** 0.6
    gnd = gnd * (1 - ff) + np.array([0.82, 0.85, 0.87])[None, None, :] * ff
    bg = np.where(hit[..., None], gnd, bg)

    img = out.reshape(H, W, 3) + bg * T.reshape(H, W, 1)
    img = np.clip(img * exposure, 0, None)
    img = img / (1.0 + img * 0.32)                 # 软肩，压高光
    img = np.clip(img, 0, 1) ** (1 / 1.02)
    return img


PRESETS = {
    # 全景：整根云柱，竖幅
    'wide': dict(W=900, H=1350, fov=52.0, aniso=0.50,
                 cam=(-5200, -4300, 500), target=(0, 0, 2600)),
    # 中景：云柱下段与雨幡
    'mid': dict(W=1300, H=950, fov=50.0, aniso=0.72,
                cam=(-2700, -2300, 620), target=(0, 0, 1900)),
    # 仰视近景：从地面附近看云柱中段
    'low': dict(W=900, H=1200, fov=60.0,
                cam=(-1500, -1400, 120), target=(0, 0, 2600)),
}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'long_k1a2b_final.npz'
    out = sys.argv[2] if len(sys.argv) > 2 else 'render_cloud.png'
    key = sys.argv[3] if len(sys.argv) > 3 else 'wide'
    p = PRESETS[key]
    t0 = time.time()

    rc, zc, cloud, rain = load_field(path)
    x, z, dens, rainv = build_voxel(rc, zc, cloud, rain)
    print('voxel', dens.shape, f'{time.time()-t0:.1f}s')

    n = fbm3(dens.shape)
    dens = np.maximum(dens * (1.0 + 0.55 * n), 0)
    rainv = np.maximum(rainv * (1.0 + 0.35 * n), 0)
    print('warp', f'{time.time()-t0:.1f}s')
    dx = float(x[1] - x[0])
    dz = float(z[1] - z[0])

    light, sunv = sun_transmission(dens, dx, dz)
    print('light', f'{time.time()-t0:.1f}s')

    img = render(dens, light, sunv, x, z, cam=p['cam'], target=p['target'],
                 fov=p['fov'], W=p['W'], H=p['H'], aniso=p.get('aniso', 1.0),
                 rain=rainv)
    im = Image.fromarray((img * 255).astype(np.uint8))
    im = im.filter(ImageFilter.UnsharpMask(radius=1.4, percent=40, threshold=2))
    im.save(out)
    print('done', out, f'{time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
