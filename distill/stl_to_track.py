#!/usr/bin/env python3
"""
stl_to_track.py - extract the centerline of a line-following track that is a MESH (.stl) in a Gazebo world.

Use it when sdf_to_track.py says "no line geometry found" because the line is a <mesh> visual.

What it does
  1. finds the <mesh> visual in the .sdf (skipping the ground plane), resolves its pose and scale,
  2. reads the STL (binary or ascii) and rasterises its footprint seen from above (5 mm pixels),
  3. thins the footprint to a 1-pixel skeleton, takes the main path, simplifies it to straight segments and
     re-fits each straight segment by least squares so corners are exact (intersection of the fitted lines),
  4. writes <name>_points.json (same format as sdf_to_track.py: points, closed, line_width, ground_rect)
     and a check image, and prints the start pose.

  python3 stl_to_track.py final_grading.sdf --stl /path/to/final_grading.stl --png final_grading_check.png

Open tracks: the two ends are placed half a line width inside the tip of the mesh (the convention used by
make_line_world.py). Use --reverse to start from the other end. Branching tracks (crossings, T junctions) are
not supported: only the longest path is kept and a warning is printed.
"""
import argparse
import json
import math
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET
from collections import deque

import cv2
import numpy as np


# ---------------------------------------------------------------- SDF helpers
def floats(text, n):
    v = [float(t) for t in (text or '').split()]
    return (v + [0.0] * n)[:n]


def pose_matrix(el):
    p = el.find('pose') if el is not None else None
    x, y, z, r, pt, yw = floats(p.text if p is not None else '', 6)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(pt), math.sin(pt), math.cos(yw), math.sin(yw)
    R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                  [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                  [-sp, cp * sr, cp * cr]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, [x, y, z]
    return T


def find_mesh_and_ground(sdf):
    world = ET.parse(sdf).getroot().find('world')
    mesh, ground = None, None
    for model in world.findall('model'):
        mT = pose_matrix(model)
        for link in model.findall('link'):
            lT = mT @ pose_matrix(link)
            for vis in link.findall('visual'):
                T = lT @ pose_matrix(vis)
                geom = vis.find('geometry')
                if geom is None:
                    continue
                m = geom.find('mesh')
                if m is not None and mesh is None:
                    sc = m.find('scale')
                    mesh = dict(uri=(m.find('uri').text or '').strip(), T=T,
                                scale=floats(sc.text, 3) if sc is not None else [1.0, 1.0, 1.0],
                                name=model.get('name'))
                pl = geom.find('plane')
                if pl is not None and ground is None:
                    s = floats(pl.find('size').text, 2)
                    ground = (T[0, 3] - s[0] / 2, T[1, 3] - s[1] / 2, T[0, 3] + s[0] / 2, T[1, 3] + s[1] / 2)
    return mesh, ground


# ---------------------------------------------------------------- STL
def read_stl(path):
    data = open(path, 'rb').read()
    if len(data) >= 84:
        n = struct.unpack('<I', data[80:84])[0]
        if len(data) == 84 + 50 * n:
            dt = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
            return np.frombuffer(data, dtype=dt, count=n, offset=84)['v'].astype(np.float64)
    txt = data.decode('latin-1')
    v = np.array(re.findall(r'vertex\s+(\S+)\s+(\S+)\s+(\S+)', txt), dtype=np.float64)
    if len(v) == 0 or len(v) % 3:
        sys.exit('could not parse the STL (neither binary nor ascii)')
    return v.reshape(-1, 3, 3)


# ---------------------------------------------------------------- skeleton
def zhang_suen(img):
    img = (img > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P = np.pad(img, 1)
            p2, p3, p4, p5 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:], P[2:, 2:]
            p6, p7, p8, p9 = P[2:, 1:-1], P[2:, :-2], P[1:-1, :-2], P[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(a.astype(np.int16) for a in nb)
            A = sum(((nb[i] == 0) & (nb[(i + 1) % 8] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                c = ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                c = ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            m = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if m.any():
                img[m] = 0
                changed = True
    return img


def skeletonize(mask):
    try:
        from skimage.morphology import skeletonize as sk
        return sk(mask > 0).astype(np.uint8)
    except Exception:
        return zhang_suen(mask)


NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def neighbour_count(sk):
    return cv2.filter2D(sk.astype(np.float32), -1, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.float32),
                        borderType=cv2.BORDER_CONSTANT)


def prune(sk, iters):
    sk = sk.copy()
    for _ in range(iters):
        ends = (sk == 1) & (neighbour_count(sk) <= 1)
        if not ends.any():
            break
        sk[ends] = 0
    return sk


def bfs(sk, src, blocked=None):
    H, W = sk.shape
    dist = {src: 0}
    parent = {src: None}
    q = deque([src])
    last = src
    while q:
        y, x = q.popleft()
        last = (y, x)
        for dy, dx in NB8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and sk[ny, nx] and (ny, nx) not in dist:
                dist[(ny, nx)] = dist[(y, x)] + 1
                parent[(ny, nx)] = (y, x)
                q.append((ny, nx))
    return last, parent, dist


def longest_path(sk):
    ys, xs = np.nonzero(sk)
    # largest connected component
    n, lab = cv2.connectedComponents(sk.astype(np.uint8), connectivity=8)
    if n > 2:
        sizes = [(lab == i).sum() for i in range(1, n)]
        keep = 1 + int(np.argmax(sizes))
        sk = (lab == keep).astype(np.uint8)
        ys, xs = np.nonzero(sk)
    a, _, _ = bfs(sk, (int(ys[0]), int(xs[0])))
    b, parent, _ = bfs(sk, a)
    path, cur = [], b
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return np.array([(x, y) for y, x in path], np.float64), sk      # (x, y) pixel coords


# ---------------------------------------------------------------- geometry
def fit_line(pts):
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0]), np.array([vx, vy])


def line_intersect(p1, d1, p2, d2):
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-6:
        return None
    r = p2 - p1
    t = (r[0] * d2[1] - r[1] * d2[0]) / den
    return p1 + t * d1


def refine_vertices(path, V, closed, margin_px, min_len_px):
    """Re-fit each straight segment of `path` between simplified vertices V and intersect neighbours."""
    n = len(V)
    segs = n if closed else n - 1
    lines = [None] * segs
    for i in range(segs):
        a, b = V[i], V[(i + 1) % n]
        d = b - a
        L = np.linalg.norm(d)
        if L < min_len_px:
            continue
        u = d / L
        t = (path - a) @ u
        off = np.abs((path - a) @ np.array([-u[1], u[0]]))
        sel = path[(t > margin_px) & (t < L - margin_px) & (off < 0.6 * margin_px + 3)]
        if len(sel) >= 8:
            lines[i] = fit_line(sel)
    W = V.copy()
    rng = range(n) if closed else range(1, n - 1)
    for i in rng:
        l0, l1 = lines[(i - 1) % segs], lines[i % segs]
        if l0 is None or l1 is None:
            continue
        q = line_intersect(l0[0], l0[1], l1[0], l1[1])
        if q is not None and np.linalg.norm(q - V[i]) < 3 * margin_px:
            W[i] = q
    if not closed:                                  # snap the ends onto their fitted line
        for i, k in ((0, 0), (n - 1, segs - 1)):
            if lines[k] is not None:
                p0, d = lines[k]
                W[i] = p0 + d * float((V[i] - p0) @ d)
    return W


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sdf')
    ap.add_argument('--stl', default=None, help='path of the .stl (default: looked up from the model:// uri)')
    ap.add_argument('-o', '--out', default=None, help='output JSON (default: <sdf>_points.json)')
    ap.add_argument('--png', default=None, help='write a check image')
    ap.add_argument('--res', type=float, default=0.005, help='raster resolution [m/pixel]')
    ap.add_argument('--tol', type=float, default=0.012, help='polyline simplification tolerance [m]')
    ap.add_argument('--scale', type=float, nargs=3, default=None, help='override the mesh scale from the sdf')
    ap.add_argument('--closed', action='store_true', help='force a closed loop')
    ap.add_argument('--open', action='store_true', help='force an open path')
    ap.add_argument('--reverse', action='store_true', help='start from the other end / drive the other way round')
    a = ap.parse_args()

    mesh, ground = find_mesh_and_ground(a.sdf)
    if mesh is None:
        sys.exit('no <mesh> visual found in the sdf')
    stl = a.stl
    if stl is None:
        base = os.path.basename(mesh['uri'])
        hits = [os.path.join(r, base) for r, _, f in os.walk(os.path.dirname(os.path.abspath(a.sdf))) if base in f]
        if not hits:
            sys.exit(f"cannot find {base} next to the sdf; pass --stl PATH  (try: find / -name {base} 2>/dev/null)")
        stl = hits[0]
    print(f"mesh: {mesh['uri']} (model '{mesh['name']}')  file: {stl}")

    tris = read_stl(stl)
    sc = np.array(a.scale if a.scale else mesh['scale'])
    T = mesh['T']
    P = tris.reshape(-1, 3) * sc
    P = (T[:3, :3] @ P.T).T + T[:3, 3]
    tris = P.reshape(-1, 3, 3)
    lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    print(f'  {len(tris)} triangles, world bounds x[{lo[0]:.3f},{hi[0]:.3f}] y[{lo[1]:.3f},{hi[1]:.3f}] z[{lo[2]:.4f},{hi[2]:.4f}] (scale {sc.tolist()})')

    res = a.res
    pad = 0.15
    x0, y1 = lo[0] - pad, hi[1] + pad
    Wp, Hp = int((hi[0] - lo[0] + 2 * pad) / res) + 1, int((hi[1] - lo[1] + 2 * pad) / res) + 1
    if Wp * Hp > 60e6:
        sys.exit(f'raster would be {Wp}x{Hp}; increase --res')
    S = 16
    px = np.stack([(tris[:, :, 0] - x0) / res * S, (y1 - tris[:, :, 1]) / res * S], -1)
    mask = np.zeros((Hp, Wp), np.uint8)
    cv2.fillPoly(mask, [np.round(t).astype(np.int32) for t in px], 255, shift=4)
    mask = (mask > 0).astype(np.uint8)
    print(f'  raster {Wp}x{Hp} px, line covers {mask.mean() * 100:.2f}% of it')

    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    sk = skeletonize(mask)
    width_px = 2.0 * float(np.median(dt[sk > 0]))
    width = width_px * res
    print(f'  line width ~ {width:.3f} m')

    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = [c for c, h in zip(cnts, hier[0]) if h[3] != -1 and cv2.contourArea(c) > (3 * width_px) ** 2]
    closed = bool(holes) if not (a.closed or a.open) else a.closed
    print(f"  shape: {'closed loop' if closed else 'open path'}")

    sk = prune(sk, int(round(1.2 * width_px)))
    if closed:                                        # cut a 3 cm gap so the loop becomes a path
        ys, xs = np.nonzero(sk)
        k = len(ys) // 3
        cv2.circle(sk, (int(xs[k]), int(ys[k])), max(3, int(0.015 / res)), 0, -1)
    path, sk = longest_path(sk)
    if len(path) < 10:
        sys.exit('skeleton too short - wrong scale or not a line mesh?')
    others = (sk > 0).sum() - len(path)
    if others > 0.15 * len(path):
        print(f'  WARNING: {others} skeleton pixels are off the main path (branching track?). Only the longest path is used.')

    eps = max(1.0, a.tol / res)
    V = cv2.approxPolyDP(path.astype(np.float32).reshape(-1, 1, 2), eps, closed).reshape(-1, 2).astype(np.float64)
    if closed and np.linalg.norm(V[0] - V[-1]) < 2 * eps and len(V) > 3:
        V = V[:-1]
    V = refine_vertices(path, V, closed, margin_px=0.7 * width_px + 2, min_len_px=3 * width_px)

    if not closed:                                    # ends: half a line width inside the mesh tip
        for i, j in ((0, 1), (len(V) - 1, len(V) - 2)):
            d = V[i] - V[j]
            d /= np.linalg.norm(d)
            q = V[i].copy()
            for _ in range(int(0.5 / res)):
                nxt = q + d
                xi, yi = int(round(nxt[0])), int(round(nxt[1]))
                if not (0 <= yi < Hp and 0 <= xi < Wp) or not mask[yi, xi]:
                    break
                q = nxt
            V[i] = q - d * (width_px / 2)
    if a.reverse:
        V = V[::-1]

    world = np.column_stack([x0 + V[:, 0] * res, y1 - V[:, 1] * res])
    out = a.out or os.path.splitext(a.sdf)[0] + '_points.json'
    with open(out, 'w') as f:
        json.dump({'points': np.round(world, 4).tolist(), 'closed': bool(closed), 'line_width': round(float(width), 4),
                   'ground_rect': list(ground) if ground else None}, f)
    seg = np.diff(np.vstack([world, world[:1]]) if closed else world, axis=0)
    length = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
    yaw0 = math.atan2(world[1][1] - world[0][1], world[1][0] - world[0][0])
    print(f'  {len(world)} vertices, centerline length {length:.2f} m')
    print('  vertices:', [tuple(round(float(c), 3) for c in p) for p in world][:16], '...' if len(world) > 16 else '')
    print(f'  suggested start: x:={world[0][0]:.3f} y:={world[0][1]:.3f} yaw:={yaw0:.4f}')
    print(f'  wrote {out}')

    if a.png:
        img = cv2.cvtColor(mask * 200, cv2.COLOR_GRAY2BGR)
        img = 255 - img
        pts = np.round(V).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], closed, (0, 0, 255), 1)
        for p in V:
            cv2.circle(img, (int(round(p[0])), int(round(p[1]))), 4, (255, 0, 0), -1)
        cv2.circle(img, (int(round(V[0][0])), int(round(V[0][1]))), 7, (0, 160, 0), -1)
        scale = min(1.0, 1600.0 / max(img.shape))
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(a.png, img)
        print(f'  wrote {a.png}  (mesh in grey, fitted centerline red, vertices blue, start green)')


if __name__ == '__main__':
    main()
