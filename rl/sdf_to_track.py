#!/usr/bin/env python3
"""
sdf_to_track.py - extract the centerline of a line-following track from a Gazebo world (.sdf).

Understands the two patterns used by the example worlds:
  * straight strips: <plane> or <box> visuals, size "L W" (L along the link's x axis). Consecutive strips
    are joined at the intersection of their axes, so corners are exact; the result is a closed polygon if
    the last strip meets the first, otherwise an open polyline.
  * rings: two concentric <cylinder> visuals in one model (dark outer, lighter inner) -> a circle whose
    centerline radius is the mean of the two radii and whose line width is their difference.
Large planes (>= 5 m) / models named *ground* are treated as the floor and give the ground extent.

Output: <name>_points.json (read by rl_env.py / train_rl.py / eval_rl.py) containing
points, closed, line_width, ground_rect. Also prints the suggested robot start pose.

  python3 sdf_to_track.py hexagon.sdf --png hexagon_check.png
"""
import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np


def floats(text, n=None):
    v = [float(t) for t in (text or '').split()]
    return v if n is None else (v + [0.0] * n)[:n]


def pose_xyyaw(el):
    p = el.find('pose') if el is not None else None
    v = floats(p.text if p is not None else '', 6)
    return v[0], v[1], v[5]


def compose(a, b):
    ax, ay, at = a
    bx, by, bt = b
    return (ax + math.cos(at) * bx - math.sin(at) * by, ay + math.sin(at) * bx + math.cos(at) * by, at + bt)


def luminance(visual):
    m = visual.find('material')
    if m is not None:
        d = m.find('diffuse') if m.find('diffuse') is not None else m.find('ambient')
        if d is not None:
            r, g, b = floats(d.text, 3)
            return 0.299 * r + 0.587 * g + 0.114 * b
    return 0.5


def intersect(c1, y1, c2, y2):
    d1, d2 = np.array([math.cos(y1), math.sin(y1)]), np.array([math.cos(y2), math.sin(y2)])
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-6:
        return None
    r = np.array(c2) - np.array(c1)
    t = (r[0] * d2[1] - r[1] * d2[0]) / den
    return np.array(c1) + t * d1


def extract(path):
    world = ET.parse(path).getroot().find('world')
    strips, rings, ground = [], [], None
    for model in world.findall('model'):
        mpose = pose_xyyaw(model)
        circles = []
        for link in model.findall('link'):
            lpose = compose(mpose, pose_xyyaw(link))
            for vis in link.findall('visual'):
                wpose = compose(lpose, pose_xyyaw(vis))
                geom = vis.find('geometry')
                if geom is None:
                    continue
                shape = geom[0]
                if shape.tag in ('plane', 'box'):
                    size = floats(shape.find('size').text, 3)
                    L, W = size[0], size[1]
                    if 'ground' in (model.get('name') or '').lower() or min(L, W) >= 5.0:
                        ground = (wpose[0] - L / 2, wpose[1] - W / 2, wpose[0] + L / 2, wpose[1] + W / 2)
                    elif min(L, W) <= 0.5:
                        if W > L:                      # long axis is local y
                            L, W, wpose = W, L, (wpose[0], wpose[1], wpose[2] + math.pi / 2)
                        strips.append(dict(c=(wpose[0], wpose[1]), yaw=wpose[2], L=L, W=W, name=link.get('name')))
                elif shape.tag == 'cylinder':
                    circles.append(dict(c=(wpose[0], wpose[1]), r=float(shape.find('radius').text),
                                        lum=luminance(vis), name=link.get('name')))
        if len(circles) >= 2:
            circles.sort(key=lambda c: -c['r'])
            outer, inner = circles[0], circles[1]
            if outer['lum'] < inner['lum'] - 0.1 and math.dist(outer['c'], inner['c']) < 1e-6:
                rings.append(dict(c=outer['c'], R=(outer['r'] + inner['r']) / 2, W=outer['r'] - inner['r']))
    return strips, rings, ground


def strips_to_track(strips):
    n = len(strips)
    if n == 0:
        return None
    if n == 1:
        s = strips[0]
        d = np.array([math.cos(s['yaw']), math.sin(s['yaw'])])
        c = np.array(s['c'])
        return np.array([c - d * s['L'] / 2, c + d * s['L'] / 2]), False, s['W']
    V = [intersect(strips[i]['c'], strips[i]['yaw'], strips[(i + 1) % n]['c'], strips[(i + 1) % n]['yaw']) for i in range(n)]
    def on_strip(p, s):
        d = np.array([math.cos(s['yaw']), math.sin(s['yaw'])])
        return p is not None and abs(np.dot(p - np.array(s['c']), d)) <= s['L'] / 2 + 1e-6
    closed = all(on_strip(V[i], strips[i]) and on_strip(V[i], strips[(i + 1) % n]) for i in range(n))
    width = float(np.median([s['W'] for s in strips]))
    if closed:
        return np.array(V), True, width
    # open chain: interior vertices are the intersections of consecutive strips
    V = [intersect(strips[i]['c'], strips[i]['yaw'], strips[i + 1]['c'], strips[i + 1]['yaw']) for i in range(n - 1)]
    overlaps = [(strips[i]['L'] - np.linalg.norm(V[i] - V[i - 1])) / 2 for i in range(1, n - 1)]
    ov = float(np.median(overlaps)) if overlaps else 0.0
    def far_end(s, v):
        d = np.array([math.cos(s['yaw']), math.sin(s['yaw'])])
        c = np.array(s['c'])
        e1, e2 = c - d * (s['L'] / 2 - ov), c + d * (s['L'] / 2 - ov)
        return e1 if np.linalg.norm(e1 - v) > np.linalg.norm(e2 - v) else e2
    return np.array([far_end(strips[0], V[0])] + V + [far_end(strips[-1], V[-1])]), False, width


def ring_to_track(ring, step=0.02):
    n = max(24, int(math.ceil(2 * math.pi * ring['R'] / step)))
    a = -math.pi / 2 + np.arange(n) * 2 * math.pi / n          # start at the bottom, drive counter-clockwise
    pts = np.column_stack([ring['c'][0] + ring['R'] * np.cos(a), ring['c'][1] + ring['R'] * np.sin(a)])
    return pts, True, ring['W']


def draw_check(path_png, strips, rings, pts, closed, ground):
    import cv2
    allp = [np.asarray(pts)]
    lo, hi = np.min(np.vstack(allp), 0) - 0.6, np.max(np.vstack(allp), 0) + 0.6
    ppm = 150
    W, H = int((hi[0] - lo[0]) * ppm) + 1, int((hi[1] - lo[1]) * ppm) + 1
    img = np.full((H, W, 3), 235, np.uint8)
    to = lambda p: (int((p[0] - lo[0]) * ppm), int(H - 1 - (p[1] - lo[1]) * ppm))
    for s in strips:                                   # the strips exactly as the SDF places them
        d, nrm = np.array([math.cos(s['yaw']), math.sin(s['yaw'])]), np.array([-math.sin(s['yaw']), math.cos(s['yaw'])])
        c = np.array(s['c'])
        quad = [c + d * s['L'] / 2 + nrm * s['W'] / 2, c + d * s['L'] / 2 - nrm * s['W'] / 2,
                c - d * s['L'] / 2 - nrm * s['W'] / 2, c - d * s['L'] / 2 + nrm * s['W'] / 2]
        cv2.fillPoly(img, [np.array([to(q) for q in quad], np.int32)], (30, 30, 30))
    for r in rings:
        cv2.circle(img, to(r['c']), int(r['R'] * ppm), (30, 30, 30), max(1, int(r['W'] * ppm)))
    cv2.polylines(img, [np.array([to(p) for p in pts], np.int32)], closed, (0, 0, 255), 1)
    cv2.circle(img, to(pts[0]), 5, (0, 160, 0), -1)
    cv2.imwrite(path_png, img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sdf')
    ap.add_argument('-o', '--out', default=None, help='output JSON (default: <sdf>_points.json)')
    ap.add_argument('--png', default=None, help='write a check image (SDF strips in black, extracted centerline in red)')
    a = ap.parse_args()
    strips, rings, ground = extract(a.sdf)
    if rings:
        pts, closed, width = ring_to_track(rings[0])
        kind = f"ring: centre {rings[0]['c']}, centerline radius {rings[0]['R']:.3f} m"
    elif strips:
        pts, closed, width = strips_to_track(strips)
        kind = f"{len(strips)} strips -> {'closed polygon' if closed else 'open polyline'} with {len(pts)} vertices"
    else:
        sys.exit('no line geometry found (expected thin <plane>/<box> strips or two concentric cylinders)')
    out = a.out or os.path.splitext(a.sdf)[0] + '_points.json'
    with open(out, 'w') as f:
        json.dump({'points': np.asarray(pts).tolist(), 'closed': bool(closed), 'line_width': float(width),
                   'ground_rect': list(ground) if ground else None}, f)
    p0, p1 = np.asarray(pts)[0], np.asarray(pts)[1]
    yaw0 = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    length = float(np.sum(np.hypot(*np.diff(np.vstack([pts, pts[:1]]) if closed else np.asarray(pts), axis=0).T)))
    print(f'{os.path.basename(a.sdf)}: {kind}')
    print(f'  line width {width:.3f} m, centerline length {length:.2f} m, ground rect {ground}')
    if len(pts) <= 12:
        print('  vertices:', [tuple(round(float(c), 3) for c in p) for p in pts])
    print(f'  suggested start: x={p0[0]:.3f} y={p0[1]:.3f} yaw={yaw0:.4f}')
    print(f'  wrote {out}')
    if a.png:
        draw_check(a.png, strips, rings, np.asarray(pts), closed, ground)
        print(f'  wrote {a.png}')


if __name__ == '__main__':
    main()
