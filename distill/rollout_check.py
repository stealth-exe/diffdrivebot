#!/usr/bin/env python3
"""
rollout_check.py - closed-loop check of the label policy on a track, without Gazebo.

It drives a kinematic unicycle with exactly the labels that make_distill_dataset.py would write
(teacher, hybrid or expert) and reports how well that policy follows the centerline. A student that fits
these labels will behave similarly (minus its own approximation error). Use it to compare modes and to
tune --scale / --a0 / --a1 BEFORE spending time on training.

  python3 rollout_check.py --teacher model.pt --track line_world_points.json --mode hybrid
  python3 rollout_check.py --teacher model.pt --track line_world_points.json --mode teacher   # shows why a blend is needed
  python3 rollout_check.py --track line_world_points.json --mode expert                      # no checkpoint needed
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from distill_lib import (Cam, LabelCfg, StudentCfg, TeacherCfg, TeacherStyle, Teacher, Track, parse_points,
                         random_track, simulate)


def draw(track, results, path, px_per_m=140):
    pts = track.points
    lo = pts.min(0) - 0.6
    allxy = np.array([(x, y) for r in results for x, y, _ in r['traj']] + [tuple(p) for p in pts])
    lo, hi = np.minimum(lo, allxy.min(0) - 0.2), np.maximum(pts.max(0) + 0.6, allxy.max(0) + 0.2)
    W, H = [int((hi - lo)[i] * px_per_m) + 1 for i in (0, 1)]
    img = np.full((H, W, 3), 245, np.uint8)
    to = lambda p: (int((p[0] - lo[0]) * px_per_m), int(H - 1 - (p[1] - lo[1]) * px_per_m))
    cv2.polylines(img, [np.array([to(p) for p in pts], np.int32)], track.closed, (90, 90, 90), 5)
    for r, col in zip(results, [(0, 0, 220), (220, 120, 0), (0, 160, 0)]):
        tr = np.array([to(p) for p in r['traj']], np.int32)
        cv2.polylines(img, [tr.reshape(-1, 1, 2)], False, col, 2)
    cv2.circle(img, to(pts[0]), 6, (0, 140, 0), -1)
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--teacher', default=None,
                    help='teacher checkpoint; required for --mode teacher/hybrid, ignored for --mode expert')
    ap.add_argument('--track', nargs='*', default=[])
    ap.add_argument('--points', default=None, help='"x,y x,y ..." instead of --track')
    ap.add_argument('--random', type=int, default=0, help='also test N random sharp-cornered tracks')
    ap.add_argument('--mode', choices=['teacher', 'hybrid', 'expert'], default='hybrid')
    ap.add_argument('--scale', type=float, default=LabelCfg.scale)
    ap.add_argument('--road-width', type=float, default=TeacherStyle.road_width_T)
    ap.add_argument('--a0', type=float, default=LabelCfg.blend_a0_deg)
    ap.add_argument('--a1', type=float, default=LabelCfg.blend_a1_deg)
    ap.add_argument('--lookahead', type=float, default=LabelCfg.lookahead)
    ap.add_argument('--steer-sign', type=float, default=TeacherCfg.steer_sign)
    ap.add_argument('--student-xacro', default=os.path.join(HERE, '..', 'urdf', 'diffdrive_bot.xacro'))
    ap.add_argument('--backend', choices=['auto', 'torch', 'numpy'], default='auto')
    ap.add_argument('--scripts-dir', default=None)
    ap.add_argument('--out', default='rollout.png')
    a = ap.parse_args()
    if a.mode != 'expert' and not a.teacher:
        ap.error('--teacher is required for --mode teacher and --mode hybrid')

    scfg = StudentCfg.from_xacro(a.student_xacro) if os.path.exists(a.student_xacro) else StudentCfg()
    scam = scfg.camera()
    tcfg = TeacherCfg(); tcfg.steer_sign = a.steer_sign
    cfg = LabelCfg(scale=a.scale, v_max=scfg.max_linear, w_max=scfg.max_angular, lookahead=a.lookahead,
                   blend_a0_deg=a.a0, blend_a1_deg=a.a1, mode=a.mode)
    teacher = None
    if a.mode != 'expert':
        teacher = Teacher(a.teacher, backend=a.backend, scripts_dir=a.scripts_dir)
    style = TeacherStyle(road_width_T=a.road_width)

    tracks = [Track.load(f) for f in a.track]
    if a.points:
        tracks.append(Track(parse_points(a.points), False, 'points'))
    rng = np.random.default_rng(1)
    tracks += [random_track(rng, min_sep=max(0.5, 1.5 * a.road_width / a.scale + 0.1)) for _ in range(a.random)]
    if not tracks:
        tracks = [Track(np.array([[0, 0], [2.5, 0], [2.5, 1.4], [0, 1.4], [0, 2.8], [2.5, 2.8]]), False, 'zigzag')]

    backend = teacher.backend if teacher is not None else 'not used'
    print(f'mode={a.mode}  teacher backend={backend}  scale={a.scale}  blend={a.a0}->{a.a1} deg')
    results = []
    for i, t in enumerate(tracks):
        r = simulate(t, teacher, tcfg, cfg, style, scam, seed=i)
        results.append(r)
        print(f"  track {i} ({t.name}, {t.length:.1f} m): {r['reason']:14s} progress={r['progress']:.2f} "
              f"mean err={r['mean_err'] * 100:.1f} cm  max err={r['max_err'] * 100:.1f} cm  "
              f"look-ahead target visible {r['target_visible'] * 100:.0f}% of the time  ({r['seconds_sim']:.0f} s)")
        draw(t, [r], a.out if len(tracks) == 1 else a.out.replace('.png', f'_{i}.png'))
    print('trajectory image(s) written next to', a.out)


if __name__ == '__main__':
    main()
