#!/usr/bin/env python3
"""
probe_teacher.py - check that the teacher produces sensible steering on the synthetic "road" renders,
and calibrate the road width.

For each candidate road width it renders a straight road and a pair of curves from the teacher's camera,
asks the teacher for steering, and measures whether the response is *restoring*:
  * car left of the road centre / pointing left  -> steer right
  * left curve -> steer left, right curve -> steer right
The teacher output is mirror-averaged (see Teacher.__call__) and its sign is reported so you can verify
TeacherCfg.steer_sign. Writes probe_montage.png (synthetic teacher views) so you can compare them with a
real frame from your Ackermann simulation (--frame real.png appends it).

  python3 probe_teacher.py --teacher /tmp/diffdrivebot/model.pt
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from distill_lib import DistField, Teacher, TeacherCfg, TeacherStyle, render_teacher, teacher_camera


def arc(R_T, left, scale):
    R = R_T / scale
    ang = np.linspace(0, 150 / R_T, 200)
    sg = 1 if left else -1
    return np.vstack([[[-6, 0]], np.column_stack([R * np.sin(ang), sg * R * (1 - np.cos(ang))])])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--teacher', required=True)
    ap.add_argument('--scale', type=float, default=14.0)
    ap.add_argument('--widths', type=float, nargs='*', default=[2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0])
    ap.add_argument('--no-symmetrize', action='store_true')
    ap.add_argument('--backend', choices=['auto', 'torch', 'numpy'], default='auto')
    ap.add_argument('--scripts-dir', default=None)
    ap.add_argument('--frame', default=None, help='optional real camera frame to place next to the synthetic ones')
    ap.add_argument('--out', default='probe_montage.png')
    a = ap.parse_args()

    tcfg = TeacherCfg()
    T = Teacher(a.teacher, backend=a.backend, scripts_dir=a.scripts_dir, symmetrize=not a.no_symmetrize)
    s = a.scale
    rng = np.random.default_rng(0)
    cam = teacher_camera(tcfg, s)
    straight = np.array([[-8.0, 0.0], [60.0, 0.0]])
    fs = DistField(straight, False, rng=rng, texture=True)
    fl, fr_ = DistField(arc(8, True, s), False, rng=rng, texture=True), DistField(arc(8, False, s), False, rng=rng, texture=True)

    print(f'teacher backend: {T.backend}, mirror-averaged: {not a.no_symmetrize}')
    print('steering is reported with the RAW network sign (positive = what the network calls "positive").')
    print(f'{"width[m]":>8} | raw steer: car LEFT of centre | car RIGHT of centre | left curve | right curve | restoring score (higher = better)')
    best, rows = None, []
    for W in a.widths:
        st = TeacherStyle(road_width_T=W)
        def raw(pose, field):
            return T(render_teacher(cam, pose, field, st, s, rng))[1]
        off_l = np.mean([raw((10, o / s, 0), fs) for o in (0.5, 1.0, 1.5, 2.0)])
        off_r = np.mean([raw((10, -o / s, 0), fs) for o in (0.5, 1.0, 1.5, 2.0)])
        cl, cr = raw((0, 0, 0), fl), raw((0, 0, 0), fr_)
        # Restoring behaviour: a car left of the centre steers RIGHT while a left curve makes it steer LEFT,
        # so (off_l - off_r) and (cl - cr) must have OPPOSITE signs, whatever the network's sign convention.
        score = (abs(off_l - off_r) + abs(cl - cr)) / 2 if np.sign(off_l - off_r) != np.sign(cl - cr) else -(abs(off_l - off_r) + abs(cl - cr)) / 2
        rows.append((W, off_l, off_r, cl, cr, score))
        print(f'{W:8.1f} | {off_l:+10.3f}                   | {off_r:+10.3f}          | {cl:+8.3f}   | {cr:+8.3f}    | {score:+.3f}')
        if best is None or score > best[1]:
            best = (W, score, np.sign(cl - cr))
    W, score, sgn = best
    print(f'\nbest road width: {W} m (score {score:+.3f}).  Use --road-width {W} in the other tools.')
    print('sign check: a LEFT curve gives raw steering', 'POSITIVE' if sgn > 0 else 'NEGATIVE',
          '-> teacher "positive steering" means', 'LEFT (steer_sign=+1)' if sgn > 0 else 'RIGHT (steer_sign=-1, the default)')
    if score < 0.05:
        print('WARNING: weak or non-restoring response - the synthetic road does not look like the teacher training data.')

    st = TeacherStyle(road_width_T=W)
    tiles = [render_teacher(cam, (10, o / s, 0), fs, st, s, rng) for o in (-1.0, 0.0, 1.0)]
    tiles += [render_teacher(cam, (0, 0, 0), fl, st, s, rng), render_teacher(cam, (0, 0, 0), fr_, st, s, rng)]
    if a.frame and os.path.exists(a.frame):
        f = cv2.imread(a.frame, 0)
        tiles.append(cv2.resize(f, (256, 192)))
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(a.out, np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
