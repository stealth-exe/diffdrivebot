#!/usr/bin/env python3
"""
make_distill_dataset.py - build a training set for the differential-drive student by
distilling the Ackermann AutopilotNet (teacher).

For every sampled robot pose (x, y, yaw) on a centerline track:
  1. render what the TEACHER's camera would see if the centerline were a road,
     and ask the teacher for (speed, steering)            -> teacher label
  2. convert the teacher's steering to a curvature and then to (v, omega) for the bot
  3. compute a geometric look-ahead "expert" label        -> used where the teacher cannot help
                                                             (large heading errors / sharp corners)
  4. blend the two labels
  5. render what the STUDENT's camera sees (dark line on a light floor) and save it
     with the blended label.

The output folder is directly usable by the existing train.py / dataset.py / eval.py
(CSV columns: image_id, velocity, steering_angle, max_velocity, max_steering_angle; here
`steering_angle` holds the angular velocity omega [rad/s]).

Example:
  python3 make_distill_dataset.py --teacher /tmp/diffdrivebot/model.pt \\
      --out /tmp/diffdrivebot/dataset_diffdrive --num-samples 15000 \\
      --track /tmp/diffdrivebot/line_world_points.json --random-tracks 30

With --mode expert the teacher is never used and --teacher may be omitted.
"""
import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from distill_lib import (Body, Cam, LabelContext, LabelCfg, StudentCfg, StudentStyle, TeacherCfg, TeacherStyle,
                         Teacher, Track, jitter_cam, line_pixel_fraction, make_label, random_track,
                         render_student, teacher_camera, point_visible, expert_label, self_occlusion_fraction)


def sample_pose(track, rng):
    """Mix of on-line states, recoveries (offset / rotated) and corner approaches."""
    if track.corners and rng.random() < 0.30:        # on-line approach to a vertex (the critical states)
        cs, _ = track.corners[int(rng.integers(len(track.corners)))]
        s = cs - rng.uniform(0.0, 0.45)
        s = (s % track.length) if track.closed else min(max(s, 0.0), track.length - 0.3)
        qx, qy, tyaw = track.point_at(s)
        e, he = rng.normal(0.0, 0.025), rng.normal(0.0, math.radians(4))
        return (qx - math.sin(tyaw) * e, qy + math.cos(tyaw) * e, tyaw + he)
    if track.corners and rng.random() < 0.65:
        cs, _ = track.corners[int(rng.integers(len(track.corners)))]
        s = cs + rng.normal(0.0, 0.25)
    else:
        s = rng.uniform(0.0, track.length)
    s = (s % track.length) if track.closed else min(max(s, 0.0), track.length - 0.3)
    qx, qy, tyaw = track.point_at(s)
    e = rng.normal(0.0, 0.04) if rng.random() < 0.55 else rng.uniform(-0.30, 0.30)
    r = rng.random()
    if r < 0.55:
        he = rng.normal(0.0, math.radians(8))
    elif r < 0.90:
        he = rng.uniform(-math.radians(60), math.radians(60))
    else:
        he = rng.uniform(-math.radians(100), math.radians(100))
    return (qx - math.sin(tyaw) * e, qy + math.cos(tyaw) * e, tyaw + he)


def run_unit(job):
    """Generate `n` samples on one track. Returns (rows, skipped)."""
    (unit, track_src, n, first_id, seed, p) = job
    rng = np.random.default_rng(seed)
    tcfg = TeacherCfg() if not p['teacher_xacro'] else TeacherCfg.from_xacro(p['teacher_xacro'])
    tcfg.steer_sign = p['steer_sign']
    scfg = StudentCfg() if not p['student_xacro'] else StudentCfg.from_xacro(p['student_xacro'])
    cfg = LabelCfg(scale=p['scale'], corner_radius_T=p['corner_radius'], v_max=p['v_max'], w_max=p['w_max'],
                   v_floor_norm=p['v_floor'], lookahead=p['lookahead'], k_heading=p['k_heading'],
                   blend_a0_deg=p['a0'], blend_a1_deg=p['a1'], mode=p['mode'])
    tstyle = TeacherStyle(road_width_T=p['road_width'])
    track = Track(*track_src) if isinstance(track_src, tuple) else Track.load(track_src)
    teacher = None
    if p['mode'] != 'expert':
        teacher = p.get('_teacher') or Teacher(p['teacher'], backend=p['backend'], scripts_dir=p['scripts_dir'],
                                               symmetrize=not p['no_symmetrize'])
    ctx = LabelContext(track, cfg, tstyle, rng)
    ctx.set_ground_rect(p['ground_margin'])
    tcam = teacher_camera(tcfg, cfg.scale)
    scam = scfg.camera()
    body = scfg.body or Body()
    sun = tuple(float(v) for v in p['sun_dir'].split(','))
    lw = track.line_width or p['line_width']
    rows, skipped, k = [], 0, 0
    while k < n:
        pose = sample_pose(track, rng)
        # the image must contain the part of the path the label talks about
        *_, tx, ty = expert_label(track, pose[0], pose[1], pose[2], cfg)
        cam_j = jitter_cam(scam, rng)
        if not point_visible(cam_j, pose, tx, ty):
            skipped += 1
            continue
        sstyle = StudentStyle.sample(rng, lw, p['shadow_prob'])
        img = render_student(cam_j, pose, ctx.lfield, sstyle, ctx.ground_rect, rng, body=body, sun=sun)
        if line_pixel_fraction(img) < p['min_line_frac']:
            skipped += 1
            continue
        lab = make_label(ctx, teacher, tcfg, tcam, pose, rng)
        image_id = first_id + k
        cv2.imwrite(os.path.join(p['out'], f'{image_id}.png'), img)
        rows.append(dict(image_id=image_id, velocity=lab['v'], steering_angle=lab['w'],
                         max_velocity=cfg.v_max, max_steering_angle=cfg.w_max,
                         x=pose[0], y=pose[1], yaw=pose[2], track=unit,
                         teacher_v=lab['teacher_v'], teacher_w=lab['teacher_w'],
                         expert_v=lab['expert_v'], expert_w=lab['expert_w'],
                         expert_weight=lab['weight'], alpha=lab['alpha'], lateral=lab['lateral']))
        k += 1
    return rows, skipped


def montage(out_dir, rows, n=24, cols=6):
    rng = np.random.default_rng(0)
    pick = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    tiles = []
    for i in pick:
        r = rows[i]
        im = cv2.cvtColor(cv2.imread(os.path.join(out_dir, f"{r['image_id']}.png"), 0), cv2.COLOR_GRAY2BGR)
        im = cv2.resize(im, (256, 192))
        cv2.putText(im, f"v={r['velocity']:.2f} w={r['steering_angle']:+.2f}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(im, f"expert={r['expert_weight']:.2f}", (4, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1, cv2.LINE_AA)
        tiles.append(im)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(os.path.join(out_dir, 'preview.png'),
                np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--teacher', default=None,
                    help='teacher checkpoint (model.pt); required for --mode teacher/hybrid, ignored for --mode expert')
    ap.add_argument('--out', required=True, help='dataset output folder')
    ap.add_argument('--num-samples', type=int, default=15000)
    ap.add_argument('--track', nargs='*', default=[], help='track JSON files from make_line_world.py (<out>_points.json)')
    ap.add_argument('--random-tracks', type=int, default=30, help='number of random sharp-cornered tracks to add')
    ap.add_argument('--user-share', type=float, default=0.5, help='fraction of samples drawn from --track files')
    ap.add_argument('--teacher-xacro', default=None, help="teacher's vehicle.xacro (defaults are its values)")
    ap.add_argument('--student-xacro', default=os.path.join(HERE, '..', 'urdf', 'diffdrive_bot.xacro'))
    ap.add_argument('--scale', type=float, default=LabelCfg.scale, help='virtual teacher world scale (x student world)')
    ap.add_argument('--road-width', type=float, default=TeacherStyle.road_width_T, help='teacher road width [teacher m]')
    ap.add_argument('--corner-radius', type=float, default=LabelCfg.corner_radius_T, help='teacher road corner radius [teacher m]')
    ap.add_argument('--mode', choices=['teacher', 'hybrid', 'expert'], default='hybrid')
    ap.add_argument('--a0', type=float, default=LabelCfg.blend_a0_deg)
    ap.add_argument('--a1', type=float, default=LabelCfg.blend_a1_deg)
    ap.add_argument('--lookahead', type=float, default=LabelCfg.lookahead)
    ap.add_argument('--k-heading', type=float, default=LabelCfg.k_heading)
    ap.add_argument('--v-floor', type=float, default=LabelCfg.v_floor_norm)
    ap.add_argument('--v-max', type=float, default=None, help='default: from the student xacro')
    ap.add_argument('--w-max', type=float, default=None, help='default: from the student xacro')
    ap.add_argument('--steer-sign', type=float, default=TeacherCfg.steer_sign)
    ap.add_argument('--no-symmetrize', action='store_true', help='do not mirror-average the teacher output')
    ap.add_argument('--line-width', type=float, default=0.10, help='line width [m] for tracks without one (random tracks); '
                                                                  'track JSONs from sdf_to_track.py carry their own')
    ap.add_argument('--shadow-prob', type=float, default=0.6, help="fraction of images with the robot's own shadow (0 if your worlds have cast_shadows=false)")
    ap.add_argument('--sun-dir', default='-0.5,0.1,-0.9', help="direction the light travels (the world's <light><direction>)")
    ap.add_argument('--ground-margin', type=float, default=2.0, help='floor margin around the track when the JSON has no ground_rect')
    ap.add_argument('--min-line-frac', type=float, default=0.002, help='drop samples where the line is (almost) invisible (the look-ahead target must also be in view)')
    ap.add_argument('--backend', choices=['auto', 'torch', 'numpy'], default='auto')
    ap.add_argument('--scripts-dir', default=None, help='folder with model.py / dataset.py (for the torch backend)')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if a.mode != 'expert' and not a.teacher:
        ap.error('--teacher is required for --mode teacher and --mode hybrid')

    scfg = StudentCfg.from_xacro(a.student_xacro) if os.path.exists(a.student_xacro) else StudentCfg()
    p = dict(vars(a))
    p.update(student_xacro=a.student_xacro if os.path.exists(a.student_xacro) else None,
             v_max=a.v_max or scfg.max_linear, w_max=a.w_max or scfg.max_angular,
             scale=a.scale, road_width=a.road_width, corner_radius=a.corner_radius, v_floor=a.v_floor,
             a0=a.a0, a1=a.a1, k_heading=a.k_heading, no_symmetrize=a.no_symmetrize)
    os.makedirs(a.out, exist_ok=True)
    occl = self_occlusion_fraction(scfg.camera(), scfg.body or Body())
    print(f'robot body hides {occl * 100:.1f}% of the camera image')
    if occl > 0.005:
        print('WARNING: the chassis is visible in the camera image. The renderer paints it in, but the lost-line reflex and the '
              'line detector will count it as line. Move the camera ahead of the front edge (camera_forward >= body_length/2).')

    # track list -> jobs (one job = one track, so only one set of distance fields is alive at a time)
    rng = np.random.default_rng(a.seed)
    virt_w = a.road_width / a.scale
    sources, weights = [], []
    for f in a.track:
        sources.append(f)
        weights.append(a.user_share / len(a.track))
    n_rand = a.random_tracks if (a.random_tracks > 0 or not a.track) else 0
    n_rand = n_rand or 30
    for i in range(n_rand):
        t = random_track(rng, min_sep=max(0.5, 1.5 * virt_w + 0.1))
        sources.append((t.points.tolist(), False, f'random{i}'))
        weights.append((1.0 - (a.user_share if a.track else 0.0)) / n_rand)
    weights = np.array(weights) / np.sum(weights)
    counts = np.floor(weights * a.num_samples).astype(int)
    counts[0] += a.num_samples - counts.sum()
    first, jobs = 0, []
    for u, (src, c) in enumerate(zip(sources, counts)):
        if c > 0:
            jobs.append((u if isinstance(src, tuple) else os.path.basename(src), src, int(c), first, a.seed * 100003 + u, p))
            first += int(c)

    t0 = time.time()
    rows, skipped = [], 0
    if a.workers > 1:
        with mp.Pool(a.workers, maxtasksperchild=1) as pool:
            for k, (r, s) in enumerate(pool.imap_unordered(run_unit, jobs), 1):
                rows += r; skipped += s
                print(f'  [{k}/{len(jobs)}] {len(rows)} samples  ({time.time() - t0:.0f}s)', flush=True)
    else:
        if a.mode != 'expert':
            teacher = Teacher(a.teacher, backend=a.backend, scripts_dir=a.scripts_dir, symmetrize=not a.no_symmetrize)
            print(f'teacher backend: {teacher.backend}  (symmetrised: {not a.no_symmetrize})')
            p['_teacher'] = teacher
        for k, j in enumerate(jobs, 1):
            r, s = run_unit(j)
            rows += r; skipped += s
            print(f'  [{k}/{len(jobs)}] {len(rows)} samples  ({time.time() - t0:.0f}s)', flush=True)

    rows.sort(key=lambda r: r['image_id'])
    with open(os.path.join(a.out, 'dataset.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    montage(a.out, rows)

    v = np.array([r['velocity'] for r in rows]); om = np.array([r['steering_angle'] for r in rows])
    wt = np.array([r['expert_weight'] for r in rows])
    stats = dict(samples=len(rows), skipped_target_or_line_not_visible=skipped,
                 v_mean=float(v.mean()), omega_abs_mean=float(np.abs(om).mean()),
                 robot_body_in_image=round(occl, 4),
                 frac_omega_saturated=float((np.abs(om) > 0.95 * p['w_max']).mean()),
                 mean_teacher_share=float(1 - wt.mean()), frac_expert_dominated=float((wt > 0.5).mean()),
                 seconds=round(time.time() - t0, 1))
    meta = dict(stats=stats, max_linear_velocity=p['v_max'], max_angular_velocity=p['w_max'],
                params={k: v for k, v in p.items() if not k.startswith('_')})
    with open(os.path.join(a.out, 'distill_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    print(json.dumps(stats, indent=2))
    print(f"Wrote {len(rows)} samples to {a.out}  (see preview.png)")
    print(f"IMPORTANT: set max_linear_velocity={p['v_max']} and max_angular_velocity={p['w_max']} "
          f"in diffdrive_parameters.yaml (they were used to normalise the labels).")


if __name__ == '__main__':
    main()
