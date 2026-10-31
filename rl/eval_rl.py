#!/usr/bin/env python3
"""
eval_rl.py - run a policy in the environment and report how well it follows the line.

  python3 eval_rl.py --policy baseline --track hexagon_points.json circle_2_5_points.json   # geometric look-ahead controller
  python3 eval_rl.py --policy ppo --model rl_out/ppo_final.zip --track ... --random 20        # a trained checkpoint
  python3 eval_rl.py --policy jit --model rl_policy.pt  --track ...                           # the exported policy

Policies: baseline (privileged geometric controller; NOT a teacher), ppo, jit, random, straight, stand, spin.
User tracks are run once from the start of the line with a clean, Gazebo-like appearance; --random N adds N random tracks
with full domain randomisation. Writes a trajectory image per user track.
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rl_env import LineFollowEnv, expert_action
from linefollow_lib import Track


def load_policy(kind, model_path):
    if kind == 'baseline':
        return lambda env, obs: expert_action(env)
    if kind == 'random':
        return lambda env, obs: env.action_space.sample()
    fixed = {'straight': [1.0, 0.0], 'stand': [-1.0, 0.0], 'spin': [-1.0, 1.0]}
    if kind in fixed:
        return lambda env, obs: np.array(fixed[kind], np.float32)
    if kind == 'ppo':
        from stable_baselines3 import PPO
        m = PPO.load(model_path, device='cpu')
        return lambda env, obs: m.predict(obs, deterministic=True)[0]
    if kind == 'jit':
        import torch
        net = torch.jit.load(model_path, map_location='cpu').eval()

        def act(env, obs):
            with torch.no_grad():
                return net(torch.from_numpy(obs[None]))[0].numpy()
        return act
    raise SystemExit(f'unknown policy {kind}')


def run_episode(env, policy, options, seed):
    obs, _ = env.reset(seed=seed, options=options)
    R, dists, traj, term, trunc, info = 0.0, [], [], False, False, {}
    while not (term or trunc):
        obs, r, term, trunc, info = env.step(policy(env, obs))
        R += r
        dists.append(info['dist'])
        traj.append((env.x, env.y))
    result = 'success' if info['is_success'] else ('left the line' if term else 'timeout')
    return dict(R=R, steps=len(dists), result=result, mean=float(np.mean(dists)), max=float(np.max(dists)), traj=traj)


def draw(track, traj, path, ppm=140):
    pts = np.vstack([track.points, np.array(traj)])
    lo, hi = pts.min(0) - 0.4, pts.max(0) + 0.4
    W, H = int((hi[0] - lo[0]) * ppm) + 1, int((hi[1] - lo[1]) * ppm) + 1
    img = np.full((H, W, 3), 245, np.uint8)
    to = lambda p: (int((p[0] - lo[0]) * ppm), int(H - 1 - (p[1] - lo[1]) * ppm))
    cv2.polylines(img, [np.array([to(p) for p in track.points], np.int32)], track.closed, (90, 90, 90), 5)
    cv2.polylines(img, [np.array([to(p) for p in traj], np.int32)], False, (0, 0, 220), 2)
    cv2.circle(img, to(traj[0]), 5, (0, 150, 0), -1)
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--policy', default='baseline')
    ap.add_argument('--model', default=None)
    ap.add_argument('--track', nargs='*', default=[])
    ap.add_argument('--random', type=int, default=0, help='number of random tracks (full domain randomisation)')
    ap.add_argument('--student-xacro', default=os.path.join(HERE, '..', 'urdf', 'diffdrive_bot.xacro'))
    ap.add_argument('--out', default='eval')
    a = ap.parse_args()

    env = LineFollowEnv(student_xacro=a.student_xacro, tracks=a.track, seed=0)
    policy = load_policy(a.policy, a.model)
    print(f'policy: {a.policy}  (return scale: baseline about +25..+40 per finished episode, random actions about +1)')
    for i, tr in enumerate(env.user_tracks):
        r = run_episode(env, policy, dict(track=tr, start_s=0.0, lateral=0.0, heading=0.0, clean=True), 100 + i)
        draw(tr, r['traj'], f'{a.out}_track{i}.png')
        print(f"  {os.path.basename(a.track[i]):28s} {r['result']:13s} return {r['R']:6.1f}  steps {r['steps']:4d}  "
              f"lateral error mean {r['mean'] * 100:4.1f} cm  max {r['max'] * 100:5.1f} cm   -> {a.out}_track{i}.png")
    if a.random:
        rs = [run_episode(env, policy, {'track': __import__('linefollow_lib').random_track(env.rng, min_sep=0.6)}, 1000 + k)
              for k in range(a.random)]
        print(f"  {a.random} random tracks (randomised appearance + dynamics): success "
              f"{np.mean([r['result'] == 'success' for r in rs]) * 100:.0f}%  mean return {np.mean([r['R'] for r in rs]):.1f}  "
              f"mean/max lateral error {np.mean([r['mean'] for r in rs]) * 100:.1f}/{np.max([r['max'] for r in rs]) * 100:.1f} cm")


if __name__ == '__main__':
    main()
