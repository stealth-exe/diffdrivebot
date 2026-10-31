#!/usr/bin/env python3
"""
export_policy.py - turn a trained PPO checkpoint into a standalone TorchScript policy for the ROS node.

  python3 export_policy.py --model rl_out/ppo_final.zip --out /tmp/diffdrivebot/rl_policy.pt

The exported module maps a uint8 tensor (1, 3, 96, 128) of stacked grayscale frames to 2 actions in [-1, 1]
(speed, turn) - the deterministic policy mean. It needs no stable-baselines3 at runtime. The script checks the export
against the checkpoint on real observations from the environment before writing it.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--student-xacro', default=os.path.join(HERE, '..', 'urdf', 'diffdrive_bot.xacro'))
    a = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    from rl_env import LineFollowEnv

    model = PPO.load(a.model, device='cpu')
    policy = model.policy.eval()

    class Deployable(torch.nn.Module):
        def __init__(self, p):
            super().__init__()
            self.p = p

        def forward(self, obs_u8):
            return torch.clamp(self.p.get_distribution(obs_u8).distribution.mean, -1.0, 1.0)

    env = LineFollowEnv(student_xacro=a.student_xacro, seed=1)
    obs, _ = env.reset(seed=1)
    wrapper = Deployable(policy).eval()
    traced = torch.jit.trace(wrapper, torch.from_numpy(obs[None]), check_trace=False)

    worst = 0.0
    for _ in range(30):
        for _ in range(10):
            obs, _, term, trunc, _ = env.step(model.predict(obs, deterministic=True)[0])
            if term or trunc:
                obs, _ = env.reset()
        ref = model.predict(obs, deterministic=True)[0]
        with torch.no_grad():
            out = traced(torch.from_numpy(obs[None]))[0].numpy()
        worst = max(worst, float(np.abs(ref - out).max()))
    print(f'export check: max |checkpoint - exported| over 30 real observations = {worst:.2e}')
    if worst > 1e-3:
        raise SystemExit('export does not match the checkpoint - not written')
    traced.save(a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
