#!/usr/bin/env python3
"""
train_rl.py - train the line follower with PPO from pixels (no teacher, no expert labels).

  pip install stable-baselines3 gymnasium           # inside the container's venv, once

  python3 train_rl.py --track /tmp/diffdrivebot/hexagon_points.json \
                              /tmp/diffdrivebot/circle_2_5_points.json \
                      --out /tmp/diffdrivebot/rl_out --timesteps 1500000 --n-envs 4

Every `--checkpoint-every` steps a checkpoint (ppo_<steps>_steps.zip) is written; ppo_final.zip at the end.
Watch  rollout/ep_rew_mean  and  rollout/success_rate  (a geometric baseline scores about +25..+40 and 100 %; random actions about +1).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='rl_out')
    ap.add_argument('--track', nargs='*', default=[], help='track JSON files from sdf_to_track.py / make_line_world.py')
    ap.add_argument('--user-share', type=float, default=0.5, help='fraction of episodes on --track tracks (rest: random tracks)')
    ap.add_argument('--timesteps', type=int, default=1_500_000)
    ap.add_argument('--n-envs', type=int, default=4, help='parallel environments (processes); about the number of free CPU cores')
    ap.add_argument('--n-steps', type=int, default=256, help='steps per environment per PPO update')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--n-epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--ent-coef', type=float, default=0.003)
    ap.add_argument('--log-std-init', type=float, default=-0.5, help='initial exploration noise (actions live in [-1, 1])')
    ap.add_argument('--shadow-prob', type=float, default=0.6, help='0 if your worlds have cast_shadows=false')
    ap.add_argument('--line-width', type=float, default=0.10)
    ap.add_argument('--student-xacro', default=os.path.join(HERE, '..', 'urdf', 'diffdrive_bot.xacro'))
    ap.add_argument('--checkpoint-every', type=int, default=100_000)
    ap.add_argument('--resume', default=None, help='continue from a checkpoint .zip')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--torch-threads', type=int, default=2)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from rl_env import LineFollowEnv

    torch.set_num_threads(args.torch_threads)
    os.makedirs(args.out, exist_ok=True)

    class LineCNN(BaseFeaturesExtractor):
        """Small CNN for 3 stacked 96x128 grayscale frames (inputs arrive as floats in [0, 1])."""

        def __init__(self, observation_space, features_dim=256):
            super().__init__(observation_space, features_dim)
            c = observation_space.shape[0]
            self.cnn = nn.Sequential(nn.Conv2d(c, 16, 5, stride=2), nn.ReLU(),
                                     nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(), nn.Flatten())
            with torch.no_grad():
                n = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
            self.fc = nn.Sequential(nn.Linear(n, features_dim), nn.ReLU())

        def forward(self, obs):
            return self.fc(self.cnn(obs))

    env_kwargs = dict(student_xacro=args.student_xacro, tracks=list(args.track), user_share=args.user_share,
                      line_width=args.line_width, shadow_prob=args.shadow_prob)
    check_env(LineFollowEnv(seed=args.seed, **env_kwargs), warn=True)         # fail early on API mistakes
    venv = make_vec_env(LineFollowEnv, n_envs=args.n_envs, seed=args.seed, vec_env_cls=SubprocVecEnv,
                        env_kwargs=env_kwargs)

    if args.resume:
        model = PPO.load(args.resume, env=venv, device=args.device)
    else:
        model = PPO('CnnPolicy', venv, learning_rate=lambda p: args.lr * max(p, 0.1), n_steps=args.n_steps,
                    batch_size=args.batch_size, n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=args.ent_coef, vf_coef=0.5, max_grad_norm=0.5,
                    policy_kwargs=dict(features_extractor_class=LineCNN,
                                       features_extractor_kwargs=dict(features_dim=256),
                                       net_arch=dict(pi=[128], vf=[128]), log_std_init=args.log_std_init),
                    tensorboard_log=os.path.join(args.out, 'tb'), seed=args.seed, device=args.device, verbose=1)

    ckpt = CheckpointCallback(save_freq=max(args.checkpoint_every // args.n_envs, 1), save_path=args.out,
                              name_prefix='ppo')
    model.learn(total_timesteps=args.timesteps, callback=ckpt, log_interval=5,
                reset_num_timesteps=not args.resume)
    final = os.path.join(args.out, 'ppo_final')
    model.save(final)
    print(f'saved {final}.zip - next: python3 export_policy.py --model {final}.zip --out <dir>/rl_policy.pt')


if __name__ == '__main__':
    main()
