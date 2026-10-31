"""
rl_env.py - Gymnasium environment: a differential-drive bot learns to follow a centerline from camera pixels.

  observation  last `frame_stack` grayscale frames of the bot's own camera (uint8, shape (C, 96, 128)), rendered by the
               same ground-plane renderer as before: dark line on a light floor, robot body, robot shadow
  action       2 numbers in [-1, 1]:  a[0] -> speed   v = (a[0] + 1) / 2 * v_max      (never reverses)
                                      a[1] -> turn    w = a[1] * w_max                (+ = left)
               (the same normalised outputs the deployed node scales by max_linear / max_angular velocity)
  step         0.1 s of simulated time; first-order actuator lag, motor gains and the lag are randomised
  reward       progress along the centerline, weighted by closeness to it (zero beyond 25 cm)
               + potential-based shaping for improving the look-ahead heading error (rewards pivoting at a corner)
               - a small action-jerk penalty. -5 and the episode ends when the bot is more than 35 cm from the line,
               +5 when it finishes (open track: reaches the end; closed track: one lap). All scaled by `reward_scale`.
               There is deliberately NO per-step penalty: with one, an agent learns to end episodes early or to stall.

The centerline is PRIVILEGED information: it is used for the reward and for termination only. The policy never sees it.
Tracks: your own (track JSON files from sdf_to_track.py) mixed with random ones (sharp corners, arcs, polygons, ellipses).
Domain randomisation per episode: floor/line brightness, noise, blur, vignette, robot shadow, camera height/pitch jitter.

Needs only numpy + OpenCV (gymnasium is used if installed; a tiny stand-in is used otherwise so it can be tested anywhere).
"""
import math
import os
import sys
from collections import deque

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from linefollow_lib import (Body, Cam, DistField, ExpertCfg, StudentCfg, StudentStyle, Track, expert_label,
                            jitter_cam, random_track, render_student, self_occlusion_fraction)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:                                           # stand-in so the env can be tested without gymnasium
    class _Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low, self.high, self.shape, self.dtype = low, high, tuple(shape), dtype

        def sample(self):
            return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)

    class spaces:                                             # noqa: N801
        Box = _Box

    class gym:                                                # noqa: N801
        class Env:
            def reset(self, seed=None, options=None):
                return None


class LineFollowEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, student_xacro=None, tracks=(), user_share=0.5, line_width=0.10, shadow_prob=0.6,
                 obs_hw=(96, 128), frame_stack=3, dt=0.1, v_max=None, w_max=None, off_line=0.35,
                 field_res=0.005, reward_scale=0.1, seed=None):
        cv2.setNumThreads(1)
        xacro = student_xacro or os.path.join(HERE, '..', 'urdf', 'diffdrive_bot.xacro')
        self.scfg = StudentCfg.from_xacro(xacro) if os.path.exists(xacro) else StudentCfg()
        self.body = self.scfg.body or Body()
        h, w = obs_hw
        c = self.scfg
        self.cam = Cam(c.cam_height, c.cam_forward, c.cam_pitch, c.cam_hfov, w, h)
        occl = self_occlusion_fraction(self.cam, self.body)
        if occl > 0.005:
            print(f'WARNING: the robot body hides {occl * 100:.1f}% of the camera image. Keep camera_forward >= '
                  f'body_length / 2 in diffdrive_bot.xacro, otherwise the policy will see a body that is not there in Gazebo '
                  f'or the other way round.')
        self.v_max = float(v_max or c.max_linear)
        self.w_max = float(w_max or c.max_angular)
        self.dt, self.off_line, self.field_res, self.reward_scale = dt, off_line, field_res, reward_scale
        self.user_share, self.line_width, self.shadow_prob = user_share, line_width, shadow_prob
        self.frame_stack = frame_stack
        self.user_tracks = [t if isinstance(t, Track) else Track.load(t) for t in tracks]
        self._user_fields = {}
        self.observation_space = spaces.Box(0, 255, (frame_stack, h, w), np.uint8)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.rng = np.random.default_rng(seed)
        self.track = None

    # ------------------------------------------------------------------ helpers
    def _field_for(self, tr):
        if tr in self.user_tracks:
            if id(tr) not in self._user_fields:
                self._user_fields[id(tr)] = DistField(tr.points, tr.closed, res=self.field_res, margin=1.0)
            return self._user_fields[id(tr)]
        return DistField(tr.points, tr.closed, res=self.field_res, margin=1.0)

    def _render(self):
        return render_student(self.cam_j, (self.x, self.y, self.th), self.field, self.style, self.ground,
                              self.rng, body=self.body)

    def _geometry(self):
        i, s, e, tyaw, dist = self.track.nearest(self.x, self.y)
        tx, ty, _ = self.track.point_at(s + 0.25)
        bx = math.cos(self.th) * (tx - self.x) + math.sin(self.th) * (ty - self.y)
        by = -math.sin(self.th) * (tx - self.x) + math.cos(self.th) * (ty - self.y)
        return s, e, dist, math.atan2(by, bx)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        if hasattr(super(), 'reset') and hasattr(gym, 'make'):
            super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        rng, opt = self.rng, (options or {})

        tr = opt.get('track')
        if tr is None:
            if self.user_tracks and rng.random() < self.user_share:
                tr = self.user_tracks[int(rng.integers(len(self.user_tracks)))]
            else:
                tr = random_track(rng, min_sep=0.6)
        self.track, self.field = tr, self._field_for(tr)
        if tr.ground_rect:
            self.ground = tr.ground_rect
        else:
            lo, hi = tr.points.min(0) - 2.0, tr.points.max(0) + 2.0
            self.ground = (lo[0], lo[1], hi[0], hi[1])
        L = tr.length

        # start pose near the line (sometimes badly placed so that it must learn to recover)
        if 'start_s' in opt:
            s0 = opt['start_s']
        elif tr.closed:
            s0 = rng.uniform(0, L)
        else:
            s0 = 0.0 if rng.random() < 0.5 else rng.uniform(0, max(0.0, L - 1.0))
        qx, qy, tyaw = tr.point_at(s0)
        e = opt.get('lateral', float(np.clip(rng.normal(0, 0.05), -0.2, 0.2)))
        he = opt.get('heading', rng.normal(0, math.radians(15)) if rng.random() < 0.8
                     else rng.uniform(-math.radians(70), math.radians(70)))
        self.x, self.y, self.th = qx - math.sin(tyaw) * e, qy + math.cos(tyaw) * e, tyaw + he
        self.v = self.w = 0.0
        self.prev_a = np.zeros(2, np.float32)

        # per-episode randomisation (constant within the episode so the frame stack stays consistent)
        lw = tr.line_width or self.line_width
        if opt.get('clean'):
            self.style = StudentStyle(line_width=lw, floor=0.8, line=0.02, bg=0.8, gain=1.0, noise=0.01,
                                      shadow=0.5, body_gray=0.3)
            self.cam_j, self.tau, self.gain_v, self.gain_w = self.cam, 0.15, 1.0, 1.0
        else:
            self.style = StudentStyle.sample(rng, lw, self.shadow_prob)
            self.cam_j = jitter_cam(self.cam, rng)
            self.tau = rng.uniform(0.08, 0.30)
            self.gain_v, self.gain_w = rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1)

        s, _, dist, alpha0 = self._geometry()
        self.s_prev, self.total, self.steps, self.alpha_prev = s, 0.0, 0, alpha0
        remaining = L - s0 if not tr.closed else L
        self.max_steps = int(np.clip(1.2 * remaining / (0.3 * self.v_max * self.dt), 200, 1500))
        frame = self._render()
        self.frames = deque([frame] * self.frame_stack, maxlen=self.frame_stack)
        return np.stack(self.frames), {'track': tr.name, 'length': L}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32).reshape(2), -1.0, 1.0)
        v_cmd, w_cmd = (a[0] + 1.0) / 2.0 * self.v_max, a[1] * self.w_max
        n_sub = 5
        h = self.dt / n_sub
        k = 1.0 - math.exp(-h / self.tau)
        for _ in range(n_sub):
            self.v += (v_cmd - self.v) * k
            self.w += (w_cmd - self.w) * k
            vv, ww = self.v * self.gain_v, self.w * self.gain_w
            th_mid = self.th + 0.5 * ww * h
            self.x += vv * math.cos(th_mid) * h
            self.y += vv * math.sin(th_mid) * h
            self.th += ww * h
        self.steps += 1

        tr, L = self.track, self.track.length
        s, e, dist, alpha = self._geometry()
        ds = s - self.s_prev
        if tr.closed:
            ds = (ds + L / 2) % L - L / 2
        self.s_prev = s
        self.total += ds
        unit = self.v_max * self.dt
        prog = float(np.clip(ds, -0.5 * unit, 1.5 * unit) / unit)

        w_lat = max(0.0, 1.0 - dist / 0.25)
        r = (prog * w_lat + 0.5 * (abs(self.alpha_prev) - abs(alpha))
             - 0.02 * float(np.abs(a - self.prev_a).sum()))
        self.prev_a, self.alpha_prev = a, alpha

        terminated, success = False, False
        if dist > self.off_line:
            r -= 5.0
            terminated = True
        else:
            done = (self.total >= L) if tr.closed else (s >= L - 0.15)
            if done:
                r += 5.0
                terminated = success = True
        truncated = (not terminated) and self.steps >= self.max_steps
        r *= self.reward_scale

        self.frames.append(self._render())
        info = {'lateral': e, 'dist': dist, 'alpha': alpha, 'v': self.v, 'w': self.w, 'is_success': success,
                'progress': (self.total if tr.closed else s) / L}
        return np.stack(self.frames), float(r), terminated, truncated, info


# ---------------------------------------------------------------------- scripted policies (baselines / tests)
def expert_action(env, cfg=None):
    """Geometric look-ahead controller using PRIVILEGED state. A baseline to compare against, not a teacher."""
    cfg = cfg or ExpertCfg(v_max=env.v_max, w_max=env.w_max)
    v, w, *_ = expert_label(env.track, env.x, env.y, env.th, cfg)
    return np.array([2.0 * v / env.v_max - 1.0, w / env.w_max], np.float32)


def make_env(rank, seed, **kwargs):
    """Factory for vectorised training (picklable, no torch import)."""
    def _init():
        return LineFollowEnv(seed=seed + rank, **kwargs)
    return _init
