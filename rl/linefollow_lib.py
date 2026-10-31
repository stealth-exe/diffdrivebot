"""
linefollow_lib.py - shared code for the reinforcement-learning line follower (no teacher network, no expert labels).

Contents: robot/camera description read from diffdrive_bot.xacro, tracks (polylines with sharp corners, arcs, closed
loops), a fast ground-plane camera renderer (dark line on a light floor, robot body and shadow, domain randomisation),
and a geometric look-ahead controller that is used ONLY as a baseline to compare the learned policy against.

Coordinate conventions (REP-103): +x forward, +y left, +z up, yaw CCW, positive angular velocity = turn LEFT.
"""
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import cv2
import numpy as np


def load_xacro_args(path):
    args = {}
    root = ET.parse(path).getroot()
    for el in root.iter():
        if el.tag.endswith('}arg') or el.tag == 'arg':
            name, default = el.get('name'), el.get('default')
            if name is not None and default is not None:
                try:
                    args[name] = float(default)
                except ValueError:
                    args[name] = default
    return args


@dataclass
class Cam:
    """Pinhole camera rigidly attached to a robot reference point."""
    height: float       # camera height above the ground [m]
    forward: float      # camera offset ahead of the reference point [m]
    pitch: float        # downward pitch [rad]
    hfov: float         # horizontal field of view [rad]
    width: int = 256
    px_height: int = 192


@dataclass
class Body:
    """Robot geometry in the robot frame (origin = wheel-axle centre) for self-occlusion and shadows."""
    length: float = 0.24
    width: float = 0.20
    z_bottom: float = 0.035          # chassis underside above the ground
    z_top: float = 0.095             # chassis top above the ground
    wheel_radius: float = 0.05
    wheel_width: float = 0.03
    wheel_sep: float = 0.23
    mast_x: float = 0.10
    cam_height: float = 0.22
    cam_forward: float = 0.13


@dataclass
class StudentCfg:
    """Diff-drive bot parameters (read from diffdrive_bot.xacro)."""
    cam_height: float = 0.22
    cam_forward: float = 0.13
    cam_pitch: float = 0.8
    cam_hfov: float = 1.8
    img_w: int = 256
    img_h: int = 192
    max_linear: float = 0.4
    max_angular: float = 2.0
    body: Body = None

    @staticmethod
    def from_xacro(path):
        a = load_xacro_args(path)
        body = Body(length=a['body_length'], width=a['body_width'], z_bottom=a['ground_clearance'],
                    z_top=a['ground_clearance'] + a['body_height'], wheel_radius=a['wheel_radius'],
                    wheel_width=a['wheel_width'], wheel_sep=a['body_width'] + a['wheel_width'],
                    mast_x=a['camera_forward'] - 0.03, cam_height=a['camera_height'],
                    cam_forward=a['camera_forward'])
        return StudentCfg(
            cam_height=a['camera_height'], cam_forward=a['camera_forward'],
            cam_pitch=a['camera_pitch'], cam_hfov=a['camera_fov'],
            img_w=int(a['image_width']), img_h=int(a['image_height']),
            max_linear=a['max_linear_velocity'], max_angular=a['max_angular_velocity'], body=body)

    def camera(self):
        return Cam(self.cam_height, self.cam_forward, self.cam_pitch, self.cam_hfov, self.img_w, self.img_h)


def parse_points(text):
    pts = []
    for tok in text.split():
        x, y = tok.split(',')
        pts.append((float(x), float(y)))
    if len(pts) < 2:
        raise ValueError('need at least two points')
    return pts


def resample_polyline(P, closed, step):
    """Return (rows[s,x,y,yaw], total_length) sampled every `step` metres."""
    P = np.asarray(P, float)
    pts = np.vstack([P, P[:1]]) if closed else P
    seg = np.diff(pts, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    keep = L > 1e-9
    seg, L, starts = seg[keep], L[keep], pts[:-1][keep]
    out, s0 = [], 0.0
    for p0, d, l in zip(starts, seg, L):
        n = max(1, int(round(l / step)))
        frac = (np.arange(n) / n)
        xy = p0[None, :] + frac[:, None] * d[None, :]
        yaw = np.full(n, math.atan2(d[1], d[0]))
        out.append(np.column_stack([s0 + frac * l, xy, yaw]))
        s0 += l
    last = pts[-1]
    out.append(np.array([[s0, last[0], last[1], out[-1][-1, 3]]]))
    return np.vstack(out), s0


class Track:
    """A centerline made of straight segments (corners may be sharp)."""

    def __init__(self, points, closed=False, name='track', line_width=None, ground_rect=None):
        self.points = np.asarray(points, float)
        self.closed = bool(closed)
        self.name = name
        self.line_width = line_width
        self.ground_rect = tuple(ground_rect) if ground_rect else None
        self.dense, self.length = resample_polyline(self.points, self.closed, 0.01)
        self._compute_corners()

    @staticmethod
    def load(path):
        with open(path) as f:
            d = json.load(f)
        return Track(d['points'], d.get('closed', False), os.path.basename(path),
                     d.get('line_width'), d.get('ground_rect'))

    def _compute_corners(self):
        P = self.points
        n = len(P)
        seglen = np.hypot(*(np.roll(P, -1, 0) - P).T)
        if not self.closed:
            seglen = seglen[:-1]
        cum = np.concatenate([[0], np.cumsum(seglen)])
        idx = range(n) if self.closed else range(1, n - 1)
        self.corners = []                           # (arclength, turning angle rad)
        for i in idx:
            a, b, c = P[i - 1], P[i], P[(i + 1) % n]
            u, v = b - a, c - b
            ang = math.atan2(u[0] * v[1] - u[1] * v[0], u @ v)
            if abs(ang) > math.radians(15):
                self.corners.append((cum[i % len(cum)] if i < len(cum) else self.length, ang))

    def nearest(self, x, y):
        d2 = (self.dense[:, 1] - x) ** 2 + (self.dense[:, 2] - y) ** 2
        i = int(np.argmin(d2))
        s, qx, qy, yaw = self.dense[i]
        e = math.cos(yaw) * (y - qy) - math.sin(yaw) * (x - qx)    # + = robot is left of the line
        return i, float(s), float(e), float(yaw), math.sqrt(float(d2[i]))

    def point_at(self, s):
        """Point (x, y, yaw) at arclength s. Interpolates on the true arclength column: the resampled
        points are NOT exactly 1 cm apart when the polyline has short segments (e.g. a circle)."""
        if self.closed:
            s = s % self.length
        s = min(max(s, 0.0), self.length)
        S = self.dense[:, 0]
        i = int(np.clip(np.searchsorted(S, s), 0, len(S) - 1))
        return (float(np.interp(s, S, self.dense[:, 1])), float(np.interp(s, S, self.dense[:, 2])),
                float(self.dense[i, 3]))

    def start_pose(self):
        return float(self.dense[0, 1]), float(self.dense[0, 2]), float(self.dense[0, 3])


def _path_ok(pts, closed, min_sep):
    """True if no two parts of the path that are far apart ALONG the path come closer than min_sep."""
    dense, L = resample_polyline(pts, closed, 0.05)
    P = dense[:-1, 1:3] if closed else dense[:, 1:3]
    S = dense[:len(P), 0]
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    G = np.abs(S[:, None] - S[None, :])
    if closed:
        G = np.minimum(G, L - G)
    return not ((G > 1.5 * min_sep) & (D < min_sep)).any()


def _arc_points(pos, heading, radius, sweep, step=0.05):
    """Points of a circular arc starting at pos/heading; sweep > 0 turns left."""
    n = max(2, int(math.ceil(abs(sweep) * radius / step)))
    sg = 1.0 if sweep > 0 else -1.0
    centre = pos + radius * sg * np.array([-math.sin(heading), math.cos(heading)])
    a0 = math.atan2(*(pos - centre)[::-1])
    pts = [centre + radius * np.array([math.cos(a0 + sweep * k / n), math.sin(a0 + sweep * k / n)])
           for k in range(1, n + 1)]
    return pts, heading + sweep


def random_track(rng, min_sep=1.2, kind=None, max_tries=200):
    """
    Random training track. kind: 'open' (straight pieces with sharp corners and arcs),
    'polygon' (closed regular polygon, like a hexagon), 'ellipse' (closed, circle-like) or None = random mix.
    """
    kind = kind or rng.choice(['open', 'polygon', 'ellipse'], p=[0.5, 0.2, 0.3])
    turns = np.radians([90, -90, 45, -45, 30, -30, 135, -135, 0])
    probs = np.array([.25, .25, .10, .10, .05, .05, .05, .05, .10])
    probs /= probs.sum()
    for _ in range(max_tries):
        if kind == 'polygon':
            n = int(rng.integers(3, 9))
            side = rng.uniform(0.9, 2.5)
            R = side / (2 * math.sin(math.pi / n))
            rot = rng.uniform(0, 2 * math.pi)
            ang = rot + 2 * math.pi * np.arange(n) / n * rng.choice([-1, 1])
            pts, closed = np.column_stack([R * np.cos(ang), R * np.sin(ang)]), True
        elif kind == 'ellipse':
            a = rng.uniform(0.8, 3.0)
            b = a * rng.uniform(0.6, 1.0)
            n = int(math.ceil(2 * math.pi * max(a, b) / 0.03))
            t = np.arange(n) * 2 * math.pi / n * rng.choice([-1, 1])
            rot = rng.uniform(0, math.pi)
            xy = np.column_stack([a * np.cos(t), b * np.sin(t)])
            c, s = math.cos(rot), math.sin(rot)
            pts, closed = xy @ np.array([[c, s], [-s, c]]), True
        else:
            pos, heading, pts, closed = np.zeros(2), 0.0, [np.zeros(2)], False
            for _k in range(int(rng.integers(4, 8))):
                if rng.random() < 0.3:                                  # arc piece
                    arc, heading = _arc_points(pos, heading, rng.uniform(0.7, 3.0),
                                               rng.choice([-1, 1]) * rng.uniform(math.radians(30), math.radians(150)))
                    pts += arc
                    pos = pts[-1]
                else:                                                   # straight piece
                    pos = pos + rng.uniform(0.9, 2.4) * np.array([math.cos(heading), math.sin(heading)])
                    pts.append(pos)
                heading += float(rng.choice(turns, p=probs))
            pts = np.array(pts)
        if _path_ok(pts, closed, min_sep):
            return Track(pts, closed, f'random-{kind}')
    return Track(np.array([[0, 0], [2, 0], [2, 2]]), False, 'fallback')


class DistField:
    """Distance (m) from any ground point to a polyline, via an OpenCV distance transform."""

    def __init__(self, poly, closed=False, margin=3.0, rng=None, texture=False, res=None):
        poly = np.asarray(poly, float)
        lo, hi = poly.min(0) - margin, poly.max(0) + margin
        res = res or max(0.005, float((hi - lo).max()) / 3000.0)
        W = int(math.ceil((hi[0] - lo[0]) / res)) + 1
        H = int(math.ceil((hi[1] - lo[1]) / res)) + 1
        img = np.full((H, W), 255, np.uint8)
        px = np.round((poly - lo) / res).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [px], closed, 0, 1)
        self.dt = (cv2.distanceTransform(img, cv2.DIST_L2, 5) * res).astype(np.float32)
        self.lo, self.res, self.shape = lo, res, (H, W)
        self.extent = (lo[0], lo[1], hi[0], hi[1])
        self.texmap = None
        if texture:
            rng = rng or np.random.default_rng(0)
            coarse = cv2.resize(rng.standard_normal((max(2, H // 40), max(2, W // 40))).astype(np.float32),
                                (W, H), interpolation=cv2.INTER_CUBIC)
            fine = cv2.resize(rng.standard_normal((max(2, H // 6), max(2, W // 6))).astype(np.float32),
                              (W, H), interpolation=cv2.INTER_LINEAR)
            t = 0.6 * coarse + 0.4 * fine
            self.texmap = (t / (t.std() + 1e-6)).astype(np.float32)

    def _maps(self, gx, gy):
        return ((gx - self.lo[0]) / self.res).astype(np.float32), ((gy - self.lo[1]) / self.res).astype(np.float32)

    def sample(self, gx, gy):
        mx, my = self._maps(gx, gy)
        return cv2.remap(self.dt, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1e3)

    def tex(self, gx, gy):
        if self.texmap is None:
            return 0.0
        mx, my = self._maps(gx, gy)
        return cv2.remap(self.texmap, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


_DC = {}


def _camera_dirs(W, H, hfov):
    """Pixel ray directions in the camera frame (x forward, y left, z up). Depends only on the image size and
    FOV, so it is cached (a handful of entries at most; NOT keyed by pitch, which is jittered per sample)."""
    key = (W, H, round(hfov, 6))
    if key not in _DC:
        if len(_DC) >= 8:
            _DC.clear()
        fx = (W / 2) / math.tan(hfov / 2)
        uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        _DC[key] = np.stack([np.ones_like(uu), -(uu - W / 2) / fx, -(vv - H / 2) / fx])
    return _DC[key]


def _rays(W, H, hfov, pitch):
    """Pixel ray directions in the robot frame for a camera pitched down by `pitch` radians."""
    dc = _camera_dirs(W, H, hfov)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return np.stack([cp * dc[0] + sp * dc[2], dc[1], -sp * dc[0] + cp * dc[2]])


def project(cam, pose):
    """Ground-plane intersection of every pixel ray. Returns gx, gy, valid, footprint."""
    x, y, th = pose
    d = _rays(cam.width, cam.px_height, cam.hfov, cam.pitch)
    c, s = math.cos(th), math.sin(th)
    cx, cy = x + cam.forward * c, y + cam.forward * s
    valid = d[2] < -1e-4
    t = np.where(valid, cam.height / np.where(valid, -d[2], 1.0), 0.0)
    rx, ry = t * d[0], t * d[1]
    gx, gy = cx + c * rx - s * ry, cy + s * rx + c * ry
    gxu, gxv = np.gradient(gx, axis=1), np.gradient(gx, axis=0)
    gyu, gyv = np.gradient(gy, axis=1), np.gradient(gy, axis=0)
    fp = np.clip(np.sqrt(np.maximum(np.abs(gxu * gyv - gxv * gyu), 1e-12)), 1e-4, 5.0)
    return gx, gy, valid, fp


def pixel_of_ground_point(cam, pose, px, py):
    """Pixel (u, v) where the ground point (px, py) appears, or None if it is behind the camera."""
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    dx, dy = px - (x + cam.forward * c), py - (y + cam.forward * s)
    rx, ry, rz = c * dx + s * dy, -s * dx + c * dy, -cam.height
    cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
    xc, zc = cp * rx - sp * rz, sp * rx + cp * rz
    if xc <= 1e-6:
        return None
    fx = (cam.width / 2) / math.tan(cam.hfov / 2)
    return cam.width / 2 - fx * ry / xc, cam.px_height / 2 - fx * zc / xc


def point_visible(cam, pose, px, py, margin=4):
    uv = pixel_of_ground_point(cam, pose, px, py)
    return uv is not None and margin <= uv[0] < cam.width - margin and margin <= uv[1] < cam.px_height - margin


def interval_cov(d, a, b, fp):
    """Fraction of a pixel footprint (width fp, centred at d) that overlaps [a, b]."""
    lo, hi = np.maximum(a, d - fp / 2), np.minimum(b, d + fp / 2)
    return np.clip(hi - lo, 0, None) / fp


@dataclass
class StudentStyle:
    """Appearance of the student's world: a dark line on a light floor (randomised per sample)."""
    line_width: float = 0.10
    floor: float = 0.85
    line: float = 0.03
    bg: float = 0.85
    gain: float = 1.0
    noise: float = 0.01
    blur: float = 0.0
    vignette: float = 0.0
    shadow: float = 0.0          # 0 = no robot shadow, else darkening strength in the shadow
    body_gray: float = 0.3

    @staticmethod
    def sample(rng, line_width=0.10, shadow_prob=0.6):
        return StudentStyle(
            line_width=line_width * rng.uniform(0.8, 1.25),
            floor=rng.uniform(0.60, 0.98), line=rng.uniform(0.0, 0.20),
            bg=rng.uniform(0.60, 0.95), gain=rng.uniform(0.80, 1.15),
            noise=rng.uniform(0.003, 0.02), blur=rng.choice([0.0, 0.0, 0.6, 1.0]),
            vignette=rng.uniform(0.0, 0.25),
            shadow=rng.uniform(0.3, 0.65) if rng.random() < shadow_prob else 0.0,
            body_gray=rng.uniform(0.15, 0.5))


def jitter_cam(cam, rng, amount=1.0):
    return Cam(cam.height * (1 + 0.06 * amount * rng.uniform(-1, 1)), cam.forward,
               cam.pitch + 0.03 * amount * rng.uniform(-1, 1), cam.hfov, cam.width, cam.px_height)


SUN_DIR = (-0.5, 0.1, -0.9)      # direction the light travels, as in the example worlds' <light>


def _in_rect(px, py, cx, cy, hx, hy):
    return (np.abs(px - cx) <= hx) & (np.abs(py - cy) <= hy)


def body_mask(cam, body):
    """Pixels in which the robot's own chassis top hides the ground (camera behind the front edge)."""
    d = _rays(cam.width, cam.px_height, cam.hfov, cam.pitch)
    down = d[2] < -1e-4
    t = np.where(down, (cam.height - body.z_top) / np.where(down, -d[2], 1.0), 0.0)
    x, y = cam.forward + t * d[0], t * d[1]
    return down & (cam.height > body.z_top) & _in_rect(x, y, 0.0, 0.0, body.length / 2, body.width / 2)


def self_occlusion_fraction(cam, body):
    return float(body_mask(cam, body).mean())


def render_student(cam, pose, field, style, ground_rect, rng=None, body=None, sun=SUN_DIR):
    """What the bot's camera sees in the line-only world. Returns uint8 gray."""
    gx, gy, valid, fp = project(cam, pose)
    d = field.sample(gx, gy)
    cov = interval_cov(d, -style.line_width / 2, style.line_width / 2, fp)
    img = style.floor * (1 - cov) + style.line * cov
    x0, y0, x1, y1 = ground_rect
    inside = (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)
    if body is not None and style.shadow > 0:
        th = pose[2]
        c, s = math.cos(th), math.sin(th)
        rx, ry = c * (gx - pose[0]) + s * (gy - pose[1]), -s * (gx - pose[0]) + c * (gy - pose[1])
        k_w = np.array([sun[0], sun[1]]) / -sun[2]
        k = np.array([c * k_w[0] + s * k_w[1], -s * k_w[0] + c * k_w[1]])   # shift per metre of height, robot frame
        shade = np.zeros(rx.shape, bool)
        parts = [(0.0, 0.0, body.length / 2, body.width / 2, body.z_bottom, body.z_top),
                 (0.0, body.wheel_sep / 2, body.wheel_radius, body.wheel_width / 2, 0.0, 2 * body.wheel_radius),
                 (0.0, -body.wheel_sep / 2, body.wheel_radius, body.wheel_width / 2, 0.0, 2 * body.wheel_radius),
                 (body.mast_x, 0.0, 0.01, 0.01, body.z_top, body.cam_height),
                 (body.cam_forward, 0.0, 0.015, 0.025, body.cam_height - 0.015, body.cam_height)]
        for (cx, cy, hx, hy, z0, z1) in parts:
            for z in np.linspace(z0, z1, 6):
                shade |= _in_rect(rx - z * k[0], ry - z * k[1], cx, cy, hx, hy)
        shade = cv2.GaussianBlur(shade.astype(np.float32), (0, 0), 1.0)
        img = img * (1 - style.shadow * shade)
    img = np.where(valid & inside, img, style.bg)
    if body is not None:
        bm = body_mask(cam, body)
        if bm.any():
            img = np.where(bm, style.body_gray, img)
    if style.vignette > 0:
        H, W = img.shape
        yy, xx = np.mgrid[0:H, 0:W]
        r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
        img = img * (1 - style.vignette * r2 / 2)
    img = img * style.gain
    if style.blur > 0:
        img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), style.blur)
    if rng is not None and style.noise > 0:
        img = img + rng.normal(0, style.noise, img.shape)
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


@dataclass
class ExpertCfg:
    """Parameters of the geometric look-ahead BASELINE controller (not a teacher: nothing is learned from it)."""
    v_max: float = 0.4
    w_max: float = 2.0
    lookahead: float = 0.25
    k_heading: float = 2.5


def expert_label(track, x, y, th, cfg):
    i, s, e, tyaw, dist = track.nearest(x, y)
    tx, ty, _ = track.point_at(s + cfg.lookahead)
    bx = math.cos(th) * (tx - x) + math.sin(th) * (ty - y)
    by = -math.sin(th) * (tx - x) + math.cos(th) * (ty - y)
    alpha = math.atan2(by, bx)                          # + = target is to the left
    v = cfg.v_max * max(0.0, math.cos(alpha)) ** 2
    w = max(-cfg.w_max, min(cfg.w_max, cfg.k_heading * alpha))
    return v, w, alpha, e, s, tx, ty


def line_pixel_fraction(gray_u8, rel=0.6, min_contrast=25):
    """Fraction of pixels that are clearly darker than the median (i.e. belong to the line)."""
    med = float(np.median(gray_u8))
    g = gray_u8.astype(np.float32)
    dark = (g < rel * med) & ((med - g) > min_contrast)
    return float(dark.mean())
