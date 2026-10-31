"""
distill_lib.py - shared code for distilling the Ackermann AutopilotNet (teacher)
onto a differential-drive bot (student) that follows a centerline.

Everything here is plain numpy/OpenCV/PIL (torch is optional).

Coordinate conventions (REP-103): +x forward, +y left, +z up, yaw CCW,
positive angular velocity / positive steering = turn LEFT.
"""
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


# --------------------------------------------------------------------------
# xacro argument reader (single source of truth for robot dimensions)
# --------------------------------------------------------------------------
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
class TeacherCfg:
    """Geometry of the Ackermann car the teacher was trained on (defaults = vehicle.xacro)."""
    wheelbase: float = 1.4
    cam_height: float = 1.95     # above ground
    cam_forward: float = 1.6     # ahead of the rear axle (the bicycle-model reference)
    cam_pitch: float = 0.1
    cam_hfov: float = 1.396263
    max_velocity: float = 1.0
    max_steer: float = 0.6108
    steer_sign: float = -1.0   # measured with probe_teacher.py: restoring behaviour needs the sign inverted

    @staticmethod
    def from_xacro(path):
        a = load_xacro_args(path)
        wheel_offset = a['body_length'] / 2 - a['wheel_radius']
        cam_x = a['body_length'] / 2 - a['camera_stick_size'] / 2
        return TeacherCfg(
            wheelbase=2 * wheel_offset,
            cam_height=a['camera_height'] + 1.5 * a['wheel_radius'],
            cam_forward=cam_x + wheel_offset,
            cam_pitch=a['camera_pitch'], cam_hfov=a['camera_fov'],
            max_velocity=a['max_velocity'], max_steer=a['max_steering_angle'])


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
    x: float = 0.0                   # chassis centre ahead of the axle (negative = behind it)


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
                    cam_forward=a['camera_forward'], x=a.get('body_x', 0.0))
        return StudentCfg(
            cam_height=a['camera_height'], cam_forward=a['camera_forward'],
            cam_pitch=a['camera_pitch'], cam_hfov=a['camera_fov'],
            img_w=int(a['image_width']), img_h=int(a['image_height']),
            max_linear=a['max_linear_velocity'], max_angular=a['max_angular_velocity'], body=body)

    def camera(self):
        return Cam(self.cam_height, self.cam_forward, self.cam_pitch, self.cam_hfov, self.img_w, self.img_h)


# --------------------------------------------------------------------------
# Tracks (polylines) and geometry helpers
# --------------------------------------------------------------------------
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


def fillet_polyline(points, radius, closed=False, arc_step_deg=4.0):
    """Round the corners of a polyline with circular arcs (used for the teacher's road only)."""
    P = np.asarray(points, float)
    n = len(P)
    out = [] if closed else [P[0]]
    idx = range(n) if closed else range(1, n - 1)
    for i in idx:
        a, b, c = P[i - 1], P[i], P[(i + 1) % n]
        d1, d2 = b - a, c - b
        L1, L2 = np.hypot(*d1), np.hypot(*d2)
        u1, u2 = d1 / L1, d2 / L2
        cross = u1[0] * u2[1] - u1[1] * u2[0]
        phi = math.atan2(abs(cross), u1 @ u2)
        if phi < math.radians(6):
            out.append(b)
            continue
        t = min(radius * math.tan(phi / 2), 0.5 * min(L1, L2))
        r = t / math.tan(phi / 2)
        A, B = b - u1 * t, b + u2 * t
        left = cross > 0
        nrm = np.array([-u1[1], u1[0]]) if left else np.array([u1[1], -u1[0]])
        C = A + nrm * r
        a0 = math.atan2(*(A - C)[::-1])
        delta = phi if left else -phi
        m = max(2, int(math.ceil(abs(delta) / math.radians(arc_step_deg))))
        for k in range(m + 1):
            ang = a0 + delta * k / m
            out.append(C + r * np.array([math.cos(ang), math.sin(ang)]))
    if not closed:
        out.append(P[-1])
    return np.asarray(out)


def _seg_dist(p, q, r, s):
    """Distance between segments pq and rs (2D)."""
    def pt_seg(a, b, c):
        ab = b - a
        t = np.clip(((c - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        return np.hypot(*(a + t * ab - c))
    def inter(p, q, r, s):
        d1, d2 = q - p, s - r
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-12:
            return False
        t = ((r - p)[0] * d2[1] - (r - p)[1] * d2[0]) / den
        u = ((r - p)[0] * d1[1] - (r - p)[1] * d1[0]) / den
        return 0 <= t <= 1 and 0 <= u <= 1
    if inter(p, q, r, s):
        return 0.0
    return min(pt_seg(p, q, r), pt_seg(p, q, s), pt_seg(r, s, p), pt_seg(r, s, q))


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


# --------------------------------------------------------------------------
# Distance fields (fast ground-plane rendering without ray-casting a mesh)
# --------------------------------------------------------------------------
class DistField:
    """Distance (m) from any ground point to a polyline, via an OpenCV distance transform."""

    def __init__(self, poly, closed=False, margin=3.0, rng=None, texture=False):
        poly = np.asarray(poly, float)
        lo, hi = poly.min(0) - margin, poly.max(0) + margin
        res = max(0.005, float((hi - lo).max()) / 3000.0)
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


# --------------------------------------------------------------------------
# Camera projection and image formation
# --------------------------------------------------------------------------
_RAYS = {}


def _rays(W, H, hfov, pitch):
    key = (W, H, round(hfov, 6), round(pitch, 6))
    if key not in _RAYS:
        fx = (W / 2) / math.tan(hfov / 2)
        uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        dc = np.stack([np.ones_like(uu), -(uu - W / 2) / fx, -(vv - H / 2) / fx])
        cp, sp = math.cos(pitch), math.sin(pitch)
        _RAYS[key] = np.stack([cp * dc[0] + sp * dc[2], dc[1], -sp * dc[0] + cp * dc[2]]).astype(np.float64)
    return _RAYS[key]


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
class TeacherStyle:
    """Appearance of the teacher's training world (measured from a screenshot of grass_track)."""
    road_width_T: float = 3.0      # full road width in TEACHER-world metres (calibrated with probe_teacher.py)
    stripe_frac: float = 0.08      # edge-stripe width as a fraction of the road width
    grass: float = 0.32
    grass_tex: float = 0.045
    road: float = 0.15
    stripe: float = 0.82
    sky: float = 0.855
    noise: float = 0.007


def teacher_camera(tcfg, scale, W=256, H=192):
    """The teacher's camera, expressed in student-world metres (world scaled by 1/scale)."""
    return Cam(tcfg.cam_height / scale, tcfg.cam_forward / scale, tcfg.cam_pitch, tcfg.cam_hfov, W, H)


def render_teacher(cam, pose, field, style, scale, rng=None):
    """What the teacher's camera would see if the centerline were a road. Returns uint8 gray."""
    gx, gy, valid, fp = project(cam, pose)
    d = field.sample(gx, gy)
    W = style.road_width_T / scale
    hw, st = W / 2, style.stripe_frac * W
    c_road = interval_cov(d, -(hw - st), hw - st, fp)
    c_str = interval_cov(d, hw - st, hw, fp) + interval_cov(d, -hw, -(hw - st), fp)
    grass = style.grass + style.grass_tex * field.tex(gx, gy)
    img = grass * (1 - c_road - c_str) + style.road * c_road + style.stripe * c_str
    img = np.where(valid, img, style.sky)
    if rng is not None and style.noise > 0:
        img = img + rng.normal(0, style.noise, img.shape)
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


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
    """Pixels in which the robot's own chassis top or wheels hide the ground."""
    d = _rays(cam.width, cam.px_height, cam.hfov, cam.pitch)
    down = d[2] < -1e-4
    den = np.where(down, -d[2], 1.0)
    mask = np.zeros(down.shape, bool)
    if cam.height > body.z_top:
        t = np.where(down, (cam.height - body.z_top) / den, 0.0)
        x, y = cam.forward + t * d[0], t * d[1]
        mask |= down & _in_rect(x, y, getattr(body, 'x', 0.0), 0.0, body.length / 2, body.width / 2)
    zw = 2 * body.wheel_radius
    if cam.height > zw:
        t = np.where(down, (cam.height - zw) / den, 0.0)
        x, y = cam.forward + t * d[0], t * d[1]
        for sg in (1.0, -1.0):
            mask |= down & _in_rect(x, y, 0.0, sg * body.wheel_sep / 2, body.wheel_radius, body.wheel_width / 2)
    return mask


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
        parts = [(getattr(body, 'x', 0.0), 0.0, body.length / 2, body.width / 2, body.z_bottom, body.z_top),
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


# --------------------------------------------------------------------------
# Teacher network wrapper (torch if available, otherwise a numpy forward pass)
# --------------------------------------------------------------------------
class Teacher:
    def __init__(self, ckpt_path, backend='auto', scripts_dir=None, h=96, w=128, symmetrize=True):
        self.h, self.w = h, w
        self.symmetrize = symmetrize
        self.backend = backend
        if backend in ('auto', 'torch'):
            try:
                if scripts_dir:
                    sys.path.insert(0, scripts_dir)
                import torch
                from model import AutopilotNet
                from dataset import preprocessing_transform
                self.torch = torch
                self.net = AutopilotNet(h=h, w=w, inputs=1)
                ck = torch.load(ckpt_path, map_location='cpu')
                self.net.load_state_dict(ck['state_dict'])
                self.net.eval()
                self.transform = preprocessing_transform(h, w)
                self.backend = 'torch'
                return
            except Exception as e:                       # noqa
                if backend == 'torch':
                    raise
                print(f'[teacher] torch backend unavailable ({type(e).__name__}); using numpy forward pass')
        from np_teacher import NumpyAutopilot
        self.np_net = NumpyAutopilot(ckpt_path)
        self.backend = 'numpy'

    def _raw(self, gray_u8):
        pil = Image.fromarray(np.ascontiguousarray(gray_u8), 'L')
        if self.backend == 'torch':
            with self.torch.no_grad():
                out = self.net(self.transform(pil).unsqueeze(0))[0]
            return float(out[0]), float(out[1])
        x = np.asarray(pil.resize((self.w, self.h), Image.BILINEAR), np.float32) / 255.0
        return self.np_net.forward((x - 0.5) / 0.5)

    def __call__(self, gray_u8, symmetrize=None):
        """gray_u8: (H, W) uint8 -> (velocity_norm, steering_norm).

        Training used random horizontal flips with the steering sign inverted, so the ideal
        network is odd under mirroring. Averaging the prediction on the image and its mirror
        removes the network's left/right bias: v = (v1+v2)/2, steer = (s1-s2)/2.
        """
        sym = self.symmetrize if symmetrize is None else symmetrize
        v1, s1 = self._raw(gray_u8)
        if not sym:
            return v1, s1
        v2, s2 = self._raw(gray_u8[:, ::-1])
        return 0.5 * (v1 + v2), 0.5 * (s1 - s2)


# --------------------------------------------------------------------------
# Labels: teacher (distilled) + geometric expert, blended
# --------------------------------------------------------------------------
@dataclass
class LabelCfg:
    scale: float = 14.0             # virtual teacher world is `scale` x bigger than the student world
    corner_radius_T: float = 3.0    # corner radius of the teacher's road, in teacher metres
    v_max: float = 0.4
    w_max: float = 2.0
    v_floor_norm: float = 0.25      # minimum distilled speed (fraction of v_max) so the bot never stalls
    lookahead: float = 0.25         # expert look-ahead distance [m]
    k_heading: float = 2.5          # expert heading gain [1/s]
    blend_a0_deg: float = 20.0      # below this look-ahead heading error the teacher is trusted fully
    blend_a1_deg: float = 50.0      # above this the expert takes over completely
    mode: str = 'hybrid'            # teacher | hybrid | expert
    corner_eps: float = 0.05          # fully turned when the axle is this close to a vertex [m]
    corner_ramp: float = 0.06         # the turn-in is blended in over this distance before corner_eps [m]
    corner_slow: float = 0.4          # speed fraction at a vertex (1 = no slowing down)
    corner_slow_dist: float = 0.25    # slowing starts this far before the vertex [m]


def expert_label(track, x, y, th, cfg):
    i, s, e, tyaw, dist = track.nearest(x, y)
    s_t = s + cfg.lookahead
    ds = [((cs - s) % track.length if track.closed else cs - s) for cs, _ in track.corners]
    ds = [d for d in ds if cfg.corner_eps < d < cfg.lookahead]
    if ds:
        dv = min(ds)                      # distance to the next vertex
        u = min(max((cfg.corner_eps + cfg.corner_ramp - dv) / cfg.corner_ramp, 0.0), 1.0)
        s_t = s + dv + (cfg.lookahead - dv) * u    # at the vertex far away -> past it when close
    tx, ty, _ = track.point_at(s_t)
    bx = math.cos(th) * (tx - x) + math.sin(th) * (ty - y)
    by = -math.sin(th) * (tx - x) + math.cos(th) * (ty - y)
    alpha = math.atan2(by, bx)                          # + = target is to the left
    v = cfg.v_max * max(0.0, math.cos(alpha)) ** 2
    dn = [d for d in ((cs - s) % track.length if track.closed else cs - s for cs, _ in track.corners) if d >= 0.0]
    if dn and cfg.corner_slow < 1.0:        # slow down towards a vertex so the turn is (almost) a pivot in place
        v *= cfg.corner_slow + (1.0 - cfg.corner_slow) * min(max((min(dn) - cfg.corner_eps) / cfg.corner_slow_dist, 0.0), 1.0)
    w = max(-cfg.w_max, min(cfg.w_max, cfg.k_heading * alpha))
    return v, w, alpha, e, s, tx, ty


def teacher_label(teacher, img_u8, tcfg, cfg):
    v_out, st_out = teacher(img_u8)
    v_norm = min(max(v_out, 0.0), 1.0)
    v = max(v_norm, cfg.v_floor_norm) * cfg.v_max
    delta = max(-1.0, min(1.0, tcfg.steer_sign * st_out)) * tcfg.max_steer
    kappa_T = math.tan(delta) / tcfg.wheelbase          # path curvature the teacher wants [1/m, teacher world]
    w = max(-cfg.w_max, min(cfg.w_max, kappa_T * cfg.scale * v))
    return v, w, v_out, st_out


def blend_weight(alpha, cfg):
    a0, a1 = math.radians(cfg.blend_a0_deg), math.radians(cfg.blend_a1_deg)
    t = min(max((abs(alpha) - a0) / max(a1 - a0, 1e-6), 0.0), 1.0)
    return t * t * (3 - 2 * t)


class LabelContext:
    """Pre-built fields for one track."""

    def __init__(self, track, cfg, style, rng):
        self.track, self.cfg, self.style = track, cfg, style
        self.teacher_poly = fillet_polyline(track.points, cfg.corner_radius_T / cfg.scale, track.closed)
        self.tfield = DistField(self.teacher_poly, track.closed, rng=rng, texture=True)
        self.lfield = DistField(track.points, track.closed)
        self.ground_rect = None

    def set_ground_rect(self, margin=2.0):
        if self.track.ground_rect:
            self.ground_rect = tuple(self.track.ground_rect)
            return
        lo, hi = self.track.points.min(0) - margin, self.track.points.max(0) + margin
        self.ground_rect = (lo[0], lo[1], hi[0], hi[1])


def make_label(ctx, teacher, tcfg, teacher_cam, pose, rng=None):
    cfg = ctx.cfg
    x, y, th = pose
    ev, ew, alpha, e_lat, s, tx, ty = expert_label(ctx.track, x, y, th, cfg)
    out = dict(expert_v=ev, expert_w=ew, alpha=alpha, lateral=e_lat, s=s, tx=tx, ty=ty)
    if cfg.mode == 'expert':
        out.update(v=ev, w=ew, teacher_v=float('nan'), teacher_w=float('nan'), weight=1.0,
                   t_vout=float('nan'), t_steer=float('nan'))
        return out
    img = render_teacher(teacher_cam, pose, ctx.tfield, ctx.style, cfg.scale, rng)
    tv, tw, v_out, st_out = teacher_label(teacher, img, tcfg, cfg)
    wgt = 0.0 if cfg.mode == 'teacher' else blend_weight(alpha, cfg)
    out.update(v=(1 - wgt) * tv + wgt * ev, w=(1 - wgt) * tw + wgt * ew,
               teacher_v=tv, teacher_w=tw, weight=wgt, t_vout=v_out, t_steer=st_out)
    return out


# --------------------------------------------------------------------------
# Line visibility (shared by the dataset filter and the runtime "lost line" reflex)
# --------------------------------------------------------------------------
def line_pixel_fraction(gray_u8, rel=0.6, min_contrast=25):
    """Fraction of pixels that are clearly darker than the median (i.e. belong to the line)."""
    med = float(np.median(gray_u8))
    g = gray_u8.astype(np.float32)
    dark = (g < rel * med) & ((med - g) > min_contrast)
    return float(dark.mean())


# --------------------------------------------------------------------------
# Closed-loop kinematic check of a label policy (no Gazebo needed)
# --------------------------------------------------------------------------
def simulate(track, teacher, tcfg, cfg, style, student_cam, seed=0, dt=0.1, tau=0.15,
             max_steps=2500, lost_dist=0.6):
    """Drive a unicycle with the labels as commands (first-order actuator lag tau). Closed tracks: one lap."""
    rng = np.random.default_rng(seed)
    ctx = LabelContext(track, cfg, style, rng)
    ctx.set_ground_rect()
    tcam = teacher_camera(tcfg, cfg.scale)
    x, y, th = track.start_pose()
    v = w = 0.0
    dists, vis, traj, prog = [], 0, [], 0.0
    s_prev, total = None, 0.0
    reason, k = 'timeout', 0
    for k in range(max_steps):
        lab = make_label(ctx, teacher, tcfg, tcam, (x, y, th), rng)
        a = dt / (tau + dt)
        v += a * (lab['v'] - v)
        w += a * (lab['w'] - w)
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        th += w * dt
        _, s, _, _, dist = track.nearest(x, y)
        if track.closed:
            s_prev = s if s_prev is None else s_prev
            total += (s - s_prev + track.length / 2) % track.length - track.length / 2
            s_prev = s
            prog = total
        else:
            prog = max(prog, s)
        dists.append(dist)
        traj.append((x, y, v))
        vis += point_visible(student_cam, (x, y, th), lab['tx'], lab['ty'])
        if dist > lost_dist:
            reason = 'lost the line'
            break
        if (track.closed and prog >= track.length) or (not track.closed and prog > track.length - 0.15):
            reason = 'finished'
            break
    return dict(finished=reason == 'finished', reason=reason, progress=max(prog, 0.0) / track.length,
                max_err=max(dists), mean_err=float(np.mean(dists)), target_visible=vis / (k + 1),
                seconds_sim=(k + 1) * dt, traj=traj)
