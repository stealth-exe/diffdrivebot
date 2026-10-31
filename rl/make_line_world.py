#!/usr/bin/env python3
"""
Generate a Gazebo Harmonic world with a painted centerline made of straight
segments (so 90-degree corners are exact), plus a centerline CSV that an
expert controller or a labeller can use as ground truth. UNTESTED in Gazebo.

Example:
  python3 make_line_world.py \
      --points "0,0 4,0 4,3 1,3 1,5" \
      --width 0.08 \
      --robot /tmp/diffdrivebot/diffdrive_robot.sdf \
      --out /tmp/diffdrivebot/line_world.sdf

Outputs:
  <out>.sdf              the world
  <out>_centerline.csv   columns: s, x, y, yaw (points every --step metres)
  <out>_points.json      the track definition, read by train_rl.py / eval_rl.py

The world is built like the example worlds (hexagon.sdf, circle_2_5.sdf): 0.8 grey floor, black line, a
directional sun with shadows, and the Sensors system loaded at world level (so diffdrive_bot.xacro must NOT
load it again - its default sensors_plugin:=false is right). Pass --no-world-sensors to leave it out.

Run it with:  gz sim -r -v 3 <out>.sdf
"""
import argparse
import csv
import json
import math
import os


def parse_points(text):
    pts = []
    for tok in text.split():
        x, y = tok.split(",")
        pts.append((float(x), float(y)))
    if len(pts) < 2:
        raise SystemExit("Need at least two points.")
    return pts


def segments(points, closed):
    segs = list(zip(points[:-1], points[1:]))
    if closed:
        segs.append((points[-1], points[0]))
    return segs


def segment_box(p0, p1, width, idx, rgb):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return ""
    yaw = math.atan2(dy, dx)
    cx, cy = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
    # Extend each end by half the line width so segments overlap and
    # right-angle corners are filled in with no gap.
    size_x = length + width
    r, g, b = rgb
    return f"""
    <model name="line_{idx}">
      <static>true</static>
      <pose>{cx:.5f} {cy:.5f} 0.0005 0 0 {yaw:.6f}</pose>
      <link name="link">
        <visual name="visual">
          <geometry><box><size>{size_x:.5f} {width:.5f} 0.001</size></box></geometry>
          <material>
            <ambient>{r} {g} {b} 1</ambient>
            <diffuse>{r} {g} {b} 1</diffuse>
          </material>
        </visual>
      </link>
    </model>"""


def resample(points, closed, step):
    rows = []
    s_total = 0.0
    for p0, p1 in segments(points, closed):
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue
        yaw = math.atan2(dy, dx)
        n = max(1, int(round(length / step)))
        for k in range(n):
            t = k / n
            rows.append((s_total + t * length, p0[0] + t * dx, p0[1] + t * dy, yaw))
        s_total += length
    last = points[0] if closed else points[-1]
    rows.append((s_total, last[0], last[1], rows[-1][3]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", default="0,0 4,0 4,3 1,3 1,5",
                    help='Corner points in metres, e.g. "0,0 4,0 4,3"')
    ap.add_argument("--closed", action="store_true", help="Join the last point back to the first")
    ap.add_argument("--width", type=float, default=0.10, help="Line width in metres")
    ap.add_argument("--margin", type=float, default=2.0, help="Ground margin around the track (m)")
    ap.add_argument("--step", type=float, default=0.02, help="Centerline CSV spacing (m)")
    ap.add_argument("--robot", default=None, help="Absolute path to diffdrive_robot.sdf to include")
    ap.add_argument("--render-engine", default="ogre2", help="ogre2 or ogre")
    ap.add_argument("--no-world-sensors", action="store_true",
                    help="do not load the Sensors system in the world (then spawn the robot with sensors_plugin:=true)")
    ap.add_argument("--out", default="line_world.sdf")
    args = ap.parse_args()

    pts = parse_points(args.points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    gx0, gx1 = min(xs) - args.margin, max(xs) + args.margin
    gy0, gy1 = min(ys) - args.margin, max(ys) + args.margin
    gcx, gcy = (gx0 + gx1) / 2.0, (gy0 + gy1) / 2.0
    gw, gh = gx1 - gx0, gy1 - gy0

    line_models = "".join(
        segment_box(p0, p1, args.width, i, (0, 0, 0))
        for i, (p0, p1) in enumerate(segments(pts, args.closed))
    )

    robot_xml = ""
    if args.robot:
        first, second = pts[0], pts[1]
        yaw0 = math.atan2(second[1] - first[1], second[0] - first[0])
        robot_xml = f"""
    <include>
      <name>robot</name>
      <uri>file://{os.path.abspath(args.robot)}</uri>
      <pose>{first[0]:.4f} {first[1]:.4f} 0.07 0 0 {yaw0:.6f}</pose>
    </include>"""

    sensors_xml = ""
    if not args.no_world_sensors:
        sensors_xml = f"""    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>{args.render_engine}</render_engine>
    </plugin>"""

    world = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="line_world">
    <physics name="1ms" type="ignored">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
{sensors_xml}
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name="ground">
      <static>true</static>
      <pose>{gcx:.4f} {gcy:.4f} 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>{gw:.3f} {gh:.3f}</size></plane></geometry>
          <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>{gw:.3f} {gh:.3f}</size></plane></geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
{line_models}
{robot_xml}
  </world>
</sdf>
"""

    with open(args.out, "w") as f:
        f.write(world)

    csv_path = os.path.splitext(args.out)[0] + "_centerline.csv"
    rows = resample(pts, args.closed, args.step)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["s", "x", "y", "yaw"])
        for r in rows:
            w.writerow([f"{v:.5f}" for v in r])

    points_path = os.path.splitext(args.out)[0] + "_points.json"
    with open(points_path, "w") as f:
        json.dump({"points": [list(p) for p in pts], "closed": bool(args.closed),
                   "line_width": args.width, "ground_margin": args.margin,
                   "ground_rect": [gx0, gy0, gx1, gy1]}, f, indent=1)

    print(f"Wrote {args.out}")
    print(f"Wrote {points_path}")
    print(f"Wrote {csv_path} ({len(rows)} points, {rows[-1][0]:.2f} m long)")
    print(f"Run: gz sim -r -v 3 {args.out}")


if __name__ == "__main__":
    main()
