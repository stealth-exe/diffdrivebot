#!/usr/bin/env python3
"""
start_pose.py - print a good spawn pose "x y yaw" for a track JSON (<name>_points.json).

The robot is put in the middle of the first side, pointing along it, so it does not start on a corner.

  read X Y YAW < <(python3 start_pose.py tracks/final_grading_points.json)
  ros2 launch diffdrivebot diffdrive_sim.launch.py world:=... x:=$X y:=$Y yaw:=$YAW

  --vertex   start exactly on the first vertex instead
"""
import json
import math
import sys

args = [a for a in sys.argv[1:] if not a.startswith('--')]
if not args:
    sys.exit(__doc__)
p = json.load(open(args[0]))['points']
(x0, y0), (x1, y1) = p[0], p[1]
yaw = math.atan2(y1 - y0, x1 - x0)
if '--vertex' in sys.argv:
    print(f'{x0:.4f} {y0:.4f} {yaw:.4f}')
else:
    print(f'{(x0 + x1) / 2:.4f} {(y0 + y1) / 2:.4f} {yaw:.4f}')
