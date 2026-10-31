#!/usr/bin/env python3
"""
mirror_track.py - write the mirror image (y -> -y) of one or more track JSON files.

Mirroring turns every left turn into a right turn (and a counter-clockwise loop into a clockwise one). Training on
both a track and its mirror image keeps the student from favouring one turning direction.

  python3 mirror_track.py tracks/hexagon_points.json tracks/loop0_points.json
  -> tracks/hexagon_mirror_points.json, tracks/loop0_mirror_points.json
"""
import json
import os
import sys

if len(sys.argv) < 2:
    sys.exit(__doc__)
for path in sys.argv[1:]:
    d = json.load(open(path))
    d['points'] = [[x, -y] for x, y in d['points']]
    if d.get('ground_rect'):
        x0, y0, x1, y1 = d['ground_rect']
        d['ground_rect'] = [x0, -y1, x1, -y0]
    base = os.path.basename(path)
    if not base.endswith('_points.json'):
        sys.exit(f'{path}: expected a file named <name>_points.json')
    out = os.path.join(os.path.dirname(path), base[:-len('_points.json')] + '_mirror_points.json')
    json.dump(d, open(out, 'w'))
    print('wrote', out)
