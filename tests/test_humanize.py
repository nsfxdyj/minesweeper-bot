# -*- coding: utf-8 -*-
"""Humanize module tests: paths start/end correctly, stay near the segment,
and click jitter always lands inside the 25 px cell."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from msbot.humanize import gen_path, click_point, action_delay


def test_paths():
    for _ in range(500):
        x0, y0, x1, y1 = (100 + _ % 500, 100, 600 - _ % 300, 400 + _ % 200)
        pts = gen_path(x0, y0, x1, y1)
        ex, ey, _ = pts[-1]
        assert abs(ex - x1) < 1.5 and abs(ey - y1) < 1.5, (pts[-1], (x1, y1))
        assert len(pts) >= 2
        for x, y, dt in pts:
            assert 0 < dt < 0.05
            # never stray absurdly far from the straight segment
            assert abs(x - x0) + abs(x - x1) < 3 * abs(x1 - x0) + 400
    print("OK: 500 paths end on target with sane shape/timing")


def test_click_point():
    for _ in range(1000):
        x, y = click_point(500, 300)
        assert 493 <= x <= 507 and 293 <= y <= 307
    print("OK: click jitter stays inside the cell")


def test_delay():
    ds = [action_delay() for _ in range(200)]
    assert all(d > 0.02 for d in ds)
    assert len(set(round(d, 4) for d in ds)) > 100     # not constant
    print("OK: delays are randomised")


if __name__ == "__main__":
    test_paths()
    test_click_point()
    test_delay()
