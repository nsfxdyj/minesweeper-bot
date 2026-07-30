# -*- coding: utf-8 -*-
"""
Human-like mouse behaviour for live-stream / anti-robot scenarios.

  gen_path()   : cubic-Bezier trajectory with ease-out velocity profile,
                 random curvature, per-step random timing and occasional
                 overshoot + correction — no two moves look the same
  click_point(): a random point inside the target cell instead of the
                 dead-centre pixel every time
  pause()      : randomised inter-action rhythm with rare longer
                 "thinking" pauses
"""
import math
import random


class GameAbort(Exception):
    """Raised when the user aborts the bot (hotkey / corner slam)."""


def gen_path(x0, y0, x1, y1):
    """-> list of (x, y, dt_seconds) waypoints from (x0,y0) to (x1,y1)."""
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist < 3:
        return [(x1, y1, random.uniform(0.003, 0.012))]

    # 12 % of longer moves overshoot the target a little, then correct
    correct_back = None
    if dist > 80 and random.random() < 0.12:
        over = random.uniform(3, 9)
        ux, uy = (x1 - x0) / dist, (y1 - y0) / dist
        correct_back = (x1, y1)
        x1, y1 = x1 + ux * over, y1 + uy * over

    # two random control points -> different curve every move
    spread = dist * random.uniform(0.15, 0.35)
    t1, t2 = random.uniform(0.15, 0.45), random.uniform(0.55, 0.85)
    c1 = (x0 + (x1 - x0) * t1 + random.uniform(-spread, spread),
          y0 + (y1 - y0) * t1 + random.uniform(-spread, spread))
    c2 = (x0 + (x1 - x0) * t2 + random.uniform(-spread, spread),
          y0 + (y1 - y0) * t2 + random.uniform(-spread, spread))

    steps = max(8, min(60, int(dist / random.uniform(7, 14))))
    pts = []
    for i in range(steps + 1):
        t = i / steps
        te = 1.0 - (1.0 - t) ** 3            # ease-out: fast start, soft land
        mt = 1.0 - te
        x = mt**3 * x0 + 3 * mt**2 * te * c1[0] + 3 * mt * te**2 * c2[0] + te**3 * x1
        y = mt**3 * y0 + 3 * mt**2 * te * c1[1] + 3 * mt * te**2 * c2[1] + te**3 * y1
        pts.append((x, y, random.uniform(0.003, 0.014)))

    if correct_back:
        bx, by = correct_back
        for i in range(1, 6):                # small corrective hop
            t = i / 5
            pts.append((x1 + (bx - x1) * t, y1 + (by - y1) * t,
                        random.uniform(0.004, 0.012)))
    return pts


def click_point(cx, cy, half=7):
    """Random point near (cx, cy) within +-half px (inside a 25 px cell)."""
    return (cx + random.randint(-half, half),
            cy + random.randint(-half, half))


def action_delay(base=0.05):
    """Randomised delay between actions."""
    time_sleep = base * random.uniform(0.6, 1.9)
    if random.random() < 0.03:               # rare "thinking" pause
        time_sleep += random.uniform(0.4, 1.1)
    return time_sleep


def pre_click_delay():
    return random.uniform(0.04, 0.18)


def hold_time():
    """Mouse button hold duration."""
    return random.uniform(0.03, 0.09)
