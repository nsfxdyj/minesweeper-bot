# -*- coding: utf-8 -*-
"""
Pacing controller v2: dynamic delay strategy, converges to target ±10%.

Closed loop on schedule error, evaluated before every action batch:

    error = target * (done / total) - elapsed

  error > 0  ahead of schedule (too fast) -> raise `scale` (longer
             inter-action delays) and idle-wander until the schedule
             for the post-batch progress is reached
  error < 0  behind schedule -> cut `scale` toward 0.12 (minimal
             delays), skip waiting entirely; if the linear projection
             still overshoots 105% of target, hurry even harder

A batch that finishes the game (all mines flagged / every unknown cell
acted on) is held back until `target - estimated_batch_time`, so the
winning click lands within a fraction of a second of the target.

Idle time is spent "thinking": irregular SMALL movements around the
next candidate click points (hover, approach-and-pull-back, tiny
circles, micro-jitter, plain staring) — never clicks, never large
regular sweeps.
"""
import math
import random
import time

from . import humanize


class Pacer:
    MIN_SCALE = 0.12          # fastest: delays nearly gone
    MAX_SCALE = 3.0           # slowest: very deliberate movements

    def __init__(self, target, total, delay_base=0.03):
        self.target = float(target)
        self.total = max(total, 1)
        self.delay_base = delay_base
        self.t0 = None
        self.scale = 1.0
        self.action_cost = None     # observed per-action seconds (EMA)
        self.overhead = 0.3         # observed per-round scan/plan cost (EMA)
        self._m = None              # last wander position (x, y)

    # ------------------------------------------------------------------
    def begin(self):
        """Start the clock (called right after the first click)."""
        self.t0 = time.time()

    def elapsed(self):
        return 0.0 if self.t0 is None else time.time() - self.t0

    def projected(self, done):
        """Linear projection of the finish time at the current rate."""
        e = self.elapsed()
        if done <= 0 or e <= 0:
            return None
        rate = done / e
        return e + max(self.total - done, 0) / rate

    def observe(self, dt, n_actions):
        """Feed back the measured duration of an applied batch so the
        finishing hold uses real costs instead of a guess."""
        if n_actions <= 0 or dt <= 0:
            return
        c = dt / n_actions
        self.action_cost = c if self.action_cost is None \
            else 0.7 * self.action_cost + 0.3 * c

    def observe_overhead(self, dt):
        """Feed back the round's scan/plan cost (everything that is
        neither waiting nor clicking).  The schedule pre-pays this
        amount so rounds land ON their marks instead of systematically
        late."""
        if dt <= 0:
            return
        self.overhead = min(5.0, max(0.05,
                                     0.7 * self.overhead + 0.3 * dt))

    def estimate_batch_time(self, n_actions):
        """Cost of applying a batch (pure clicking)."""
        unit = self.action_cost if self.action_cost is not None else 0.20
        return n_actions * unit

    # ------------------------------------------------------------------
    def adjust(self, done):
        """Recompute the delay multiplier from the schedule error.

        Integral-proportional hybrid with memory:
          late (behind schedule or projection overruns target) -> decay
          the scale geometrically round by round until MIN_SCALE, so
          persistent lateness drives delays to the floor;
          ahead -> approach a proportional target scale (slow down).
        Returns the new scale; primary evidence for the
        'slow down when fast / hurry when slow' requirement."""
        e = self.elapsed()
        if e <= 0:
            return self.scale
        unit = max(0.08 * self.target, 0.5)
        want = self.target * min(done / self.total, 1.0)
        err = want - e                       # >0 -> ahead (too fast)
        proj = self.projected(done)
        late = err < 0 or (proj is not None and proj > self.target * 1.02)
        if late:
            factor = 0.45 if (proj is not None
                              and proj > self.target * 1.08) else 0.65
            self.scale = max(self.MIN_SCALE, self.scale * factor)
        else:
            desired = min(self.MAX_SCALE, 1.0 + max(err, 0.0) / unit)
            alpha = 0.6 if desired > self.scale else 0.35
            self.scale = alpha * desired + (1 - alpha) * self.scale
            self.scale = min(self.MAX_SCALE,
                             max(self.MIN_SCALE, self.scale))
        return self.scale

    # ------------------------------------------------------------------
    def before_batch(self, done, incoming, finishing,
                     backend, focus, decoys, area):
        """Dynamic delay control, executed before each action batch.

        The error is measured against the schedule for the post-batch
        progress `done + incoming`: the batch may not start before its
        own schedule, and `scale` reflects whether we are ahead of or
        behind where we are about to be."""
        done_next = min(done + incoming, self.total)
        self.adjust(done_next)
        waited = 0.0
        if finishing:
            hold_to = self.target - self.estimate_batch_time(incoming) \
                - self.overhead - 0.10
            if self.elapsed() < hold_to:
                waited += self.wait_until(hold_to, backend, focus,
                                          decoys, area)
            self.scale = min(self.scale, 0.6)   # finish crisply
            return waited
        # the batch should COMPLETE by its schedule, not start at it:
        # subtract the estimated batch cost AND the observed per-round
        # overhead (scans/planning), so rounds land on their marks
        want = self.target * done_next / self.total
        start_at = want - self.estimate_batch_time(incoming) - self.overhead
        if self.elapsed() < start_at:
            waited += self.wait_until(start_at, backend, focus,
                                      decoys, area)
        return waited

    def wait_until(self, when, backend, focus, decoys, area):
        """Idle — wandering near the candidate points — until `when`.
        Returns the seconds actually waited."""
        t0 = self.elapsed()
        while True:
            left = when - self.elapsed()
            if left <= 0:
                return self.elapsed() - t0
            if getattr(backend, "stop_requested", lambda: False)():
                raise humanize.GameAbort
            self._think(backend, focus, decoys, area,
                        min(left, random.uniform(0.25, 0.7)))

    # ------------------------------------------------------------------
    # "thinking": small irregular moves between the next candidate points
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp(x, y, area):
        x0, y0, w, h = area
        return (min(max(x, x0), x0 + w - 1),
                min(max(y, y0), y0 + h - 1))

    def _move(self, backend, x, y):
        backend.human_move_to(x, y)
        self._m = (x, y)

    def _think(self, backend, focus, decoys, area, budget):
        """One small thinking episode; never clicks, honours `budget`."""
        move = getattr(backend, "human_move_to", None)
        end = time.time() + budget

        def left():
            return end - time.time()

        def nap(a, b):
            s = min(random.uniform(a, b), max(left(), 0.0))
            if s > 0:
                time.sleep(s)

        if move is None or not focus:
            nap(0.15, 0.45)
            return

        # mostly hover the real next targets, sometimes a decoy number
        pool = focus if (not decoys or random.random() < 0.75) else decoys
        fx, fy = random.choice(pool)
        roll = random.random()

        if roll < 0.46:
            # small hover near a candidate point
            tx, ty = fx + random.gauss(0, 8), fy + random.gauss(0, 8)
            self._move(backend, *self._clamp(tx, ty, area))
            nap(0.10, 0.42)
        elif roll < 0.64:
            # hesitation: drift partway to another candidate, maybe pull back
            gx, gy = random.choice(focus)
            ox, oy = self._m if self._m else (fx, fy)
            u = random.uniform(0.45, 0.8)
            self._move(backend, *self._clamp(ox + (gx - ox) * u,
                                             oy + (gy - oy) * u, area))
            nap(0.08, 0.25)
            if random.random() < 0.6 and left() > 0.05:
                v = random.uniform(0.0, 0.35)
                self._move(backend, *self._clamp(ox + (gx - ox) * v,
                                                 oy + (gy - oy) * v, area))
                nap(0.08, 0.30)
        elif roll < 0.78:
            # tiny circle: sizing up the cell
            r = random.uniform(5, 13)
            a0 = random.uniform(0, math.tau)
            for i in range(1, random.randint(3, 6) + 1):
                if left() <= 0.03:
                    break
                a = a0 + math.tau * i / random.randint(4, 6)
                self._move(backend, *self._clamp(fx + r * math.cos(a),
                                                 fy + r * math.sin(a), area))
                time.sleep(min(random.uniform(0.03, 0.09), max(left(), 0)))
            nap(0.05, 0.25)
        elif roll < 0.90:
            # micro fidget in place
            ox, oy = self._m if self._m else (fx, fy)
            for _ in range(random.randint(2, 4)):
                if left() <= 0.03:
                    break
                self._move(backend, *self._clamp(ox + random.uniform(-5, 5),
                                                 oy + random.uniform(-5, 5),
                                                 area))
                time.sleep(min(random.uniform(0.04, 0.12), max(left(), 0)))
            nap(0.05, 0.20)
        else:
            # plain staring — the classic do-nothing "invalid operation"
            nap(0.45, 1.05)
