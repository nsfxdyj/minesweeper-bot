# -*- coding: utf-8 -*-
"""
Property test: on randomly generated boards with ground-truth mines,
solver.infer() must NEVER mark a safe cell as mine or a mine as safe.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import numpy as np

from msbot.vision import UNKNOWN, FLAG
from msbot.solver import Solver


def make_board(rows, cols, n_mines, rng):
    mines = set(rng.sample([(r, c) for r in range(rows) for c in range(cols)],
                           n_mines))
    numbers = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            if (r, c) in mines:
                numbers[r, c] = -9
            else:
                numbers[r, c] = sum(
                    (nr, nc) in mines
                    for nr in range(max(0, r - 1), min(rows, r + 2))
                    for nc in range(max(0, c - 1), min(cols, c + 2))
                    if (nr, nc) != (r, c))
    return mines, numbers


def reveal(numbers, mines, start, rng, extra=0.0):
    """Simulate flood-fill reveal from a safe start cell -> visible board."""
    rows, cols = numbers.shape
    vis = np.full((rows, cols), UNKNOWN, dtype=np.int8)
    stack = [start]
    while stack:
        r, c = stack.pop()
        if (r, c) in mines or vis[r, c] != UNKNOWN:
            continue
        vis[r, c] = numbers[r, c]
        if numbers[r, c] == 0:
            for nr in range(max(0, r - 1), min(rows, r + 2)):
                for nc in range(max(0, c - 1), min(cols, c + 2)):
                    if vis[nr, nc] == UNKNOWN:
                        stack.append((nr, nc))
    # randomly reveal extra numbered cells to create richer mid-game states
    for r in range(rows):
        for c in range(cols):
            if vis[r, c] == UNKNOWN and (r, c) not in mines and rng.random() < extra:
                vis[r, c] = numbers[r, c]
    return vis


def test_solver_never_wrong():
    rng = random.Random(42)
    checked = 0
    for trial in range(300):
        rows = cols = 9
        mines, numbers = make_board(rows, cols, 10, rng)
        start = rng.choice([cell for cell in
                            [(r, c) for r in range(9) for c in range(9)]
                            if cell not in mines])
        vis = reveal(numbers, mines, start, rng, extra=rng.random() * 0.3)
        # mark a few random real mines as flags (must be correct flags)
        for m in mines:
            if rng.random() < 0.2:
                vis[m] = FLAG
        solver = Solver(rows, cols, 10)
        safe, mined = solver.infer(vis)
        for cell in safe:
            assert cell not in mines, f"safe cell {cell} is a mine! trial {trial}"
        for cell in mined:
            assert cell in mines, f"flagged cell {cell} is safe! trial {trial}"
        checked += len(safe) + len(mined)
    print(f"OK: 300 random boards, {checked} inferred cells, zero mistakes")


if __name__ == "__main__":
    test_solver_never_wrong()
