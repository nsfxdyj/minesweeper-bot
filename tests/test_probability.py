# -*- coding: utf-8 -*-
"""
Exact-probability engine tests:

1. sanity: probabilities of all unknown cells sum to the remaining mine count
2. soundness: p==0 cells are never mines, p==1 cells always are
   (checked against ground truth on random boards)
3. agreement: every cell the subset inference marks safe/mine must have
   exact probability 0 / 1
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from msbot.vision import UNKNOWN, FLAG
from msbot.solver import Solver
from msbot.probability import exact_probabilities, best_guess
from tests.test_solver import make_board, reveal


def test_probabilities():
    rng = random.Random(7)
    checked_cells = 0
    for trial in range(60):
        rows, cols = 9, 9
        mines, numbers = make_board(rows, cols, 10, rng)
        start = rng.choice([cell for cell in
                            [(r, c) for r in range(9) for c in range(9)]
                            if cell not in mines])
        vis = reveal(numbers, mines, start, rng, extra=rng.random() * 0.3)
        for m in mines:                       # some correct flags
            if rng.random() < 0.3:
                vis[m] = FLAG

        res = exact_probabilities(vis, 10)
        assert res is not None, f"trial {trial}: intractable 9x9 board?!"
        probs, free_p = res
        for cell, p in probs.items():
            assert -1e-9 <= p <= 1 + 1e-9, (cell, p)
        assert 0.0 <= free_p <= 1.0

        # 1. expected total mines matches
        n_flagged = int((vis == FLAG).sum())
        total = sum(probs.values()) + free_p * (
            int((vis == UNKNOWN).sum()) - len(probs))
        assert abs(total - (10 - n_flagged)) < 1e-6, \
            f"trial {trial}: expected {10 - n_flagged}, got {total}"

        # 2. soundness vs ground truth
        for cell, p in probs.items():
            if p == 0.0:
                assert cell not in mines, f"trial {trial}: p=0 on a mine {cell}"
            if p == 1.0:
                assert cell in mines, f"trial {trial}: p=1 on safe {cell}"
            checked_cells += 1

        # 3. agreement with subset inference
        safe, mined = Solver(9, 9, 10).infer(vis)
        for cell in safe:
            assert probs[cell] == 0.0, (cell, probs[cell])
        for cell in mined:
            assert probs[cell] == 1.0, (cell, probs[cell])

        guess = best_guess(vis, 10)
        assert guess is not None and 0.0 <= guess[1] <= 1.0

    print(f"OK: 60 boards, {checked_cells} frontier cells, "
          "probabilities consistent & sound")


if __name__ == "__main__":
    test_probabilities()
