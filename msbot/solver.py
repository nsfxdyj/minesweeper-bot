# -*- coding: utf-8 -*-
"""
Decision layer: constraint-propagation minesweeper solver.

Given the visible board it returns guaranteed-safe clicks and guaranteed-mine
cells using two inference mechanisms:

  1. single-constraint rules
       count == 0          -> every unknown cell in the constraint is safe
       count == len(cells) -> every unknown cell is a mine
  2. subset differencing
       A subset of B  ->  B - A is a new constraint with count nB - nA

If no certain move exists it falls back to a probability estimate and picks
the least risky cell (frontier constraints first, global density otherwise).
"""
from .vision import UNKNOWN, FLAG

# level -> mine count (used for the global guess density)
MINES_BY_SHAPE = {(9, 9): 10, (16, 16): 40, (16, 30): 99}


class Solver:
    def __init__(self, rows, cols, mines=None):
        self.rows, self.cols = rows, cols
        self.mines = mines if mines is not None else MINES_BY_SHAPE.get((rows, cols))

    # ------------------------------------------------------------------
    def _neighbors(self, r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    yield nr, nc

    def _constraints(self, board):
        """Build (frozenset(cells), remaining_mine_count) for each number."""
        cons = {}
        for r in range(self.rows):
            for c in range(self.cols):
                n = int(board[r, c])
                if n < 1:
                    continue
                unknown = frozenset(
                    (nr, nc) for nr, nc in self._neighbors(r, c)
                    if board[nr, nc] == UNKNOWN)
                if not unknown:
                    continue
                flagged = sum(1 for nr, nc in self._neighbors(r, c)
                              if board[nr, nc] == FLAG)
                cons[unknown] = n - flagged        # same set -> same residual
        return [(cells, n) for cells, n in cons.items() if n >= 0]

    # ------------------------------------------------------------------
    def infer(self, board):
        """Return (safe:set, mines:set) of cells certain to be safe / mined."""
        cons = self._constraints(board)
        safe, mines = set(), set()
        changed = True
        while changed:
            changed = False
            # normalise against already-known cells
            norm, seen = [], set()
            for cells, n in cons:
                n -= len(cells & mines)          # newly found mines consume count
                cells = cells - safe - mines
                if not cells:
                    continue
                if n <= 0:
                    if not cells <= safe:
                        safe |= cells; changed = True
                elif n == len(cells):
                    if not cells <= mines:
                        mines |= cells; changed = True
                elif (cells, n) not in seen:
                    seen.add((cells, n)); norm.append((cells, n))
            cons = norm
            # subset differencing: A < B  ->  B - A has count nB - nA
            for i in range(len(cons)):
                for j in range(len(cons)):
                    if i == j:
                        continue
                    a, na = cons[i]
                    b, nb = cons[j]
                    if a < b:
                        diff, dn = b - a, nb - na
                        if diff and dn == 0 and not diff <= safe:
                            safe |= diff; changed = True
                        elif diff and dn == len(diff) and not diff <= mines:
                            mines |= diff; changed = True
                        elif diff and 0 < dn < len(diff) and (diff, dn) not in seen:
                            seen.add((diff, dn)); cons.append((diff, dn))
        return safe, mines

    # ------------------------------------------------------------------
    def guess(self, board):
        """Pick the unknown cell with the lowest estimated mine probability."""
        cons = self._constraints(board)
        risk = {}
        for cells, n in cons:
            p = n / len(cells)
            for cell in cells:
                risk[cell] = max(risk.get(cell, 0.0), p)

        frontier = set(risk)
        best, best_p = None, 2.0
        for cell, p in risk.items():
            if p < best_p:
                best, best_p = cell, p

        # non-frontier cells: global density, only better if fewer mines left
        rest = [(r, c) for r in range(self.rows) for c in range(self.cols)
                if board[r, c] == UNKNOWN and (r, c) not in frontier]
        if rest and self.mines is not None:
            flagged = int((board == FLAG).sum())
            p_global = max(self.mines - flagged, 0) / (len(rest) + len(frontier) or 1)
            if p_global < best_p:
                import random
                best = random.choice(rest)
        if best is None:                          # no info at all
            import random
            unknowns = [(r, c) for r in range(self.rows) for c in range(self.cols)
                        if board[r, c] == UNKNOWN]
            best = random.choice(unknowns) if unknowns else None
        return best

    # ------------------------------------------------------------------
    def next_actions(self, board, exact=True):
        """-> dict(clicks=[(r,c)], flags=[(r,c)], guess=(r,c)|None)

        Cheap subset inference first; only when it runs dry, invoke the
        exact probability engine — it finds certain moves the inference
        misses (p=0 / p=1) and otherwise gives the least risky guess.
        """
        safe, mines = self.infer(board)
        clicks = set(safe)
        flags = set(mines)
        guess = None
        if not clicks and exact:
            from .probability import exact_probabilities
            res = exact_probabilities(board, self.mines)
            if res is not None:
                probs, free_p = res
                for cell, p in probs.items():
                    if p == 0.0:
                        clicks.add(cell)
                    elif p == 1.0:
                        flags.add(cell)
                if not clicks:
                    guess = self._exact_guess(board, probs, free_p)
        if not clicks and guess is None:
            guess = self.guess(board)
        return {"clicks": sorted(clicks), "flags": sorted(flags),
                "guess": guess}

    def _exact_guess(self, board, probs, free_p):
        """Pick the lowest-probability cell from exact results."""
        if not probs:
            return None
        best = min(probs, key=lambda c: probs[c])
        if free_p < probs[best]:
            frontier = set(probs)
            rest = [(r, c) for r in range(self.rows) for c in range(self.cols)
                    if board[r, c] == UNKNOWN and (r, c) not in frontier]
            if rest:
                import random
                return random.choice(rest)
        return best
