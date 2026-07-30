# -*- coding: utf-8 -*-
"""
Exact mine-probability engine (constraint enumeration).

For every unknown cell touching a numbered cell ("frontier"), enumerate ALL
mine assignments consistent with the visible constraints, then combine the
per-component solution counts with the global mine budget to get an exact
probability for every unknown cell:

    P(cell) = sum over all consistent configurations where cell is a mine
              (number of completions of the other components and the
              unconstrained cells) / total number of configurations

Probability 0 => certainly safe, probability 1 => certainly a mine — this
strictly subsumes the subset inference in solver.infer().
"""
import math

from .vision import UNKNOWN, FLAG

MAX_SOLUTIONS = 200_000        # per-component enumeration cap
MAX_COMPONENT = 36             # cells per component cap


def _neighbors(r, c, rows, cols):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc


def _constraints(board):
    """(frozenset(cells), residual_mine_count) per visible number."""
    rows, cols = board.shape
    cons = {}
    for r in range(rows):
        for c in range(cols):
            n = int(board[r, c])
            if n < 1:
                continue
            unknown = frozenset(
                (nr, nc) for nr, nc in _neighbors(r, c, rows, cols)
                if board[nr, nc] == UNKNOWN)
            if not unknown:
                continue
            flagged = sum(1 for nr, nc in _neighbors(r, c, rows, cols)
                          if board[nr, nc] == FLAG)
            cons[unknown] = n - flagged
    return [(cells, n) for cells, n in cons.items() if n >= 0]


def _enumerate_component(cells, cons):
    """Brute-force one independent component with constraint pruning.

    Returns (dist, cell_mine, truncated):
      dist[m]      -> number of solutions using exactly m mines
      cell_mine[c] -> {m: number of solutions where cell c is a mine}
    """
    n = len(cells)
    idx = {c: i for i, c in enumerate(cells)}
    ccons = [([idx[c] for c in cs], k) for cs, k in cons]
    occ = [0] * n
    cell_cons = [[] for _ in range(n)]
    for ci, (inds, _) in enumerate(ccons):
        for i in inds:
            occ[i] += 1
            cell_cons[i].append(ci)
    order = sorted(range(n), key=lambda i: -occ[i])   # most-constrained first

    rem = [len(inds) for inds, _ in ccons]            # unassigned cells left
    need = [k for _, k in ccons]                      # mines still to place
    assign = [0] * n
    dist = {}
    cell_mine = {c: {} for c in cells}
    truncated = [False]
    sols = [0]

    def rec(o, mines):
        if sols[0] >= MAX_SOLUTIONS:
            truncated[0] = True
            return
        if o == n:
            sols[0] += 1
            dist[mines] = dist.get(mines, 0) + 1
            for i in range(n):
                if assign[i]:
                    d = cell_mine[cells[i]]
                    d[mines] = d.get(mines, 0) + 1
            return
        i = order[o]
        for v in (0, 1):
            ok = True
            for ci in cell_cons[i]:
                rem[ci] -= 1
                need[ci] -= v
                if need[ci] < 0 or need[ci] > rem[ci]:
                    ok = False
            if ok:
                assign[i] = v
                rec(o + 1, mines + v)
            for ci in cell_cons[i]:
                rem[ci] += 1
                need[ci] += v
            if truncated[0]:
                return

    rec(0, 0)
    return dist, cell_mine, truncated[0]


def _components(frontier, cons):
    """Split frontier cells into independent components (union-find on the
    constraint hypergraph)."""
    parent = {c: c for c in frontier}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for cells, _ in cons:
        cells = list(cells)
        for c in cells[1:]:
            union(cells[0], c)
    comps = {}
    for c in frontier:
        comps.setdefault(find(c), []).append(c)
    comp_cons = {root: [] for root in comps}
    for cells, k in cons:
        root = find(next(iter(cells)))
        comp_cons[root].append((cells, k))
    return [(comps[r], comp_cons[r]) for r in comps]


def exact_probabilities(board, mines_total):
    """-> (probs, free_prob) or None if intractable / mine count unknown.

    probs: dict (r, c) -> exact mine probability for every frontier cell
    free_prob: mine probability for each unconstrained unknown cell
    """
    if mines_total is None:
        return None
    cons = _constraints(board)
    frontier = set().union(*(c for c, _ in cons)) if cons else set()
    rows, cols = board.shape
    unknowns = [(r, c) for r in range(rows) for c in range(cols)
                if board[r, c] == UNKNOWN]
    n_free = len(unknowns) - len(frontier)
    M = mines_total - int((board == FLAG).sum())
    if M < 0 or M > len(unknowns):
        return None
    if not frontier:                       # no information yet
        p = M / len(unknowns) if unknowns else 0.0
        return {}, p

    comp_data = []
    for cells, ccons in _components(frontier, cons):
        dist, cell_mine, truncated = _enumerate_component(sorted(cells), ccons)
        if truncated:                  # too many solutions -> give up exactly
            return None
        comp_data.append((sorted(cells), dist, cell_mine))

    # prefix/suffix convolutions of per-component mine-count distributions
    k = len(comp_data)
    dists = [cd[1] for cd in comp_data]

    def conv(a, b):
        out = {}
        for x, cx in a.items():
            for y, cy in b.items():
                out[x + y] = out.get(x + y, 0.0) + cx * cy
        return out

    prefix = [{0: 1.0}]
    for d in dists:
        prefix.append(conv(prefix[-1], d))
    suffix = [{0: 1.0}]
    for d in reversed(dists):
        suffix.append(conv(suffix[-1], d))
    suffix.reverse()

    def binom(n, kk):
        return float(math.comb(n, kk)) if 0 <= kk <= n else 0.0

    total = 0.0
    for s, ways in prefix[k].items():
        total += ways * binom(n_free, M - s)
    if total <= 0:
        return None

    probs = {}
    for i, (cells, dist, cell_mine) in enumerate(comp_data):
        others = conv(prefix[i], suffix[i + 1])
        for cell in cells:
            num = 0.0
            for m, cnt in cell_mine[cell].items():
                for f, ways in others.items():
                    num += cnt * ways * binom(n_free, M - m - f)
            probs[cell] = num / total

    free_num = 0.0
    for s, ways in prefix[k].items():
        free_num += ways * binom(n_free - 1, M - s - 1)
    return probs, free_num / total


def best_guess(board, mines_total):
    """Exact-probability guess -> (cell, probability) or None (intractable)."""
    res = exact_probabilities(board, mines_total)
    if res is None:
        return None
    probs, free_p = res
    if not probs:
        # nothing on the frontier: pick any unknown (all share free_p)
        rows, cols = board.shape
        for r in range(rows):
            for c in range(cols):
                if board[r, c] == UNKNOWN:
                    return (r, c), free_p
        return None
    best_cell = min(probs, key=lambda c: probs[c])
    if free_p < probs[best_cell]:
        rows, cols = board.shape
        frontier = set(probs)
        for r in range(rows):
            for c in range(cols):
                if board[r, c] == UNKNOWN and (r, c) not in frontier:
                    return (r, c), free_p
    return best_cell, probs[best_cell]
