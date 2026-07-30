# -*- coding: utf-8 -*-
"""
Execution layer: backend-agnostic game loop.

A backend must provide:
    grab()              -> full-screen RGB ndarray (H, W, 3, uint8)
    left_click(x, y)    -> absolute pixel coordinates
    right_click(x, y)

The player locates the board with vision.locate_board(), reads it with
vision.read_board(), asks Solver for the next actions and executes them.
"""
import time

import numpy as np

from .vision import UNKNOWN, FLAG, BLAST, MINE, locate_board, read_board
from .solver import Solver, MINES_BY_SHAPE
from . import humanize
from .humanize import GameAbort
from .pace import Pacer


class GameOver(Exception):
    pass


class GameStopped(GameAbort):
    """Raised when the user presses the stop hotkey."""


class Player:
    def __init__(self, backend, delay=0.05, verbose=True):
        self.backend = backend
        self.delay = delay
        self.verbose = verbose
        self.geo = None
        self._prev_numbers = None      # {(r,c): n} of the last trusted board

    def _log(self, *args):
        if self.verbose:
            print(*args)

    # ------------------------------------------------------------------
    def scan(self):
        """Grab a screenshot, (re-)locate the board and read all cells.

        Re-locating every round costs ~1 ms and protects against the page
        layout shifting (e.g. ads loading above the board)."""
        img = self.backend.grab()
        geo = locate_board(img)
        if geo is not None:
            if self.geo is None:
                self._log(f"[board] origin=({geo.x0},{geo.y0}) "
                          f"{geo.rows}x{geo.cols}")
            self.geo = geo
        if self.geo is None:
            raise RuntimeError("board not found in screenshot")
        return img, read_board(img, self.geo)

    def scan_stable(self):
        """Read until two consecutive snapshots agree — cascade openings
        render progressively and a mid-animation read is garbage."""
        prev = None
        for _ in range(30):
            _, board = self.scan()
            if prev is not None and (board == prev).all():
                return board
            prev = board
            time.sleep(0.04)
        return prev

    def _check_stop(self):
        if getattr(self.backend, "stop_requested", lambda: False)():
            raise GameStopped

    # ------------------------------------------------------------------
    # read validation: never act on a board that cannot be true.
    # A *consistent* misread (page re-rendered at a fractional offset,
    # popup covering cells, ...) passes scan_stable, so stability alone
    # is not enough — the board must also make logical sense.
    # ------------------------------------------------------------------
    @staticmethod
    def _board_consistent(board, mines):
        """Static sanity: every revealed number must be satisfiable."""
        rows, cols = board.shape
        n_flags = int((board == FLAG).sum())
        if mines is not None and n_flags > mines:
            return False, f"flags {n_flags} > mines {mines}"
        for r in range(rows):
            for c in range(cols):
                n = board[r, c]
                if not 1 <= n <= 8:
                    continue
                f = u = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < rows and 0 <= cc < cols:
                            if board[rr, cc] == FLAG:
                                f += 1
                            elif board[rr, cc] == UNKNOWN:
                                u += 1
                if f > n or n > f + u:
                    return False, f"({r},{c})={n} flags={f} unknowns={u}"
        return True, ""

    def _check_transition(self, board):
        """Revealed numbers never change (except into blast/mine on a
        loss).  -> ('ok'|'shift'|'bad', info); 'shift' = the whole board
        moved by whole cells (ad push) and everything realigns."""
        prev = self._prev_numbers
        if not prev:
            return "ok", None
        rows, cols = board.shape

        def fits(dr, dc):
            for (r, c), n in prev.items():
                rr, cc = r + dr, c + dc
                if not (0 <= rr < rows and 0 <= cc < cols):
                    return False
                v = board[rr, cc]
                if not (v == n or v == BLAST or v == MINE):
                    return False
            return True

        if fits(0, 0):
            return "ok", None
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if (dr or dc) and fits(dr, dc):
                    return "shift", (dr, dc)
        for (r, c), n in prev.items():
            v = board[r, c]
            if not (v == n or v == BLAST or v == MINE):
                return "bad", f"({r},{c}) was {n}, now {v}"
        return "bad", "unknown"

    def _accept_board(self, board, shift=None):
        if self._prev_numbers is None:
            self._prev_numbers = {}
        if shift:
            dr, dc = shift
            self._prev_numbers = {(r + dr, c + dc): n
                                  for (r, c), n in self._prev_numbers.items()}
        rows, cols = board.shape
        for r in range(rows):
            for c in range(cols):
                if 1 <= board[r, c] <= 8:
                    self._prev_numbers[(r, c)] = int(board[r, c])

    def _trusted_board(self, board, mines, retries=5, pause=0.4):
        """Validate a fresh scan; re-read on suspicion.  Returns a
        trusted board, or None — the caller must then stop safely
        instead of clicking."""
        why = ""
        for _ in range(retries + 1):
            ok, why = self._board_consistent(board, mines)
            tok, tinfo = self._check_transition(board) if ok \
                else ("bad", None)
            if ok and tok in ("ok", "shift"):
                if tok == "shift":
                    self._log(f"[guard] board shifted {tinfo} cells, "
                              f"realigned")
                self._accept_board(board, tinfo if tok == "shift" else None)
                return board
            why = why or tinfo
            time.sleep(pause)
            board = self.scan_stable()
        self._log(f"[guard] board failed validation: {why} -> stop safely")
        return None

    def _apply(self, actions, scale=1.0, verify_geo=False):
        seq = [("right", r, c) for r, c in actions["flags"]]
        targets = list(actions["clicks"])
        if actions["guess"] is not None:
            targets.append(actions["guess"])
            self._log(f"[guess] {actions['guess']}")
        seq += [("left", r, c) for r, c in targets]
        for i, (button, r, c) in enumerate(seq):
            self._check_stop()
            # long batches give ads time to shift the layout mid-batch:
            # re-locate every few steps and click with fresh coordinates
            if verify_geo and i and i % 4 == 0:
                geo = locate_board(self.backend.grab())
                if geo is not None and (geo.x0, geo.y0) != \
                        (self.geo.x0, self.geo.y0):
                    self._log(f"[geo] board moved to ({geo.x0},{geo.y0}) "
                              f"mid-batch, adjusting")
                    self.geo = geo
            x, y = humanize.click_point(*self.geo.center(r, c))
            if button == "right":
                self.backend.right_click(x, y)
            else:
                self.backend.left_click(x, y)
            time.sleep(humanize.action_delay(self.delay * scale))
        return len(seq)

    # ------------------------------------------------------------------
    def play_game(self, max_rounds=400):
        """Play one game. Returns 'win' or 'lose'."""
        # wait up to 30 s for the board to appear (user may still be
        # alt-tabbing to the game tab)
        for attempt in range(30):
            try:
                img, board = self.scan()
                break
            except RuntimeError:
                if attempt == 29:
                    raise
                time.sleep(1)
        rows, cols = board.shape
        solver = Solver(rows, cols, MINES_BY_SHAPE.get((rows, cols)))

        # opening move: centre cell (first click is guaranteed safe)
        x, y = self.geo.center(rows // 2, cols // 2)
        self.backend.left_click(x, y)
        time.sleep(self.delay * 2)

        last_board, stale = None, 0
        for rnd in range(max_rounds):
            self._check_stop()
            board = self._trusted_board(self.scan_stable(), solver.mines)
            if board is None:
                return "inconsistent"
            if (board == BLAST).any() or (board == MINE).any():
                self._log(f"[round {rnd}] BOOM -> lose")
                return "lose"
            n_unknown = int((board == UNKNOWN).sum())
            n_flags = int((board == FLAG).sum())
            # the site ends the game when every mine is correctly flagged
            if n_unknown == 0 or \
                    (solver.mines is not None and n_flags >= solver.mines):
                self._log(f"[round {rnd}] cleared -> win")
                return "win"

            actions = solver.next_actions(board)
            if not actions["clicks"] and not actions["flags"] \
                    and actions["guess"] is None:
                return "stuck"
            self._apply(actions)
            time.sleep(self.delay)
            if last_board is not None and (board == last_board).all():
                stale += 1
                if stale >= 6:
                    self._log("no progress for 6 rounds -> stale")
                    return "stale"
            else:
                stale = 0
            last_board = board.copy()
        return "timeout"

    def restart(self):
        """Click the smiley to start a fresh game."""
        x, y = self.geo.restart_point()
        self.backend.left_click(x, y)
        self._prev_numbers = None
        time.sleep(0.5)

    def _focus_points(self, board, actions, max_pts=5, max_decoys=4,
                      radius=3):
        """Pixel points the mouse should hover around while "thinking":
        the cluster of upcoming action cells near the first target,
        plus a few revealed number cells nearby as decoys."""
        cells = list(actions["flags"]) + list(actions["clicks"])
        if actions["guess"] is not None:
            cells.append(actions["guess"])
        if not cells:
            return [], []
        r0, c0 = cells[0]
        near = [(r, c) for r, c in cells
                if abs(r - r0) <= radius and abs(c - c0) <= radius]
        chosen = (near or cells[:1])[:max_pts]
        focus = [self.geo.center(r, c) for r, c in chosen]
        chosen_set = set(chosen)
        decoys, seen = [], set()
        rows, cols = board.shape
        for r, c in chosen:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols \
                            and (rr, cc) not in chosen_set \
                            and (rr, cc) not in seen \
                            and 1 <= board[rr, cc] <= 8:
                        seen.add((rr, cc))
                        decoys.append(self.geo.center(rr, cc))
            if len(decoys) >= max_decoys:
                break
        return focus, decoys[:max_decoys]

    # ------------------------------------------------------------------
    def play_paced(self, target, max_rounds=600):
        """Play one game stretched to ~`target` seconds.

        Returns (result, elapsed).  The clock starts at the first click,
        mirroring the site's own timer; the artificial delay is spread
        over the action gaps and the mouse keeps wandering meanwhile."""
        for attempt in range(30):
            try:
                img, board = self.scan()
                break
            except RuntimeError:
                if attempt == 29:
                    raise
                time.sleep(1)
        rows, cols = board.shape
        mines = MINES_BY_SHAPE.get((rows, cols))
        solver = Solver(rows, cols, mines)
        self._prev_numbers = None          # fresh game, no history yet
        # work metric = every cell that must be resolved: safe cells to
        # open AND mines to flag (resolved counts opened + flagged, so
        # total must be the full board or the schedule finishes early)
        total_work = rows * cols
        pacer = Pacer(target, total_work, delay_base=self.delay)

        x, y = humanize.click_point(*self.geo.center(rows // 2, cols // 2))
        self.backend.left_click(x, y)
        pacer.begin()
        time.sleep(humanize.action_delay(0.08))

        last_board, stale = None, 0
        for rnd in range(max_rounds):
            self._check_stop()
            rt0 = time.time()
            board = self._trusted_board(self.scan_stable(), mines)
            if board is None:
                return "inconsistent", pacer.elapsed()

            # Re-plan loop: the paced idle can last minutes, plenty of
            # time for ads to reload and shift the board.  After every
            # idle the board is re-read; if it no longer matches the
            # world the plan was made for, discard the plan — clicking
            # stale coordinates is how mines get hit.
            for replan in range(4):
                if (board == BLAST).any() or (board == MINE).any():
                    self._log(f"[round {rnd}] BOOM -> lose "
                              f"({pacer.elapsed():.1f}s)")
                    return "lose", pacer.elapsed()
                n_unknown = int((board == UNKNOWN).sum())
                n_flags = int((board == FLAG).sum())
                if n_unknown == 0 or \
                        (mines is not None and n_flags >= mines):
                    self._log(f"[round {rnd}] cleared in "
                              f"{pacer.elapsed():.1f}s")
                    return "win", pacer.elapsed()

                actions = solver.next_actions(board)
                if not actions["clicks"] and not actions["flags"] \
                        and actions["guess"] is None:
                    return "stuck", pacer.elapsed()
                resolved = int((board != UNKNOWN).sum())
                n_flag = len(actions["flags"])
                incoming = n_flag + len(actions["clicks"]) \
                    + (1 if actions["guess"] is not None else 0)
                # a batch ends the game when it flags the last mines or
                # acts on every remaining unknown cell
                finishing = (mines is not None
                             and n_flags + n_flag >= mines) \
                    or n_unknown <= incoming
                focus, decoys = self._focus_points(board, actions)
                area = (self.geo.x0, self.geo.y0, cols * 25, rows * 25)
                self._log(f"[pace] r{rnd} done={resolved}/{total_work} "
                          f"inc={incoming} fin={int(finishing)} "
                          f"scale={pacer.scale:.2f} "
                          f"el={pacer.elapsed():.1f}"
                          + (f" replan={replan}" if replan else ""))
                waited = pacer.before_batch(resolved, incoming, finishing,
                                            self.backend, focus, decoys,
                                            area)
                board2 = self.scan_stable()
                if (board2 == board).all():
                    break
                board2 = self._trusted_board(board2, mines)
                if board2 is None:
                    return "inconsistent", pacer.elapsed()
                if (board2 == board).all():
                    break
                self._log(f"[pace] r{rnd} board shifted during idle "
                          f"-> re-plan (no stale clicks)")
                board = board2
            else:
                self._log("[pace] page keeps shifting -> stale")
                return "stale", pacer.elapsed()

            t_a = time.time()
            n_applied = self._apply(actions, scale=pacer.scale,
                                    verify_geo=True)
            dt_a = time.time() - t_a
            pacer.observe(dt_a, n_applied)
            pacer.observe_overhead(time.time() - rt0 - waited - dt_a)

            if last_board is not None and (board == last_board).all():
                stale += 1
                if stale >= 6:
                    return "stale", pacer.elapsed()
            else:
                stale = 0
            last_board = board.copy()
        return "timeout", pacer.elapsed()

    # ------------------------------------------------------------------
    def play_session(self, games=1):
        results = []
        for g in range(games):
            self._log(f"===== game {g + 1}/{games} =====")
            try:
                res = self.play_game()
            except GameOver:
                res = "lose"
            except GameAbort:
                self._log("stopped by user (F8 / corner slam)")
                results.append("stopped")
                break
            results.append(res)
            self._log(f"result: {res}")
            if g + 1 < games:
                self.restart()
        wins = results.count("win")
        self._log(f"===== session over: {wins}/{len(results)} wins =====")
        return results
