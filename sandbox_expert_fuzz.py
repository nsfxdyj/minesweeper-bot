# -*- coding: utf-8 -*-
"""
Expert-mode fuzz with post-mortem loss analysis.

Plays paced expert games (16x30 / 99 mines) on headless Chromium at
device_scale_factor=1.5 — the same resampled render the vision layer sees
on a 150 % Windows display — with random targets in [200, 500] s, through
the same Player.play_paced loop the exe uses.  The physical mouse is
never touched.

Every solver decision is recorded (board snapshot + actions).  On a loss
the post-mortem board (all mines revealed) is read back and the death is
classified:

  GAMBLE   the blasted cell was the solver's explicit guess and its exact
           probability was > 0  -> legitimate bad luck
  SAFE-CLICK  the blasted cell was in actions["clicks"], i.e. the solver
           claimed it safe -> logic or vision BUG
  MISREAD  revealed numbers on the pre-loss board disagree with the
           post-mortem board -> vision BUG (upstream of any decision)

    python sandbox_expert_fuzz.py [sessions] [minutes] [seed] [lo] [hi]

Log: verifier/runs/<timestamp>-v5-expert-fuzz.log
"""
import asyncio
import faulthandler
import json
import os
import random
import sys
import threading
import time

import numpy as np
from PIL import Image
from playwright.async_api import async_playwright

import msbot.player as mp
from msbot.player import Player
from msbot.backend_playwright import AsyncPlaywrightBackend
from msbot.solver import Solver
from msbot.vision import (UNKNOWN, FLAG, BLAST, MINE,
                          locate_board, read_board)

DSF = 1.5                       # simulate 150 % display scaling
LOG_DIR = os.path.join(os.path.dirname(__file__), "verifier", "runs")
OUT_DIR = os.path.join(os.path.dirname(__file__), "debug_out")

_hist = threading.local()       # per-session decision history


class RecSolver(Solver):
    """Solver that records (board, actions) for the calling thread."""

    def next_actions(self, board, exact=True):
        actions = super().next_actions(board, exact)
        if not hasattr(_hist, "log"):
            _hist.log = []
        _hist.log.append((board.copy(), dict(actions)))
        return actions


mp.Solver = RecSolver           # Player picks this up at construction


class ScaledFacade:
    """Sync wrapper; vision works in physical pixels (dsf=1.5 screenshots)
    while Playwright clicks in CSS pixels — convert at the boundary,
    exactly like the DPI-aware Windows backend maps physical pixels."""
    def __init__(self, backend, loop):
        self.b = backend
        self.loop = loop
        self.moves = []

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def grab(self):
        return self._run(self.b.grab())

    def left_click(self, x, y):
        self._run(self.b.left_click(x / DSF, y / DSF))

    def right_click(self, x, y):
        self._run(self.b.right_click(x / DSF, y / DSF))

    def human_move_to(self, x, y):
        self.moves.append((x, y))
        self._run(self.b.human_move_to(x / DSF, y / DSF))

    def stop_requested(self):
        return False


def log(logf, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    logf.write(line + "\n")
    logf.flush()


def board_list(b):
    return [[int(v) for v in row] for row in b]


def analyse_loss(player, facade, pre_board, actions):
    """Classify a death using the post-mortem board.  -> dict verdict."""
    img = facade.grab()
    geo = locate_board(img)
    post = read_board(img, geo if geo is not None else player.geo)
    blasts = [tuple(x) for x in np.argwhere(post == BLAST)]
    out = {"blasts": blasts,
           "post_mines": int((post == MINE).sum()),
           "post_flags": int((post == FLAG).sum()),
           "pre_flags": int((pre_board == FLAG).sum()),
           "actions": {k: (list(map(list, v)) if isinstance(v, list) else v)
                       for k, v in actions.items()}}

    # vision misread check: revealed numbers may not change post-mortem
    misreads = []
    for r in range(pre_board.shape[0]):
        for c in range(pre_board.shape[1]):
            n = pre_board[r, c]
            if 1 <= n <= 8 and post[r, c] != n:
                misreads.append([(r, c), int(n), int(post[r, c])])
    out["misreads"] = misreads

    verdicts = []
    if misreads:
        verdicts.append("MISREAD")
    for br, bc in blasts:
        cell = (int(br), int(bc))
        if any(tuple(x) == cell for x in actions.get("clicks", [])):
            verdicts.append("SAFE-CLICK")
        elif actions.get("guess") is not None \
                and tuple(actions["guess"]) == cell:
            # a real gamble — but the exact engine must agree it was risky
            p_val = None
            try:
                from msbot.probability import exact_probabilities
                res = exact_probabilities(pre_board, 99)
                if res is not None:
                    probs, free_p = res
                    p_val = probs.get(cell, free_p)
            except Exception as e:                           # noqa: BLE001
                p_val = f"err:{e!r}"
            out["guess_p"] = p_val
            verdicts.append("GAMBLE" if not isinstance(p_val, float)
                            or p_val > 0 else "P0-GUESS-BUG")
        else:
            verdicts.append("UNKNOWN-DEATH")
    out["verdict"] = "+".join(sorted(set(verdicts))) or "?"
    return out


def session_worker(idx, page_box, facade_box, loop_box, lo, hi, seed,
                   deadline, logf, result):
    rng = random.Random(seed)
    games = 0
    try:
        facade_box[0]  # page already loaded by main
        player = None
        while time.time() < deadline:
            target = rng.uniform(lo, hi)
            if player is None:
                player = Player(facade_box[0], delay=0.03, verbose=True)
            _hist.log = []
            t0 = time.time()
            try:
                res, elapsed = player.play_paced(target)
            except Exception as e:                           # noqa: BLE001
                result.setdefault("events", []).append(
                    {"game": games, "res": "error", "err": repr(e)})
                log(logf, f"s{idx} g{games}: ERROR {e!r} -> reload page")
                try:
                    awaitable = page_box[0].reload(timeout=60000)
                    asyncio.run_coroutine_threadsafe(awaitable, loop_box[0])
                    time.sleep(6)
                    asyncio.run_coroutine_threadsafe(
                        page_box[0].click("text=专家"), loop_box[0])
                    time.sleep(2)
                    player = None
                    continue
                except Exception as e2:                      # noqa: BLE001
                    log(logf, f"s{idx}: reload failed {e2!r}")
                    break
            games += 1
            ev = {"game": games, "target": round(target, 1),
                  "res": res, "elapsed": round(elapsed, 1)}
            if res == "inconsistent":
                # capture the rejected board + screenshot for forensics
                stamp = time.strftime("%H%M%S")
                png = os.path.join(
                    OUT_DIR, f"inc_{stamp}_s{idx}_g{games}.png")
                try:
                    Image.fromarray(facade_box[0].grab()).save(png)
                    if _hist.log:
                        np.savetxt(png.replace(".png", "_board.txt"),
                                   _hist.log[-1][0], fmt="%d")
                    ev["evidence"] = png
                except Exception as e:                       # noqa: BLE001
                    ev["evidence"] = f"save-failed:{e!r}"
                log(logf, f"s{idx} g{games}: INCONSISTENT at "
                          f"{elapsed:.0f}s/{target:.0f}s -> {png}")
            elif res == "lose" and _hist.log:
                pre_board, actions = _hist.log[-1]
                try:
                    ev["loss"] = analyse_loss(player, facade_box[0],
                                              pre_board, actions)
                except Exception as e:                       # noqa: BLE001
                    ev["loss"] = {"verdict": f"analysis-err:{e!r}"}
                log(logf, f"s{idx} g{games}: LOSE at {elapsed:.0f}s/"
                    f"{target:.0f}s verdict={ev['loss']['verdict']} "
                    f"blasts={ev['loss'].get('blasts')} "
                    f"guess_p={ev['loss'].get('guess_p')} "
                    f"misreads={len(ev['loss'].get('misreads', []))}")
            else:
                log(logf, f"s{idx} g{games}: {res} {elapsed:.0f}s/"
                    f"{target:.0f}s")
            result.setdefault("events", []).append(ev)
            # next game: click the smiley, keep going until the deadline
            try:
                player.restart()
            except Exception as e:                           # noqa: BLE001
                log(logf, f"s{idx}: restart failed {e!r}")
                break
            _ = t0
        result["games"] = games
    except Exception as e:                                   # noqa: BLE001
        result["fatal"] = repr(e)
        log(logf, f"s{idx}: FATAL {e!r}")


async def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 45
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else int(time.time())
    lo = float(sys.argv[4]) if len(sys.argv) > 4 else 200.0
    hi = float(sys.argv[5]) if len(sys.argv) > 5 else 500.0

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{stamp}-v5-expert-fuzz.log")
    logf = open(log_path, "w", encoding="utf-8")
    faulthandler.dump_traceback_later(minutes * 60 + 300, file=logf,
                                      exit=True)
    deadline = time.time() + minutes * 60
    log(logf, f"seed={seed} sessions={k} budget={minutes}min "
              f"targets=[{lo},{hi}] dsf={DSF}")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True,
                                           args=["--no-sandbox"])
        loop = asyncio.get_running_loop()
        threads, results = [], []
        for i in range(k):
            page = await browser.new_page(
                viewport={"width": 1500, "height": 1000},
                device_scale_factor=DSF)
            for attempt in range(3):
                try:
                    await page.goto("https://www.minesweeper.cn/",
                                    timeout=60000)
                    break
                except Exception as e:                       # noqa: BLE001
                    if attempt == 2:
                        raise
                    log(logf, f"s{i}: goto failed ({e}); retrying")
            await page.wait_for_timeout(4000)
            await page.click("text=专家")
            await page.wait_for_timeout(1200)
            facade = ScaledFacade(AsyncPlaywrightBackend(page), loop)
            res = {}
            results.append(res)
            th = threading.Thread(
                target=session_worker,
                args=(i, [page], [facade], [loop], lo, hi,
                      seed * 1000 + i, deadline, logf, res),
                daemon=True)
            th.start()
            threads.append(th)

        while time.time() < deadline + 600:
            if all(not th.is_alive() for th in threads):
                break
            await asyncio.sleep(1.0)
        hung = [i for i, th in enumerate(threads) if th.is_alive()]
        for i in hung:
            results[i]["HANG"] = True
            log(logf, f"s{i}: *** still running at cutoff ***")
        await browser.close()

    log(logf, "---- summary ----")
    total_g = total_w = 0
    losses = []
    jdump = lambda o: json.dumps(
        o, ensure_ascii=False,
        default=lambda x: int(x) if isinstance(x, np.integer) else str(x))
    for i, res in enumerate(results):
        log(logf, f"s{i}: {jdump(res)}")
        for ev in res.get("events", []):
            if ev.get("res") == "error":
                continue
            total_g += 1
            total_w += ev["res"] == "win"
            if ev["res"] == "lose":
                losses.append((i, ev["game"],
                               ev.get("loss", {}).get("verdict")))
    log(logf, f"games={total_g} wins={total_w} "
              f"rate={(total_w / total_g * 100 if total_g else 0):.0f}%")
    for i, g, v in losses:
        log(logf, f"loss s{i} g{g}: {v}")
    bug_losses = [v for _, _, v in losses
                  if v and not v.startswith("GAMBLE")]
    log(logf, f"verdict: {'BUG SUSPECTED' if bug_losses else 'no bug evidence'}"
              f" ({len(losses)} losses, {len(bug_losses)} suspicious)")
    logf.close()
    print(f"log: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
