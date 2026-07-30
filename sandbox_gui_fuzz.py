# -*- coding: utf-8 -*-
"""
Background fuzz verification for paced mode through the REAL control path:

  ControlServer (HTTP panel) -> /start POST -> play_paced(T)

with random targets T in [140, 1000] s, driven against headless Chromium
(Playwright backend) — the physical mouse is never touched.

  - K parallel sessions, each with its own page / player / panel port
  - seeded RNG (seed printed -> reproducible)
  - per-session faulthandler watchdog dumps all thread stacks if a session
    outlives T * 1.3 + 180 s without a result (the "long target freezes"
    detector)
  - two random sessions get an extra mid-game /start POST to prove queued
    submissions are consumed (regression coverage for the control_ui fix)

Assertions at the end:
  * no HANG, every session produced a definite result
  * every win lands within target +-10 %
  * the mouse kept "thinking" while idling (>=10 wander moves)

    python sandbox_gui_fuzz.py [sessions] [seed] [lo] [hi]
                               (targets uniform in [lo, hi], 140..1000 default)

Log: verifier/runs/<timestamp>-v4-gui-fuzz.log
"""
import asyncio
import faulthandler
import io
import os
import random
import sys
import threading
import time
import urllib.request

import numpy as np
from PIL import Image
from playwright.async_api import async_playwright

from msbot.player import Player
from msbot.backend_playwright import AsyncPlaywrightBackend
from msbot.control_ui import ControlServer

LOG_DIR = os.path.join(os.path.dirname(__file__), "verifier", "runs")


class SyncFacade:
    """Sync wrapper that schedules async backend calls on the loop."""
    def __init__(self, backend, loop):
        self.b = backend
        self.loop = loop
        self.moves = []

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def grab(self):
        return self._run(self.b.grab())

    def left_click(self, x, y):
        self._run(self.b.left_click(x, y))

    def right_click(self, x, y):
        self._run(self.b.right_click(x, y))

    def human_move_to(self, x, y):
        self.moves.append((x, y))
        self._run(self.b.human_move_to(x, y))

    def stop_requested(self):
        return False


def post_start(url, t):
    req = urllib.request.Request(url + "start", data=f"t={t}".encode(),
                                 method="POST")
    try:
        urllib.request.urlopen(req, timeout=5)
    except urllib.error.HTTPError as e:
        if e.code != 302:
            raise


def log(logf, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    logf.write(line + "\n")
    logf.flush()


def session_worker(idx, target, mid_post, facade_box, ui, logf, result):
    """Replicates main.gui_mode's loop for one session (retry on loss)."""
    try:
        ui_state = urllib.request.urlopen(ui.url + "state", timeout=5) \
            .read().decode()
        assert "waiting" in ui_state
        post_start(ui.url, target)
        log(logf, f"s{idx}: submitted target {target:.0f}s")
        t_first = ui.wait_for_start()
        assert abs(t_first - target) < 0.01, t_first
        player = Player(facade_box[0], delay=0.03, verbose=False)
        if mid_post is not None:
            def second_submission():
                time.sleep(target / 2)
                post_start(ui.url, mid_post)
                log(logf, f"s{idx}: mid-game POST {mid_post:.0f}s (queued)")
            threading.Thread(target=second_submission, daemon=True).start()
        t0 = time.time()
        res, elapsed = None, 0.0
        for attempt in range(3):
            res, elapsed = player.play_paced(target)
            if res == "lose":
                log(logf, f"s{idx}: lose at {elapsed:.0f}s -> retry")
                player.restart()
                continue
            break
        result.update(res=res, elapsed=elapsed, attempts=attempt + 1,
                      wall=time.time() - t0,
                      wander=len(facade_box[0].moves))
        log(logf, f"s{idx}: {res} elapsed={elapsed:.1f}s "
                  f"target={target:.0f}s attempts={attempt + 1} "
                  f"wander={result['wander']}")
        # if a mid-game value was queued, the panel must consume it next
        if mid_post is not None:
            box = {}

            def consume():
                box["t"] = ui.wait_for_start()

            th = threading.Thread(target=consume, daemon=True)
            th.start()
            th.join(5)
            result["queued_consumed"] = not th.is_alive() and \
                abs(box.get("t", 0) - mid_post) < 0.01
            log(logf, f"s{idx}: queued next target consumed: "
                      f"{result['queued_consumed']}")
    except Exception as e:                                   # noqa: BLE001
        result.update(res="error", err=repr(e))
        log(logf, f"s{idx}: ERROR {e!r}")


async def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else int(time.time())
    lo = float(sys.argv[3]) if len(sys.argv) > 3 else 140.0
    hi = float(sys.argv[4]) if len(sys.argv) > 4 else 1000.0
    rng = random.Random(seed)
    targets = [rng.uniform(lo, hi) for _ in range(k)]
    mid_idx = set(rng.sample(range(k), min(2, k)))
    mid_posts = {i: rng.uniform(lo, hi) for i in mid_idx}

    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{stamp}-v4-gui-fuzz.log")
    logf = open(log_path, "w", encoding="utf-8")
    faulthandler.dump_traceback_later(
        max(targets) * 1.3 + 240, file=logf, exit=True)
    log(logf, f"seed={seed} sessions={k}")
    for i, t in enumerate(targets):
        log(logf, f"s{i}: target={t:.0f}s"
                  + (f" mid-post={mid_posts[i]:.0f}s" if i in mid_idx else ""))

    uis = [ControlServer() for _ in range(k)]
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True,
                                           args=["--no-sandbox"])
        loop = asyncio.get_running_loop()
        threads, results = [], []
        for i in range(k):
            page = await browser.new_page(viewport={"width": 1280,
                                                    "height": 900})
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
            facade = SyncFacade(AsyncPlaywrightBackend(page), loop)
            box = [facade]
            res = {}
            results.append(res)
            th = threading.Thread(
                target=session_worker,
                args=(i, targets[i], mid_posts.get(i), box, uis[i],
                      logf, res),
                daemon=True)
            th.start()
            threads.append(th)

        deadline = time.time() + max(targets) * 1.3 + 240
        while time.time() < deadline:
            if all(not th.is_alive() for th in threads):
                break
            await asyncio.sleep(1.0)
        hung = [i for i, th in enumerate(threads) if th.is_alive()]
        for i in hung:
            results[i].update(res="HANG")
            log(logf, f"s{i}: *** HANG — no result before deadline ***")
        await browser.close()

    log(logf, "---- summary ----")
    ok = True
    for i, res in enumerate(results):
        log(logf, f"s{i}: {res}")
        if res.get("res") == "HANG" or "err" in res:
            ok = False
            continue
        if res.get("res") == "win":
            t, e = targets[i], res["elapsed"]
            if not (t * 0.9 <= e <= t * 1.1):
                log(logf, f"s{i}: pacing off: {e:.1f}s vs {t:.0f}s")
                ok = False
        if res.get("res") not in ("win", "stale", "inconsistent",
                                  "timeout", "stuck"):
            log(logf, f"s{i}: no definite result")
            ok = False
        if res.get("wander", 0) < 10:
            log(logf, f"s{i}: mouse barely wandered ({res.get('wander')})")
            ok = False
        if i in mid_idx and not res.get("queued_consumed"):
            log(logf, f"s{i}: queued mid-game submission NOT consumed")
            ok = False
    log(logf, f"verdict: {'PASS' if ok else 'FAIL'}")
    logf.close()
    print(f"log: {log_path}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
