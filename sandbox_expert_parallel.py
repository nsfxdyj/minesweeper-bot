# -*- coding: utf-8 -*-
"""
Parallel expert-mode paced test (verifier v3 / C4).

N browser contexts play expert (16x30/99) games simultaneously, each
paced to ~target seconds, ONE attempt each (no retry — we measure the
raw outcome).  Assertions:
  * every win lands within +-10% of target
  * win count >= ceil(N*0.62)  (baseline full-speed win rate is ~85 %;
    anything much lower reproduces the customer's "keeps hitting mines")

usage: python sandbox_expert_parallel.py [target=600] [n=8]
"""
import asyncio
import math
import sys
import threading
import time

from playwright.async_api import async_playwright

from msbot.player import Player
from msbot.backend_playwright import AsyncPlaywrightBackend


class SyncFacade:
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


async def main():
    target = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True,
                                           args=["--no-sandbox"])
        loop = asyncio.get_running_loop()
        results, threads = {}, []
        for i in range(n):
            ctx = await browser.new_context(
                viewport={"width": 1280, "height": 900})
            page = await ctx.new_page()
            await page.goto("https://www.minesweeper.cn/", timeout=60000)
            await page.wait_for_timeout(1500)
            await page.click("text=专家")
            await page.wait_for_timeout(2500)
            facade = SyncFacade(AsyncPlaywrightBackend(page), loop)
            player = Player(facade, delay=0.03, verbose=False)
            player._log = lambda *a, _i=i: print(f"[t{_i}]", *a, flush=True)

            def worker(_i=i, _p=player):
                try:
                    res, el = _p.play_paced(target)
                    results[_i] = (res, round(el, 1))
                except Exception as e:            # noqa: BLE001
                    results[_i] = ("error", repr(e))
                print(f"[t{_i}] finished: {results[_i]}", flush=True)

            threads.append(threading.Thread(target=worker, daemon=True))
        for th in threads:
            th.start()
        t0 = time.time()
        while any(th.is_alive() for th in threads):
            await asyncio.sleep(1.0)
            if time.time() - t0 > target * 2 + 240:
                print("WATCHDOG: giving up on stragglers")
                break
        await browser.close()

    print("=" * 56)
    wins = 0
    pacing_fail = []
    for i in sorted(results):
        res, el = results[i]
        tag = ""
        if res == "win":
            wins += 1
            if not (target * 0.9 <= el <= target * 1.1):
                pacing_fail.append((i, el))
                tag = "  <- PACING OFF"
        print(f"  game {i}: {res}  {el}s{tag}")
    need = math.ceil(n * 0.62)
    print(f"wins: {wins}/{n} (need >= {need})  "
          f"guard-events: see [tN] lines above")
    ok = wins >= need and not pacing_fail and len(results) == n
    print("C4", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
