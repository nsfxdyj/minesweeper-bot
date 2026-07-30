# -*- coding: utf-8 -*-
"""
Sandbox test for paced mode:
  1. ControlServer serves the panel page and /state
  2. play_paced(T) finishes a real basic game in ~T seconds (+-10%)
  3. loss -> immediate retry handled by caller (loop until win, max 3)
"""
import asyncio
import io
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


class SyncFacade:
    """Sync wrapper that schedules async backend calls on the loop."""
    def __init__(self, backend, loop):
        self.b = backend
        self.loop = loop
        self._pos = (640, 300)
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
    target = float(sys.argv[1]) if len(sys.argv) > 1 else 18.0

    # --- control panel sanity check ---
    ui = ControlServer()
    page_txt = urllib.request.urlopen(ui.url, timeout=5).read().decode()
    assert "扫雷机器人控制台" in page_txt, "panel page broken"
    state = urllib.request.urlopen(ui.url + "state", timeout=5).read().decode()
    assert "waiting" in state, "state endpoint broken"
    print(f"OK: control panel at {ui.url}")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        await page.goto("https://www.minesweeper.cn/", timeout=60000)
        await page.wait_for_timeout(4000)

        backend = AsyncPlaywrightBackend(page)
        loop = asyncio.get_running_loop()
        facade = SyncFacade(backend, loop)
        player = Player(facade, delay=0.03, verbose=True)

        result = {}
        def worker():
            for _ in range(3):
                try:
                    res, elapsed = player.play_paced(target)
                except Exception as e:
                    result["err"] = repr(e)
                    return
                if res == "lose":
                    player.restart()
                    continue
                result["res"] = res
                result["t"] = elapsed
                return

        t0 = time.time()
        th = threading.Thread(target=worker, daemon=True)
        th.start()
        while th.is_alive():
            await asyncio.sleep(0.5)
            if time.time() - t0 > target * 6 + 60:
                break

        await browser.close()

    assert "res" in result, f"no result: {result}"
    res, elapsed = result["res"], result["t"]
    print(f"result={res} elapsed={elapsed:.1f}s target={target}s")
    assert res == "win", f"unexpected result {res}"
    assert target * 0.9 <= elapsed <= target * 1.1, \
        f"pacing off: {elapsed:.1f}s vs target {target}s (+-10%)"
    print(f"wander moves during game: {len(facade.moves)}")
    assert len(facade.moves) >= 10, "mouse barely wandered while idling"
    print("OK: paced mode works, timing within +-10%, mouse kept thinking")


if __name__ == "__main__":
    asyncio.run(main())
