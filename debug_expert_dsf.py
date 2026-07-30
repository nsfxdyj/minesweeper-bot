# -*- coding: utf-8 -*-
"""
Debug helper: play full-speed expert games at device_scale_factor=1.5 and
capture evidence when the consistency guard aborts ("inconsistent").

For every aborted game it saves:
  debug_out/inc_<ts>_s<session>_g<game>.png   screenshot at failure
  debug_out/inc_<ts>_s<session>_g<game>.txt   the board the guard rejected
and prints the guard's rejection reason ([guard] lines, verbose=True).

    python debug_expert_dsf.py [sessions] [games_each]
"""
import asyncio
import io
import os
import sys
import threading
import time

import numpy as np
from PIL import Image
from playwright.async_api import async_playwright

from msbot.player import Player
from msbot.backend_playwright import AsyncPlaywrightBackend
from sandbox_expert_fuzz import ScaledFacade, DSF

OUT = os.path.join(os.path.dirname(__file__), "debug_out")


def session_worker(idx, facade, games, target, log_lines):
    player = Player(facade, delay=0.03, verbose=True)
    import contextlib
    for g in range(games):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                if target:
                    res, _ = player.play_paced(target)
                else:
                    res = player.play_game()
        except Exception as e:                               # noqa: BLE001
            res = f"error:{e!r}"
        text = buf.getvalue()
        guards = [l for l in text.splitlines() if "[guard]" in l]
        log_lines.append(f"s{idx} g{g}: {res} guards={guards[-2:]}")
        if res == "inconsistent":
            stamp = int(time.time())
            png = os.path.join(OUT, f"inc_{stamp}_s{idx}_g{g}.png")
            img = facade.grab()
            Image.fromarray(img).save(png)
            txt = png.replace(".png", ".txt")
            with open(txt, "w") as f:
                f.write(text)
            log_lines.append(f"  -> saved {png}")
        if g + 1 < games:
            try:
                player.restart()
            except Exception as e:                           # noqa: BLE001
                log_lines.append(f"s{idx}: restart failed {e!r}")
                break


async def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    target = float(sys.argv[3]) if len(sys.argv) > 3 else 0
    os.makedirs(OUT, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True,
                                           args=["--no-sandbox"])
        loop = asyncio.get_running_loop()
        threads, log_lines = [], []
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
                    print(f"s{i}: goto failed ({e}); retrying")
            await page.wait_for_timeout(4000)
            await page.click("text=专家")
            await page.wait_for_timeout(1200)
            facade = ScaledFacade(AsyncPlaywrightBackend(page), loop)
            th = threading.Thread(target=session_worker,
                                  args=(i, facade, games, target,
                                        log_lines),
                                  daemon=True)
            th.start()
            threads.append(th)
        t0 = time.time()
        budget = games * (target * 1.5 + 120) if target \
            else games * 240 + 120
        while any(th.is_alive() for th in threads) and \
                time.time() - t0 < budget:
            await asyncio.sleep(1.0)
        await browser.close()
    for l in log_lines:
        print(l)


if __name__ == "__main__":
    asyncio.run(main())
