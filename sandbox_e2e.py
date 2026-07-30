# -*- coding: utf-8 -*-
"""
Sandbox end-to-end test: drives the REAL https://www.minesweeper.cn/ game
in headless Chromium through the full pipeline
(screenshot -> locate -> classify -> solve -> real mouse events).

    python3 sandbox_e2e.py [games] [level]
    level: basic (9x9/10) | middle (16x16/40) | expert (16x30/99)
"""
import asyncio
import sys
import time

import numpy as np
from playwright.async_api import async_playwright

from msbot.vision import UNKNOWN, FLAG, BLAST, MINE, locate_board, read_board
from msbot.solver import Solver, MINES_BY_SHAPE

LAST_DUMP = None

LEVEL_LINK = {"basic": "基础", "middle": "中级", "expert": "专家"}


async def play_game(page, verbose=False):
    global LAST_DUMP
    png = await page.screenshot()
    import io
    from PIL import Image
    img = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
    geo = locate_board(img)
    assert geo is not None, "board not located"
    solver = Solver(geo.rows, geo.cols, MINES_BY_SHAPE.get((geo.rows, geo.cols)))

    async def grab_board():
        nonlocal geo
        png = await page.screenshot()
        im = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
        g = locate_board(im)          # re-locate every round: ads can shift layout
        if g is not None:
            geo = g
        return read_board(im, geo)

    def center(r, c):
        return geo.x0 + c * 25 + 12, geo.y0 + r * 25 + 12

    pos = [(geo.x0 + geo.cols * 12, geo.y0 + geo.rows * 12)]  # mouse tracker

    # opening click
    x, y = center(geo.rows // 2, geo.cols // 2)
    await page.mouse.click(x, y)
    await page.wait_for_timeout(100)

    async def read_stable():
        """Read until two consecutive snapshots agree (cascades render
        progressively; a mid-animation read is garbage)."""
        prev = None
        for _ in range(30):
            b = await grab_board()
            if prev is not None and (b == prev).all():
                return b
            prev = b
            await page.wait_for_timeout(40)
        return prev

    guesses = 0
    last_board, stale = None, 0
    for rnd in range(500):
        board = await read_stable()
        if (board == BLAST).any() or (board == MINE).any():
            await page.screenshot(path=f"/tmp/lose_{int(time.time()*10)}.png")
            if LAST_DUMP:
                b0, a0 = LAST_DUMP
                np.savetxt("/tmp/lose_board.txt", b0, fmt="%d")
                with open("/tmp/lose_actions.txt", "w") as f:
                    f.write(str(a0))
            return "lose", guesses, rnd
        n_flags = int((board == FLAG).sum())
        if int((board == UNKNOWN).sum()) == 0 or \
                (solver.mines is not None and n_flags >= solver.mines):
            return "win", guesses, rnd

        actions = solver.next_actions(board)
        LAST_DUMP = (board.copy(), dict(actions))
        # exercise the same humanised click pipeline as the Windows backend
        async def hclick(r, c, button="left"):
            from msbot.humanize import gen_path, click_point
            tx, ty = click_point(*center(r, c))
            cur = pos[0]
            for px, py, dt in gen_path(cur[0], cur[1], tx, ty):
                await page.mouse.move(px, py)
                await page.wait_for_timeout(int(dt * 1000))
            pos[0] = (tx, ty)
            await page.mouse.click(tx, ty, button=button)

        for r, c in actions["flags"]:
            await hclick(r, c, "right")
            await page.wait_for_timeout(15)
        targets = list(actions["clicks"])
        if actions["guess"] is not None:
            guesses += 1
            targets.append(actions["guess"])
        if not targets and not actions["flags"]:
            return "stuck", guesses, rnd
        for r, c in targets:
            await hclick(r, c)
            await page.wait_for_timeout(15)
        await page.wait_for_timeout(30)
        # no-progress guard
        if last_board is not None and (board == last_board).all():
            stale += 1
            if stale >= 6:
                await page.screenshot(path=f"/tmp/stale_{int(time.time())}.png")
                return "stale", guesses, rnd
        else:
            stale = 0
        last_board = board.copy()
    return "timeout", guesses, 500


async def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    level = sys.argv[2] if len(sys.argv) > 2 else "basic"
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        await page.goto("https://www.minesweeper.cn/", timeout=60000)
        await page.wait_for_timeout(4000)
        if level != "basic":
            await page.click(f"text={LEVEL_LINK[level]}")
            await page.wait_for_timeout(1000)

        results, total_g, t0 = [], 0, time.time()
        for g in range(games):
            res, guesses, rounds = await play_game(page)
            results.append(res)
            total_g += guesses
            print(f"game {g+1:>3}: {res:<7} guesses={guesses:<3} rounds={rounds}")
            if g + 1 < games:
                # click the smiley to restart
                png = await page.screenshot()
                import io
                from PIL import Image
                img = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
                geo = locate_board(img)
                if geo is None:
                    await page.screenshot(path="/tmp/restart_fail.png")
                    print("DEBUG: locate failed, screenshot -> /tmp/restart_fail.png")
                    raise SystemExit(1)
                await page.mouse.click(geo.x0 + geo.cols * 25 // 2, geo.y0 - 25)
                await page.wait_for_timeout(400)
        wins = results.count("win")
        dt = time.time() - t0
        print(f"\n== {level}: {wins}/{games} wins "
              f"({wins/games*100:.0f}%), {total_g} guesses, {dt:.0f}s ==")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
