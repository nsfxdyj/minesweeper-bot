# -*- coding: utf-8 -*-
"""
Async Playwright backend used for sandbox / CI end-to-end tests.

The bot reads the board purely from screenshots (same vision code path as
the Windows backend) and clicks through Playwright's mouse, which dispatches
real browser input events at viewport coordinates — so this exercises the
full pipeline: screenshot -> locate -> classify -> solve -> click.
"""
import io

import numpy as np
from PIL import Image


class AsyncPlaywrightBackend:
    def __init__(self, page):
        self.page = page

    async def grab(self):
        png = await self.page.screenshot()
        return np.array(Image.open(io.BytesIO(png)).convert("RGB"))

    async def left_click(self, x, y):
        await self.page.mouse.click(x, y)

    async def right_click(self, x, y):
        await self.page.mouse.click(x, y, button="right")

    async def human_move_to(self, x, y):
        """Idle mouse wander along a human path (no click)."""
        from . import humanize
        cur = self.page.mouse
        x0, y0 = getattr(self, "_mx", x), getattr(self, "_my", y)
        for px, py, dt in humanize.gen_path(x0, y0, x, y):
            await cur.move(px, py)
            await self.page.wait_for_timeout(max(int(dt * 1000), 1))
        self._mx, self._my = x, y
