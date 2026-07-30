# -*- coding: utf-8 -*-
"""
Windows backend: screen capture via `mss`, humanised mouse control via
raw Win32 API (ctypes -> user32: SetCursorPos / mouse_event /
GetAsyncKeyState).  No pyautogui dependency.

Stop the bot two ways, both system-wide (window focus does not matter):
  - press F8
  - slam the mouse into a screen corner and hold it there (~0.4 s)

Run main.py, then focus the browser tab showing
https://www.minesweeper.cn/ (zoom must be 100 %).
"""
import ctypes
import time

import numpy as np

from . import humanize
from .humanize import GameAbort

STOP_KEY = 0x77                            # VK_F8

# mouse_event flags
_MOVE = 0x0001
_LEFTDOWN, _LEFTUP = 0x0002, 0x0004
_RIGHTDOWN, _RIGHTUP = 0x0008, 0x0010


class WindowsBackend:
    def __init__(self, stop_key=STOP_KEY):
        import mss
        self._u = ctypes.windll.user32
        # mss captures *physical* pixels.  A DPI-unaware process gets
        # virtualised (scaled) coordinates from GetCursorPos/SetCursorPos/
        # GetSystemMetrics, so clicks would land at pos*scale and the
        # corner-slam stop zone would sit off-screen.  Opt into per-monitor
        # awareness to keep one coordinate space everywhere.  Harmless when
        # the exe manifest already declared awareness (call just fails).
        for set_aware in (
                lambda: self._u.SetProcessDpiAwarenessContext(
                    ctypes.c_void_p(-4)),          # PER_MONITOR_AWARE_V2
                lambda: ctypes.windll.shcore.SetProcessDpiAwareness(2),
                lambda: self._u.SetProcessDPIAware()):
            try:
                if set_aware():
                    break
            except Exception:
                continue
        self._sct = mss.mss()
        self._mon = self._sct.monitors[1]          # primary monitor
        self._stop_key = stop_key
        self._corner_since = None
        self._scr_w = self._u.GetSystemMetrics(0)
        self._scr_h = self._u.GetSystemMetrics(1)

    # ---------------- stop detection ----------------
    def stop_requested(self):
        """True when F8 is held, or the mouse rests in a screen corner."""
        if self._u.GetAsyncKeyState(self._stop_key) & 0x8000:
            return True
        pt = ctypes.wintypes_POINT() if hasattr(ctypes, "wintypes_POINT") \
            else self._point()
        self._u.GetCursorPos(ctypes.byref(pt))
        corner = (pt.x <= 1 or pt.x >= self._scr_w - 2) and \
                 (pt.y <= 1 or pt.y >= self._scr_h - 2)
        now = time.time()
        if corner:
            if self._corner_since is None:
                self._corner_since = now
            return now - self._corner_since > 0.4
        self._corner_since = None
        return False

    @staticmethod
    def _point():
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
        return POINT()

    # ---------------- screen ----------------
    def grab(self):
        shot = self._sct.grab(self._mon)
        return np.asarray(shot)[:, :, :3][:, :, ::-1]   # BGRA -> RGB

    # ---------------- humanised mouse ----------------
    def _pos(self):
        pt = self._point()
        self._u.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    def _human_move(self, x, y):
        x0, y0 = self._pos()
        for px, py, dt in humanize.gen_path(x0, y0, x, y):
            if self.stop_requested():
                raise GameAbort
            self._u.SetCursorPos(int(round(px)), int(round(py)))
            time.sleep(dt)

    def _human_click(self, x, y, button):
        self._human_move(x, y)
        time.sleep(humanize.pre_click_delay())
        down, up = (_LEFTDOWN, _LEFTUP) if button == "left" \
            else (_RIGHTDOWN, _RIGHTUP)
        self._u.mouse_event(down, 0, 0, 0, 0)
        time.sleep(humanize.hold_time())
        self._u.mouse_event(up, 0, 0, 0, 0)

    def left_click(self, x, y):
        self._human_click(int(x), int(y), "left")

    def right_click(self, x, y):
        self._human_click(int(x), int(y), "right")

    def human_move_to(self, x, y):
        """Idle mouse wander (no click) used by the pacing controller."""
        self._human_move(int(x), int(y))
        # SetCursorPos does NOT reset the Windows idle timer — during long
        # paced waits the screen could otherwise sleep/lock mid-game.
        # A zero-delta mouse_event counts as real input and keeps it awake.
        now = time.time()
        if now - getattr(self, "_last_keepalive", 0.0) > 25:
            self._u.mouse_event(_MOVE, 0, 0, 0, 0)
            self._last_keepalive = now
