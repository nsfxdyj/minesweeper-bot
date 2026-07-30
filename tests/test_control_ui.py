# -*- coding: utf-8 -*-
"""
Control panel regression tests — no browser needed, drives the real HTTP
server and the wait_for_start() synchronisation directly.

  1. a submission made WHILE a game is running must be consumed by the next
     wait_for_start() call instead of being swallowed (used to leave the
     panel stuck in "waiting" forever — looked like a freeze)
  2. the normal path still blocks until a POST arrives
"""
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from msbot.control_ui import ControlServer


def post(url, t):
    req = urllib.request.Request(url + "start", data=f"t={t}".encode(),
                                 method="POST")
    try:
        urllib.request.urlopen(req, timeout=5)
    except urllib.error.HTTPError as e:
        if e.code != 302:                       # redirect is the OK path
            raise


def test_queued_submission_not_swallowed():
    ui = ControlServer()
    post(ui.url, 240)                           # user presses start mid-game
    t0 = time.time()
    t = ui.wait_for_start()                     # game ends, next round starts
    dt = time.time() - t0
    assert t == 240, f"expected queued 240, got {t}"
    assert dt < 1.0, f"wait_for_start blocked {dt:.1f}s on a queued value"
    print("OK: submission made during play is queued, not swallowed")


def test_blocks_until_post():
    ui = ControlServer()
    got = {}

    def worker():
        got["t"] = ui.wait_for_start()

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    th.join(0.5)
    assert th.is_alive(), "wait_for_start returned without any submission"
    post(ui.url, 100)
    th.join(2)
    assert not th.is_alive(), "wait_for_start still blocked after POST"
    assert got["t"] == 100, got
    print("OK: blocks while idle, returns on POST")


if __name__ == "__main__":
    test_queued_submission_not_swallowed()
    test_blocks_until_post()
