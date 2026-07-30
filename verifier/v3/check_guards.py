# -*- coding: utf-8 -*-
"""
verifier v3 — C1/C2 一致性误读防护检查

C1: 静态矛盾棋盘 -> 拒绝行动、安全停止、零危险点击
C2: 数字格跨轮改变 -> 拒绝行动；整板平移 -> 自动重对齐继续

用法: python verifier/v3/check_guards.py   退出码 0 = 通过
"""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from msbot.player import Player                     # noqa: E402
from msbot.vision import BoardGeo, UNKNOWN          # noqa: E402


class StubBackend:
    def __init__(self):
        self.lefts = []
        self.rights = []
        self.moves = []

    def grab(self):
        return np.zeros((400, 800, 3), dtype=np.uint8)

    def left_click(self, x, y):
        self.lefts.append((x, y))

    def right_click(self, x, y):
        self.rights.append((x, y))

    def human_move_to(self, x, y):
        self.moves.append((x, y))

    def stop_requested(self):
        return False


def non_centre_clicks(backend, geo):
    cx, cy = geo.center(4, 4)
    return [(x, y) for x, y in backend.lefts
            if abs(x - cx) > 8 or abs(y - cy) > 8]


def main():
    failures = []
    geo = BoardGeo(x0=100, y0=100, rows=9, cols=9)

    # ---------------- C1: 静态矛盾 -> 安全停止 -------------------------
    print("== C1 静态一致性防护 ==")
    intro = np.full((9, 9), UNKNOWN, dtype=int)
    bad = np.ones((9, 9), dtype=int)      # 全是数字 1……
    bad[4, 4] = 3                          # ……唯独中间是 3：不可能
    backend = StubBackend()
    p = Player(backend, delay=0.01, verbose=False)
    p.geo = geo
    with patch.object(Player, "scan",
                      return_value=(backend.grab(), intro)), \
         patch.object(Player, "scan_stable", side_effect=[bad] * 8):
        res, _ = p.play_paced(0.05)
    nc = non_centre_clicks(backend, geo)
    print(f"  结果={res}  非开局点击={nc}  右键={backend.rights}")
    if res != "inconsistent" or nc or backend.rights:
        failures.append(f"C1: 矛盾棋盘未被拦下 res={res} clicks={nc}")

    # ---------------- C2a: 数字跨轮改变 -> 安全停止 --------------------
    print("== C2a 跨轮转移防护（数字改变）==")
    b0 = np.full((9, 9), UNKNOWN, dtype=int)
    b0[3, 3] = 2
    b1 = b0.copy()
    b1[3, 3] = 5                           # 上一轮是 2，这轮变 5：不可能
    backend2 = StubBackend()
    p2 = Player(backend2, delay=0.01, verbose=False)
    p2.geo = geo
    acts = {"flags": [], "clicks": [(0, 0)], "guess": None}
    scans = [b0, b0] + [b1] * 8            # r0 顶读+复检，之后全是 b1
    with patch.object(Player, "scan",
                      return_value=(backend2.grab(), intro)), \
         patch.object(Player, "scan_stable", side_effect=scans), \
         patch("msbot.player.Solver") as MS:
        MS.return_value.next_actions.return_value = acts
        MS.return_value.mines = 10
        res2, _ = p2.play_paced(0.05)
    nc2 = non_centre_clicks(backend2, geo)
    exp_ok = any(abs(x - geo.center(0, 0)[0]) <= 8
                 and abs(y - geo.center(0, 0)[1]) <= 8 for x, y in nc2)
    print(f"  结果={res2}  非开局点击数={len(nc2)}（应只有r0的1次）")
    if res2 != "inconsistent" or len(nc2) > 1 or not exp_ok:
        failures.append(f"C2a: 数字突变未被拦下 res={res2} clicks={nc2}")

    # ---------------- C2b: 整板平移 -> 重对齐 --------------------------
    print("== C2b 平移重对齐 ==")
    bA = np.full((9, 9), UNKNOWN, dtype=int)
    bA[3, 3] = 2
    bA[5, 6] = 4
    bB = np.full((9, 9), UNKNOWN, dtype=int)
    bB[3, 4] = 2                           # 全部右移一格
    bB[5, 7] = 4
    p3 = Player(StubBackend(), delay=0.01, verbose=False)
    p3._accept_board(bA)
    tok, info = p3._check_transition(bB)
    print(f"  平移检测: {tok} {info}")
    if tok != "shift" or info != (0, 1):
        failures.append(f"C2b: 平移未识别 tok={tok} info={info}")
    trusted = p3._trusted_board(bB, 10)
    aligned = trusted is not None and p3._prev_numbers.get((3, 4)) == 2 \
        and p3._prev_numbers.get((5, 7)) == 4
    print(f"  重对齐后通过校验: {trusted is not None}  坐标已迁移: {aligned}")
    if trusted is None or not aligned:
        failures.append("C2b: 平移后未能继续游戏")

    print("=" * 50)
    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("C1/C2 ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
