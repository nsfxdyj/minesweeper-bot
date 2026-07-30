# -*- coding: utf-8 -*-
"""
verifier v2 — B1/B2/B3 自动检查（长目标踩雷 bug 修复）

B1: 等待后棋盘移位 -> 弃用旧计划，点击新棋盘计划点
B2: _apply 批中重定位 -> 后续点击跟随新原点
B3: Windows 后端 keep-alive 存在（源码级 + 节流逻辑）

用法: python verifier/v2/check_stale_fix.py   退出码 0 = 通过
"""
import sys
import time
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


def board_with(fill, **cells):
    b = np.full((9, 9), UNKNOWN, dtype=int)
    b[0, 0] = fill
    for (r, c), v in cells.items():
        b[r, c] = v
    return b


def main():
    failures = []

    # ---------------- B1: 等待后移位 -> 重规划 -------------------------
    print("== B1 等待后移位重规划 ==")
    geo = BoardGeo(x0=100, y0=100, rows=9, cols=9)
    board_a = board_with(2)              # 等待前的世界（角格 2，3 邻格可满足）
    board_b = board_a.copy()             # 等待中级联晚到：多出数字，旧数字不变
    board_b[2, 2] = 1
    board_win = board_b.copy()           # 收尾：10 面旗 -> 触发胜利
    for r, c in [(0, 1), (1, 0), (5, 0), (6, 0), (7, 0), (8, 0),
                 (5, 5), (6, 5), (7, 5), (8, 5)]:
        board_win[r, c] = -2             # FLAG

    backend = StubBackend()
    player = Player(backend, delay=0.01, verbose=False)
    player.geo = geo

    scans = [board_a,      # round 0 顶读
             board_b,      # before_batch 等待后复检：变了！
             board_b,      # 重规划后等待后复检：稳定
             board_win]    # round 1 顶读 -> 胜利收尾
    acts_a = {"flags": [], "clicks": [(0, 0)], "guess": None}
    acts_b = {"flags": [], "clicks": [(1, 1)], "guess": None}

    def fake_actions(b):
        return acts_b if b[2, 2] == 1 else acts_a

    with patch.object(Player, "scan",
                      return_value=(backend.grab(), board_a)), \
         patch.object(Player, "scan_stable", side_effect=scans), \
         patch("msbot.player.Solver") as MockSolver:
        MockSolver.return_value.next_actions.side_effect = fake_actions
        MockSolver.return_value.mines = 10
        t0 = time.time()
        res, _ = player.play_paced(0.05)
    assert res == "win", f"B1 流程异常: {res}"

    exp_b = geo.center(1, 1)               # 重规划后的目标格
    bad_a = geo.center(0, 0)               # 旧计划的目标格
    ok_b = any(abs(x - exp_b[0]) <= 8 and abs(y - exp_b[1]) <= 8
               for x, y in backend.lefts)
    ok_a = any(abs(x - bad_a[0]) <= 8 and abs(y - bad_a[1]) <= 8
               for x, y in backend.lefts)
    # 第一次点击是开局中心点，要排除
    centre = geo.center(4, 4)
    non_centre = [(x, y) for x, y in backend.lefts
                  if abs(x - centre[0]) > 8 or abs(y - centre[1]) > 8]
    print(f"  非开局点击={non_centre}")
    print(f"  命中新计划点(1,1): {ok_b}  误中旧计划点(0,0): {ok_a}")
    if not ok_b or ok_a:
        failures.append("B1: 等待后移位仍按旧坐标点击")

    # ---------------- B2: 批中重定位 ----------------------------------
    print("== B2 批中移位修正 ==")
    geo2 = BoardGeo(x0=140, y0=120, rows=9, cols=9)   # 移位后原点
    backend2 = StubBackend()
    p2 = Player(backend2, delay=0.01, verbose=False)
    p2.geo = geo
    five = {"flags": [], "guess": None,
            "clicks": [(0, i) for i in range(5)]}
    with patch("msbot.player.locate_board", return_value=geo2):
        p2._apply(five, scale=0.5, verify_geo=True)
    # 前 4 次用旧原点，第 5 次（i=4, 触发重定位）用新原点
    old_exp = geo.center(0, 0)
    new_exp = geo2.center(0, 4)
    first_ok = abs(backend2.lefts[0][0] - old_exp[0]) <= 8
    last_ok = abs(backend2.lefts[4][0] - new_exp[0]) <= 8 and \
        abs(backend2.lefts[4][1] - new_exp[1]) <= 8
    print(f"  首击旧原点: {first_ok}  第5击新原点: {last_ok}")
    if not (first_ok and last_ok):
        failures.append("B2: 批中重定位未生效")

    # ---------------- B3: keep-alive ----------------------------------
    print("== B3 Windows keep-alive ==")
    src = Path(sys.path[0], "msbot", "backend_windows.py").read_text(
        encoding="utf-8")
    has_flag = "_MOVE = 0x0001" in src
    has_call = "mouse_event(_MOVE, 0, 0, 0, 0)" in src
    has_gate = "_last_keepalive" in src and "> 25" in src
    print(f"  MOUSEEVENTF_MOVE 常量: {has_flag}  零位移注入: {has_call}"
          f"  25s 节流: {has_gate}")
    if not (has_flag and has_call and has_gate):
        failures.append("B3: keep-alive 缺失或节流参数漂移")

    print("=" * 50)
    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("B1/B2/B3 ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
