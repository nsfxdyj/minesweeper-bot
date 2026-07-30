# -*- coding: utf-8 -*-
"""
Windows entry point — GUI (browser control panel) mode by default:

    python main.py            # open the control panel, input target seconds
    python main.py --cli 20   # legacy no-GUI mode: play N games full speed

GUI flow:
  1. a tiny local web page opens (works with Edge / any default browser)
  2. input the target time in seconds and press start
  3. the bot paces itself to finish each game in about that time
  4. lose  -> instant retry, clock reset, same target
     win   -> pause; input the next time and press start again

Stop any time, system-wide: press F8 or slam the mouse into a screen
corner and hold it ~0.4 s.
"""
import sys
import time

from msbot.player import Player
from msbot.backend_windows import WindowsBackend
from msbot.humanize import GameAbort


def gui_mode():
    from msbot.control_ui import ControlServer

    ui = ControlServer()
    print(f"control panel: {ui.url}")
    ui.open_browser()
    print("input the target seconds in the page and press start.")
    print("press F8 (or slam mouse into a screen corner) to stop.")

    player = None
    played = wins = 0
    while True:
        target = ui.wait_for_start()
        print(f"[target] {target}s")
        try:
            if player is None:
                player = Player(WindowsBackend(), delay=0.03)
            while True:                                    # retry on loss
                ui.set(detail=f"本局目标 {target}s，进行中……")
                res, elapsed = player.play_paced(target)
                if res == "lose":
                    played += 1
                    ui.set(status="lost",
                           detail=f"踩雷（{elapsed:.1f}s），立即重试，目标 {target}s")
                    print(f"[lose] {elapsed:.1f}s -> retry")
                    player.restart()
                    continue
                played += 1
                if res == "win":
                    wins += 1
                    ui.set(status="won",
                           detail=f"完成！用时 {elapsed:.1f}s（目标 {target}s）\n"
                                  f"战绩 {wins}/{played}\n输入下一个时间并开始")
                    print(f"[win] {elapsed:.1f}s / target {target}s")
                else:
                    ui.set(status="stopped",
                           detail=f"本局异常结束（{res}）\n输入时间重新开始")
                break
        except GameAbort:
            ui.set(status="stopped", detail="已停止（F8/角落急停）\n输入时间重新开始")
            print("[stopped] by user")
        except RuntimeError as e:
            ui.set(status="stopped", detail=f"错误：{e}\n确认扫雷页面可见后重新输入时间")
            print(f"[error] {e}")


def cli_mode(games):
    print("Minesweeper bot starting in 5 s - focus the game tab now.")
    print("press F8 (or slam mouse into a screen corner) to stop")
    time.sleep(5)
    player = Player(WindowsBackend(), delay=0.03)
    results = player.play_session(games)
    print(f"done: {results.count('win')}/{len(results)} wins")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        cli_mode(int(sys.argv[2]) if len(sys.argv) > 2 else 10_000)
    else:
        try:
            gui_mode()
        except KeyboardInterrupt:
            print("\nbye")


if __name__ == "__main__":
    main()
