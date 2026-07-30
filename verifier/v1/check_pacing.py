# -*- coding: utf-8 -*-
"""
verifier v1 — 动态配速 + 拟人游走 自动检查（A1-A5）

用法: python verifier/v1/check_pacing.py
退出码 0 = 全部通过
"""
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from msbot.pace import Pacer            # noqa: E402

AREA = (100, 100, 400, 225)
FOCUS = [(200, 150), (225, 150), (200, 175), (225, 160), (210, 175)]
DECOYS = [(180, 140), (250, 150)]


class FakeBackend:
    def __init__(self):
        self.moves = []
        self.left = 0
        self.right = 0

    def human_move_to(self, x, y):
        self.moves.append((time.time(), x, y))

    def left_click(self, x, y):
        self.left += 1

    def right_click(self, x, y):
        self.right += 1

    def stop_requested(self):
        return False


def run_sim(regime, target, seed):
    """模拟一局：均匀/前快/后重 三种进度节奏。"""
    random.seed(seed)
    total = 71
    pacer = Pacer(target, total, delay_base=0.03)
    be = FakeBackend()
    pacer.begin()
    done, scales = 0, []
    while done < total:
        if regime == "uniform":
            inc = random.randint(2, 5)
        elif regime == "front":
            inc = random.randint(8, 14) if done < total * 0.35 \
                else random.randint(1, 3)
        else:  # "slow": 每步带固定重开销，制造落后局面
            inc = random.randint(1, 3) if done < total * 0.65 \
                else random.randint(6, 12)
        inc = min(inc, total - done)
        finishing = done + inc >= total
        pacer.before_batch(done, inc, finishing, be, FOCUS, DECOYS, AREA)
        scales.append(pacer.scale)
        t = time.time()
        for _ in range(inc):
            cost = 0.03 * pacer.scale * random.uniform(0.6, 1.9)
            if regime == "slow":
                cost += 0.16            # 不可压缩的固定成本
            time.sleep(max(cost, 0.001))
        pacer.observe(time.time() - t, inc)
        time.sleep(0.06 if regime != "slow" else 0.10)
        done += inc
    return pacer.elapsed(), scales, be


def main():
    failures = []

    # ---- A1 收束 ±10% & A2 动态调整 --------------------------------
    print("== A1/A2 仿真收束与动态调整 ==")
    cases = [("uniform", 10, 2), ("front", 10, 2), ("slow", 15.5, 2)]
    for regime, target, n_seeds in cases:
        for seed in range(n_seeds):
            el, scales, be = run_sim(regime, target, 1000 + seed)
            ok = 0.9 * target <= el <= 1.1 * target
            print(f"  {regime:7s} T={target:4.1f}s seed={seed} "
                  f"elapsed={el:5.2f}s  scale[{min(scales):.2f}.."
                  f"{max(scales):.2f}]  {'OK' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"A1 {regime} seed={seed}: {el:.2f}s "
                                f"not in [{(0.9*target):.1f},{(1.1*target):.1f}]")
            if regime == "front" and max(scales) < 1.3:
                failures.append(f"A2 front: 未观察到放慢 max={max(scales):.2f}")
            if regime == "slow" and min(scales) > 0.55:
                failures.append(f"A2 slow: 未观察到加速 min={min(scales):.2f}")

    # ---- A3/A4/A5 游走行为 ------------------------------------------
    print("== A3/A4/A5 游走行为（6 秒等待窗口）==")
    random.seed(7)
    pacer = Pacer(60, 71)
    be = FakeBackend()
    pacer.begin()
    pacer.wait_until(6.0, be, FOCUS, DECOYS, AREA)

    pool = FOCUS + DECOYS
    far = 0
    big_jumps = 0
    steps, dts = [], []
    prev = None
    for t, x, y in be.moves:
        d = min(((x - px) ** 2 + (y - py) ** 2) ** 0.5 for px, py in pool)
        if d > 60:
            far += 1
        if prev is not None:
            pj = ((x - prev[1]) ** 2 + (y - prev[2]) ** 2) ** 0.5
            steps.append(pj)
            dts.append(t - prev[0])
            if pj > 150:
                big_jumps += 1
        prev = (t, x, y)

    print(f"  移动次数={len(be.moves)}  离池>60px={far}  大跳>150px={big_jumps}")
    if far:
        failures.append(f"A3: {far} 个点远离候选池")
    if big_jumps:
        failures.append(f"A3: {big_jumps} 次大跳")

    if len(dts) >= 20:
        cv = statistics.pstdev(dts) / statistics.mean(dts)
        distinct = len({round(s, 1) for s in steps})
        print(f"  间隔CV={cv:.2f}  不同步长数={distinct}/{len(steps)}")
        if cv < 0.25:
            failures.append(f"A4: 间隔太规律 CV={cv:.2f}")
        if distinct < min(10, len(steps) // 2):
            failures.append(f"A4: 步长太单一 distinct={distinct}")
    else:
        failures.append(f"A4: 移动次数太少 {len(dts)}")

    print(f"  无效操作期间点击: left={be.left} right={be.right}")
    if be.left or be.right:
        failures.append("A5: 等待期间发生点击")

    # ---- 汇总 --------------------------------------------------------
    print("=" * 50)
    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
