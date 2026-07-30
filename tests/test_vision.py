# -*- coding: utf-8 -*-
"""
Vision regression test: locate + read two real game screenshots and compare
cell-by-cell against human-verified labels.

  won_board.png       9x9  won game (10 flags, rest opened)
  mid_postmortem.png  16x16 lost game (blast + revealed mines + navy 4!)
"""
import os

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from msbot.vision import locate_board, read_board

FIX = os.path.dirname(__file__) + "/fixtures"


def check(png, expected_txt, rows, cols):
    img = np.array(Image.open(os.path.join(FIX, png)).convert("RGB"))
    geo = locate_board(img)
    assert geo is not None, f"board not located in {png}"
    assert (geo.rows, geo.cols) == (rows, cols), (geo.rows, geo.cols)
    board = read_board(img, geo)
    expected = np.loadtxt(os.path.join(FIX, expected_txt), dtype=int)
    diff = int((board != expected).sum())
    assert diff == 0, f"{png}: {diff} cells misread\n{board}\nvs\n{expected}"
    print(f"OK {png}: located at ({geo.x0},{geo.y0}), all {rows*cols} cells correct")


def test_all():
    check("won_board.png", "won_board_expected.txt", 9, 9)
    check("mid_postmortem.png", "mid_postmortem_expected.txt", 16, 16)


def check_scaled(png, expected_txt, rows, cols, scale):
    """Simulate OS display scaling (e.g. 1.5 = 150 %): the same board
    rendered bigger must still locate with the true shape and read every
    cell correctly (a 16x16@37.5px frame used to fake-match as 24x24@25px).
    """
    img = Image.open(os.path.join(FIX, png)).convert("RGB")
    w, h = img.size
    img = np.array(img.resize((int(w * scale), int(h * scale)),
                              Image.BILINEAR))
    geo = locate_board(img)
    assert geo is not None, f"board not located in {png} x{scale}"
    assert (geo.rows, geo.cols) == (rows, cols), \
        f"x{scale}: detected {geo.rows}x{geo.cols} cell={geo.cell:.2f}"
    board = read_board(img, geo)
    expected = np.loadtxt(os.path.join(FIX, expected_txt), dtype=int)
    diff = int((board != expected).sum())
    assert diff == 0, f"{png} x{scale}: {diff} cells misread\n{board}\nvs\n{expected}"
    print(f"OK {png} x{scale}: {rows}x{cols} cell={geo.cell:.1f}, "
          f"all {rows*cols} cells correct")


def test_scaled():
    for scale in (1.5, 1.25, 2.0):
        check_scaled("won_board.png", "won_board_expected.txt", 9, 9, scale)
        check_scaled("mid_postmortem.png", "mid_postmortem_expected.txt",
                     16, 16, scale)


if __name__ == "__main__":
    test_all()
    test_scaled()
