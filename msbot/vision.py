# -*- coding: utf-8 -*-
"""
Perception layer: locate the minesweeper board inside a full screenshot and
classify every cell from raw pixels.

Works for the classic-rendered board on https://www.minesweeper.cn/:
  - canvas frame : 2 px of rgb(204,204,204) then 2 px of rgb(128,128,128)
                   on the top/left; white on bottom/right
  - cell size    : 25 x 25 px (browser zoom must stay at 100 %)

The canvas sits at a fractional CSS offset, so the compositor renders some
frames crisp (exact palette) and some slightly resampled (blended greys).
All detection therefore works on *structural* features and hue rules with
tolerances instead of exact colours.
"""
from dataclasses import dataclass

import numpy as np

CELL = 25                      # cell edge in pixels at 100 % zoom

# ---- cell state codes ------------------------------------------------------
UNKNOWN = -1                   # closed cell
EMPTY   =  0                   # opened, no neighbouring mine
FLAG    = -2                   # closed + red flag
BLAST   = -3                   # the mine that exploded (red background)
MINE    = -4                   # revealed mine after game over (black glyph)


@dataclass
class BoardGeo:
    """Pixel geometry of the board inside a screenshot."""
    x0: int          # left edge of cell (0,0)
    y0: int          # top edge of cell (0,0)
    rows: int
    cols: int
    cell: int = CELL

    def center(self, r, c):
        """Absolute pixel coordinates of the centre of cell (r, c)."""
        return (self.x0 + c * self.cell + self.cell // 2,
                self.y0 + r * self.cell + self.cell // 2)

    def restart_point(self):
        """The smiley button sits centred ~25 px above the board."""
        return (self.x0 + self.cols * self.cell // 2, self.y0 - 25)


# ---------------------------------------------------------------------------
# board localisation
# ---------------------------------------------------------------------------
def _runs(mask_row):
    """(start, length) of every True-run in a 1-D boolean array."""
    out, start = [], None
    for i, v in enumerate(mask_row):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - start)); start = None
    if start is not None:
        out.append((start, len(mask_row) - start))
    return out


def locate_board(img):
    """Find the board inside an RGB screenshot (H, W, 3 uint8) -> BoardGeo|None.

    The canvas top border is a long horizontal run of ~rgb(128) grey; the
    bottom border is a matching white run.  The pair must line up in x,
    have the same length and span an integral number of 25 px cells — this
    rejects the similar-looking header frame around the counters/smiley.
    """
    a = np.asarray(img).astype(np.int16)
    h, w, _ = a.shape
    grey = a.max(axis=2) - a.min(axis=2)                # channel spread
    g128 = (np.abs(a - 128) <= 8).all(axis=2)           # ~rgb(128) grey
    white = (a >= 240).all(axis=2) & (grey <= 12)       # near-white

    for y in range(h):
        for x_start, length in _runs(g128[y]):
            if length < 220:
                continue
            x0, y0 = x_start + 2, y + 2
            width = length - 2
            cols = width / CELL
            if abs(cols - round(cols)) > 0.15 or y0 + 100 >= h:
                continue
            cols = round(cols)
            # matching white bottom frame
            for yy in range(y0 + CELL * 8 - 2, min(y0 + CELL * 42 + 2, h)):
                for bx, bl in _runs(white[yy]):
                    if abs(bx - x_start) <= 4 and abs(bl - length) <= 6:
                        rows_f = (yy - y0) / CELL
                        if abs(rows_f - round(rows_f)) <= 0.1:
                            rows = round(rows_f)
                            if rows >= 8:
                                return BoardGeo(x0=x0, y0=y0,
                                                rows=rows, cols=cols)
    return None


# ---------------------------------------------------------------------------
# cell classification
# ---------------------------------------------------------------------------
def classify_cell(cell):
    """Classify one 25x25 RGB cell image -> state code / number."""
    a = np.asarray(cell, dtype=np.int16)

    # structural closed/open test: closed cells have a bright white raised
    # top edge (always renders at 255); opened cells have a dark sunken
    # border and measure <=169 in every observed render variant.
    top_strip = a[0:3, 3:22].mean()
    closed = top_strip > 200

    inner = a[3:22, 3:22].reshape(-1, 3)
    r, g, b = inner[:, 0], inner[:, 1], inner[:, 2]
    red    = int(((r > 150) & (r - g > 100) & (r - b > 100)).sum())
    dark   = int((inner.max(axis=1) < 80).sum())
    if closed:
        return FLAG if red > 10 else UNKNOWN

    # exploded mine: red *background* -> the inner corners are red;
    # a number-3 glyph never reaches the corners
    corners = (inner[0], inner[18], inner[342], inner[360])
    for cr, cg, cb in corners:
        if cr > 200 and cg < 80 and cb < 80:
            return BLAST
    if dark > 170:                           # revealed mine glyph (~190 px)
        return MINE

    blue   = int(((b > 170) & (b - r > 100) & (b - g > 100)).sum())
    green  = int(((g > 80) & (g - r > 50) & (g - b > 50)).sum())
    navy   = int(((b > 60) & (b < 170) & (b - r > 40) & (b - g > 40)).sum())
    maroon = int(((r > 80) & (r < 170) & (r - g > 60) & (r - b > 60)).sum())
    teal   = int(((g > 80) & (b > 80) & (r < 90)
                  & (np.abs(g - b) < 45)).sum())
    if blue > 8:   return 1
    if green > 8:  return 2
    if red > 8:    return 3
    if navy > 8:   return 4
    if maroon > 8: return 5
    if teal > 8:   return 6
    if dark > 8:   return 7
    greyish = int(((inner.max(axis=1) - inner.min(axis=1) < 14)
                   & (inner[:, 0] > 90) & (inner[:, 0] < 175)).sum())
    if greyish > 15:
        return 8
    return EMPTY


def read_board(img, geo: BoardGeo):
    """Slice the screenshot into cells and classify each -> (rows, cols) int8."""
    a = np.asarray(img)
    board = np.full((geo.rows, geo.cols), UNKNOWN, dtype=np.int8)
    for r in range(geo.rows):
        for c in range(geo.cols):
            y, x = geo.y0 + r * geo.cell, geo.x0 + c * geo.cell
            board[r, c] = classify_cell(a[y:y + geo.cell, x:x + geo.cell])
    return board
