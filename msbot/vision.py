# -*- coding: utf-8 -*-
"""
Perception layer: locate the minesweeper board inside a full screenshot and
classify every cell from raw pixels.

Works for the classic-rendered board on https://www.minesweeper.cn/:
  - canvas frame : 2 px of rgb(204,204,204) then 2 px of rgb(128,128,128)
                   on the top/left; white on bottom/right
  - cell size    : 25 x 25 px at 100 % zoom; larger pitches (OS display
                   scaling / browser zoom) are auto-detected from the frame
                   geometry and the board crop is resampled back to 25 px
                   cells for classification

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
    cell: float = CELL      # real cell pitch; >25 under OS display scaling

    def center(self, r, c):
        """Absolute pixel coordinates of the centre of cell (r, c)."""
        return (self.x0 + c * self.cell + self.cell / 2,
                self.y0 + r * self.cell + self.cell / 2)

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


# the site's three standard boards: (rows, cols)
SHAPES = ((9, 9), (16, 16), (16, 30))


def _grid_score(a, x0, y0, rows, cols, cell):
    """Evidence that true cell boundaries sit at x0/y0 + k*cell.

    Classic cells always show a bevel / grid line at their edges (raised
    white+grey when closed, sunken dark when open), so the pixel gradient
    peaks on true grid lines and is flat mid-cell.  Used to disambiguate
    border readings, e.g. a 16x16 board rendered at 37.5 px cells (150 %
    display scaling) whose outer frame *also* divides into 24 fake 25 px
    cells — the fake grid has lines landing mid-cell and scores lower.
    """
    a = np.asarray(a)
    y1 = min(int(round(y0 + rows * cell)), a.shape[0])
    x1 = min(int(round(x0 + cols * cell)), a.shape[1])
    if y1 - y0 < 40 or x1 - x0 < 40:
        return -1e9
    reg = a[y0:y1, x0:x1].astype(np.int16)
    gx = np.abs(np.diff(reg, axis=1)).sum(axis=2).mean(axis=0)
    gy = np.abs(np.diff(reg, axis=0)).sum(axis=2).mean(axis=1)

    def profile(prof, n, s):
        on, off = [], []
        for k in range(1, n):
            i, m = int(round(k * s)), int(round(k * s + s / 2))
            if i < len(prof):
                on.append(prof[i])
            if m < len(prof):
                off.append(prof[m])
        return float(np.mean(on) - np.mean(off)) if on and off else 0.0

    return profile(gx, cols, cell) + profile(gy, rows, cell)


def _candidates(width, height):
    """All plausible (rows, cols, cell) interpretations of a border pair."""
    out = []
    for rows, cols in SHAPES:
        cx, cy = width / cols, height / rows
        # sane render scale (67 %..240 % zoom) and square cells
        if 16 <= cx <= 60 and abs(cx - cy) <= 0.05 * cx:
            out.append((rows, cols, (cx + cy) / 2.0))
    # custom sizes rendered at the canonical 25 px
    rf, cf = height / CELL, width / CELL
    if abs(rf - round(rf)) <= 0.1 and abs(cf - round(cf)) <= 0.15 \
            and round(rf) >= 8:
        out.append((round(rf), round(cf), float(CELL)))
    return out


def _rows_with_run(mask, min_len):
    """-> set of row indices that can contain a True-run >= min_len.

    A row can only contain such a run if some length-`min_len` window is
    all-True; that prefilter is one vectorised cumsum over the whole mask,
    so the slow per-pixel `_runs` only executes on rows actually inspected.
    """
    cs = np.cumsum(mask, axis=1, dtype=np.int32)
    full = (cs[:, min_len:] - cs[:, :-min_len]) == min_len
    return set(np.nonzero(full.any(axis=1))[0].tolist())


def locate_board(img):
    """Find the board inside an RGB screenshot (H, W, 3 uint8) -> BoardGeo|None.

    The canvas top border is a long horizontal run of ~rgb(128) grey; the
    bottom border is a matching white run.  The pair must line up in x and
    have the same length.  A frame can correspond to several cell grids
    (the real one, plus fake 25 px grids under display scaling), so every
    plausible interpretation of every border pair is scored against the
    internal grid lines and the global best wins — this also rejects the
    similar-looking header frame around the counters/smiley.
    """
    a = np.asarray(img)
    h, w, _ = a.shape
    mn = a.min(axis=2)
    mx = a.max(axis=2)
    g128 = (mn >= 120) & (mx <= 136)                         # ~rgb(128) grey
    white = (mn >= 240) & ((mx - mn) <= 12)                  # near-white

    grey_cand = _rows_with_run(g128, 220)
    white_cand = _rows_with_run(white, 214)   # 220 - 6 px length tolerance
    white_cache = {}
    best, best_score = None, 1.0            # score floor: reject non-board UI
    scored = {}                             # geometry -> score (dedupe)
    for y in grey_cand:
        y0 = y + 2
        if y0 + 100 >= h:
            continue
        for x_start, length in (r for r in _runs(g128[y]) if r[1] >= 220):
            x0, width = x_start + 2, length - 2
            # matching white bottom frame
            lo = min(y0 + CELL * 8 - 2, h)
            hi = min(y0 + CELL * 42 + 2, h)
            for yy in range(lo, hi):
                if yy not in white_cand:
                    continue
                if yy not in white_cache:
                    white_cache[yy] = [r for r in _runs(white[yy])
                                       if r[1] >= 214]
                for bx, bl in white_cache[yy]:
                    if abs(bx - x_start) > 4 or abs(bl - length) > 6:
                        continue
                    height = yy - y0
                    for rows, cols, cell in _candidates(width, height):
                        key = (x0, y0, rows, cols, round(cell, 1))
                        if key not in scored:
                            scored[key] = _grid_score(a, x0, y0,
                                                      rows, cols, cell)
                        s = scored[key]
                        if s > best_score:
                            best_score = s
                            best = BoardGeo(x0=x0, y0=y0, rows=rows,
                                            cols=cols, cell=cell)
    return best


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
    """Slice the screenshot into cells and classify each -> (rows, cols) int8.

    When the board is rendered at a non-canonical pitch (OS display scaling
    or browser zoom, e.g. 37.5 px cells at 150 %), the board crop is first
    resampled back to 25 px cells so the classification rules — tuned for
    25 px structure — apply unchanged."""
    a = np.asarray(img)
    if abs(geo.cell - CELL) > 0.51:
        from PIL import Image
        w = int(round(geo.cols * geo.cell))
        h = int(round(geo.rows * geo.cell))
        crop = a[geo.y0:geo.y0 + h, geo.x0:geo.x0 + w]
        a = np.array(Image.fromarray(crop).resize(
            (geo.cols * CELL, geo.rows * CELL), Image.BILINEAR))
        ox = oy = 0
    else:
        ox, oy = geo.x0, geo.y0
    board = np.full((geo.rows, geo.cols), UNKNOWN, dtype=np.int8)
    for r in range(geo.rows):
        for c in range(geo.cols):
            y, x = oy + r * CELL, ox + c * CELL
            board[r, c] = classify_cell(a[y:y + CELL, x:x + CELL])
    return board
