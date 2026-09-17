"""Paint. Cards, buttons, badges, bars -- the pieces the screen is assembled from.

THE SAME PALETTE AS THE DESKTOP WINDOW, copied rather than imported: app/theme.py belongs
to a Qt program that cannot be imported on a phone, and a phone that drifts to its own
greens would stop looking like the same product. The numbers here came from app/theme.py
and are meant to be kept in step by hand, which is a two-line job on the day the brand
changes and a dependency on torch and PySide6 if it were done any other way.

EVERYTHING DRAWS INTO ONE BGR ARRAY. There is no widget tree, no layout engine and no
retained state: `screen.py` composes the whole 1080x2160 picture every frame and hands it
to Android as a bitmap. That sounds wasteful and is not -- the camera frame underneath it
has to be redrawn anyway, and a phone that composes one image has no invalidation bugs.

TAP TARGETS COME BACK FROM THE DRAWING. Every button returns the rectangle it drew, and
screen.py keeps them in a list to test a tap against. A control cannot be drawn in one
place and hit in another, because there is only one set of coordinates.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:                                             # pragma: no cover
    cv2 = None

# --- palette, BGR because that is what cv2 works in ------------------------------------
BG = (244, 247, 244)[::-1]
SURFACE = (255, 255, 255)
LINE = (225, 231, 223)[::-1]
LINE_STRONG = (207, 218, 211)[::-1]
INK = (23, 33, 28)[::-1]
INK_SOFT = (91, 107, 98)[::-1]
INK_MUTED = (148, 163, 160)[::-1]

GREEN_900 = (1, 94, 19)[::-1]
GREEN_700 = (16, 134, 68)[::-1]
GREEN_500 = (23, 160, 80)[::-1]
GREEN_TINT = (231, 244, 236)[::-1]
WARN = (232, 131, 58)[::-1]
DANGER = (192, 57, 43)[::-1]

BADGE = {
    "ok": ((220, 252, 231)[::-1], (22, 101, 52)[::-1]),
    "over": ((254, 243, 199)[::-1], (146, 64, 14)[::-1]),
    "short": ((254, 226, 226)[::-1], (153, 27, 27)[::-1]),
    "none": ((238, 242, 239)[::-1], (91, 107, 98)[::-1]),
}

#: The marker on a counted pill: the app's own green with a white rim, the same pairing
#: the bench uses -- black read as one more shadow between two pills, and the green ties a
#: marked pill to the number it is counted into. The rim is what keeps it visible on a
#: dark capsule.
MARK = (68, 134, 16)                        # #108644, GREEN_700 above
MARK_RIM = (255, 255, 255)
ROI_LINE = (90, 220, 110)
ROI_BAND = (140, 245, 160)
STALE_EDGE = (60, 60, 220)


def rounded(img, rect, radius, colour, thickness=-1):
    """A rounded rectangle, filled or stroked. cv2 has no such call, so: four and four."""
    x, y, w, h = [int(v) for v in rect]
    r = int(min(radius, w / 2, h / 2))
    if r <= 0:
        cv2.rectangle(img, (x, y), (x + w, y + h), colour, thickness, cv2.LINE_AA)
        return rect
    if thickness < 0:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), colour, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), colour, -1)
        for cx, cy in ((x + r, y + r), (x + w - r, y + r),
                       (x + r, y + h - r), (x + w - r, y + h - r)):
            cv2.circle(img, (cx, cy), r, colour, -1, cv2.LINE_AA)
        return rect
    t = int(thickness)
    cv2.line(img, (x + r, y), (x + w - r, y), colour, t, cv2.LINE_AA)
    cv2.line(img, (x + r, y + h), (x + w - r, y + h), colour, t, cv2.LINE_AA)
    cv2.line(img, (x, y + r), (x, y + h - r), colour, t, cv2.LINE_AA)
    cv2.line(img, (x + w, y + r), (x + w, y + h - r), colour, t, cv2.LINE_AA)
    for cx, cy, a0, a1 in ((x + r, y + r, 180, 270), (x + w - r, y + r, 270, 360),
                           (x + w - r, y + h - r, 0, 90), (x + r, y + h - r, 90, 180)):
        cv2.ellipse(img, (cx, cy), (r, r), 0, a0, a1, colour, t, cv2.LINE_AA)
    return rect


def card(img, rect, radius=28):
    rounded(img, rect, radius, SURFACE, -1)
    rounded(img, rect, radius, LINE, 2)
    return rect


def button(img, text, rect, txt, px, kind="ghost", enabled=True):
    """One tappable thing. Returns its rect, which is also its tap target.

    THE DISABLED LOOK IS NOT A GREY TINT OVER THE ENABLED ONE. A button that is off has to
    read as off from a metre away, so it loses its fill as well as its contrast -- the same
    decision the desktop window makes, for the same operator.
    """
    x, y, w, h = [int(v) for v in rect]
    if kind == "primary":
        fill = GREEN_700 if enabled else LINE
        ink = SURFACE if enabled else INK_MUTED
        rounded(img, rect, 18, fill, -1)
    elif kind == "danger":
        rounded(img, rect, 18, SURFACE, -1)
        rounded(img, rect, 18, DANGER if enabled else LINE, 2)
        ink = DANGER if enabled else INK_MUTED
    elif kind == "chip":
        rounded(img, rect, h // 2, GREEN_700 if enabled == "on" else BG, -1)
        rounded(img, rect, h // 2, GREEN_700 if enabled == "on" else LINE, 2)
        ink = SURFACE if enabled == "on" else INK_SOFT
    else:
        rounded(img, rect, 18, SURFACE, -1)
        rounded(img, rect, 18, LINE_STRONG if enabled else LINE, 2)
        ink = INK if enabled else INK_MUTED
    text.draw(img, txt, x + w // 2, y + (h - text.height(px)) // 2, px, ink, align="centre")
    return rect


def badge(img, text, centre_x, y, txt, px, kind):
    """The verdict, in a tinted pill sized to its words."""
    bg, fg = BADGE.get(kind, BADGE["none"])
    w = text.measure(txt, px) + int(px * 1.6)
    h = text.height(px) + int(px * 0.5)
    rect = (int(centre_x - w / 2), int(y), int(w), int(h))
    rounded(img, rect, h // 2, bg, -1)
    text.draw(img, txt, rect[0] + w // 2, rect[1] + (h - text.height(px)) // 2, px, fg,
              align="centre")
    return rect


def tag(img, text, right_x, y, txt, px=24):
    """A status word, right-aligned, in plain type. Returns the box it filled.

    NO PILL AND NO OUTLINE, which is the same decision app/window.py's QLabel#chip makes
    and for the same reason: this is not a control. It says which part of the picture is
    being counted and what a pass costs, and on a header whose other occupants are real
    buttons, anything wearing a shape is read as another one.

    The rect comes back so the caller can put the next word to the left of it -- these
    are laid out from the right-hand edge inwards, because the one that changes width
    every frame is the one nearest the edge.
    """
    w = text.measure(txt, px)
    rect = (int(right_x - w), int(y), int(w), text.height(px))
    text.draw(img, txt, rect[0], rect[1], px, INK_SOFT)
    return rect


def progress(img, rect, fraction, colour):
    x, y, w, h = [int(v) for v in rect]
    rounded(img, rect, h // 2, LINE, -1)
    filled = int(max(0.0, min(1.0, fraction)) * w)
    if filled > h // 2:
        rounded(img, (x, y, filled, h), h // 2, colour, -1)
    return rect


def fit(frame, box):
    """Scale a camera frame into a box, keeping its shape. (image, x, y, scale)."""
    bx, by, bw, bh = [int(v) for v in box]
    h, w = frame.shape[:2]
    scale = min(bw / w, bh / h)
    new = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                     interpolation=cv2.INTER_AREA)
    x = bx + (bw - new.shape[1]) // 2
    y = by + (bh - new.shape[0]) // 2
    return new, x, y, scale


def paste(img, piece, x, y):
    """Copy a picture in, clipped to whatever of it lands on the canvas."""
    h, w = piece.shape[:2]
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    img[y0:y1, x0:x1] = piece[y0 - y:y1 - y, x0 - x:x1 - x]


def dim_outside(img, rect, strength=0.45):
    """Darken everything but the rectangle -- what makes a selection read as a selection."""
    x, y, w, h = [int(v) for v in rect]
    inside = img[y:y + h, x:x + w].copy()
    img[:] = (img.astype(np.float32) * (1 - strength)).astype(np.uint8)
    if inside.size:
        img[y:y + h, x:x + w] = inside


def corner_ticks(img, rect, colour, length=34, width=6):
    """Elbows at the corners instead of dots, which would sit on top of a pill."""
    x, y, w, h = [int(v) for v in rect]
    for cx, cy, sx, sy in ((x, y, 1, 1), (x + w, y, -1, 1),
                           (x + w, y + h, -1, -1), (x, y + h, 1, -1)):
        cv2.line(img, (cx, cy), (cx + sx * length, cy), colour, width, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + sy * length), colour, width, cv2.LINE_AA)


def hit(rect, x, y) -> bool:
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh
