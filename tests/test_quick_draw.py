# -*- coding: utf-8 -*-
"""The phone's quick path draws the SAME picture as the long way round, pixel for pixel.

compose() scales only the parts of the screen that change on an ordinary frame, and takes
the pane's answer -- black, transparent -- without computing it. Both are only safe while
every live method stays inside its patch of the screen; this holds them to it, on several
screen shapes and in every state the quick path is allowed to draw.
"""
import os, sys, tempfile, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

FAIL = []
def check(label, got, want):
    ok = got == want
    print(("  ok  " if ok else "  FAIL") + f"  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok: FAIL.append(label)

from model3.phone import screen as sm

FR = np.full((480, 640, 3), 90, np.uint8)
BOXES = np.array([[100 + 30 * i, 200, 124 + 30 * i, 224] for i in range(12)], np.float32)
ROI = [(80, 60), (80, 420), (560, 420), (560, 60)]


def both(sc, count, ms=281.0):
    """The quick picture and alpha, then the long one, from the same state.

    The figure is held steady first: the round button comes alive half a second after it
    stops moving, and a pair of composes straddling that moment differ for a reason that
    has nothing to do with the path.
    """
    sc._steady_n, sc._steady_at = count, time.time() - 10
    sc._chrome_key = None
    sc.quick_draw = True
    quick = sc.compose(FR, BOXES, count, ms)
    q_alpha = None if sc.hole_alpha is None else sc.hole_alpha.copy()
    took_quick = sc._quick(FR, 0.0)
    sc._chrome_key = None
    sc.quick_draw = False
    slow = sc.compose(FR, BOXES, count, ms)
    s_alpha = None if sc.hole_alpha is None else sc.hole_alpha.copy()
    sc.quick_draw = True
    return took_quick, quick, slow, q_alpha, s_alpha


def same(label, sc, count):
    took, q, s, qa, sa = both(sc, count)
    check(f"{label}: took the quick path", took, True)
    diff = int(np.abs(q.astype(int) - s.astype(int)).max())
    where = np.argwhere(np.abs(q.astype(int) - s.astype(int)).max(axis=2) > 0)
    check(f"{label}: same pixels", diff, 0)
    if diff:
        print("        first differences at", where[:5].tolist())
    check(f"{label}: same transparency", bool(np.array_equal(qa, sa)), True)


for view in ((1280, 800), (2340, 1080), (1920, 1080)):
    sm.use(*view)
    print(f"--- {view[0]}x{view[1]}: canvas {sm.W}x{sm.H} -> {sm.OUT_W}x{sm.OUT_H}, "
          f"pane {sm.PANE}")
    check("the canvas is on the grid", (sm.W % 3, sm.OUT_W * 3, sm.OUT_H * 3),
          (0, sm.W * 2, sm.H * 2))
    check("so is the pane", tuple(v % 3 for v in sm.PANE), (0, 0, 0, 0))
    check("still 4:3", sm.PANE[2] * 3, sm.PANE[3] * 4)
    sc = sm.Screen(os.path.join(ROOT, "model3", "phone", "assets"), tempfile.mkdtemp(),
                   target=60)
    sc.compose(FR, BOXES, 12, 281.0)
    same("no region yet", sc, 12)
    sc.roi = list(ROI)
    same("counting", sc, 12)
    sc.take_round(FR)
    sc.toast = None                     # the pour's toast goes the long way; see below
    same("one pour banked, clearing", sc, 12)
    sc.clearing = False
    same("two pours", sc, 30)
    sc.target = 0
    same("no target", sc, 7)
    sc.target = 60
    sc._act("setup", None, None)
    same("in the camera settings", sc, 7)
    sc.roi = [(64, 48), (64, 431), (575, 431), (575, 48)]
    sc.grab = 0
    same("a corner being dragged", sc, 7)
    sc.grab = None
    sc._act("setup-cancel", None, None)
    sc.say("กำหนดกรอบแล้ว")
    same("a note in the footer", sc, 7)

    print("--- and the frames it must NOT take")
    sc.typing = ""
    check("the pad is up: long way", sc._quick(FR, 0.0), False)
    sc.typing = None
    sc.toast = ("บันทึกแล้ว", "", time.time())
    check("a toast: long way", sc._quick(FR, 0.0), False)
    sc.toast = None
    check("a dead camera: long way", sc._quick(FR, 9.0), False)
    check("no frame yet: long way", sc._quick(None, 0.0), False)
    sc.page = "records"
    check("the records page: long way", sc._quick(FR, 0.0), False)
    sc.page = "count"

print()
print("FAILURES:", ", ".join(FAIL) if FAIL else "none")
sys.exit(1 if FAIL else 0)
