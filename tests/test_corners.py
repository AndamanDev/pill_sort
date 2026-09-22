# -*- coding: utf-8 -*-
"""The corner geometry, and the two copies of it agreeing.

app/window.py and model3/phone/screen.py each carry quad(). Neither can import the other
-- the bench must not depend on the phone's package, and the phone cannot have Qt near it
-- so this is what keeps them the same function rather than a comment asking nicely.
"""
import os, sys, random, itertools
os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                          # app/ and model3/ read relative paths
sys.path.insert(0, ROOT); os.chdir(ROOT)

import numpy as np, cv2
from PySide6.QtWidgets import QApplication
QApplication.instance() or QApplication([])

from app.window import quad as bench_quad, MIN_ROI, ROI_POINTS
from model3.phone import screen as screen_mod
from model3.phone.screen import quad as phone_quad

FAIL = []
def check(label, got, want):
    ok = got == want
    print(("  ok  " if ok else "  FAIL") + f"  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok: FAIL.append(label)

SIZE = (640, 480)

def area(pts):
    return abs(sum(pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1]
                   for i in range(4))) / 2.0

print("--- the two copies are one function")
rng = random.Random(7)
cases = [
    [(100, 80), (500, 80), (500, 400), (100, 400)],          # tapped round the rim
    [(100, 80), (500, 400), (500, 80), (100, 400)],          # tapped as a bow tie
    [(500, 400), (100, 80), (100, 400), (500, 80)],          # tapped at random
    [(120, 90), (520, 140), (480, 420), (90, 360)],          # a tray seen at an angle
    [(0, 0), (639, 0), (639, 479), (0, 479)],                # the whole picture
    [(-50, -50), (900, -20), (900, 700), (-80, 700)],        # dragged off every edge
    [(300, 200), (300, 200), (300, 200), (300, 200)],        # four taps on one spot
    [(10, 200), (600, 202), (610, 205), (20, 203)],          # a sliver: wide and flat
    [(100, 100), (150, 100), (150, 150), (100, 150)],        # exactly at the floor
    [(300, 200), (301, 200), (301, 201), (300, 201)],        # a speck
]
for _ in range(300):
    cases.append([(rng.randint(-100, 740), rng.randint(-100, 580)) for _ in range(4)])

differ = [c for c in cases if bench_quad(c, SIZE) != phone_quad(c, SIZE)]
check("every case agrees", differ, [])

print("--- any tap order gives the same shape")
rim = [(100, 80), (500, 80), (500, 400), (100, 400)]
shapes = {tuple(bench_quad(list(order), SIZE)) for order in itertools.permutations(rim)}
check("all 24 orders -> one shape", len(shapes), 1)
check("and it is the rim", sorted(shapes.pop()), sorted(rim))

print("--- the shape never crosses itself")

def side(a, b, c):
    """Which side of ab does c fall on. Zero means the three are in a line."""
    v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (v > 0) - (v < 0)

def cross(p1, p2, p3, p4):
    """Do the open segments p1p2 and p3p4 meet. Touching at a shared end does not count."""
    d1, d2 = side(p3, p4, p1), side(p3, p4, p2)
    d3, d4 = side(p1, p2, p3), side(p1, p2, p4)
    return d1 * d2 < 0 and d3 * d4 < 0

# A quadrilateral crosses itself exactly when one pair of OPPOSITE edges meets; adjacent
# edges always share a corner. Two pairs, two tests, and no room for an opinion.
crossed = []
for c in cases:
    pts = bench_quad(c, SIZE)
    if pts is None:
        continue
    if (cross(pts[0], pts[1], pts[2], pts[3])
            or cross(pts[1], pts[2], pts[3], pts[0])):
        crossed.append((c, pts))
check("no bow ties survive", crossed, [])
print(f"       checked {len(cases)} tap sets, "
      f"{sum(1 for c in cases if bench_quad(c, SIZE))} of them usable")

print("--- the same four corners come back the same way round, whatever the tap order")
unstable = []
for c in cases[:40]:
    shapes = {tuple(bench_quad(list(o), SIZE) or ()) for o in itertools.permutations(c)}
    if len(shapes) != 1:
        unstable.append((c, shapes))
check("one answer per corner set", unstable, [])

print("--- what is refused, and why")
check("four taps on one spot", bench_quad([(300, 200)] * 4, SIZE), None)
check("a speck", bench_quad([(300, 200), (301, 200), (301, 201), (300, 201)], SIZE), None)
# 1750 square pixels -- comfortably over a 40x40 floor, and three pixels deep. This is
# the case an area test alone lets through, and the reason there is a second test.
check("a flat sliver", bench_quad([(10, 200), (600, 202), (610, 205), (20, 203)], SIZE), None)
check("a thin diagonal kite",
      bench_quad([(100, 100), (500, 400), (300, 250), (301, 249)], SIZE), None)
check("three taps", bench_quad([(100, 80), (500, 80), (500, 400)], SIZE), None)
check("five taps", bench_quad([(100, 80), (500, 80), (500, 400), (100, 400), (1, 1)], SIZE), None)
check("no picture yet", bench_quad(rim, (0, 0)), None)
ok_floor = bench_quad([(100, 100), (100 + MIN_ROI, 100),
                       (100 + MIN_ROI, 100 + MIN_ROI), (100, 100 + MIN_ROI)], SIZE)
check("exactly the floor is allowed", ok_floor is not None, True)

print("--- everything comes back inside the picture")
out = []
for c in cases:
    pts = bench_quad(c, SIZE)
    if pts and not all(0 <= x < SIZE[0] and 0 <= y < SIZE[1] for x, y in pts):
        out.append((c, pts))
check("no corner outside the frame", out, [])

print("--- and cv2 agrees the shape holds what it looks like it holds")
pts = bench_quad([(120, 90), (520, 140), (480, 420), (90, 360)], SIZE)
poly = np.array(pts, np.int32)
inside = cv2.pointPolygonTest(poly, (300.0, 250.0), False)
outside = cv2.pointPolygonTest(poly, (10.0, 10.0), False)
check("a pill in the middle counts", inside >= 0, True)
check("a pill on the bench does not", outside < 0, True)

print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
