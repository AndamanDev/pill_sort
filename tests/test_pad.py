# -*- coding: utf-8 -*-
"""The number pad: modal while it is up, closed only on purpose."""
import os, sys, tempfile, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                          # app/ and model3/ read relative paths
import numpy as np, cv2
from model3.phone import screen as screen_mod
from model3.phone.screen import Screen

screen_mod.use(2340, 1080)
W, H, OUT_W, OUT_H = screen_mod.W, screen_mod.H, screen_mod.OUT_W, screen_mod.OUT_H
REC = tempfile.mkdtemp(prefix="pad_")
sc = Screen(os.path.join(ROOT, "model3", "phone", "assets"), REC, target=60)
FRAME = np.full((480, 640, 3), 40, np.uint8)
B = np.zeros((0, 4), np.float32)

FAIL = []
def check(label, got, want):
    ok = got == want
    print(("  ok  " if ok else "  FAIL") + f"  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok: FAIL.append(label)

def paint():
    sc._chrome_key = None
    return sc.compose(FRAME, B, 0, 8.0)

def rect(name):
    paint()
    return dict(sc.hits).get(name)

def tap(canvas_xy):
    """A tap in the coordinates Android sends -- the SCALED screen, as touch() expects."""
    x, y = canvas_xy
    ox, oy = int(x * OUT_W / W), int(y * OUT_H / H)
    sc.touch("down", ox, oy, FRAME.shape)
    return sc.touch("up", ox, oy, FRAME.shape)

def centre(r):
    return (r[0] + r[2] // 2, r[1] + r[3] // 2)

print("--- opening it")
paint()
sc._act("type-target", None, None)
check("the pad is up", sc.typing, "")
paint()
names = dict(sc.hits)
check("it has digits", "key-7" in names, True)
# The button reads ตกลง and is registered as key-ok: the Thai is the drawing,
# the ascii is the name _typed() switches on.
check("it has ok", "key-ok" in names, True)
check("and it has a cross", "key-close" in names, True)

cross = names["key-close"]
pad_w, pad_h = 560, 700
pad = ((W - pad_w) // 2, (H - pad_h) // 2, pad_w, pad_h)
print(f"       cross at {cross}, pad at {pad}")
check("the cross is inside the pad",
      pad[0] <= cross[0] and cross[0] + cross[2] <= pad[0] + pad[2]
      and pad[1] <= cross[1] and cross[1] + cross[3] <= pad[1] + pad[3], True)
check("in its top-right corner",
      cross[0] > pad[0] + pad[2] * 0.7 and cross[1] < pad[1] + pad[3] * 0.2, True)
# 72 design px on a 2340-wide canvas shown at 1560: about 48 real px, Android's own floor.
check("big enough for a thumb", cross[2] >= 72 and cross[3] >= 72, True)

print("--- typing works")
for d in ("1", "2", "0"):
    tap(centre(rect(f"key-{d}")))
check("typed", sc.typing, "120")

print("--- a tap OUTSIDE the pad does nothing at all")
for label, point in (("the target row behind it", (120, 545)),
                     ("a preset chip", (120, 636)),
                     ("the save button", (400, 924)),
                     ("the camera picture", (1570, 574)),
                     ("the records button", (W - 600, 44)),
                     ("empty background", (30, 1050))):
    before = (sc.typing, sc.target, sc.page, sc.arming)
    tap(point)
    after = (sc.typing, sc.target, sc.page, sc.arming)
    check(f"{label} changed nothing", after, before)

print("--- the digits still work after all that")
tap(centre(rect("key-ลบ")))
check("backspace", sc.typing, "12")
tap(centre(rect("key-ล้าง")))
check("clear", sc.typing, "")
for d in ("9", "0"):
    tap(centre(rect(f"key-{d}")))
check("typed again", sc.typing, "90")

print("--- the cross closes it and leaves the target alone")
was = sc.target
tap(centre(rect("key-close")))
check("closed", sc.typing, None)
check("target untouched", sc.target, was)

print("--- and OK closes it by setting the target")
sc._act("type-target", None, None)
for d in ("7", "5"):
    tap(centre(rect(f"key-{d}")))
tap(centre(rect("key-ok")))
check("closed", sc.typing, None)
check("target set", sc.target, 75)

print("--- the screen behind works again once it is shut")
# The old + stepped the target by one; it opens the pad now, which is what proves the
# screen behind is live again.
check("the pad is shut", sc.typing, None)
tap(centre(rect("target+")))
check("ป้อนจำนวน reaches the screen again and opens the pad", sc.typing, "")
tap(centre(rect("key-close")))

print("--- no corner of the counting region lands behind a pad")
sc._act("roi", None, None)
check("armed", sc.arming, True)
sc._act("type-target", None, None)
check("pad up over it", sc.typing, "")
px, py, pw, ph = screen_mod.PANE
tap((px + 40, py + ph - 40))            # on the video, clear of the pad
check("no corner was placed", len(sc.pending), 0)
tap(centre(rect("key-close")))
tap((px + 40, py + ph - 40))
check("and it places one again once the pad is gone", len(sc.pending), 1)

print("--- the calendar is modal too, and keeps its own close")
sc.page = "records"
paint()
sc._act("pick-from", None, None)
check("calendar up", sc.picking, "from")
before = (sc.picking, sc.page)
tap((60, 1040))
check("a tap beside it changes nothing", (sc.picking, sc.page), before)
paint()
r = dict(sc.hits).get("pick-close")
check("it has a close", r is not None, True)
tap(centre(r))
check("which closes it", sc.picking, "")

shutil.rmtree(REC, ignore_errors=True)
print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
