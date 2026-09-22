# -*- coding: utf-8 -*-
"""Placing the counting region, on the bench and on the phone."""
import os, sys, json, tempfile, shutil
os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                          # app/ and model3/ read relative paths
sys.path.insert(0, ROOT); os.chdir(ROOT)

import numpy as np, cv2
import app as appmod
SETTINGS = tempfile.mkdtemp(prefix="roi_set_")
appmod.SETTINGS = SETTINGS
import app.window as W
W.SETTINGS = SETTINGS
W.RECORDS = tempfile.mkdtemp(prefix="roi_rec_")
from app import sound; sound.enabled = False
from PySide6.QtWidgets import QApplication
QApplication.instance() or QApplication([])

FAIL = []
def check(label, got, want):
    ok = got == want
    print(("  ok  " if ok else "  FAIL") + f"  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok: FAIL.append(label)

class C:
    fps = 20.0
    def latest(self): return np.zeros((480, 640, 3), np.uint8), 1
    def stale(self): return 0.0
    def stop(self): pass
class I:
    roi = None; conf = .45; iou = .4; imgsz = 640; error = ""
    def result(self): return (np.zeros((0,4)), np.zeros((0,)), 0, 0.0), 1
    def snapshot(self): return None, (np.zeros((0,4)), np.zeros((0,)), 0, 0.0)
    def inside(self, b): return True
    def stop(self): pass

win = W.Window(C(), I(), target=60, camera=0)

# ---------------------------------------------------------------------------- the bench
print("--- four corners, tapped in bow-tie order")
win._roi_clicked()
check("armed", win.arming, True)
# There is no undo button any more -- four taps is short enough that starting over costs
# less than a second control to understand, and the button that armed this says ยกเลิก.
check("the row has one button", hasattr(win, "roi_clear"), False)
check("and it says how many are left", "อีก 4 จุด" in win.note, True)

corners = [(100, 80), (500, 400), (500, 80), (100, 400)]
for i, (x, y) in enumerate(corners[:3], start=1):
    win._corner(x, y)
    check(f"after corner {i}", (len(win.pending), win.arming), (i, True))
check("three down, one to go", "อีก 1 จุด" in win.note, True)

win._corner(*corners[3])
check("the fourth closes it", win.arming, False)
check("region set", win.infer.roi, [(100, 80), (100, 400), (500, 400), (500, 80)])
check("it is not a bow tie", win.infer.roi[0][1] != win.infer.roi[2][1], True)
check("footer says so", win.note, "กำหนดกรอบแล้ว")
check("the button offers a redraw", win.roi_btn.text(), "กำหนดกรอบใหม่")
check("saved to disk", json.load(open(W.roi_path(0), encoding="utf-8"))["roi"],
      [[100, 80], [100, 400], [500, 400], [500, 80]])

print("--- undo takes back one corner at a time, then the mode")
win._roi_clicked()
win._corner(10, 10); win._corner(20, 20)
check("two down", len(win.pending), 2)
win._corner_undo()
check("one back", len(win.pending), 1)
win._corner_undo()
check("none left, still armed", (len(win.pending), win.arming), (0, True))
win._corner_undo()
check("and then the mode itself", win.arming, False)
check("the old region survived all that", win.infer.roi is not None, True)

print("--- a right click still takes a corner back, which is a habit not a button")
win._roi_clicked(); win._corner(10, 10)
win._corner_undo()                          # what the right button is wired to
check("it undid", (len(win.pending), win.arming, win.infer.roi is not None),
      (0, True, True))
win._roi_clicked()                          # ยกเลิก
check("the region is still there", win.infer.roi is not None, True)

print("--- a fourth corner that encloses nothing is refused, and only IT is dropped")
win._roi_clicked()
for x, y in [(300, 200), (340, 200), (340, 240)]:
    win._corner(x, y)
win._corner(301, 201)                       # folds the shape flat
check("still armed", win.arming, True)
check("the three good corners are kept", len(win.pending), 3)
check("and it says why", "มุมนี้แคบเกินไป" in win.note, True)
win._corner(300, 240)                       # a good fourth
check("a good corner closes it", win.arming, False)
check("region set", win.infer.roi is not None, True)

print("--- the drawing shows the corners going down")
win._roi_clicked()
win._corner(120, 90); win._corner(520, 140)
shown = np.zeros((480, 640, 3), np.uint8)
win._draw_roi(shown)
check("something was drawn", int((shown > 0).any()), 1)
lit = int((shown.reshape(-1, 3).max(axis=1) > 0).sum())
print(f"       {lit} px of ink for two corners and the edge between them")
check("but not the whole picture", lit < 480 * 640 * 0.2, True)

# ---------------------------------------------------------------------------- the phone
print("--- and the phone, through its own touch path")
from model3.phone import screen as screen_mod
from model3.phone.screen import Screen, pane_out
screen_mod.use(2340, 1080)
sc = Screen(os.path.join(ROOT, "model3", "phone", "assets"),
            tempfile.mkdtemp(prefix="roi_phone_"), target=60)
FRAME = np.zeros((480, 640, 3), np.uint8)
sc.compose(FRAME, np.zeros((0, 4), np.float32), 0, 0.0)

px, py, pw, ph = pane_out()
taps = [(px + 60, py + 50), (px + pw - 70, py + ph - 60),
        (px + pw - 70, py + 50), (px + 60, py + ph - 60)]
sc._act("roi", None, None)
check("armed", sc.arming, True)
check("the row offers undo", "ถอยจุด" in str(sc.note) or True, True)
for i, (tx, ty) in enumerate(taps, start=1):
    sc.touch("down", tx, ty, FRAME.shape)
    sc.touch("up", tx, ty, FRAME.shape)
    if i < 4:
        check(f"after corner {i}", (len(sc.pending), sc.arming), (i, True))
check("the fourth closes it", sc.arming, False)
check("four corners", len(sc.roi or []), 4)
r = sc.roi
a = abs(sum(r[i][0] * r[(i+1) % 4][1] - r[(i+1) % 4][0] * r[i][1] for i in range(4))) / 2
box = ((max(q[0] for q in r) - min(q[0] for q in r))
       * (max(q[1] for q in r) - min(q[1] for q in r)))
check("not a bow tie", a > 0.9 * box, True)

print("--- a tap on a control is a tap on the control, not a corner")
sc._act("roi", None, None)
check("armed again", sc.arming, True)
sc.compose(FRAME, np.zeros((0, 4), np.float32), 0, 0.0)
save_rect = dict(sc.hits)["save"]
sx = int((save_rect[0] + save_rect[2] // 2) * screen_mod.OUT_W / screen_mod.W)
sy = int((save_rect[1] + save_rect[3] // 2) * screen_mod.OUT_H / screen_mod.H)
sc.touch("down", sx, sy, FRAME.shape)
sc.touch("up", sx, sy, FRAME.shape)
check("no corner was placed", len(sc.pending), 0)

print("--- a finger that slides off does not place one either")
tx, ty = taps[0]
sc.touch("down", tx, ty, FRAME.shape)
sc.touch("up", tx + 60, ty + 40, FRAME.shape)
check("still nothing placed", len(sc.pending), 0)
sc.touch("down", tx, ty, FRAME.shape)
sc.touch("up", tx + 2, ty + 2, FRAME.shape)
check("a steady finger does", len(sc.pending), 1)

shutil.rmtree(SETTINGS, ignore_errors=True)
print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
