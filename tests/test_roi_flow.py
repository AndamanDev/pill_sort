# -*- coding: utf-8 -*-
"""Setting the counting region, on the bench and on the phone.

A new region is a rectangle (default_quad), and each corner is then dragged onto the
tray's own. A drop that leaves no usable shape puts back only that drag.
"""
import os, sys, json, tempfile, shutil
os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                          # app/ and model3/ read relative paths

import numpy as np, cv2
import app as appmod
SETTINGS = tempfile.mkdtemp(prefix="roi_set_")
appmod.SETTINGS = SETTINGS
import app.window as W
W.SETTINGS = SETTINGS
W.RECORDS = tempfile.mkdtemp(prefix="roi_rec_")
from app import sound; sound.enabled = False
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication
QApplication.instance() or QApplication([])

FAIL = []
def check(label, got, want):
    ok = got == want
    print(("  ok  " if ok else "  FAIL") + f"  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok: FAIL.append(label)

FRAME = np.zeros((480, 640, 3), np.uint8)
class C:
    fps = 20.0
    def latest(self): return FRAME, 1
    def stale(self): return 0.0
    def stop(self): pass
class I:
    roi = None; conf = .45; iou = .4; imgsz = 640; error = ""
    def result(self): return (np.zeros((0,4)), np.zeros((0,)), 0, 0.0), 1
    def snapshot(self): return None, (np.zeros((0,4)), np.zeros((0,)), 0, 0.0)
    def inside(self, b): return True
    def stop(self): pass

win = W.Window(C(), I(), target=60, camera=0)
RECT = [(64, 48), (64, 431), (575, 431), (575, 48)]    # 10% in from each edge of 640x480

# ---------------------------------------------------------------------------- the bench
print("--- กำหนดกรอบนับ starts from a rectangle")
check("no region yet", win.infer.roi, None)
win._roi_clicked()
check("it opened the settings", win.setup, True)
check("the default rectangle", win.infer.roi, RECT)
check("the corners are handed to the picture", win.view.handles, RECT)
check("the footer says to drag them", win.note, W.ROI_DRAG_NOTE)
check("the button offers a redraw", win.roi_btn.text(), "กำหนดกรอบใหม่")

print("--- one corner dragged, the others stay")
win._corner_moved(0, 100, 80)
win._corner_moved(0, 110, 90)
check("it follows", win.infer.roi[0], (110, 90))
win._corner_dropped()
check("dropped", sorted(win.infer.roi), [(64, 431), (110, 90), (575, 48), (575, 431)])

print("--- dragged across another corner is not a bow tie")
i = win.infer.roi.index((575, 48))          # top right, dragged to the far left
win._corner_moved(i, 20, 300)
win._corner_dropped()
r = win.infer.roi
a = abs(sum(r[k][0] * r[(k+1) % 4][1] - r[(k+1) % 4][0] * r[k][1] for k in range(4))) / 2
check("still four corners", len(r), 4)
check("and a shape with area", a > 1000, True)

print("--- dragged past the edge of the picture, it stops at the edge")
win._roi_clicked()                          # กำหนดกรอบใหม่: back to the rectangle
check("the redraw is the rectangle again", win.infer.roi, RECT)
win._corner_moved(0, -50, -40)
win._corner_dropped()
check("clamped", (0, 0) in win.infer.roi, True)

print("--- a drop that squeezes the shape flat puts back only that drag")
win._roi_clicked()
before = list(win.infer.roi)
i = win.infer.roi.index((575, 431))
win._corner_moved(i, 64, 432)               # onto the line between two others
win._corner_moved(i, 65, 431)
win._corner_dropped()
check("still a region", win.infer.roi is not None, True)
check("not on disk until the settings are saved", os.path.isfile(W.roi_path(0)), False)

print("--- and a sliver is refused")
win.infer.roi = [(100, 100), (100, 120), (400, 120), (400, 100)]
win._corner_moved(1, 100, 110); win._corner_dropped()
check("put back", win.infer.roi, [(100, 100), (100, 120), (400, 120), (400, 100)])
check("and it says why", "มุมนี้แคบเกินไป" in win.note, True)
win.infer.roi = list(RECT)
win._setup_save()
check("บันทึก closes the settings", win.setup, False)
check("saved to disk", json.load(open(W.roi_path(0), encoding="utf-8"))["roi"],
      [list(p) for p in RECT])
check("the handles are gone with them", win.view.handles, None)

print("--- outside the settings, a drag does nothing")
win._corner_moved(0, 300, 300); win._corner_dropped()
check("the region did not move", win.infer.roi, RECT)

print("--- the mouse, through the picture itself")
win._setup_open()
v = win.view
v.resize(640, 480)
v.show_frame(FRAME)                         # scale 1, no letterbox
def mouse(kind, x, y, held=True):
    btn = Qt.LeftButton
    ev = QMouseEvent(kind, QPointF(x, y), QPointF(x, y), btn,
                     btn if held else Qt.NoButton, Qt.NoModifier)
    {QEvent.MouseButtonPress: v.mousePressEvent, QEvent.MouseMove: v.mouseMoveEvent,
     QEvent.MouseButtonRelease: v.mouseReleaseEvent}[kind](ev)
mouse(QEvent.MouseButtonPress, 300, 300)    # nowhere near a corner
mouse(QEvent.MouseMove, 320, 320)
mouse(QEvent.MouseButtonRelease, 320, 320, held=False)
check("a press away from the corners moves nothing", win.infer.roi, RECT)
mouse(QEvent.MouseButtonPress, 70, 52)      # within GRAB_PX of (64, 48)
mouse(QEvent.MouseMove, 30, 20)
mouse(QEvent.MouseButtonRelease, 30, 20, held=False)
check("a press on a corner drags it", (30, 20) in win.infer.roi, True)
check("and only it", sum(p in win.infer.roi for p in RECT), 3)

print("--- the drawing shows a ring on every corner while they may be moved")
shown = np.zeros((480, 640, 3), np.uint8)
win._draw_roi(shown)
r = max(9, round(480 / 36))
check("a ring at every corner", [tuple(shown[y, min(x + r, 639)]) == W.ROI_BAND
                                  for x, y in win.infer.roi], [True] * 4)
win._setup_cancel()
check("ยกเลิก puts the saved region back", win.infer.roi, RECT)

# ---------------------------------------------------------------------------- the phone
print("--- and the phone, through its own touch path")
from model3.phone import screen as screen_mod
from model3.phone.screen import Screen, pane_out
screen_mod.use(2340, 1080)
sc = Screen(os.path.join(ROOT, "model3", "phone", "assets"),
            tempfile.mkdtemp(prefix="roi_phone_"), target=60)
sc.compose(FRAME, np.zeros((0, 4), np.float32), 0, 0.0)

def out(cx, cy):
    """Design canvas -> the output coordinates Android sends touches in."""
    return int(cx * screen_mod.OUT_W / screen_mod.W), int(cy * screen_mod.OUT_H / screen_mod.H)

sc._act("roi", None, None)
check("into the settings", sc.setup, True)
check("the default rectangle", sc.roi, RECT)
sc.compose(FRAME, np.zeros((0, 4), np.float32), 0, 0.0)

tl = sc._to_canvas(*RECT[0], FRAME.shape)
dx, dy = out(*tl)
sc.touch("down", dx + 4, dy + 4, FRAME.shape)
check("a finger on a corner picks it up", sc.grab, 0)
sc.touch("long", dx + 4, dy + 4, FRAME.shape)
check("holding still does not let go", sc.grab, 0)
px, py, pw, ph = pane_out()
sc.touch("move", px + 20, py + 20, FRAME.shape)
check("it follows the finger", sc.roi[0] != RECT[0], True)
sc.touch("up", px + 20, py + 20, FRAME.shape)
check("let go", sc.grab, None)
check("still four corners", len(sc.roi), 4)
r = sc.roi
a = abs(sum(r[i][0] * r[(i+1) % 4][1] - r[(i+1) % 4][0] * r[i][1] for i in range(4))) / 2
box = ((max(q[0] for q in r) - min(q[0] for q in r))
       * (max(q[1] for q in r) - min(q[1] for q in r)))
check("not a bow tie", a > 0.6 * box, True)

print("--- a finger on the picture away from the corners moves nothing")
was = list(sc.roi)
cx, cy = out(*sc._to_canvas(320, 240, FRAME.shape))
sc.touch("down", cx, cy, FRAME.shape)
check("nothing picked up", sc.grab, None)
sc.touch("move", cx + 40, cy + 40, FRAME.shape)
sc.touch("up", cx + 40, cy + 40, FRAME.shape)
check("the region did not move", sc.roi, was)

print("--- outside the settings the corners cannot be moved")
sc._act("setup-save", None, None)
check("saved and closed", sc.setup, False)
x0, y0 = out(*sc._to_canvas(*sc.roi[0], FRAME.shape))
sc.touch("down", x0, y0, FRAME.shape)
check("nothing picked up", sc.grab, None)
sc.touch("up", x0, y0, FRAME.shape)

print("--- and the new words have pictures")
def missing(t):
    return [p for p, d in sc.text._runs(t) if sc.text._entry(p, d) is None]
for t in ("ลากมุมกรอบให้ตรงมุมถาด แล้วกดบันทึก", "กำหนดกรอบแล้ว ลากมุมบนภาพเพื่อปรับ",
          "มุมนี้แคบเกินไป ลากให้ห่างจากมุมอื่น"):
    check(t, missing(t), [])

shutil.rmtree(SETTINGS, ignore_errors=True)
print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
