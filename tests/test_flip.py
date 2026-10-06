# -*- coding: utf-8 -*-
"""Showing the picture the right way round, without moving anything that counts."""
import os, sys, json, glob, tempfile, shutil
os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                          # app/ and model3/ read relative paths
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, cv2

import app as appmod
SETTINGS = tempfile.mkdtemp(prefix="flip_set_")
appmod.SETTINGS = SETTINGS
import app.window as W
W.SETTINGS = SETTINGS
W.RECORDS = tempfile.mkdtemp(prefix="flip_rec_")
from app import sound; sound.enabled = False
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication
QApplication.instance() or QApplication([])

FAIL = []
def check(label, got, want):
    ok = got == want
    print(("  ok  " if ok else "  FAIL") + f"  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok: FAIL.append(label)

# A frame with a bright mark on the LEFT, so a mirror is visible rather than argued about.
FRAME = np.zeros((480, 640, 3), np.uint8)
FRAME[200:280, 40:120] = 255

print("--- the flip is OFF until somebody turns it on")
# It was briefly on, on the strength of a bench report that the picture looked mirrored.
# That turned out to be the marks being computed with a stretch where the surface
# centre-crops -- right in the middle of the tray, wrong at the rim. CameraX's back
# camera mirrors nothing, so off is what the hardware does.
check("a camera nobody has set up", W.load_flip(0), False)
W.save_flip(0, False)
check("an off is remembered", W.load_flip(0), False)
W.save_flip(0, True)
check("so is an on", W.load_flip(0), True)

print("--- the camera thread does not know the setting exists")
from app.worker import Capture
check("Capture has no flip at all", hasattr(Capture(0), "flip"), False)

class C:
    fps = 20.0
    def latest(self): return FRAME, 1
    def stale(self): return 0.0
    def stop(self): pass
class I:
    roi = None; conf = .45; iou = .4; imgsz = 640; error = ""
    boxes = np.array([[40.0, 200.0, 120.0, 280.0]], np.float32)   # on the bright mark
    def result(self): return (self.boxes, np.ones((1,), np.float32), 1, 8.0), 1
    def snapshot(self): return FRAME, (self.boxes, np.ones((1,), np.float32), 1, 8.0)
    def inside(self, b): return True
    def stop(self): pass

W.save_flip(0, False)
win = W.Window(C(), I(), target=60, camera=0)
check("the window picked the setting up", win.flip, False)
check("and told the view", win.view.flip, False)

print("--- the picture is mirrored, and only the picture")
shots = {}
real_show = win.view.show_frame
def grab(bgr):
    shots["last"] = bgr.copy()
    return real_show(bgr)
win.view.show_frame = grab

win._tick()
plain = shots["last"]
# A corner of the bright block, NOT its centre: the model's dot is painted over the
# centre in the app's green, so sampling there measures the dot rather than the mark.
check("unflipped: the mark is on the left", int(plain[210, 45].max()) > 200, True)
check("and nothing is on the right", int(plain[210, 594].max()), 0)

win._flip_clicked()
win._tick()
mirrored = shots["last"]
check("flipped: the picture is exactly the mirror of it",
      int(np.abs(cv2.flip(plain, 1).astype(int) - mirrored.astype(int)).max()), 0)
check("so the mark moved to the right", int(mirrored[210, 594].max()) > 200, True)
check("and the left is empty now", int(mirrored[210, 45].max()), 0)
# The dot is the app's green (68, 134, 16) rather than the mark's white, so finding that
# exact colour at the mirror of the box centre is what proves the MARK turned with the
# picture -- which it must, because it was drawn before the flip rather than after it.
check("the model's dot turned with the picture",
      tuple(int(v) for v in mirrored[240, 640 - 1 - 80]), (68, 134, 16))

print("--- and a tap comes back through the same mirror")
win.view._geom = (0.0, 0.0, 1.0)            # a 1:1 picture at the widget's origin
win.view._fw = 640
win.view.arming = True
check("the view is flipped", win.view.flip, True)
check("flipped, a tap at 100 is 539", win.view._to_frame(QPointF(100, 50))[0], 539.0)
win.view.set_flip(False)
check("unflipped, it is 100", win.view._to_frame(QPointF(100, 50))[0], 100.0)
win.view.set_flip(True)
check("the mirror is its own inverse",
      win.view._to_frame(QPointF(640 - 1 - 539, 50))[0], 539.0)

print("--- the counting region does not move when the picture turns")
win.view.set_flip(win.flip)
win._roi_clicked()
for x, y in ((100, 80), (500, 80), (500, 400), (100, 400)):
    win._corner(x, y)
win._setup_save()
before = list(win.infer.roi)
saved = json.load(open(W.roi_path(0), encoding="utf-8"))["roi"]
win._flip_clicked()
check("the region is untouched", win.infer.roi, before)
check("and so is its file", json.load(open(W.roi_path(0), encoding="utf-8"))["roi"], saved)
win._flip_clicked()
check("still untouched", win.infer.roi, before)

print("--- and neither does the record")
win._flip_clicked()
win._tick()
win._save()
rec = json.load(open(max(glob.glob(os.path.join(W.RECORDS, "count_*.json")),
                         key=os.path.getmtime), encoding="utf-8"))
check("the boxes are in the camera's coordinates", rec["boxes"][0][0], 40.0)

print("--- replacing the region does NOT undo the flip")
# There is no "clear the frame" button any more -- a region is what makes counting
# possible, so removing one leaves the window unable to do its job. What an operator
# actually does is draw a DIFFERENT one, which is what this stands in for.
flip_now = win.flip
win._roi_clicked()
check("armed for a new one", win.arming, True)
check("flip kept", W.load_flip(0), flip_now)
win._setup_cancel()

# ------------------------------------------------------------------------------- phone
print("--- the phone does it the same way, in its own coordinates")
from model3.phone import records as rec_store
from model3.phone import screen as screen_mod
from model3.phone.screen import Screen
PH = tempfile.mkdtemp(prefix="flip_phone_")
check("off until turned on", rec_store.load_flip(PH), False)
rec_store.save_flip(PH, True)
check("an on is remembered", rec_store.load_flip(PH), True)
rec_store.save_flip(PH, False)
rec_store.save_roi(PH, (640, 480), [(1, 1), (2, 2), (3, 3), (4, 4)])
rec_store.save_roi(PH, (640, 480), None)
check("clearing the region leaves the flip alone", rec_store.load_flip(PH), False)

screen_mod.use(2340, 1080)
sc = Screen(os.path.join(ROOT, "model3", "phone", "assets"), PH, target=60)
check("the screen picks it up", sc.flip, False)
sc.compose(FRAME, np.zeros((0, 4), np.float32), 0, 0.0)
sc.roi = [(100, 80), (100, 400), (500, 400), (500, 80)]
roi_before = list(sc.roi)
# THE BUTTON HAS GONE FROM THE PHONE, the mechanism has not: the bench still has its own
# and the pane's mapping still has to undo a mirror correctly if one is ever set. Called
# directly, because there is no longer a control that calls it here.
check("no flip button on the phone any more", "flip" in dict(sc.hits), False)
sc._flip()
check("flip on", sc.flip, True)
check("written", rec_store.load_flip(PH), True)
check("the region did not move", sc.roi, roi_before)

print("--- the mark and the tap use one mirror, so they meet on the tablet")
sc.roi = None
def mark_x(flip, frame_x):
    """Where a tablet at frame_x lands on the canvas."""
    sc.flip = flip
    sc._over = []
    sc.compose(FRAME, np.array([[frame_x - 5, 200, frame_x + 5, 220]], np.float32), 1, 8.0)
    dots = [g for k, g, _c, _w in sc._over if k == "circle"]
    return dots[0][0] if dots else None

left_plain = mark_x(False, 60)
left_flipped = mark_x(True, 60)
px, py, pw, ph = screen_mod.PANE
check("unflipped, a tablet near the left edge marks near the left of the pane",
      left_plain - px < pw // 3, True)
check("flipped, it marks near the right", (px + pw) - left_flipped < pw // 3, True)

# WITHIN A FRAME PIXEL, not exactly: the mark is drawn at a whole pixel of the scaled-down
# screen, which is more than one frame pixel wide, so the way back can land on the next
# one. Exact equality held only for the pane size it was written against.
sc.flip = True
check("and a tap on that mark comes back to the tablet",
      abs(sc._to_frame(left_flipped, py + 50, FRAME.shape)[0] - 60) <= 1, True)
sc.flip = False
check("unflipped too", abs(sc._to_frame(left_plain, py + 50, FRAME.shape)[0] - 60) <= 1, True)

print("--- the bridge no longer touches the frame")
sys.path.insert(0, os.path.join(ROOT, "model3", "android", "app", "src", "main", "python"))
import pillcount_android as bridge
bridge._S["cv2"] = cv2
bridge._S["screen"] = sc
rgba = cv2.cvtColor(FRAME, cv2.COLOR_BGR2RGBA)
h, w = rgba.shape[:2]
buf = np.zeros((h, w + 16, 4), np.uint8); buf[:, :w] = rgba
raw, stride = buf.tobytes(), (w + 16) * 4
sc.flip = False
a = bridge._to_bgr(raw, w, h, stride, 0, 0, 0, 0, 0)
sc.flip = True
b = bridge._to_bgr(raw, w, h, stride, 0, 0, 0, 0, 0)
check("THE MODEL SEES THE SAME IMAGE EITHER WAY",
      int(np.abs(a.astype(int) - b.astype(int)).max()), 0)

print("--- and Kotlin is told, because Kotlin owns the surface")
sc.flip = False
check("every touch carries it", json.loads(bridge.touch("up", 5, 5)).get("flip"), False)
sc.flip = True
check("and carries a change", json.loads(bridge.touch("up", 5, 5)).get("flip"), True)
kt = open(os.path.join(ROOT, "model3", "android", "app", "src", "main", "java",
                       "com", "pharmaflow", "pillcount", "MainActivity.kt"),
          encoding="utf-8").read()
check("the activity reads it at boot", 'info.optBoolean("flip"' in kt, True)
check("and on every touch", 'optBoolean("flip", mirrored)' in kt, True)
check("and turns the surface round",
      'previewView.scaleX = if (mirrored != rotated) -softZoom else softZoom' in kt, True)

shutil.rmtree(SETTINGS, ignore_errors=True)
shutil.rmtree(PH, ignore_errors=True)
print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
