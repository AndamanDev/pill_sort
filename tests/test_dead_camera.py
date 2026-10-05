# -*- coding: utf-8 -*-
"""A camera that stops is shown as a stopped camera, on either screen.

WHAT THIS IS ABOUT. Both screens used to keep the last frame on the pane and draw a red
border round it. That border is a detail beside a whole tray of pills, and the picture
under it is the most convincing thing on the screen -- it looks live because it WAS live,
a second ago. An operator glancing up from the bench reads the tray, not the border.

So the pane goes BLACK and says so in words. Black is the one thing a camera never sends,
so it cannot be misread as a view of the bench.

AND IT COMES BACK BY ITSELF. Neither screen has a "reconnect" button and neither needs
one: the bench reopens the device and the phone rebinds the camera, both on their own
clocks, and the only thing either screen does is stop being told the picture is stale.
Nothing is re-aimed and no counting region is redrawn -- which is the half of this that a
test has to hold down, because it is invisible when it works.
"""
import os, sys, tempfile, shutil, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import cv2

FAIL = []
def check(label, got, want):
    ok = got == want
    print(("  ok  " if ok else "  FAIL") + f"  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok: FAIL.append(label)

# -------------------------------------------------------------------------------- bench
import app as appmod
S = tempfile.mkdtemp(prefix="dead_set_"); appmod.SETTINGS = S
import app.window as W
W.SETTINGS = S
RECORDS = tempfile.mkdtemp(prefix="dead_rec_"); W.RECORDS = RECORDS
from app import sound; sound.enabled = False
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage
QApplication.instance() or QApplication([])

FRAME = np.full((480, 640, 3), 90, np.uint8)
class C:
    fps = 20.0
    gap = 0.0
    error = ""
    def latest(self): return FRAME, 1
    def stale(self): return self.gap
    def stop(self): pass
class I:
    roi = [(10, 10), (630, 10), (630, 470), (10, 470)]
    conf = .45; iou = .4; imgsz = 640; error = ""; n = 3
    def _o(self): return np.zeros((self.n, 4), np.float32), np.ones((self.n,), np.float32), self.n, 8.0
    def result(self): return self._o(), 1
    def snapshot(self): return FRAME, self._o()
    def inside(self, b): return True
    def stop(self): pass

cam, inf = C(), I()
win = W.Window(cam, inf, target=60, camera=0)
win.resize(1280, 800)
view = win.view
view.resize(700, 500)

def pane():
    """The pane as it is actually painted, which is the only thing the operator sees.

    COPIED, not viewed. numpy over constBits() is a window onto the QImage's own buffer,
    and the QImage dies with this function -- so the array handed back points at freed
    memory and the first thing that reads it takes the interpreter down. It segfaulted
    exactly once here, which is once more than it needs to.
    """
    img = view.grab().toImage().convertToFormat(QImage.Format_RGB32)
    h, w = img.height(), img.width()
    buf = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine() // 4, 4)
    return buf[:, :w, :3].copy()            # BGR, alpha dropped

print("--- a live camera: the picture is on the pane")
cam.gap = 0.0
win._tick()
check("not dead", view._dead, False)
check("and a picture has been handed over", view._pix is not None, True)
geom_before = view._geom

print("--- the camera stops")
cam.gap = W.STALE_S + 1.0
win._tick()
check("the pane knows", view._dead, True)
check("the save is blocked", win.save_btn.isEnabled(), False)
check("and the footer says why", win.note.startswith("ภาพจากกล้องหยุด"), True)

px = pane()
mid = px[px.shape[0] // 2 - 60:px.shape[0] // 2 + 60, :, :]
check("the middle of the pane is black", int(mid.min()), 0)
check("there is white type on it", int(px.max()), 255)
# The frame is a flat 90 grey, so ANY 90 left in the pane is the old picture showing.
check("and none of the last frame is left", int((px == 90).all(axis=2).sum()), 0)

print("--- THE HALF THAT IS INVISIBLE WHEN IT WORKS: nothing was thrown away")
check("the region is untouched", inf.roi, [(10, 10), (630, 10), (630, 470), (10, 470)])
check("and so is the mapping a tap comes back through", view._geom, geom_before)

print("--- the lead goes back in")
cam.gap = 0.0
win._tick()
check("the pane is live again", view._dead, False)
check("the save is not blocked any more", win._block, "")
px = pane()
check("the picture is back", int((px == 90).all(axis=2).sum()) > 1000, True)
check("and the black is gone", int(px.min()) > 0, True)

print("--- and it survives going away and coming back a second time")
cam.gap = W.STALE_S + 1.0; win._tick()
check("dead again", view._dead, True)
cam.gap = 0.0; win._tick()
check("and live again", view._dead, False)

shutil.rmtree(S, ignore_errors=True)
shutil.rmtree(RECORDS, ignore_errors=True)

# -------------------------------------------------------------------------------- phone
print()
print("--- and the phone does the same, through its own hole in the canvas")
from model3.phone import screen as screen_mod
from model3.phone.screen import Screen, PANE, HOLE, STALE_S
screen_mod.use(2340, 1080)
from model3.phone.screen import PANE                    # use() recomputes it
REC = tempfile.mkdtemp(prefix="dead_ph_")
sc = Screen(os.path.join(ROOT, "model3", "phone", "assets"), REC, target=60)
sc.roi = [(10, 10), (630, 10), (630, 470), (10, 470)]
F = np.full((480, 640, 3), 90, np.uint8)
B = np.zeros((0, 4), np.float32)

# `hole_alpha` IS OPACITY, NOT THE HOLE: _punch returns bitwise_not of the sentinel mask,
# so 0 is what the preview surface shows through and 255 is canvas the operator sees. The
# whole of this feature, on the phone, is that pane going from "has a 0 in it" to "is 255
# everywhere" -- nothing tells CameraX to stop, the canvas simply stops leaving a gap.
def paint(stale):
    sc._chrome_key = None
    return sc.compose(F, B, 3, 8.0, stale=stale)

print("--- live: the hole is punched, so CameraX's preview shows through it")
paint(0.0)
check("part of the pane is transparent", int(sc.hole_alpha.min()), 0)

print("--- stopped: the hole is NOT punched, so the canvas covers the preview")
img = paint(STALE_S + 1.0)
check("the pane is opaque from edge to edge", int(sc.hole_alpha.min()), 255)
check("the save is blocked", sc.blocked.startswith("ภาพจากกล้องหยุด"), True)

px, py, pw, ph = PANE
p = img[py:py + ph, px:px + pw]
check("nothing is still the sentinel", int(cv2.inRange(p, np.array(HOLE, np.uint8),
                                                      np.array(HOLE, np.uint8)).max()), 0)
check("the pane is black", int(p[ph // 2 - 40:ph // 2 + 40].min()), 0)
check("with white type on it", int(p.max()), 255)

print("--- and back again")
paint(0.0)
check("the hole is punched again", int(sc.hole_alpha.min()), 0)
check("and the region survived", sc.roi, [(10, 10), (630, 10), (630, 470), (10, 470)])

print("--- the new words have pictures, or they draw as hollow boxes")
def missing(t):
    return [q for q, d in sc.text._runs(t) if sc.text._entry(q, d) is None]
for word in (screen_mod.DEAD_HEAD,) + screen_mod.DEAD_LINES:
    check(word, missing(word), [])

shutil.rmtree(REC, ignore_errors=True)

# ------------------------------------------------------------------ the capture thread
# WHAT THE SCREEN IS BEING TOLD, as opposed to what it does with it. Everything above
# trusts Capture.stale(); this drives the real thread against a device that dies and comes
# back, because the two ways that number can lie are both invisible from the window.
print()
print("--- and the number the screen trusts is honest about a sick camera")
import app.worker as WK

state = {"alive": True, "live_caps": 0}

class FakeCap:
    def __init__(self, *a, **k):
        state["live_caps"] += 1
        self._open = state["alive"]
        self._released = False
    def isOpened(self): return self._open
    def set(self, *a): return True
    def read(self):
        if self._released or not state["alive"] or not self._open:
            return False, None
        time.sleep(0.002)
        return True, np.full((480, 640, 3), 90, np.uint8)
    def release(self):
        if not self._released:
            self._released = True
            state["live_caps"] -= 1

class OpensButSilent(FakeCap):
    """The USB failure nobody expects: the handle comes back, the pictures do not."""
    def read(self): return False, None

real_cv2, real_tries = WK.cv2, WK.REOPEN_TRIES
WK.cv2 = type("shim", (), {"VideoCapture": FakeCap, "CAP_DSHOW": 700,
                           "CAP_PROP_BUFFERSIZE": 38, "CAP_PROP_AUTO_EXPOSURE": 21,
                           "CAP_PROP_EXPOSURE": 15})()
WK.REOPEN_TRIES = 20                    # 0.2s an attempt, so this suite finishes
try:
    cap = WK.Capture(0, exposure="auto")
    check("it opens", cap.open(), True)
    cap.start()
    time.sleep(0.3)
    check("and the picture is live", cap.stale() < 0.5, True)

    print("--- the lead comes out, and the device cannot even be opened")
    state["alive"] = False
    time.sleep(W.STALE_S + 0.6)         # past the threshold, and several reopens deep
    check("the screen is told it is dead", cap.stale() > W.STALE_S, True)
    # Every failed open used to leave its DSHOW graph behind, once every couple of
    # seconds, for as long as the camera was missing.
    check("and no handle was leaked doing it", state["live_caps"], 0)

    print("--- THE LIE: it opens again, and sends nothing")
    state["alive"] = True
    WK.cv2.VideoCapture = OpensButSilent
    seq_before = cap.latest()[1]
    time.sleep(0.9)
    check("not one frame arrived", cap.latest()[1], seq_before)
    # This is the one that mattered: open() stamped the frame clock, so stale fell to
    # ~0.1s, the black pane came down, the last frozen tray went back up and the save
    # button came back to life over it.
    check("so the screen is still told it is dead", cap.stale() > W.STALE_S, True)

    print("--- and a camera that really comes back really comes back")
    WK.cv2.VideoCapture = FakeCap
    seq_before = cap.latest()[1]
    time.sleep(0.8)
    check("frames again", cap.latest()[1] > seq_before, True)
    check("and the screen is told so", cap.stale() < 0.5, True)
    cap.stop()
    time.sleep(0.2)
finally:
    WK.cv2, WK.REOPEN_TRIES = real_cv2, real_tries

print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
