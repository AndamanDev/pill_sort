"""รีเซ็ต: the prescription number and every banked pour back to zero, on both screens.

The flow it protects: set 39, pour 25 and keep it, pour the rest; นับใหม่ throws the
pours away and keeps 39, because a recount is of the same prescription; รีเซ็ต throws
both away. Two presses while pours are banked, one when there is only the number.
"""
import os, sys, time, json, glob, shutil, tempfile
os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                          # app/ and model3/ read relative paths

import numpy as np
import app as appmod

RECORDS = tempfile.mkdtemp(prefix="rec-")
appmod.RECORDS = RECORDS
import app.window as W
W.RECORDS = RECORDS
# A SETTINGS DIRECTORY OF ITS OWN, and this was a real hole. Without it the window loaded
# the counting region saved by whoever last used the real app -- so a rule that refuses to
# count until one is drawn was never once exercised, and the test passed by inheriting a
# region it never asked for.
SETTINGS = tempfile.mkdtemp(prefix="set-")
appmod.SETTINGS = SETTINGS
W.SETTINGS = SETTINGS
from app import sound
sound.enabled = False
from PySide6.QtWidgets import QApplication
qapp = QApplication.instance() or QApplication([])

import numpy as np
import cv2


class FakeCap:
    def __init__(self):
        self.frame = np.zeros((480, 640, 3), np.uint8)
        self.fps = 20.0
    def latest(self): return self.frame, 1
    def stale(self): return 0.0
    def stop(self): pass


class FakeInfer:
    roi = [(0, 0), (639, 0), (639, 479), (0, 479)]
    conf, iou, imgsz = 0.45, 0.4, 640
    error = ""
    def __init__(self, cap):
        self.cap = cap
        self.n = 0
    def _out(self):
        boxes = np.zeros((self.n, 4), np.float32)
        return boxes, np.ones((self.n,), np.float32), self.n, 42.0
    def result(self): return self._out(), 1
    def snapshot(self): return self.cap.frame, self._out()
    def inside(self, b): return True
    def stop(self): pass


FAIL = []
def check(label, got, want):
    ok = got == want
    print(("  ok  " if ok else "  FAIL") + f"  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok: FAIL.append(label)


# ======================================================================== the PC window
cap = FakeCap()
inf = FakeInfer(cap)
win = W.Window(cap, inf, target=0, camera=0)

def tray(n, hold=1.0):
    inf.n = n
    win._tick()
    if hold:
        win._steady_at -= hold
        win._tick()

print("--- PC: nothing to reset on a fresh screen")
win._tick()
check("dead", win.wipe_btn.isEnabled(), False)
check("columns blank", (win.want_lbl.text(), win.diff_lbl.text()), ("—", "—"))

print("--- PC: set 39, pour 25")
win._set_target(39)
tray(25)
check("asked for", win.want_lbl.text(), "39")
check("counted", win.count_lbl.text(), "25")
check("missing", (win.diff_title.text(), win.diff_lbl.text()), ("ยอดที่ขาด", "14"))
check("live with a number set", win.wipe_btn.isEnabled(), True)

print("--- PC: keep it, sweep, pour 16 more")
win._take_round()
tray(0)
tray(16)
check("counted on from 25", win.count_lbl.text(), "41")
check("over", (win.diff_title.text(), win.diff_lbl.text()), ("ยอดที่เกิน", "2"))

print("--- PC: นับใหม่ keeps the number")
win._reset_clicked(); win._reset_clicked()
tray(0)
check("pours gone", win.rounds, [])
check("39 kept", (win.target, win.want_lbl.text()), (39, "39"))

print("--- PC: รีเซ็ต with a pour banked takes two presses")
tray(25)
win._take_round()
win._wipe_clicked(); win._tick()
check("first press only arms", (win.target, win.rounds), (39, [25]))
check("and says so on the button", win.wipe_btn.text(), "กดอีกครั้ง")
win._wipe_clicked(); win._tick()
check("second press: everything zero", (win.target, win.rounds, win.clearing),
      (0, [], False))
check("box shows 0", win.target_edit.text(), "0")
check("columns blank again", (win.want_lbl.text(), win.diff_lbl.text()), ("—", "—"))
check("button back", win.wipe_btn.text(), "รีเซ็ต")

print("--- PC: with only a number, one press")
win._set_target(60)
win._wipe_clicked()
check("gone at once", win.target, 0)


# ======================================================================== the phone
from model3.phone import screen as screen_mod
from model3.phone.screen import Screen

screen_mod.use(2340, 1080)
sc = Screen(os.path.join(ROOT, "model3", "phone", "assets"), RECORDS, target=39)
sc.roi = [(0, 0), (639, 0), (639, 479), (0, 479)]
FRAME = np.full((480, 640, 3), 40, np.uint8)
BOXES = np.zeros((0, 4), np.float32)

def paint(count):
    return sc.compose(FRAME, BOXES, count, 42.0)

def ptray(count):
    paint(count)
    sc._steady_at -= 1.0
    paint(count)

def tap(name):
    sc._act(name, lambda: None, None)

print("--- phone: every new word has a picture")
def missing(s):
    return [p for p, d in sc.text._runs(s) if sc.text._entry(p, d) is None]
for s in ("ยอดที่ต้องการ", "ยอดสะสม", "ยอดที่ขาด", "ยอดที่เกิน", "รีเซ็ต", "รีเซ็ตแล้ว",
          "จะทิ้งยอดสะสม 25 เม็ด และจำนวนที่ต้องการ กดอีกครั้งเพื่อรีเซ็ต"):
    check(f"drawable: {s}", missing(s), [])

print("--- phone: 39, pour 25, keep, then the third column")
ptray(25)
check("missing", sc._gap(sc.total(25))[:2], ("ยอดที่ขาด", "14"))
tap("round")
ptray(0)
ptray(16)
check("over", sc._gap(sc.total(16))[:2], ("ยอดที่เกิน", "2"))

print("--- phone: รีเซ็ต takes two taps with a pour banked")
tap("zero")
check("first tap only arms", (sc.target, sc.rounds), (39, [25]))
tap("zero")
check("second tap: everything zero", (sc.target, sc.rounds, sc.clearing), (0, [], False))
check("blank", sc._gap(0)[1], "—")

print("--- phone: a tap on anything else disarms it")
sc.target = 39
ptray(25)
tap("round")
tap("zero")
tap("preset30")
tap("zero")
check("still armed only once", sc.rounds, [25])

shutil.rmtree(RECORDS, ignore_errors=True)
print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
