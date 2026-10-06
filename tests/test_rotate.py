# -*- coding: utf-8 -*-
"""The half turn, for a camera mounted on the far side of the tray -- both machines.

It is the flip's design again: the picture is turned on its way to the screen and every
tap is turned back, so the frame the model sees, the counting region and the record never
learn it exists. Unlike the flip it is a camera setting, so it is kept by the settings'
บันทึก and put back by their ยกเลิก.
"""
import os, sys, json, glob, tempfile, shutil
os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                          # app/ and model3/ read relative paths
import numpy as np, cv2
# The degree sign in "หมุนภาพ 180°" is not in cp874, which is this console's code page:
# without this, printing the check that reads it is what fails. See tests/run.py.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

import app as appmod
SETTINGS = tempfile.mkdtemp(prefix="rot_set_")
appmod.SETTINGS = SETTINGS
import app.window as W
W.SETTINGS = SETTINGS
W.RECORDS = tempfile.mkdtemp(prefix="rot_rec_")
from app import sound; sound.enabled = False
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication
QApplication.instance() or QApplication([])

FAIL = []
def check(label, got, want):
    ok = got == want
    print(("  ok  " if ok else "  FAIL") + f"  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok: FAIL.append(label)

# A bright mark in the TOP-LEFT, so a half turn is visible rather than argued about.
FRAME = np.zeros((480, 640, 3), np.uint8)
FRAME[200:280, 40:120] = 255
GREEN = (68, 134, 16)                   # the model's dot, painted over the box centre

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

print("--- off until somebody turns it on, and kept beside the flip")
check("a camera nobody has set up", W.load_rotate(0), False)
W.save_flip(0, True)
W.save_rotate(0, True)
check("an on is remembered", W.load_rotate(0), True)
check("and saving it left the flip alone", W.load_flip(0), True)
W.save_flip(0, False)
check("saving the flip left it alone too", W.load_rotate(0), True)
W.save_rotate(0, False)

win = W.Window(C(), I(), target=60, camera=0)
check("the window picked the setting up", win.rotate, False)
check("and told the view", win.view.rotate, False)
check("the button is in the camera settings", win.rotate_btn.parent() is not None, True)
check("and shows it off", win.rotate_btn.isChecked(), False)

print("--- the picture is turned, and only the picture")
shots = {}
real_show = win.view.show_frame
def grab(bgr):
    shots["last"] = bgr.copy()
    return real_show(bgr)
win.view.show_frame = grab

win._tick()
plain = shots["last"]
check("unturned: the mark is top-left", int(plain[210, 45].max()) > 200, True)
win._setup_open()
win._rotate_clicked()
# Read before the tick: with no region yet, the tick puts the "no region" prompt there.
check("the footer says so", win.note, "หมุนภาพ 180° แล้ว")
win._tick()
turned = shots["last"]
check("turned: exactly the half turn of it",
      int(np.abs(cv2.rotate(plain, cv2.ROTATE_180).astype(int) - turned.astype(int)).max()), 0)
check("so the mark is bottom-right", int(turned[480 - 1 - 210, 640 - 1 - 45].max()) > 200, True)
check("and the top-left is empty", int(turned[210, 45].max()), 0)
check("the model's dot turned with the picture",
      tuple(int(v) for v in turned[480 - 1 - 240, 640 - 1 - 80]), GREEN)
check("the button shows it on", win.rotate_btn.isChecked(), True)

print("--- with the flip as well, only the up-down is left")
win.view.set_flip(True); win.flip = True
win._tick()
check("flip + half turn is the up-down mirror",
      int(np.abs(cv2.flip(plain, 0).astype(int) - shots["last"].astype(int)).max()), 0)
win.view.set_flip(False); win.flip = False

print("--- a tap comes back through the same turn")
win.view._geom = (0.0, 0.0, 1.0)            # a 1:1 picture at the widget's origin
win.view._fw, win.view._fh = 640, 480
check("turned, a tap at (100, 50) is (539, 429)",
      win.view._to_frame(QPointF(100, 50)), (539.0, 429.0))
win.view.set_flip(True)
check("turned and flipped, only y is turned back",
      win.view._to_frame(QPointF(100, 50)), (100.0, 429.0))
win.view.set_flip(False)
win.view.set_rotate(False)
check("unturned, it is (100, 50)", win.view._to_frame(QPointF(100, 50)), (100.0, 50.0))
win.view.set_rotate(True)

print("--- ยกเลิก puts it back, and nothing was written")
check("not written while the settings are open", W.load_rotate(0), False)
win._setup_cancel()
check("off again", win.rotate, False)
check("the view too", win.view.rotate, False)
check("and the button", win.rotate_btn.isChecked(), False)
check("still not written", W.load_rotate(0), False)

print("--- the region and the corners do not move when the picture turns")
win._setup_open()
win._roi_clicked()
win._corner(100, 80)
win._corner(500, 80)
pending = list(win.pending)
win._rotate_clicked()
check("corners already tapped stay where they were", win.pending, pending)
check("and the corners can still be finished", win.arming, True)
win._corner(500, 400)
win._corner(100, 400)
before = list(win.infer.roi)
win._rotate_clicked()
win._rotate_clicked()
check("turning it does not touch the region", win.infer.roi, before)
check("still turned", win.rotate, True)

print("--- บันทึก writes it")
win._setup_save()
check("written", W.load_rotate(0), True)
check("the region too", json.load(open(W.roi_path(0), encoding="utf-8"))["roi"],
      [list(p) for p in before])
check("and the flip beside it untouched", W.load_flip(0), False)

print("--- and the record is in the camera's coordinates")
win._tick()
win._save()
rec = json.load(open(max(glob.glob(os.path.join(W.RECORDS, "count_*.json")),
                         key=os.path.getmtime), encoding="utf-8"))
check("the box is where the camera saw it", rec["boxes"][0][:2], [40.0, 200.0])
win.close()

print("--- opened again, it comes back on")
win = W.Window(C(), I(), target=60, camera=0)
check("the window", win.rotate, True)
check("the view", win.view.rotate, True)
check("the button", win.rotate_btn.isChecked(), True)
win.close()

# ------------------------------------------------------------------------------- phone
print("--- phone: the same setting, in view.json beside the flip")
from model3.phone import records as rec_store
from model3.phone import screen as screen_mod
from model3.phone.screen import Screen
PH = tempfile.mkdtemp(prefix="rot_phone_")
check("off until turned on", rec_store.load_rotate(PH), False)
rec_store.save_flip(PH, True)
rec_store.save_rotate(PH, True)
check("an on is remembered", rec_store.load_rotate(PH), True)
check("beside the flip", rec_store.load_flip(PH), True)
rec_store.save_roi(PH, (640, 480), [(1, 1), (2, 2), (3, 3), (4, 4)])
rec_store.save_roi(PH, (640, 480), None)
check("clearing the region leaves it alone", rec_store.load_rotate(PH), True)
rec_store.save_flip(PH, False)
rec_store.save_rotate(PH, False)

screen_mod.use(2340, 1080)
sc = Screen(os.path.join(ROOT, "model3", "phone", "assets"), PH, target=60)
NONE = np.zeros((0, 4), np.float32)
check("the screen picks it up", sc.rotate, False)
sc.compose(FRAME, NONE, 0, 0.0)
check("no turn button on the counting screen", "rotate" in dict(sc.hits), False)

sc.roi = [(100, 80), (100, 400), (500, 400), (500, 80)]
sc._act("setup", None, None)
sc.compose(FRAME, NONE, 0, 0.0)
hits = dict(sc.hits)
check("the settings have it", "rotate" in hits, True)

def overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]
others = [n for n in ("zoom-", "zoom+", "zoom-track", "roi", "setup-save", "setup-cancel")
          if n in hits]
check("it overlaps nothing else in the panel",
      [n for n in others if overlap(hits["rotate"], hits[n])], [])
check("every control is inside the card",
      [n for n in others + ["rotate"] if hits[n][1] + hits[n][3] > 104 + 896], [])

# Its words have pictures, or the APK draws a hollow box where they should be.
words = sc.text.index["words"]
check("its words are in the APK",
      [w for w in ("ทิศทางภาพ", "หมุนภาพ", "เลิกหมุนภาพแล้ว") if w not in words], [])
check("and so is the degree sign", "°" in sc.text.index["digits"], True)

OX, OY = screen_mod.OUT_W / screen_mod.W, screen_mod.OUT_H / screen_mod.H   # taps arrive scaled
r = hits["rotate"]
cx, cy = r[0] + r[2] // 2, r[1] + r[3] // 2
sc.touch("down", cx * OX, cy * OY, FRAME.shape); sc.touch("up", cx * OX, cy * OY, FRAME.shape)
check("a tap on it turns the picture", sc.rotate, True)
check("the footer says so", sc.note, "หมุนภาพ 180° แล้ว")
check("not written yet", rec_store.load_rotate(PH), False)
check("the region did not move", sc.roi, [(100, 80), (100, 400), (500, 400), (500, 80)])
sc._act("setup-cancel", None, None)
check("ยกเลิก puts it back", sc.rotate, False)
check("still not written", rec_store.load_rotate(PH), False)

sc._act("setup", None, None)
sc._act("rotate", None, None)
sc._act("setup-save", None, None)
check("บันทึก writes it", rec_store.load_rotate(PH), True)
check("and leaves the settings", sc.setup, False)

print("--- phone: the mark and the tap use one turn, so they meet on the tablet")
sc.roi = None
def mark(rotate, flip, fx, fy):
    """Where a tablet centred on (fx, fy) lands on the canvas."""
    sc.rotate, sc.flip = rotate, flip
    sc._over = []
    sc.compose(FRAME, np.array([[fx - 5, fy - 5, fx + 5, fy + 5]], np.float32), 1, 8.0)
    dots = [g for k, g, _c, _w in sc._over if k == "circle"]
    return dots[0][:2] if dots else None

px, py, pw, ph = screen_mod.PANE
plain = mark(False, False, 60, 60)
turned = mark(True, False, 60, 60)
check("unturned, a tablet near the top-left marks near the pane's top-left",
      (plain[0] - px < pw // 3, plain[1] - py < ph // 3), (True, True))
check("turned, it marks near the bottom-right",
      ((px + pw) - turned[0] < pw // 3, (py + ph) - turned[1] < ph // 3), (True, True))
# The half turn is about the pane's centre, which is where Kotlin pivots the surface.
check("the two are reflected through the pane's centre",
      (abs(plain[0] + turned[0] - (2 * px + pw)) <= 2,
       abs(plain[1] + turned[1] - (2 * py + ph)) <= 2), (True, True))

# WITHIN A FRAME PIXEL: the mark is drawn at a whole pixel of the scaled-down screen.
for rot, flip in ((True, False), (True, True), (False, True)):
    at = mark(rot, flip, 60, 60)
    back = sc._to_frame(at[0], at[1], FRAME.shape)
    check(f"turn={rot} flip={flip}: a tap on the mark comes back to the tablet",
          (abs(back[0] - 60) <= 1, abs(back[1] - 60) <= 1), (True, True))

print("--- phone: the model sees the same image either way")
sys.path.insert(0, os.path.join(ROOT, "model3", "android", "app", "src", "main", "python"))
import pillcount_android as bridge
bridge._S["cv2"] = cv2
bridge._S["screen"] = sc
rgba = cv2.cvtColor(FRAME, cv2.COLOR_BGR2RGBA)
h, w = rgba.shape[:2]
buf = np.zeros((h, w + 16, 4), np.uint8); buf[:, :w] = rgba
raw, stride = buf.tobytes(), (w + 16) * 4
sc.rotate = False
a = bridge._to_bgr(raw, w, h, stride, 0, 0, 0, 0, 0)
sc.rotate = True
b = bridge._to_bgr(raw, w, h, stride, 0, 0, 0, 0, 0)
check("THE SAME FRAME, TURNED OR NOT", int(np.abs(a.astype(int) - b.astype(int)).max()), 0)

print("--- phone: and Kotlin is told, because Kotlin owns the surface")
sc.flip = False
sc.rotate = False
check("every touch carries it", json.loads(bridge.touch("up", 5, 5)).get("rotate"), False)
sc.rotate = True
check("and carries a change", json.loads(bridge.touch("up", 5, 5)).get("rotate"), True)
kt = open(os.path.join(ROOT, "model3", "android", "app", "src", "main", "java",
                       "com", "pharmaflow", "pillcount", "MainActivity.kt"),
          encoding="utf-8").read()
check("the activity reads it at boot", 'info.optBoolean("rotate"' in kt, True)
check("and on every touch", 'optBoolean("rotate", rotated)' in kt, True)
check("across turns when exactly one of the two is on",
      "previewView.scaleX = if (mirrored != rotated) -softZoom else softZoom" in kt, True)
check("and down when it is turned",
      "previewView.scaleY = if (rotated) -softZoom else softZoom" in kt, True)

shutil.rmtree(SETTINGS, ignore_errors=True)
shutil.rmtree(PH, ignore_errors=True)
print()
print("FAILURES:", ", ".join(FAIL) if FAIL else "none")
sys.exit(1 if FAIL else 0)
