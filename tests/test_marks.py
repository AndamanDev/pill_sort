# -*- coding: utf-8 -*-
"""Slide the tray, keep the count, and see whether the dots come with it.

The failure this covers looked like the model giving up: put tablets down and the dots sit
on them; nudge the tray and the dots stay behind while the tablets walk out from under
them. Nothing was wrong with the counting. The screen simply was not being redrawn,
because the value it compares on was the COUNT -- which does not change when a tray of
sixty-one slides two inches to the left.
"""
import os, sys, tempfile, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                          # app/ and model3/ read relative paths
sys.path.insert(0, os.path.join(ROOT, "model3", "android", "app", "src", "main", "python"))
import numpy as np, cv2
from model3.phone import screen as screen_mod
from model3.phone.screen import Screen
import pillcount_android as bridge

screen_mod.use(2340, 1080)
REC = tempfile.mkdtemp(prefix="marks_")
sc = Screen(os.path.join(ROOT, "model3", "phone", "assets"), REC, target=60)
FRAME = np.full((480, 640, 3), 40, np.uint8)

FAIL = []
def check(label, got, want):
    ok = got == want
    print(("  ok  " if ok else "  FAIL") + f"  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok: FAIL.append(label)


def tray_at(x0, n=61):
    """n tablets in a row, starting at x0. The same count, a different place."""
    return np.array([[x0 + i * 8, 200, x0 + i * 8 + 6, 206] for i in range(n)], np.float32)


def dots(boxes, count):
    sc._over = []
    sc.compose(FRAME, boxes, count, 8.0)
    return [g[:2] for k, g, _c, _w in sc._over if k == "circle"]


print("--- the key must notice new boxes, not just a new count")
a = tray_at(50)
b = tray_at(220)                      # the tray slid; still 61 tablets
sc.marks_at = 1
k1 = sc.state_key(61, 8.0, 0.0, "")
sc.marks_at = 2                       # what the bridge does on a fresh detection
k2 = sc.state_key(61, 8.0, 0.0, "")
check("same count, new detection -> different key", k1 != k2, True)
sc.marks_at = 2
check("and nothing else changing -> same key", sc.state_key(61, 8.0, 0.0, ""), k2)

print("--- and the dots actually move with the tray")
sc.marks_at = 1
before = dots(a, 61)
sc.marks_at = 2
after = dots(b, 61)
# TWO circles a tablet: a white rim and the app's green inside it, so a tray of 61 puts
# 122 marks in the overlay. The rim is what keeps a dot visible on a dark capsule.
check("two marks a tablet", (len(before), len(after)), (122, 122))
moved = sum(1 for p, q in zip(before, after) if p != q)
check("every one of them moved", moved, 122)
print(f"       first dot {before[0]} -> {after[0]}")

print("--- the whole frame path: a slid tray is redrawn, a still one is not")
class Model:
    """Hands back whatever boxes the test sets, without a graph."""
    def __init__(self): self.boxes = tray_at(50)
    def detect(self, bgr): return self.boxes, np.ones((len(self.boxes),), np.float32)

model = Model()
bridge._S.clear()
bridge._S.update({"cv2": cv2, "screen": sc, "engine": model,
                  "pane": screen_mod.pane_out(),
                  "pane_ratio": screen_mod.PANE[2] / screen_mod.PANE[3],
                  "boxes": np.zeros((0, 4), np.float32),
                  "confs": np.zeros((0,), np.float32),
                  "count": 0, "ms": 0.0, "error": "", "geometry": "x",
                  "last_frame_at": 0.0, "last_key": None})

def camera(shade):
    f = np.full((480, 640, 3), shade, np.uint8)
    rgba = cv2.cvtColor(f, cv2.COLOR_BGR2RGBA)
    stride = (640 + 64) * 4
    buf = np.zeros((480, stride // 4, 4), np.uint8); buf[:, :640] = rgba
    return buf.tobytes(), stride

raw, stride = camera(40)
out = bridge.frame(raw, 640, 480, stride, 0, 0, 0, 0, 0)
check("the first frame is drawn", len(out) > 0, True)
check("and it counted them", bridge._S["count"], 61)

out = bridge.frame(raw, 640, 480, stride, 0, 0, 0, 0, 0)
check("an identical frame is skipped", len(out), 0)

# The tray slides: the picture changes, so the detector runs, and it returns the same
# COUNT at new places. This is the exact case that used to freeze the dots.
model.boxes = tray_at(220)
raw2, stride2 = camera(70)
import time
time.sleep(bridge.DETECT_EVERY + 0.02)          # let the pacing allow another pass
out = bridge.frame(raw2, 640, 480, stride2, 0, 0, 0, 0, 0)
check("the slid tray is drawn again", len(out) > 0, True)
check("still sixty-one", bridge._S["count"], 61)
check("and the marks counter moved", sc.marks_at > 1, True)

shutil.rmtree(REC, ignore_errors=True)
print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
