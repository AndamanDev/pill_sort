# -*- coding: utf-8 -*-
"""The camera's zoom: in the camera settings, written only on บันทึก, and never under a stale region."""
import os, sys, json, tempfile
os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                          # app/ and model3/ read relative paths

import numpy as np, cv2
import app as appmod
SETTINGS = tempfile.mkdtemp(prefix="zoom_set_")
appmod.SETTINGS = SETTINGS
import app.window as W
W.SETTINGS = SETTINGS
W.RECORDS = tempfile.mkdtemp(prefix="zoom_rec_")
from app import sound; sound.enabled = False
from app import worker
from PySide6.QtWidgets import QApplication
QApplication.instance() or QApplication([])

FAIL = []
def check(label, got, want):
    ok = got == want
    print(("  ok  " if ok else "  FAIL") + f"  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok: FAIL.append(label)

print("--- the view file holds the flip AND the zoom, and saving one keeps the other")
check("nobody has zoomed", W.load_zoom(0), None)
W.save_flip(0, True)
W.save_zoom(0, 40)
check("zoom remembered", W.load_zoom(0), 40)
check("flip survived the zoom save", W.load_flip(0), True)
W.save_flip(0, False)
check("zoom survived the flip save", W.load_zoom(0), 40)
os.remove(W.view_path(0))

print("--- the zoom is a crop of a big frame, and what leaves the thread is one size")
check("0 is the whole picture", worker.zoom_factor(worker.ZOOM_MIN), 1.0)
check("the top is ZOOM_X", worker.zoom_factor(worker.ZOOM_MAX), worker.ZOOM_X)
check("out of range is clamped", worker.zoom_factor(500), worker.ZOOM_X)

# A 1920x1080 frame whose every pixel says where it came from: column in B, row in G.
big = np.zeros((1080, 1920, 3), np.uint8)
big[:, :, 0] = (np.arange(1920) * 255 // 1919)[None, :]
big[:, :, 1] = (np.arange(1080) * 255 // 1079)[:, None]
wide, close = worker.zoomed(big, 0), worker.zoomed(big, 100)
# 1x MUST BE THE OLD PICTURE. The camera's own 640x480 is the middle 1440x1080 of its
# 1080p frame, measured; anything wider shrinks every pill the model sees.
check("1x: the old 640x480", wide.shape, (480, 640, 3))
old = cv2.resize(big[:, 240:1680], (640, 480), interpolation=cv2.INTER_AREA)
check("1x: pixel for pixel the old 4:3 middle", int(np.abs(wide.astype(int) - old).max()), 0)
check("zoomed: the same size out", close.shape, wide.shape)
check("2.25x is 640 sensor pixels -- starts a third of the way in",
      abs(int(close[240, 0, 0]) - 85) <= 2, True)
check("and stays centred", abs(int(close[240, 320, 0]) - 127) <= 2, True)
check("and is never an enlargement", 1440 / worker.ZOOM_X >= worker.OUT_W, True)
check("a 640x480 camera passes through untouched",
      worker.zoomed(np.zeros((480, 640, 3), np.uint8), 0).shape, (480, 640, 3))

print("--- the camera is asked for the size, THEN MJPEG, and its own zoom to the widest")
class FakeCap:
    def __init__(self): self.sets = []
    def isOpened(self): return True
    def set(self, prop, v): self.sets.append(prop); return True
    def get(self, prop): return 0.0
    def read(self): return True, big
    def release(self): pass
real_vc = cv2.VideoCapture
try:
    cap = FakeCap()
    cv2.VideoCapture = lambda *a, **k: cap
    c = worker.Capture(0)
    c.open()
    s = cap.sets
    check("width before MJPEG", s.index(cv2.CAP_PROP_FRAME_WIDTH) < s.index(cv2.CAP_PROP_FOURCC), True)
    check("camera zoom set to the widest", cv2.CAP_PROP_ZOOM in s, True)
    c.set_zoom(250)
    check("asked for too much, clamped", c.zoom, worker.ZOOM_MAX)
finally:
    cv2.VideoCapture = real_vc

print("--- the window")
class C:
    fps = 20.0
    def __init__(self, ok): self.zoom_ok, self.zoom, self.asked = ok, (0 if ok else None), []
    def latest(self): return np.zeros((480, 640, 3), np.uint8), 1
    def stale(self): return 0.0
    def set_zoom(self, v): self.zoom = v; self.asked.append(v)
    def stop(self): pass
class I:
    roi = None; conf = .45; iou = .4; imgsz = 640; error = ""
    def result(self): return (np.zeros((0, 4)), np.zeros((0,)), 0, 0.0), 1
    def snapshot(self): return None, (np.zeros((0, 4)), np.zeros((0,)), 0, 0.0)
    def inside(self, b): return True
    def stop(self): pass

CORNERS = [(100, 80), (500, 80), (500, 400), (100, 400)]
def outline(win):
    for x, y in CORNERS:
        if not win.arming:
            win._roi_clicked()
        win._corner(x, y)

win = W.Window(C(False), I(), camera=0)
win.show()
win._setup_open()
win._tick()
check("no zoom on the camera, no slider", win.zoom_row.isVisible(), False)
check("and the settings say so", win.no_zoom_lbl.isVisible(), True)
win.close()

cam = C(True)
win = W.Window(cam, I(), camera=0)
win.show()
win._tick()
print("--- the counting screen carries neither control, only the way in")
check("no slider beside the count", win.zoom_row.isVisible(), False)
check("no region button either", win.roi_btn.isVisible(), False)
check("the way in is there", win.setup_btn.isVisible(), True)

win._setup_open()
win._tick()
check("the settings replace the counting panel",
      (win.setup_panel.isVisible(), win.panel.isVisible(), win.setup_btn.isVisible()),
      (True, False, False))
check("a zoom camera shows the slider", win.zoom_row.isVisible(), True)
check("at 1x", (win.zoom_slider.value(), win.zoom_val.text() == "1.0×"), (0, True))
check("following the camera did not ask it for anything", cam.asked, [])

outline(win)
check("a region is drawn", win.infer.roi is not None, True)
check("but not written before บันทึก", os.path.isfile(W.roi_path(0)), False)
win._setup_save()
check("บันทึก writes it", os.path.isfile(W.roi_path(0)), True)
check("and goes back to counting", (win.setup, win.panel.isVisible()), (False, True))

print("--- zoomed inside the settings: the region goes, nothing is written")
win._setup_open()
win.zoom_in.click()
check("+ asks the camera to zoom", cam.asked, [W.ZOOM_STEP])
check("the region drawn on the wider picture is gone", win.infer.roi, None)
check("the footer says why", "กรอบเดิมไม่ตรง" in win.note, True)
check("the zoom on disk is still the 1x saved with the region", W.load_zoom(0), 0)
check("nor is the region's file touched", os.path.isfile(W.roi_path(0)), True)
win._tick()
check("the count is not shown without a region", win.count_lbl.text(), "—")
check("nor can the settings be saved", win.setup_save_btn.isEnabled(), False)
win._setup_save()
check("and a press is refused, staying in", win.setup, True)
check("saying why", "กำหนดกรอบนับก่อน" in win.note, True)

print("--- ยกเลิก puts back the zoom and the region exactly")
win._setup_cancel()
check("the camera is back at 1x", cam.zoom, 0)
check("and the slider with it", win.zoom_slider.value(), 0)
check("the region is back", win.infer.roi is not None, True)
check("and nothing was written", W.load_zoom(0), 0)
check("back to counting", win.setup, False)

print("--- the whole job: zoom, outline, บันทึก")
win._setup_open()
win.zoom_slider.setValue(60)
check("the slider drives the camera", cam.zoom, 60)
win.zoom_out.click()
check("- comes back", cam.zoom, 60 - W.ZOOM_STEP)
outline(win)
win._tick()
check("with a region the settings can be saved", win.setup_save_btn.isEnabled(), True)
check("but the count cannot be, while they are open", win.save_btn.isEnabled(), False)
win._setup_save()
check("the zoom is saved", W.load_zoom(0), 60 - W.ZOOM_STEP)
check("and the region", os.path.isfile(W.roi_path(0)), True)
win.close()

print("--- opened again, the zoom goes back on before anything is counted")
cam = C(True)
win = W.Window(cam, I(), camera=0)
check("the saved zoom is asked for at once", cam.asked, [60 - W.ZOOM_STEP])
win.show(); win._tick()
check("and the slider shows it", win.zoom_slider.value(), 60 - W.ZOOM_STEP)
win.close()

# ------------------------------------------------------------------------------ phone
print("--- phone: the same slider, in the settings panel, and the same rules")
import json as _json
from model3.phone import screen as screen_mod, records as rec_store
from model3.phone.screen import Screen
PH = tempfile.mkdtemp(prefix="zoom_ph_")
rec_store.save_flip(PH, True)
screen_mod.use(2340, 1080)
check("one factor on both machines", screen_mod.ZOOM_X, worker.ZOOM_X)
px, py, pw, ph = screen_mod.PANE
bx, by, bw, bh = screen_mod.ZOOM_BAR
check("the bar is in the panel beside the picture", bx >= px + pw, True)
check("and inside the card", by + bh <= 104 + 896, True)
check("the pane is still 4:3", abs(pw / ph - 4 / 3) < 0.01, True)

sc = Screen(os.path.join(ROOT, "model3", "phone", "assets"), PH, target=60)
FR = np.zeros((480, 640, 3), np.uint8)
sc.compose(FR, np.zeros((0, 4), np.float32), 0, 0.0)
names = [n for n, _ in sc.hits]
check("the counting screen has no zoom", any(n.startswith("zoom") for n in names), False)
check("nor a region button", "roi" in names, False)
check("only the way in", "setup" in names, True)
check("never zoomed: 0", sc.zoom, 0)

sc.roi = [(100, 80), (100, 400), (500, 400), (500, 80)]
rec_store.save_roi(PH, (640, 480), sc.roi)
sc._act("setup", None, None)
sc.compose(FR, np.zeros((0, 4), np.float32), 0, 0.0)
names = [n for n, _ in sc.hits]
check("the settings have -, + and the track",
      all(n in names for n in ("zoom-", "zoom+", "zoom-track")), True)
check("and the region, บันทึก and ยกเลิก",
      all(n in names for n in ("roi", "setup-save", "setup-cancel")), True)
check("and no counting controls", any(n in names for n in ("save", "round")), False)

track = dict(sc.hits)["zoom-track"]
tx0, tx1 = sc._zoom_track
OX, OY = screen_mod.OUT_W / screen_mod.W, screen_mod.OUT_H / screen_mod.H   # taps arrive scaled
mid_y = (track[1] + track[3] // 2) * OY
sc.touch("down", (tx0 + (tx1 - tx0) // 2) * OX, mid_y, FR.shape)
check("a finger on the track takes the knob there", sc.zoom, 50)
check("and the region drawn on the old picture is gone", sc.roi, None)
check("the footer says why", sc.note, screen_mod.ZOOM_NOTE)
sc.touch("move", tx1 * OX, mid_y, FR.shape)
check("dragged to the end", sc.zoom, 100)
sc.touch("up", tx1 * OX, mid_y, FR.shape)
check("not saved when the finger lifts", rec_store.load_zoom(PH), 0)
check("nor the region's file touched", os.path.isfile(rec_store.roi_path(PH)), True)
sc._act("setup-save", None, None)
check("บันทึก with no region is refused", sc.setup, True)

sc._act("setup-cancel", None, None)
check("ยกเลิก puts the zoom back", sc.zoom, 0)
check("and the region", sc.roi, [(100, 80), (100, 400), (500, 400), (500, 80)])
check("and leaves", sc.setup, False)

sc._act("setup", None, None)
sc.compose(FR, np.zeros((0, 4), np.float32), 0, 0.0)
sc.set_zoom(100)
sc.roi = [(10, 10), (10, 400), (600, 400), (600, 10)]
sc._act("setup-save", None, None)
check("บันทึก writes the zoom", rec_store.load_zoom(PH), 100)
check("and the region", rec_store.load_roi(PH, (640, 480))[0],
      [(10, 10), (10, 400), (600, 400), (600, 10)])
check("and the flip beside it survived", rec_store.load_flip(PH), True)
check("and goes back to counting", sc.setup, False)

sc._act("setup", None, None)
sc.compose(FR, np.zeros((0, 4), np.float32), 0, 0.0)
minus = dict(sc.hits)["zoom-"]
cx, cy = minus[0] + minus[2] // 2, minus[1] + minus[3] // 2
sc.touch("down", cx * OX, cy * OY, FR.shape); sc.touch("up", cx * OX, cy * OY, FR.shape)
check("- steps back", sc.zoom, 100 - screen_mod.ZOOM_STEP)

print("--- phone: Kotlin is told, and the bridge crops when Kotlin says it must")
sys.path.insert(0, os.path.join(ROOT, "model3", "android", "app", "src", "main", "python"))
import pillcount_android as bridge
bridge._S.update({"cv2": cv2, "screen": sc, "frame": FR, "pane_ratio": pw / ph,
                  "geometry": "x"})
reply = _json.loads(bridge.touch("up", 5, 5))
check("every touch reply carries the factor", reply.get("zoom"),
      screen_mod.zoom_factor(100 - screen_mod.ZOOM_STEP))

def raw_of(img):
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    h, w = rgba.shape[:2]
    buf = np.zeros((h, w + 32, 4), np.uint8); buf[:, :w] = rgba
    return buf.tobytes(), (w + 32) * 4

# A 1280x960 USB stream (ANALYSIS_SOFT) whose pixels say which column they came from.
soft = np.zeros((960, 1280, 3), np.uint8)
soft[:, :, 0] = (np.arange(1280) * 255 // 1279)[None, :]
raw, stride = raw_of(soft)
one = bridge._to_bgr(raw, 1280, 960, stride, 0, 0, 0, 0, 0, 1.0)
check("1x on the big stream comes out 640 across", one.shape, (480, 640, 3))
two = bridge._to_bgr(raw, 1280, 960, stride, 0, 0, 0, 0, 0, 2.0)
check("2x: still 640 across", two.shape, (480, 640, 3))
check("2x starts a quarter of the way in", abs(int(two[240, 0, 0]) - 64) <= 2, True)
check("and stays centred", abs(int(two[240, 320, 0]) - 127) <= 2, True)
rot = bridge._to_bgr(raw, 1280, 960, stride, 90, 0, 0, 0, 0, 2.0)
check("a sensor mounted sideways still crops the middle",
      abs(int(rot[rot.shape[0] // 2, rot.shape[1] // 2, 0]) - 127) <= 2, True)
small, sstride = raw_of(FR)
check("a 640 phone stream at 1x passes through untouched",
      bridge._to_bgr(small, 640, 480, sstride, 0, 0, 0, 0, 0, 1.0).shape, (480, 640, 3))

kt = open(os.path.join(ROOT, "model3", "android", "app", "src", "main", "java", "com",
                       "pharmaflow", "pillcount", "MainActivity.kt"), encoding="utf-8").read()
check("the activity reads it at boot", 'info.optDouble("zoom"' in kt, True)
check("and on every touch", 'json.optDouble("zoom"' in kt, True)
check("and hands the software part to every frame", "softZoom.toDouble()" in kt, True)

print()
print("FAILURES:", ", ".join(FAIL) if FAIL else "none")
sys.exit(1 if FAIL else 0)
