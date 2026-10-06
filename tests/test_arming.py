# -*- coding: utf-8 -*-
"""Nothing is filed while the counting region is being changed.

The region only moves inside the camera settings, and while they are open neither the
save nor a pour is allowed -- so no number is filed through a half-moved region.
"""
import os, sys, tempfile, shutil
os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                          # app/ and model3/ read relative paths
import numpy as np, glob

FAIL = []
def check(label, got, want):
    ok = got == want
    print(("  ok  " if ok else "  FAIL") + f"  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok: FAIL.append(label)

# ------------------------------------------------------------------------------- bench
import app as appmod
S = tempfile.mkdtemp(prefix="arm_set_"); appmod.SETTINGS = S
import app.window as W
W.SETTINGS = S
RECORDS = tempfile.mkdtemp(prefix="arm_rec_"); W.RECORDS = RECORDS
from app import sound; sound.enabled = False
from PySide6.QtWidgets import QApplication
QApplication.instance() or QApplication([])

FRAME = np.zeros((480, 640, 3), np.uint8)
class C:
    fps = 20.0
    def latest(self): return FRAME, 1
    def stale(self): return 0.0
    def stop(self): pass
class I:
    roi = None; conf = .45; iou = .4; imgsz = 640; error = ""
    # EXACTLY THE TARGET, so the big button is Done -- the save. Short of it the button
    # is Keep, which banks a pour rather than filing anything (see test_partial).
    n = 60
    def _o(self): return np.zeros((self.n, 4), np.float32), np.ones((self.n,), np.float32), self.n, 8.0
    def result(self): return self._o(), 1
    def snapshot(self): return FRAME, self._o()
    def inside(self, b): return True
    def stop(self): pass

inf = I()
win = W.Window(C(), inf, target=60, camera=0)

def settle():
    """Let the figure hold still: Done, like Keep, waits for it to settle."""
    win._tick(); win._steady_at -= 1.0; win._tick()

print("--- the row has one button, not two")
check("no undo button on the bench", hasattr(win, "roi_clear"), False)

print("--- set a region in the camera settings, save them, and saving works")
win._setup_open()
win._roi_clicked()
win._tick()
check("region set", win.infer.roi is not None, True)
check("but the count cannot be saved while the settings are open",
      win.save_btn.isEnabled(), False)
win._setup_save()
settle()
check("the button is Done", win._main, "done")
check("saving allowed", win.save_btn.isEnabled(), True)

print("--- press กำหนดกรอบใหม่ and it is refused while the corners may move")
win._roi_clicked()                          # opens the settings by itself
win._tick()
check("inside the settings", win.setup, True)
check("a region is on the picture, so it does not go blank",
      win.infer.roi is not None, True)
check("but saving is refused", win.save_btn.isEnabled(), False)
check("and no pour may be banked either", win._can_round, False)

print("--- and the refusal holds if the button is reached anyway")
before = len(glob.glob(os.path.join(RECORDS, "count_*.json")))
win._save()
check("nothing filed", len(glob.glob(os.path.join(RECORDS, "count_*.json"))), before)

print("--- mid-drag is still refused")
win._corner_moved(0, 30, 30); win._tick()
check("still refused", win.save_btn.isEnabled(), False)
win._corner_dropped(); win._tick()
check("the settings can be saved", win.setup_save_btn.isEnabled(), True)
win._save()
check("the count still cannot", len(glob.glob(os.path.join(RECORDS, "count_*.json"))), before)
win._setup_save(); settle()
check("saving allowed", win.save_btn.isEnabled(), True)
win._save()
check("and it files", len(glob.glob(os.path.join(RECORDS, "count_*.json"))), before + 1)

print("--- cancelling leaves the old region and allows saving again")
old_roi = list(win.infer.roi)
win._roi_clicked()
win._corner_moved(1, 300, 300); win._corner_dropped()
win._tick()
check("refused", win.save_btn.isEnabled(), False)
check("a new region is drawn", win.infer.roi != old_roi, True)
win._setup_cancel()                         # the settings' ยกเลิก
settle()
check("closed", win.setup, False)
check("the old region survived", win.infer.roi, old_roi)
check("saving allowed again", win.save_btn.isEnabled(), True)

shutil.rmtree(S, ignore_errors=True)

# ------------------------------------------------------------------------------- phone
print()
print("--- and the phone does the same")
from model3.phone import screen as screen_mod
from model3.phone.screen import Screen
screen_mod.use(2340, 1080)
REC = tempfile.mkdtemp(prefix="arm_ph_")
sc = Screen(os.path.join(ROOT, "model3", "phone", "assets"), REC, target=60)
F = np.full((480, 640, 3), 40, np.uint8); B = np.zeros((0, 4), np.float32)

def paint(n=60):
    sc._chrome_key = None
    sc.compose(F, B, n, 8.0); sc._steady_at -= 1.0
    sc._chrome_key = None
    return sc.compose(F, B, n, 8.0)

sc.roi = [(0, 0), (639, 0), (639, 479), (0, 479)]
paint()
check("saving allowed", dict(sc.hits)["save"] is not None, True)
check("the button is Done", sc.main_action(), "done")
def save_live():
    paint()
    # ui.button draws a dead control differently; the truth is the flag it was given.
    return not sc.blocked and not sc.setup
check("save is live", save_live(), True)
check("no undo button", "roi-clear" in dict(sc.hits), False)

sc._act("roi", None, None)
check("into the settings", sc.setup, True)
check("a region is there", sc.roi is not None, True)
check("save is dead", save_live(), False)
check("and no pour may be banked", sc.round_block(), "กำลังตั้งค่ากล้อง")
check("the button offers a redraw", "roi" in dict(sc.hits), True)

sc._act("setup-cancel", None, None)
check("save is live again", save_live(), True)

print("--- and the new words have pictures")
def missing(t):
    return [p for p, d in sc.text._runs(t) if sc.text._entry(p, d) is None]
check("กำลังตั้งค่ากล้อง", missing("กำลังตั้งค่ากล้อง"), [])

shutil.rmtree(REC, ignore_errors=True)
print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
