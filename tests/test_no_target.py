# -*- coding: utf-8 -*-
"""No prescription, no record.

A count with no target is a count nobody can check. The record carries eight tablets and
nothing to compare them against, so the one question anybody opens it to ask -- was this
dispensed correctly -- has no answer in it and never will.

THE SCREEN HAS BEEN SAYING SO ALL ALONG. The verdict badge reads "ยังไม่กำหนดจำนวน" in
that state; what it did not do was stop the save, so the green button sat under the badge
contradicting it. This is the third of the three things that have to be true before a
count may be filed -- a region (test_arming), tablets (test_zero), and a number asked for.
"""
import os, sys, tempfile, shutil, glob
os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                          # app/ and model3/ read relative paths
import numpy as np

FAIL = []
def check(label, got, want):
    ok = got == want
    print(("  ok  " if ok else "  FAIL") + f"  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok: FAIL.append(label)

# -------------------------------------------------------------------------------- bench
import app as appmod
S = tempfile.mkdtemp(prefix="notgt_set_"); appmod.SETTINGS = S
import app.window as W
W.SETTINGS = S
RECORDS = tempfile.mkdtemp(prefix="notgt_rec_"); W.RECORDS = RECORDS
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
    roi = [(0, 0), (639, 0), (639, 479), (0, 479)]
    conf = .45; iou = .4; imgsz = 640; error = ""; n = 8
    def _o(self): return np.zeros((self.n, 4), np.float32), np.ones((self.n,), np.float32), self.n, 8.0
    def result(self): return self._o(), 1
    def snapshot(self): return FRAME, self._o()
    def inside(self, b): return True
    def stop(self): pass

inf = I()
win = W.Window(C(), inf, target=0, camera=0)     # opened without a prescription

def tray(n, hold=1.0):
    inf.n = n; win._tick()
    if hold: win._steady_at -= hold; win._tick()

def filed():
    return len(glob.glob(os.path.join(RECORDS, "count_*.json")))

print("--- eight tablets on the tray, and no number asked for")
tray(8)
check("the count is real", win.banked() + 8, 8)
check("nothing is blocking, this is not a fault", win._block, "")
check("the badge says so", win.verdict.text(), "ยังไม่กำหนดจำนวน")
check("and the save agrees with the badge", win.save_btn.isEnabled(), False)

print("--- and the refusal holds if the button is reached anyway")
before = filed()
win._save()
check("nothing filed", filed(), before)
check("and it says why", win.note, W.NO_TARGET_NOTE)

print("--- ask for a number and it comes alive")
win._set_target(10)
win._tick()
check("live", win.save_btn.isEnabled(), True)
win._save()
check("filed", filed(), before + 1)

print("--- เคลียร์ puts it back to dead, not to a silent zero")
win._set_target(0)
win._tick()
check("dead again", win.save_btn.isEnabled(), False)

shutil.rmtree(S, ignore_errors=True)
shutil.rmtree(RECORDS, ignore_errors=True)

# -------------------------------------------------------------------------------- phone
print()
print("--- and the phone does the same")
from model3.phone import screen as screen_mod
from model3.phone.screen import Screen
screen_mod.use(2340, 1080)
REC = tempfile.mkdtemp(prefix="notgt_ph_")
sc = Screen(os.path.join(ROOT, "model3", "phone", "assets"), REC, target=0)
sc.roi = [(0, 0), (639, 0), (639, 479), (0, 479)]
F = np.full((480, 640, 3), 40, np.uint8); B = np.zeros((0, 4), np.float32)

def paint(n=0):
    sc._chrome_key = None
    sc.compose(F, B, n, 8.0); sc._steady_at -= 1.0
    sc._chrome_key = None
    return sc.compose(F, B, n, 8.0)

def savable():
    return (not sc.blocked and not sc.arming and sc.live_total() > 0 and bool(sc.target))

paint(8)
check("eight on the tray", sc.live_total(), 8)
check("but no number asked for", sc.target, 0)
check("so the save is dead", savable(), False)

print("--- a tap on the dead button answers instead of swallowing the press")
saved = []
sc._act("save", lambda: saved.append(1), None)
check("nothing was filed", saved, [])
check("and it said why", "กำหนดจำนวนก่อนจึงบันทึกได้" in sc.note, True)

print("--- ask for a number and it comes alive")
sc._want_target(10)
paint(8)
check("live", savable(), True)
sc._act("save", lambda: saved.append(1), None)
check("filed", saved, [1])

print("--- and the furniture is redrawn when the number arrives")
# self.target is already in chrome_key -- asserted here so that stays true, because the
# same omission for the TOTAL left a dead button drawn green on the board this morning.
sc.rounds = []
sc._want_target(0)
paint(8)
empty = sc.chrome_key()
sc._want_target(10)
paint(8)
check("the key moved with the number", sc.chrome_key() != empty, True)

print("--- and the words have pictures")
def missing(t):
    return [p for p, d in sc.text._runs(t) if sc.text._entry(p, d) is None]
check("ยังไม่กำหนดจำนวน กำหนดจำนวนก่อนจึงบันทึกได้",
      missing("ยังไม่กำหนดจำนวน กำหนดจำนวนก่อนจึงบันทึกได้"), [])

shutil.rmtree(REC, ignore_errors=True)
print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
