# -*- coding: utf-8 -*-
"""An empty tray files nothing, on either screen.

The window opens at zero and comes back to zero after every sweep, so "nothing on the
tray" is the state this app sits in most of the day -- and it was the one state where the
save button was live and green. A press there wrote a record saying a prescription was
dispensed with no tablets in it, which is worse than no record: it is a row in the book
that has to be explained later.

WHAT MUST NOT BREAK. Zero on the TRAY is not zero to file. Sweep 35 into the bottle and
the tray is empty again while the total is 35, and that is precisely the moment somebody
reaches for save. The rule is about the TOTAL, never the live count.
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
S = tempfile.mkdtemp(prefix="zero_set_"); appmod.SETTINGS = S
import app.window as W
W.SETTINGS = S
RECORDS = tempfile.mkdtemp(prefix="zero_rec_"); W.RECORDS = RECORDS
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
    conf = .45; iou = .4; imgsz = 640; error = ""; n = 0
    def _o(self): return np.zeros((self.n, 4), np.float32), np.ones((self.n,), np.float32), self.n, 8.0
    def result(self): return self._o(), 1
    def snapshot(self): return FRAME, self._o()
    def inside(self, b): return True
    def stop(self): pass

inf = I()
win = W.Window(C(), inf, target=60, camera=0)

def tray(n, hold=1.0):
    inf.n = n; win._tick()
    if hold: win._steady_at -= hold; win._tick()

def filed():
    return len(glob.glob(os.path.join(RECORDS, "count_*.json")))

print("--- the screen as it is found: region set, tray empty")
tray(0)
check("nothing is blocking, this is not a fault", win._block, "")
check("but the save is dead", win.save_btn.isEnabled(), False)
check("and it is not wearing the short-count coat", win.save_btn.objectName(), "primary")
check("nor shouting about a shortfall", win.save_btn.text(), "บันทึกผล")

print("--- and the button being dead is not the only thing stopping it")
before = filed()
win._save()
check("nothing filed", filed(), before)
check("and it says why", win.note, W.NOTHING_NOTE)

print("--- one tablet is enough to make it a real count")
tray(1)
check("live", win.save_btn.isEnabled(), True)
check("short, so it says so", win.save_btn.objectName(), "warn")
win._save()
check("filed", filed(), before + 1)

print("--- THE CASE THIS MUST NOT BREAK: swept into the bottle, tray empty, 35 banked")
win._clear_rounds()
tray(35)
win._take_round()
win._tick()
tray(0); tray(0)
check("the tray is empty", inf.n, 0)
check("the total is not", win.banked(), 35)
check("so the save is live", win.save_btn.isEnabled(), True)
win._confirm_partial = lambda total, rounds: True       # its own suite covers the question
win._save()
check("and it files the 35, not a zero", filed(), before + 2)
import json
rec = json.load(open(max(glob.glob(os.path.join(RECORDS, "count_*.json")),
                        key=os.path.getmtime), encoding="utf-8"))
check("the record says 35", rec["count"], 35)

print("--- with no region there is a block, and the save is dead for that reason instead")
win._clear_rounds()
inf.roi = None
tray(0)
check("blocked", win._block, W.NO_ROI_NOTE)
check("dead", win.save_btn.isEnabled(), False)
inf.roi = [(0, 0), (639, 0), (639, 479), (0, 479)]

shutil.rmtree(S, ignore_errors=True)
shutil.rmtree(RECORDS, ignore_errors=True)

# -------------------------------------------------------------------------------- phone
print()
print("--- and the phone does the same")
from model3.phone import screen as screen_mod
from model3.phone.screen import Screen
screen_mod.use(2340, 1080)
REC = tempfile.mkdtemp(prefix="zero_ph_")
sc = Screen(os.path.join(ROOT, "model3", "phone", "assets"), REC, target=60)
sc.roi = [(0, 0), (639, 0), (639, 479), (0, 479)]
F = np.full((480, 640, 3), 40, np.uint8); B = np.zeros((0, 4), np.float32)

def paint(n=0):
    sc._chrome_key = None
    sc.compose(F, B, n, 8.0); sc._steady_at -= 1.0
    sc._chrome_key = None
    return sc.compose(F, B, n, 8.0)

paint(0)
check("nothing is blocking", sc.blocked, "")
check("the save is dead", sc.live_total() > 0, False)
check("and it keeps the plain coat", sc._save_look(), "primary")

print("--- a tap on the dead button answers instead of swallowing the press")
saved = []
sc._act("save", lambda: saved.append(1), None)
check("nothing was filed", saved, [])
check("and it said why", "ยังไม่มีเม็ดยาให้บันทึก" in sc.note, True)

print("--- ten on the tray, and it is a real count again")
paint(10)
check("live", sc.live_total() > 0, True)
check("short, so it says so", sc._save_look(), "warn")

print("--- swept away: tray empty, 35 banked, still saveable")
paint(35)
sc.take_round(F)
paint(0); paint(0)
check("banked", sc.rounds, [35])
check("the tray is empty and the total is not", (sc.live_total(), sc.rounds), (35, [35]))
check("live", sc.live_total() > 0, True)

print("--- and the furniture is redrawn when the total crosses zero")
# THE RULE HAS TO BE IN THE MEMO KEY OR NOBODY EVER SEES IT. The buttons are drawn once
# into a template and copied until this key changes; the save went dead at zero and the
# screen went on showing the green one it had drawn while the tray still had tablets on
# it. With no target set the button's LOOK is "primary" either way, so nothing else in
# the key moved. Found on the board, at 0 tablets, under a green button.
sc.rounds = []
sc.target = 0

def flicker(n):
    """A count that has JUST changed -- the tray is being moved, nothing has settled.

    Which is the state the bug lived in. Steady, the key moves anyway because the round
    button comes to life; mid-movement every other element is identical at 0 and at 7,
    so this is the pair that isolates the one thing being tested.
    """
    sc._chrome_key = None
    sc.compose(F, B, n, 8.0)
    return sc.chrome_key()

empty = flicker(0)
full = flicker(7)
check("the key moved with the total", empty != full, True)
check("and with nothing else", sum(a != b for a, b in zip(empty, full)), 1)
check("and moved back", flicker(0), empty)

print("--- and the new words have pictures")
def missing(t):
    return [p for p, d in sc.text._runs(t) if sc.text._entry(p, d) is None]
check("ยังไม่มีเม็ดยาให้บันทึก วางยาบนถาดก่อน",
      missing("ยังไม่มีเม็ดยาให้บันทึก วางยาบนถาดก่อน"), [])

shutil.rmtree(REC, ignore_errors=True)
print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
