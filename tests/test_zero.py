# -*- coding: utf-8 -*-
"""An empty tray files nothing, on either screen.

The window opens at zero and comes back to zero after every sweep, so "nothing on the
tray" is the state this app sits in most of the day -- and it was the one state where the
save button was live and green. A press there wrote a record saying a prescription was
dispensed with no tablets in it, which is worse than no record: it is a row in the book
that has to be explained later.

WHAT MUST NOT BREAK. Zero on the TRAY is not zero to file. Sweep 35 into the bottle and
the tray is empty again while the total is 35, and that is precisely the moment somebody
reaches for the button. The rule is about the TOTAL, never the live count.

The big button is Keep while the total is short and Done only on the exact count, so an
empty tray now shows a dead Keep -- nothing to bank -- and 35 swept into the bottle still
counts: the pour that brings the total to the target turns the button to Done.
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
check("the button is Keep, there being nothing to be Done with", win._main, "keep")
check("but it is dead", win.save_btn.isEnabled(), False)
check("because there is nothing to bank", win._can_round, False)

print("--- and the button being dead is not the only thing stopping it")
before = filed()
win._save()
check("nothing filed", filed(), before)
check("and it says why", win.note, W.NOTHING_NOTE)

print("--- one tablet is enough to make it a real count")
tray(1)
check("Keep, and live", (win._main, win.save_btn.isEnabled()), ("keep", True))
check("orange, because it is short", win.save_btn.objectName(), "warn")
tray(0)
win._set_target(1)
tray(1)
check("asked for one: Done, and live", (win._main, win.save_btn.isEnabled()), ("done", True))
win._main_clicked()
check("filed", filed(), before + 1)
tray(0)
win._set_target(60)

print("--- THE CASE THIS MUST NOT BREAK: swept into the bottle, tray empty, 35 banked")
win._clear_rounds()
tray(35)
win._main_clicked()                     # Keep
win._tick()
tray(0); tray(0)
check("the tray is empty", inf.n, 0)
check("the total is not", win.banked(), 35)
check("short, so Keep -- dead only because the tray is empty", (win._main, win.save_btn.isEnabled()),
      ("keep", False))
# The last pour makes it exact while it is still on the tray, so the button is Done
# there -- the 35 in the bottle counts towards it as fully as the 25 in view.
tray(25)
check("35 banked + 25 on the tray: Done, and live", (win._main, win.save_btn.isEnabled()),
      ("done", True))
win._main_clicked()
check("and it files the 60, not a zero", filed(), before + 2)
import json
rec = json.load(open(max(glob.glob(os.path.join(RECORDS, "count_*.json")),
                        key=os.path.getmtime), encoding="utf-8"))
check("the record says 60", (rec["count"], rec["rounds"]), (60, [35, 25]))

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
check("the button is Keep", sc.main_action(), "keep")
check("and dead: nothing to bank", bool(sc.round_block()), True)

print("--- a tap on the dead button answers instead of swallowing the press")
saved = []
sc._act("save", lambda: saved.append(1), None)
check("nothing was filed", saved, [])
check("nor banked", sc.rounds, [])
check("and it said why", "ถาดว่าง" in sc.note, True)

print("--- ten on the tray, and it is a real count again")
paint(10)
check("Keep, and live", (sc.main_action(), sc.round_block()), ("keep", ""))

print("--- swept away: tray empty, 35 banked; the next 25 makes it Done")
paint(35)
sc._act("save", None, None)             # Keep
paint(0); paint(0)
check("banked", sc.rounds, [35])
check("the tray is empty and the total is not", (sc.live_total(), sc.rounds), (35, [35]))
check("still Keep, dead with an empty tray", (sc.main_action(), bool(sc.round_block())),
      ("keep", True))
paint(25)
check("35 banked + 25 on the tray", sc.live_total(), 60)
check("Done, and live", (sc.main_action(), not sc.blocked and sc.live_total() > 0),
      ("done", True))
saved = []
sc._act("save", lambda: saved.append(1), None)
check("files", saved, [1])

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
check("ถาดว่าง ยังไม่มีอะไรให้เก็บ", missing("ถาดว่าง ยังไม่มีอะไรให้เก็บ"), [])

shutil.rmtree(REC, ignore_errors=True)
print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
