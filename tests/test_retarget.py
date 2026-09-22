# -*- coding: utf-8 -*-
"""Changing the prescription after a pour has already gone into the bottle.

The ordinary case is that somebody corrects the number and carries on, and nothing that
has been counted should be lost by it. The awkward case is a number LOWER than what is
already banked -- because then the excess is not on the tray, it is in the bottle, and a
screen that says "take some out" is naming a remedy nobody can carry out.
"""
import os, sys, tempfile, shutil
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

# ------------------------------------------------------------------------------- phone
from model3.phone import screen as screen_mod
from model3.phone.screen import Screen
screen_mod.use(2340, 1080)
REC = tempfile.mkdtemp(prefix="rt_")
sc = Screen(os.path.join(ROOT, "model3", "phone", "assets"), REC, target=60)
sc.roi = [(0, 0), (639, 0), (639, 479), (0, 479)]
F = np.full((480, 640, 3), 40, np.uint8); B = np.zeros((0, 4), np.float32)

def paint(n):
    sc.compose(F, B, n, 8.0); sc._steady_at -= 1.0
    return sc.compose(F, B, n, 8.0)

print("--- bank a pour of 35 against a prescription of 60")
paint(35); sc.take_round(F); paint(0); paint(0)
check("banked", sc.rounds, [35])

print("--- raising the number keeps everything and just asks for more")
sc.target = 100
paint(0)
check("the pours survive", sc.rounds, [35])
check("the total is untouched", sc.total(0), 35)
check("and it says how many are left", sc._verdict(sc.total(0))[1], "ขาด 65 เม็ด")
check("saving is allowed", sc.blocked, "")

print("--- lowering it to something still above what is banked is fine too")
sc.target = 40
paint(0)
check("no block", sc.blocked, "")
check("pours kept", sc.rounds, [35])

print("--- lowering it BELOW what is banked is the awkward one")
sc.target = 30
paint(0)
check("saving is refused", bool(sc.blocked), True)
check("it does not ask for tablets off an empty tray",
      "นำออกก่อน" in sc.blocked, False)
check("it names how many are already in the bottle", "35" in sc.blocked, True)
check("and points at the control that can fix it", "นับใหม่" in sc.blocked, True)
print(f"       {sc.blocked}")

print("--- and that control does fix it")
sc.reset_rounds(); sc.reset_rounds()          # two taps, as the guard requires
paint(0)
check("nothing banked", sc.rounds, [])
check("and saving is free again", sc.blocked, "")

print("--- while too many on the TRAY still says to take some off")
sc.target = 30
paint(35)
check("blocked", bool(sc.blocked), True)
check("and this time that is the right instruction",
      "นำออกก่อน" in sc.blocked, True)

print("--- every word of both messages has a picture in the atlas")
def missing(t):
    return [p for p, d in sc.text._runs(t) if sc.text._entry(p, d) is None]
for phrase in ("เก็บแล้ว 35 เม็ด เกินจำนวนที่ตั้งไว้ กดนับใหม่",
               "เกินจำนวนที่ต้องการ นำออกก่อนจึงบันทึกได้"):
    check(f"{phrase[:26]}...", missing(phrase), [])

shutil.rmtree(REC, ignore_errors=True)

# ------------------------------------------------------------------------------- bench
print()
print("--- the bench says the same two things, and makes the same noise choices")
import app as appmod
S = tempfile.mkdtemp(prefix="rt_set_"); appmod.SETTINGS = S
import app.window as W
W.SETTINGS = S; W.RECORDS = tempfile.mkdtemp(prefix="rt_rec_")
from app import sound
beeps = []
sound.problem = lambda: beeps.append("problem")
sound.saved = lambda: None
sound.round_taken = lambda: None
from PySide6.QtWidgets import QApplication

# THE TARGET NOW ASKS BEFORE IT MOVES, whenever a pour is banked -- and a modal that
# nobody answers hangs a test for ever, which is exactly how this one was found. Answered
# here the way a person would, taking the change.
W.Ask.ask = lambda self, title, lead, detail, buttons, safe=0, lead_colour=None: 1
QApplication.instance() or QApplication([])

FRAME = np.zeros((480, 640, 3), np.uint8)
class C:
    fps = 20.0
    def latest(self): return FRAME, 1
    def stale(self): return 0.0
    def stop(self): pass
class I:
    roi = None; conf = .45; iou = .4; imgsz = 640; error = ""
    n = 0
    def _o(self): return np.zeros((self.n, 4), np.float32), np.ones((self.n,), np.float32), self.n, 8.0
    def result(self): return self._o(), 1
    def snapshot(self): return FRAME, self._o()
    def inside(self, b): return True
    def stop(self): pass

inf = I()
win = W.Window(C(), inf, target=60, camera=0)

print("--- opening with no region is not a fault and must not make a noise")
win._tick(); win._tick()
check("blocked", win._block, W.NO_ROI_NOTE)
check("kind", win._block_kind, "setup")
check("and it stayed quiet", beeps, [])

inf.roi = [(0, 0), (639, 0), (639, 479), (0, 479)]

def tray(n, hold=1.0):
    inf.n = n; win._tick()
    if hold: win._steady_at -= hold; win._tick()

tray(35); win._take_round(); win._tick(); tray(0); tray(0)
check("banked", win.rounds, [35])

# FORCED, because pressing things can no longer get here -- and that is the point.
# Taking a pour is refused once the total is over the target, and changing the target
# below what is banked now asks first and clears the pours if it is taken. So the block
# below is a guard on a state the UI has stopped producing, kept and tested because a
# guard that is never exercised is a guard nobody knows is broken.
win.target = 30
win._tick()
check("refused", bool(win._block), True)
check("not asking for an empty tray to be emptied", "นำออกก่อน" in win._block, False)
check("names the 35", "35" in win._block, True)
check("orange, not red", win._block_kind, "over")
check("and still no noise", beeps, [])

# The pours have to go first, or the excess is still in the bottle and the message is
# rightly the other one -- which is exactly what the two cases are being told apart by.
win._clear_rounds()
win._set_target(30); tray(35)
win._tick()
check("tablets on the tray get the other message", "นำออกก่อน" in win._block, True)

print("--- a real fault still makes the noise it is there for")
inf.error = "OrtException: boom"
win._tick()
check("kind", win._block_kind, "fault")
check("it made one", beeps, ["problem"])

shutil.rmtree(S, ignore_errors=True)
print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
