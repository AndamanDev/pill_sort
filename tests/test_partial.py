# -*- coding: utf-8 -*-
"""Filing a total that is SHORT, with pours already banked in it.

A short count on one tray is the stock running out and is filed without ceremony -- the
button already says so in its words and its colour. What is different mid-pour is what a
stray press destroys: the tablets in `rounds` are in a bottle, they cannot be re-counted,
and saving clears the list that is the only record of them. So that case, and only that
case, is asked about first.
"""
import os, sys, json, glob, shutil, tempfile
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
from app import sound; sound.enabled = False
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
qapp = QApplication.instance() or QApplication([])

class C:
    fps = 20.0
    def __init__(self): self.frame = np.zeros((480,640,3), np.uint8)
    def latest(self): return self.frame, 1
    def stale(self): return 0.0
    def stop(self): pass
class I:
    roi=None; conf=.45; iou=.4; imgsz=640; error=""
    def __init__(self, c): self.c=c; self.n=0
    def _o(self): return np.zeros((self.n,4),np.float32), np.ones((self.n,),np.float32), self.n, 42.0
    def result(self): return self._o(), 1
    def snapshot(self): return self.c.frame, self._o()
    def inside(self,b): return True
    def stop(self): pass

cap = C(); inf = I(cap)
win = W.Window(cap, inf, target=60, camera=0)

FAIL=[]
def check(l,g,w):
    ok = g==w
    print(("  ok  " if ok else "  FAIL")+f"  {l}: {g!r}"+("" if ok else f"  (want {w!r})"))
    if not ok: FAIL.append(l)

def tray(n, hold=1.0):
    inf.n = n; win._tick()
    if hold: win._steady_at -= hold; win._tick()

# the dialog is replaced by a recorded answer, so the test never blocks on a modal
asked = []
ANSWER = {"go": True}
def fake_confirm(total, rounds):
    if not (win.rounds and win.target and total < win.target):
        return True
    asked.append((total, list(rounds)))
    return ANSWER["go"]
win._confirm_partial = fake_confirm

print("--- with no region drawn, nothing may be counted")
win._tick()
check("the number is a dash", win.count_lbl.text(), "—")
check("the verdict says which thing is missing", win.verdict.text(),
      "ยังไม่ได้กำหนดกรอบนับ")
check("and saving is refused", win.save_btn.isEnabled(), False)

# Everything below is about counting, so give it a tray to count in: the whole frame,
# because what is being tested here is the pours, not the geometry.
inf.roi = [(0, 0), (639, 0), (639, 479), (0, 479)]

print("--- short on one tray: the button says so, and does NOT ask")
tray(47)
check("button words", win.save_btn.text(), "บันทึกว่าไม่ครบ (ขาด 13)")
check("button skin", win.save_btn.objectName(), "warn")
check("still pressable", win.save_btn.isEnabled(), True)
fm = win.save_btn.fontMetrics()
check("words fit the button", fm.horizontalAdvance(win.save_btn.text()) < win.save_btn.width() - 24, True)
win._save()
check("no question asked", asked, [])
rec = json.load(open(max(glob.glob(os.path.join(RECORDS,"count_*.json")), key=os.path.getmtime), encoding="utf-8"))
check("filed", rec["count"], 47)
check("flagged short", rec["short"], True)
check("difference", rec["difference"], -13)

print("--- complete: the button goes back to green")
tray(60)
check("button words", win.save_btn.text(), "บันทึกผล")
check("button skin", win.save_btn.objectName(), "primary")
win._save(); win._tick()
rec = json.load(open(max(glob.glob(os.path.join(RECORDS,"count_*.json")), key=os.path.getmtime), encoding="utf-8"))
check("not short", rec["short"], False)

print("--- mid-pour, answered 'go back'")
tray(35); win._take_round(); win._tick(); tray(0); tray(0)
ANSWER["go"] = False
before = len(glob.glob(os.path.join(RECORDS,"count_*.json")))
win._save(); win._tick()
check("it asked", asked, [(35, [35])])
check("nothing filed", len(glob.glob(os.path.join(RECORDS,"count_*.json"))), before)
check("the 35 is still there", win.rounds, [35])
check("note", win.note, "ยกเลิกการบันทึก")

print("--- ...then the second pour completes it, no question")
asked.clear()
tray(25)
check("total", win.count_lbl.text(), "60")
check("button green again", win.save_btn.objectName(), "primary")
win._save(); win._tick()
check("no question", asked, [])
rec = json.load(open(max(glob.glob(os.path.join(RECORDS,"count_*.json")), key=os.path.getmtime), encoding="utf-8"))
check("filed", (rec["count"], rec["rounds"], rec["short"]), (60, [35,25], False))
check("rounds cleared", win.rounds, [])

print("--- mid-pour, answered 'save anyway'")
tray(35); win._take_round(); win._tick(); tray(0); tray(0)
ANSWER["go"] = True; asked.clear()
win._save(); win._tick()
check("it asked", asked, [(35, [35])])
rec = json.load(open(max(glob.glob(os.path.join(RECORDS,"count_*.json")), key=os.path.getmtime), encoding="utf-8"))
check("filed", (rec["count"], rec["rounds"], rec["short"]), (35, [35], True))
check("rounds cleared", win.rounds, [])

print("--- no target set: nothing is filed, so nothing is asked either")
# It used to file, with `short` false. See tests/test_no_target.py for why it no longer
# does; what matters here is that the question about a short save is not raised for a
# save that is not going to happen.
win._set_target(0); asked.clear()
tray(12); win._take_round(); win._tick(); tray(0); tray(0)
before = len(glob.glob(os.path.join(RECORDS, "count_*.json")))
win._save()
check("no question", asked, [])
check("and nothing filed", len(glob.glob(os.path.join(RECORDS, "count_*.json"))), before)
check("the pour survives", win.rounds, [12])

print("--- the real dialog builds, with the safe answer as the default")
# CLEARED FIRST, THEN SET. The other way round asks -- the target does not move
# while a pour is banked without a question -- and this file drives the partial-save
# dialog itself, so a second modal in the same run answers the wrong one.
win._clear_rounds(); win._set_target(60)
tray(35); win._take_round(); win._tick(); tray(0); tray(0)
del win._confirm_partial
import app.window as WW
seen = {}

def look_then_escape():
    """Read the real card, then leave it the way a stray Escape would."""
    box = win.ask_box
    seen.update(
        title=box.title.text(), lead=box.lead.text(), detail=box.detail.text(),
        visible=not box.isHidden(),
        buttons=[box.row.itemAt(i).widget().text() for i in range(box.row.count())],
        kinds=[box.row.itemAt(i).widget().objectName() for i in range(box.row.count())],
        focus=(box.focusWidget().text()
               if hasattr(box.focusWidget(), "text") else None))
    box.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
QTimer.singleShot(0, look_then_escape)
win._save()
check("it was on screen", seen.get("visible"), True)
check("the safe answer is first and holds the focus",
      (seen.get("buttons") or [None])[0], seen.get("focus"))
check("buttons", seen.get("buttons"), ["\u0e01\u0e25\u0e31\u0e1a\u0e44\u0e1b\u0e40\u0e17\u0e15\u0e48\u0e2d", "\u0e1a\u0e31\u0e19\u0e17\u0e36\u0e01 35 \u0e40\u0e21\u0e47\u0e14"])
check("the loud coat is on the one that files", seen.get("kinds"), ["ghost", "warn"])
check("the red line is the shortfall, not the sum",
      seen.get("lead"), "\u0e02\u0e32\u0e14\u0e2d\u0e35\u0e01 25 \u0e40\u0e21\u0e47\u0e14 \u0e08\u0e32\u0e01\u0e17\u0e35\u0e48\u0e15\u0e49\u0e2d\u0e07\u0e01\u0e32\u0e23 60 \u0e40\u0e21\u0e47\u0e14")
check("one pour does not get told 35 = 35",
      seen.get("detail", "").splitlines()[0],
      "\u0e40\u0e01\u0e47\u0e1a\u0e41\u0e25\u0e49\u0e27 1 \u0e23\u0e2d\u0e1a = 35 \u0e40\u0e21\u0e47\u0e14")
check("body warns about the loss", "\u0e22\u0e2d\u0e14\u0e2a\u0e30\u0e2a\u0e21\u0e08\u0e30\u0e16\u0e39\u0e01\u0e25\u0e49\u0e32\u0e07" in seen.get("detail", ""), True)
check("dismissing it filed nothing", win.rounds, [35])

shutil.rmtree(RECORDS, ignore_errors=True)
print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
