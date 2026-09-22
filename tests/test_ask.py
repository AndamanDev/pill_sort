# -*- coding: utf-8 -*-
"""Twenty set, ten poured and banked, and then the number turns out to be five.

The ten are in a bottle. Five cannot be reached by taking anything off the tray, so
applying the new number quietly leaves a screen that refuses to save and names a remedy
the operator has to find for themselves. There are two ways on and neither is obviously
right -- the number was wrong, or the pour was -- so the screen asks, and discards
nothing until it is answered.
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
REC = tempfile.mkdtemp(prefix="ask_")
sc = Screen(os.path.join(ROOT, "model3", "phone", "assets"), REC, target=20)
sc.roi = [(0, 0), (639, 0), (639, 479), (0, 479)]
F = np.full((480, 640, 3), 40, np.uint8); B = np.zeros((0, 4), np.float32)

def paint(n=0):
    sc._chrome_key = None
    sc.compose(F, B, n, 8.0); sc._steady_at -= 1.0
    sc._chrome_key = None
    return sc.compose(F, B, n, 8.0)

def act(name):
    sc._act(name, None, None)

print("--- set 20, pour 10, bank it")
paint(10); sc.take_round(F); paint(0); paint(0)
check("banked", sc.rounds, [10])
check("target", sc.target, 20)

print("--- ANY change is now a question while a pour is banked, not only an impossible one")
act("preset10")
check("even 10, which the pours can live with, asks", sc.asking, 10)
check("and does not apply yet", sc.target, 20)
act("ask-keep")
check("keeping leaves everything", (sc.target, sc.rounds), (20, [10]))

print("--- a bigger number asks too, and keeps the pours when taken")
sc._want_target(60)
check("asking", sc.asking, 60)
act("ask-apply")
check("applied", sc.target, 60)
check("and the pours are untouched, because 60 is reachable from 10", sc.rounds, [10])

print("--- now ask for 5, which is fewer than are already in the bottle")
sc.target = 20
sc._want_target(5)
check("5 does not apply yet", sc.target, 20)
check("it is asking", sc.asking, 5)

print("--- and while it asks, nothing else on the screen is live")
paint()
names = dict(sc.hits)
check("both answers are offered", ("ask-keep" in names, "ask-apply" in names), (True, True))
before = (sc.target, sc.rounds, sc.page)
sc.touch("down", 40, 40, F.shape); sc.touch("up", 40, 40, F.shape)
check("a tap on the backdrop changes nothing", (sc.target, sc.rounds, sc.page), before)

print("--- ใช้จำนวนเดิม: the pour was right, so nothing moves")
act("ask-keep")
check("no longer asking", sc.asking, None)
check("the old number is kept", sc.target, 20)
check("and the pour survives", sc.rounds, [10])
paint()
check("saving is allowed again", sc.blocked, "")

print("--- เริ่มนับใหม่: the number was right, so the pour goes")
sc._want_target(5)
check("asking again", sc.asking, 5)
act("ask-apply")
check("no longer asking", sc.asking, None)
check("the new number is taken", sc.target, 5)
check("and the pours are gone", sc.rounds, [])
paint()
check("nothing is blocking", sc.blocked, "")

print("--- with nothing banked, changing the number is nobody's business")
sc.rounds = []
for n in (10, 11, 60, 0):
    sc.asking = None
    sc._want_target(n)
    check(f"{n} applies straight away", (sc.target, sc.asking), (n, None))

print("--- pressing the number it is already on is not a change and asks nothing")
sc.rounds = [10]
sc.target = 30
sc._want_target(30)
check("no question", sc.asking, None)

print("--- and เคลียร์ asks too, because dropping the prescription is a change as well")
sc._want_target(0)
check("asking", sc.asking, 0)
act("ask-apply")
check("cleared", sc.target, 0)
check("and the pours survive, because no target cannot be exceeded", sc.rounds, [10])

print("--- the pad sets what was typed, not zero")
sc.rounds = []
sc.typing = "45"
sc._typed("ตกลง")
check("typed 45", sc.target, 45)

print("--- and every word of the question has a picture")
def missing(t):
    return [p for p, d in sc.text._runs(t) if sc.text._entry(p, d) is None]
for phrase in ("เปลี่ยนจำนวน", "เก็บแล้ว 10 เม็ด", "มากกว่าจำนวนใหม่ 5 เม็ด",
               "ใช้จำนวนเดิม 20", "เริ่มนับใหม่", "เปลี่ยนเป็น 60 เม็ด",
               # The two sentences the card gained: why the pours cannot simply be
               # subtracted, and what it costs to insist on the new number.
               "เม็ดที่เก็บแล้วอยู่ในกระปุก ไม่ได้อยู่บนถาด จึงเอาออกไม่ได้",
               "ถ้าจะใช้ 5 ต้องเทกลับลงถาดแล้วเริ่มนับใหม่",
               "เม็ดที่เก็บไว้แล้วยังอยู่ครบ นับต่อได้เลย"):
    check(f"{phrase[:22]}", missing(phrase), [])

print("--- the screen behind the question is plainly asleep, and the tray is not")
sc.asking = None
sc._chrome_key = None
bright = paint(0).copy()
sc.asking = 5
sc._chrome_key = None
dim = paint(0).copy()
# A corner of the header, far from the card in the middle of the screen.
hy, hx = 30, dim.shape[1] - 60
check("the chrome is taken down",
      int(dim[hy, hx].max()) < int(bright[hy, hx].max()) * 0.6, True)
# AND THE VIDEO IS NOT. Dimming the pane would not dim the camera -- what is still the
# sentinel at the end of compose is what becomes transparent, so a darkened magenta stops
# being a hole and the live picture turns into a solid rectangle. The left edge of the
# pane is compared because the card overlaps its middle.
ox, oy, ow, oh = screen_mod.pane_out()
check("and the video is left exactly alone",
      np.array_equal(bright[oy:oy + oh, ox:ox + 40], dim[oy:oy + oh, ox:ox + 40]), True)

shutil.rmtree(REC, ignore_errors=True)

# ------------------------------------------------------------------------------- bench
print()
print("--- the bench asks the same question")
import app as appmod
S = tempfile.mkdtemp(prefix="ask_set_"); appmod.SETTINGS = S
import app.window as W
W.SETTINGS = S; W.RECORDS = tempfile.mkdtemp(prefix="ask_rec_")
from app import sound; sound.enabled = False
from PySide6.QtCore import QTimer
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
    conf = .45; iou = .4; imgsz = 640; error = ""; n = 10
    def _o(self): return np.zeros((self.n, 4), np.float32), np.ones((self.n,), np.float32), self.n, 8.0
    def result(self): return self._o(), 1
    def snapshot(self): return FRAME, self._o()
    def inside(self, b): return True
    def stop(self): pass

inf = I()
win = W.Window(C(), inf, target=20, camera=0)

def tray(n, hold=1.0):
    inf.n = n; win._tick()
    if hold: win._steady_at -= hold; win._tick()

tray(10); win._take_round(); win._tick(); tray(0); tray(0)
check("banked", win.rounds, [10])

seen = {}

def answer(press):
    """Let the question come up for real, then press one of its buttons.

    NOT A STUB. The thing that shipped broken was the WIDGET -- unreadable, then foreign
    to the screen around it -- so a test that replaces it with a lambda tests the one part
    that was never wrong. This drives the card the operator sees: the labels it puts up,
    the buttons it builds, and the signal from the one that gets pressed.
    """
    def go():
        box = win.ask_box
        seen.update(
            title=box.title.text(), lead=box.lead.text(), detail=box.detail.text(),
            visible=not box.isHidden(),
            buttons=[box.row.itemAt(i).widget().text() for i in range(box.row.count())],
            kinds=[box.row.itemAt(i).widget().objectName() for i in range(box.row.count())],
            focus=(box.focusWidget().text()
                   if hasattr(box.focusWidget(), "text") else None))
        box.row.itemAt(press).widget().click()
    QTimer.singleShot(0, go)

print("--- \u0e43\u0e0a\u0e49\u0e08\u0e33\u0e19\u0e27\u0e19\u0e40\u0e14\u0e34\u0e21")
answer(0)
win._set_target(5)
check("it asked", "\u0e21\u0e32\u0e01\u0e01\u0e27\u0e48\u0e32\u0e08\u0e33\u0e19\u0e27\u0e19\u0e43\u0e2b\u0e21\u0e48 5" in seen.get("lead", ""), True)
check("it was on screen, not a stub", seen.get("visible"), True)
check("the quiet coat on the safe answer, the loud one on the change",
      seen.get("kinds"), ["ghost", "warn"])
check("and the safe answer holds the focus, so Enter keeps the pours",
      seen.get("focus"), seen.get("buttons", [None])[0])
check("the old number is kept", win.target, 20)
check("the pour survives", win.rounds, [10])
check("and the box on screen was put back", win.target_edit.text(), "20")
check("nothing is left covering the screen", win.ask_box.isHidden(), True)

print("--- \u0e40\u0e23\u0e34\u0e48\u0e21\u0e19\u0e31\u0e1a\u0e43\u0e2b\u0e21\u0e48")
answer(1)
win._set_target(5)
check("the new number is taken", win.target, 5)
check("and the pours are gone", win.rounds, [])

print("--- a bigger number asks too, and keeps the pours")
seen.clear()
win.rounds = [10]
win.target = 20
answer(1)
win._set_target(60)
check("it asked", "\u0e08\u0e30\u0e40\u0e1b\u0e25\u0e35\u0e48\u0e22\u0e19\u0e08\u0e32\u0e01 20 \u0e40\u0e1b\u0e47\u0e19 60" in seen.get("lead", ""), True)
check("green, not orange: nothing is being thrown away", seen.get("kinds"),
      ["ghost", "primary"])
check("applied", win.target, 60)
check("and the pours are untouched", win.rounds, [10])

print("--- Escape is the safe answer, and a click on the dim does nothing at all")
from PySide6.QtCore import QEvent, QPoint, Qt as _Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
seen.clear()
win.rounds = [10]
win.target = 20
def escape():
    box = win.ask_box
    box.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPoint(5, 5),
                                    _Qt.LeftButton, _Qt.LeftButton, _Qt.NoModifier))
    seen["still_up"] = not box.isHidden()
    box.keyPressEvent(QKeyEvent(QEvent.KeyPress, _Qt.Key_Escape, _Qt.NoModifier))
QTimer.singleShot(0, escape)
win._set_target(5)
check("the click on the backdrop did not answer it", seen.get("still_up"), True)
check("Escape kept the old number", win.target, 20)
check("and the pours", win.rounds, [10])

print("--- with nothing banked there is no question at all")
seen.clear()
win.rounds = []
win._set_target(30)
check("applied straight away, nothing asked", (win.target, seen.get("title")), (30, None))

shutil.rmtree(S, ignore_errors=True)
print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
