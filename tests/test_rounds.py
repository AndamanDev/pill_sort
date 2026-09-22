"""Drive the window's round logic headless: fake camera, fake detector, real Window."""
import os, sys, time, json, glob, shutil, tempfile
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
from app import sound
sound.enabled = False

from PySide6.QtWidgets import QApplication
qapp = QApplication.instance() or QApplication([])


class FakeCap:
    def __init__(self):
        self.frame = np.zeros((480, 640, 3), np.uint8)
        self.fps = 20.0
    def latest(self): return self.frame, 1
    def stale(self): return 0.0
    def stop(self): pass


class FakeInfer:
    roi = None
    conf, iou, imgsz = 0.45, 0.4, 640
    error = ""
    def __init__(self, cap):
        self.cap = cap
        self.n = 0
    def _out(self):
        boxes = np.zeros((self.n, 4), np.float32)
        return boxes, np.ones((self.n,), np.float32), self.n, 42.0
    def result(self): return self._out(), 1
    def snapshot(self): return self.cap.frame, self._out()
    def inside(self, b): return True
    def stop(self): pass


cap = FakeCap()
inf = FakeInfer(cap)
win = W.Window(cap, inf, target=60, camera=0)

FAIL = []
def check(label, got, want):
    ok = got == want
    print(("  ok  " if ok else "  FAIL") + f"  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok: FAIL.append(label)

def tray(n, hold=1.0):
    """Put n pills on the tray and let the screen settle for `hold` seconds of clock."""
    inf.n = n
    win._tick()
    if hold:
        win._steady_at -= hold          # fast-forward the stillness timer
        win._tick()

print("--- with no region drawn, nothing may be counted")
win._tick()
check("the number is a dash", win.count_lbl.text(), "—")
check("the verdict says which thing is missing", win.verdict.text(),
      "ยังไม่ได้กำหนดกรอบนับ")
check("and saving is refused", win.save_btn.isEnabled(), False)

# Everything below is about counting, so give it a tray to count in: the whole frame,
# because what is being tested here is the pours, not the geometry.
inf.roi = [(0, 0), (639, 0), (639, 479), (0, 479)]

print("--- pour 1: 35 on the tray, target 60")
tray(35)
check("count shown", win.count_lbl.text(), "35")
check("verdict", win.verdict.text(), "ขาด 25 เม็ด")
check("round button live", win.round_btn.isEnabled(), True)
check("start-over dead with nothing banked", win.reset_btn.isEnabled(), False)

print("--- the number is still moving: no round may be taken")
inf.n = 34; win._tick()
check("unsettled blocks round", win.round_btn.isEnabled(), False)
check("why", win._round_block, "ตัวเลขยังไม่นิ่ง  รอสักครู่")
tray(35)
check("settled again", win.round_btn.isEnabled(), True)

print("--- take round 1")
win._take_round()
win._tick()
check("rounds", win.rounds, [35])
check("total holds at 35 with 35 still on the tray", win.count_lbl.text(), "35")
check("second press refused", win.round_btn.isEnabled(), False)
win._take_round(); win._tick()
check("and refused in the method too", win.rounds, [35])

print("--- sweep: the tray empties")
tray(0)
check("still 35 after the sweep", win.count_lbl.text(), "35")
check("clearing over", win.clearing, False)
check("ready for the next pour", win.round_btn.isEnabled(), False)   # empty tray

print("--- pour 2: 25 more")
tray(25)
check("total", win.count_lbl.text(), "60")
check("verdict", win.verdict.text(), "ครบตามจำนวน")
check("strip", win.rounds_lbl.text(), "เก็บแล้ว 1 รอบ  รวม 35   +   ในถาด 25")
check("save allowed", win.save_btn.isEnabled(), True)

print("--- one too many")
tray(26)
check("total", win.count_lbl.text(), "61")
check("save refused", win.save_btn.isEnabled(), False)
check("round refused", win.round_btn.isEnabled(), False)
tray(25)
check("save allowed again", win.save_btn.isEnabled(), True)

print("--- save")
win._save()
win._tick()
recs = glob.glob(os.path.join(RECORDS, "count_*.json"))
check("one record", len(recs), 1)
rec = json.load(open(recs[0], encoding="utf-8"))
check("count is the total", rec["count"], 60)
check("rounds", rec["rounds"], [35, 25])
check("difference", rec["difference"], 0)
check("round image kept", rec.get("round_images"), ["count_" + rec["time"][:0] + os.path.basename(recs[0])[6:-5] + "_r1.jpg"])
check("rounds cleared after save", win.rounds, [])
check("strip gone", win.rounds_lbl.text(), "")

print("--- starting over throws the whole total away, and takes two presses")
tray(35); win._take_round(); win._tick()
tray(0); tray(20)
check("total before", win.count_lbl.text(), "55")
win._reset_clicked(); win._tick()
check("the first press only arms it", win.rounds, [35])
check("and says what it will cost", "จะทิ้งยอดสะสม" in win.note, True)
check("the button says so too", win.reset_btn.text(), "กดอีกครั้ง")
win._reset_clicked(); win._tick()
check("the second press does it", win.rounds, [])
check("total after", win.count_lbl.text(), "20")
check("the button goes back", win.reset_btn.text(), "นับใหม่")
check("and goes dead with nothing to discard", win.reset_btn.isEnabled(), False)

print("--- the arming lapses, so a press a minute later is a fresh first press")
tray(35); win._take_round(); win._tick(); tray(0); tray(0)
win._reset_clicked()
win._reset_armed_at -= W.CONFIRM_S + 1
win._reset_clicked(); win._tick()
check("it arms again rather than wiping", win.rounds, [35])

print("--- it also releases a tray that was never swept")
win._clear_rounds()
tray(35); win._take_round(); win._tick()
check("clearing", win.clearing, True)
check("the button is live mid-sweep", win.reset_btn.isEnabled(), True)
win._reset_clicked(); win._reset_clicked(); win._tick()
check("clearing cancelled", win.clearing, False)
check("tray counts again", win.count_lbl.text(), "35")

print("--- single pour still behaves exactly as before")
win._clear_rounds(); tray(60)
check("total", win.count_lbl.text(), "60")
check("no strip", win.rounds_lbl.isVisible(), False)
win._save()
# newest by mtime: two saves inside one second are count_X.json and count_X-2.json,
# and "-" sorts before "." so the name order is the reverse of the write order.
newest = max(glob.glob(os.path.join(RECORDS, "count_*.json")), key=os.path.getmtime)
rec = json.load(open(newest, encoding="utf-8"))
check("count", rec["count"], 60)
check("rounds", rec["rounds"], [60])
check("no extra images", "round_images" in rec, False)

print("--- records_view reads both eras")
from app import records_view as RV
check("old record, no rounds key", RV.rounds_text({"count": 60}), "")
check("single pour", RV.rounds_text({"rounds": [60]}), "")
check("two pours", RV.rounds_text({"rounds": [35, 25]}), "35 + 25")
rows = RV.load_records(RECORDS)
check("both listed", len(rows), 2)
check("round_paths resolved", sum(len(r["round_paths"]) for r in rows), 1)

shutil.rmtree(RECORDS, ignore_errors=True)
print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
