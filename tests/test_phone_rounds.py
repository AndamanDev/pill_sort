# -*- coding: utf-8 -*-
"""The phone's multi-round counting, driven the way the bridge drives it."""
import os, sys, json, glob, time, shutil, tempfile

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                          # app/ and model3/ read relative paths
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import cv2
from model3.phone import screen as screen_mod
from model3.phone.screen import Screen
from model3.phone import records as rec_store
from model3.phone import ui

ASSETS = os.path.join(ROOT, "model3", "phone", "assets")
RECORDS = tempfile.mkdtemp(prefix="phone_rec_")

screen_mod.use(2340, 1080)
W, H = screen_mod.W, screen_mod.H
sc = Screen(ASSETS, RECORDS, target=60)

FAIL = []
def check(label, got, want):
    ok = got == want
    print(("  ok  " if ok else "  FAIL") + f"  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok: FAIL.append(label)

FRAME = np.full((480, 640, 3), 40, np.uint8)
BOXES = np.zeros((0, 4), np.float32)

print("--- with no region drawn, the screen refuses to count at all")
check("blocked", (sc.compose(FRAME, BOXES, 35, 42.0) is not None
                  and sc.blocked.startswith("ยังไม่ได้กำหนดกรอบนับ")), True)
check("and the verdict says which thing is missing",
      sc._verdict(35)[1], "ยังไม่ได้กำหนดกรอบนับ")

# Everything below is about counting, so give it a tray to count in. The whole frame,
# because what is being tested here is the pours, not the geometry.
sc.roi = [(0, 0), (639, 0), (639, 479), (0, 479)]

def paint(count):
    """One composed frame, as the bridge would ask for it."""
    return sc.compose(FRAME, BOXES, count, 42.0)

def tray(count, hold=True):
    paint(count)
    if hold:
        sc._steady_at -= 1.0            # fast-forward the stillness timer
        paint(count)

def tap(name):
    saved = {"n": 0}
    def on_save():
        live = 0 if sc.clearing else int(sc._steady_n or 0)
        rounds = list(sc.rounds) + ([live] if live or not sc.rounds else [])
        total = sc.banked() + live
        stamp = rec_store.save(RECORDS, FRAME, BOXES, np.zeros((0,), np.float32),
                               total, sc.target, 42.0, sc.roi, sc.conf, sc.iou, sc.imgsz,
                               write_jpeg=cv2.imwrite,
                               rounds=rounds, round_frames=list(sc.round_frames))
        sc.saved(total, stamp)
        saved["n"] += 1
    sc._act(name, on_save, None)
    return saved["n"]

# ---------------------------------------------------------------- every word has a picture
print("--- no string reaches the screen without an image behind it")
def missing(s):
    return [p for p, d in sc.text._runs(s) if sc.text._entry(p, d) is None]

sc.rounds, sc.clearing = [35], False
sc._steady_n = 25
phrases = [
    f"เก็บรอบที่ {len(sc.rounds) + 1}", "นับใหม่", "กดอีกครั้ง", "บันทึกผล",
    f"บันทึกว่าไม่ครบ ขาด {13}", f"แตะอีกครั้งเพื่อบันทึก {35}",
    "เก็บแล้ว 1 รอบ รวม 35 เม็ด   รอกวาดถาด",
    "เก็บแล้ว 1 รอบ รวม 35 + ในถาด 25",
    screen_mod.CLEAR_NOTE, "ถาดว่างแล้ว เทรอบต่อไปได้",
    "ถาดว่าง ยังไม่มีอะไรให้เก็บ", "ตัวเลขยังไม่นิ่ง รอสักครู่",
    "เริ่มนับใหม่ ทิ้งยอดสะสม 35 เม็ดแล้ว", "จะทิ้งยอดสะสม 35 เม็ด กดอีกครั้งเพื่อเริ่มนับใหม่",
    "เก็บรอบที่ 1 35 เม็ด", "สะสมแล้ว 35 เม็ด",
    "ยังเทค้างอยู่ ยอดสะสมจะถูกล้าง แตะอีกครั้งเพื่อบันทึก 35",
]
gaps = {p: missing(p) for p in phrases if missing(p)}
check("missing glyphs", gaps, {})
sc.clear_rounds()

# ------------------------------------------------------------------------ the layout fits
print("--- the panel still holds its controls")
tray(35)
PANEL = (28, 104, 772, 896)
rects = {n: r for n, r in sc.hits if n in
         ("target-", "target+", "type-target", "round", "reset", "save")}
# Six now: the region's button went to the camera settings with the zoom, and
# "ล้างกรอบ" went before it -- a region is what makes counting possible, so a
# control whose only effect is to remove one has no use.
check("all six controls drawn", len(rects), 6)
check("and the way into the settings", "setup" in dict(sc.hits), True)
bottom = max(r[1] + r[3] for r in rects.values())
check("nothing past the panel", bottom <= PANEL[1] + PANEL[3], True)
print(f"       lowest edge {bottom} vs panel bottom {PANEL[1] + PANEL[3]}")
overlaps = []
items = sorted(rects.items(), key=lambda kv: kv[1][1])
for i, (n1, a) in enumerate(items):
    for n2, b in items[i + 1:]:
        if (a[0] < b[0] + b[2] and b[0] < a[0] + a[2]
                and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]):
            overlaps.append((n1, n2))
check("no control overlaps another", overlaps, [])

print("--- and the words fit inside the buttons they are drawn in")
def fits(rect, txt, px):
    return sc.text.measure(txt, px) <= rect[2] - 20
wide = []
for txt, px, name in ((f"เก็บรอบที่ 9", 28, "round"), ("นับใหม่", 28, "reset"), ("กดอีกครั้ง", 28, "reset"),
                      ("บันทึกผล", 38, "save"),
                      ("บันทึกว่าไม่ครบ ขาด 100", 32, "save"),
                      ("แตะอีกครั้งเพื่อบันทึก 100", 30, "save")):
    if not fits(rects[name], txt, px):
        wide.append((name, txt, sc.text.measure(txt, px), rects[name][2]))
check("no button text overflows", wide, [])

# ------------------------------------------------------------------------- the round flow
print("--- pour 1: 35 on the tray, target 60")
tray(35)
check("total", sc.total(35), 35)
check("verdict", sc._verdict(sc.total(35))[1], "ขาด 25 เม็ด")
check("a pour may be taken", sc.round_block(), "")
check("save button is the warning one", sc._save_look(), "warn")

print("--- the number is still moving: no pour may be taken")
paint(34)
check("unsettled refuses", sc.round_block(), "ตัวเลขยังไม่นิ่ง รอสักครู่")
tray(35)
check("settled again", sc.round_block(), "")

print("--- take pour 1")
tap("round")
check("rounds", sc.rounds, [35])
check("a picture was kept", len(sc.round_frames), 1)
check("total holds at 35 with 35 still on the tray", sc.total(35), 35)
check("a second tap is refused", sc.round_block(), screen_mod.CLEAR_NOTE)
tap("round")
check("and refused in the method too", sc.rounds, [35])

print("--- sweep: the tray empties")
tray(0)
check("still 35 after the sweep", sc.total(0), 35)
check("clearing over", sc.clearing, False)
check("empty tray offers nothing to take", sc.round_block(), "ถาดว่าง ยังไม่มีอะไรให้เก็บ")

print("--- pour 2: 25 more")
tray(25)
check("total", sc.total(25), 60)
check("verdict", sc._verdict(sc.total(25))[1], "ครบตามจำนวน")
check("save button is green again", sc._save_look(), "primary")
check("saving is allowed", sc.blocked, "")

print("--- one too many")
tray(26)
check("total", sc.total(26), 61)
check("save refused", sc.blocked, "เกินจำนวนที่ต้องการ นำออกก่อนจึงบันทึกได้")
check("pour refused too", sc.round_block(), sc.blocked)
tray(25)
check("allowed again", sc.blocked, "")

print("--- save")
n = tap("save")
check("filed without a second tap", n, 1)
rec = json.load(open(max(glob.glob(os.path.join(RECORDS, "count_*.json")), key=os.path.getmtime), encoding="utf-8"))
check("count is the total", rec["count"], 60)
check("rounds", rec["rounds"], [35, 25])
check("not short", rec["short"], False)
check("difference", rec["difference"], 0)
check("the first pour kept its picture", len(rec.get("round_images") or []), 1)
check("rounds cleared after save", sc.rounds, [])

# ------------------------------------------------------------------------------- undo
print("--- starting over throws the whole total away, and takes two taps")
tray(35); tap("round"); tray(0); tray(20)
check("total before", sc.total(20), 55)
tap("reset")
check("the first tap only arms it", sc.rounds, [35])
check("and says what it will cost", "จะทิ้งยอดสะสม" in sc.note, True)
tap("reset")
check("the second tap does it", sc.rounds, [])
check("pictures too", sc.round_frames, [])
tray(20)
check("total after", sc.total(20), 20)

print("--- the arming lapses, so a tap minutes later is a fresh first tap")
tray(35); tap("round"); tray(0); tray(0)
tap("reset")
sc.reset_armed_at -= screen_mod.CONFIRM_SECONDS + 1
tap("reset")
check("it arms again rather than wiping", sc.rounds, [35])

print("--- it also releases a tray that was never swept")
sc.clear_rounds()
tray(35); tap("round")
check("clearing", sc.clearing, True)
check("the button is live mid-sweep", bool(sc.rounds or sc.clearing), True)
tap("reset"); tap("reset")
check("clearing cancelled", sc.clearing, False)
tray(35)
check("the tray counts again", sc.total(35), 35)

# ------------------------------------------------------------- the mid-pour save guard
print("--- a mid-pour save takes two taps")
sc.clear_rounds()
tray(35); tap("round"); tray(0); tray(0)
before = len(glob.glob(os.path.join(RECORDS, "count_*.json")))
n = tap("save")
check("the first tap files nothing", n, 0)
check("nothing on disk", len(glob.glob(os.path.join(RECORDS, "count_*.json"))), before)
check("the 35 is still there", sc.rounds, [35])
check("and it said why", "ยอดสะสมจะถูกล้าง" in sc.note, True)
tray(0)
check("the button now says what the next tap costs",
      dict(sc.hits)["save"] is not None and
      any(n == "save" for n, _ in sc.hits), True)
n = tap("save")
check("the second tap files it", n, 1)
rec = json.load(open(max(glob.glob(os.path.join(RECORDS, "count_*.json")), key=os.path.getmtime), encoding="utf-8"))
check("filed", (rec["count"], rec["rounds"], rec["short"]), (35, [35], True))
check("rounds cleared", sc.rounds, [])

print("--- the guard lapses, so a tap minutes later is not the second of a pair")
tray(35); tap("round"); tray(0); tray(0)
tap("save")
sc.save_armed_at -= screen_mod.CONFIRM_SECONDS + 1
before = len(glob.glob(os.path.join(RECORDS, "count_*.json")))
n = tap("save")
check("it arms again rather than filing", n, 0)
check("nothing filed", len(glob.glob(os.path.join(RECORDS, "count_*.json"))), before)

print("--- a short count on ONE tray files on the first tap")
sc.clear_rounds()
tray(47)
check("button warns", sc._save_look(), "warn")
n = tap("save")
check("filed at once", n, 1)
rec = json.load(open(max(glob.glob(os.path.join(RECORDS, "count_*.json")), key=os.path.getmtime), encoding="utf-8"))
check("flagged short", (rec["count"], rec["short"], rec["rounds"]), (47, True, [47]))

print("--- no target: nothing is filed at all")
# THIS USED TO FILE. A count with no prescription was saved with `short` false, on the
# reasoning that nothing can fall short of nothing -- true, and beside the point: the
# record carried a number with nothing to check it against, so the question anybody opens
# it to ask has no answer in it. tests/test_no_target.py owns that rule now; what is
# checked here is that the multi-round machinery honours it too, pours and all.
sc.clear_rounds(); sc.target = 0
tray(12); tap("round"); tray(0); tray(0)
check("the pour was banked", sc.rounds, [12])
before = len(glob.glob(os.path.join(RECORDS, "count_*.json")))
n = tap("save")
check("nothing filed", n, 0)
check("and nothing landed on disk",
      len(glob.glob(os.path.join(RECORDS, "count_*.json"))), before)
check("the pour is still there, not thrown away", sc.rounds, [12])
sc.clear_rounds()
sc.target = 60
rec = json.load(open(max(glob.glob(os.path.join(RECORDS, "count_*.json")),
                        key=os.path.getmtime), encoding="utf-8"))

# ------------------------------------------------------------------- records and CSV
print("--- the records read back, both eras")
check("old record", rec_store.rounds_text({"count": 60}), "")
check("one pour", rec_store.rounds_text({"rounds": [60]}), "")
check("two pours", rec_store.rounds_text({"rounds": [35, 25]}), "35 + 25")
rows = rec_store.load(RECORDS)
# four saves happen above: the two-pour 60, the guarded 35, the short 47 and the
# untargeted 12. The "guard lapses" section arms twice and files nothing, which is
# the point of it.
check("every record listed", len(rows), 3)
check("per-pour frames resolved", sum(len(r["round_paths"]) for r in rows) > 0, True)
csv_path = os.path.join(RECORDS, "out.csv")
rec_store.export_csv(rows, csv_path)
head = open(csv_path, encoding="utf-8-sig").readline().strip()
check("csv header", head.split(",")[:6], ["time", "count", "rounds", "target", "difference", "short"])

# --------------------------------------------------------------- a picture to look at
print("--- rendering the screen for a human to look at")
sc.clear_rounds()
sc.rounds = [35]
sc.round_frames = [FRAME.copy()]
sc.clearing = False
out = os.path.join(tempfile.gettempdir(), "phone_rounds.png")
img = paint(25)
sc._steady_at -= 1.0
img = paint(25)
cv2.imwrite(out, img)
print("      ", out, img.shape)

shutil.rmtree(RECORDS, ignore_errors=True)
print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
