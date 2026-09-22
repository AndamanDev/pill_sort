# -*- coding: utf-8 -*-
"""The way out of the app: present on every page, and guarded."""
import os, sys, json, time, tempfile, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                          # app/ and model3/ read relative paths
sys.path.insert(0, os.path.join(ROOT, "model3", "android", "app", "src", "main", "python"))
import numpy as np, cv2
from model3.phone import screen as screen_mod
from model3.phone.screen import Screen
import pillcount_android as bridge

screen_mod.use(2340, 1080)
W, H, OUT_W, OUT_H = screen_mod.W, screen_mod.H, screen_mod.OUT_W, screen_mod.OUT_H
REC = tempfile.mkdtemp(prefix="quit_")
sc = Screen(os.path.join(ROOT, "model3", "phone", "assets"), REC, target=60)
F = np.full((480, 640, 3), 40, np.uint8); B = np.zeros((0, 4), np.float32)

FAIL = []
def check(label, got, want):
    ok = got == want
    print(("  ok  " if ok else "  FAIL") + f"  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok: FAIL.append(label)

def paint():
    sc._chrome_key = None
    return sc.compose(F, B, 0, 8.0)

def tap(pt):
    x, y = pt
    ox, oy = int(x * OUT_W / W), int(y * OUT_H / H)
    sc.touch("down", ox, oy, F.shape)
    sc.touch("up", ox, oy, F.shape)

def box(name):
    paint()
    return dict(sc.hits).get(name)

def centre(r):
    return (r[0] + r[2] // 2, r[1] + r[3] // 2)

print("--- there is a cross, and it is on every page")
paint()
q = dict(sc.hits).get("quit")
check("the counting page has one", q is not None, True)
print(f"       at {q}, header is 0..88 tall and {W} wide")
check("in the top-right corner", q[0] + q[2] >= W - 40 and q[1] + q[3] <= 88, True)
check("big enough for a thumb", q[2] >= 72 or q[3] >= 60, True)
sc.page = "records"
check("so does the records page", box("quit") is not None, True)
sc.page = "count"

print("--- nothing it could cover is covered")
paint()
# "export" has gone back to the records page, where the rows it writes are.
rects = {n: r for n, r in sc.hits if n in ("quit", "records")}
def overlap(a, b):
    return (a[0] < b[0] + b[2] and b[0] < a[0] + a[2]
            and a[1] < b[1] + b[3] and b[1] < a[1] + a[3])
clashes = [(n1, n2) for n1 in rects for n2 in rects
           if n1 < n2 and overlap(rects[n1], rects[n2])]
check("the header's buttons do not overlap", clashes, [])
# The clock repaints a patch of the header every frame; if it reached under the cross it
# would erase it. This is that patch, from _clock.
clock_patch = (W - 286, 14, 160, 58)
check("and the clock's patch does not reach the cross", overlap(clock_patch, rects["quit"]), False)

print("--- one tap arms it, it does not close")
check("not quitting yet", sc.quitting, False)
tap(centre(box("quit")))
check("still not quitting", sc.quitting, False)
check("but armed", time.time() - sc.quit_armed_at < 1.0, True)
check("and it said so", "แตะอีกครั้ง" in sc.note, True)

print("--- the cross turns red while it is armed")
def redness(img, r):
    """How red the cross is, sampled off the picture Android actually gets.

    compose() hands back the SCALED screen, and hit boxes are in design coordinates, so
    the rect has to come down with it -- a design-sized crop of a scaled picture is empty.
    """
    k = OUT_W / W
    x0, y0 = int(r[0] * k), int(r[1] * k)
    x1, y1 = int((r[0] + r[2]) * k), int((r[1] + r[3]) * k)
    patch = img[y0:y1, x0:x1]
    assert patch.size, "the crop landed outside the picture"
    return int((patch[:, :, 2].astype(int) - patch[:, :, 1]).max())

armed_img = paint()
r = box("quit")
red = redness(armed_img, r)
sc.quit_armed_at = 0.0
red2 = redness(paint(), r)
check("armed is redder than idle", red > red2 + 40, True)

print("--- the second tap closes it")
tap(centre(box("quit")))
tap(centre(box("quit")))
check("quitting", sc.quitting, True)

print("--- and it warns about counted pours before throwing them away")
sc.quitting = False
sc.quit_armed_at = 0.0
sc.rounds = [35]
tap(centre(box("quit")))
check("did not close", sc.quitting, False)
check("named what would be lost", "จะทิ้งยอดสะสม" in sc.note, True)
check("and how many", "35" in sc.note, True)

print("--- the arming lapses, so a tap minutes later is a fresh first tap")
sc.quit_armed_at -= screen_mod.CONFIRM_SECONDS + 1
tap(centre(box("quit")))
check("armed again rather than closing", sc.quitting, False)

print("--- and a tap elsewhere does not close it either")
sc.quitting = False
sc.quit_armed_at = 0.0
tap(centre(box("quit")))                        # arm
tap(centre(box("records")))                     # something else
sc.page = "count"
tap(centre(box("quit")))                        # would be the 'second' tap
check("the run was broken by the other press", sc.quitting, False)

print("--- the reply Kotlin acts on")
sc.quitting = False
bridge._S["screen"] = sc
bridge._S["cv2"] = cv2
check("quiet while nothing asked", json.loads(bridge.touch("up", 5, 5)).get("quit"), False)
sc.quitting = True
check("and true once it is", json.loads(bridge.touch("up", 5, 5)).get("quit"), True)

kt = open(os.path.join(ROOT, "model3", "android", "app", "src", "main", "java",
                       "com", "pharmaflow", "pillcount", "MainActivity.kt"),
          encoding="utf-8").read()
check("the activity reads it", 'optBoolean("quit", false)' in kt, True)
check("and finishes the task", "finishAndRemoveTask()" in kt, True)

print("--- and every word of it has a picture in the atlas")
def missing(t):
    return [p for p, d in sc.text._runs(t) if sc.text._entry(p, d) is None]
for phrase in ("ปิดโปรแกรม แตะอีกครั้งเพื่อยืนยัน",
               "จะทิ้งยอดสะสม 35 เม็ด แตะอีกครั้งเพื่อยืนยัน",
               "ปิดโปรแกรม"):
    check(f"{phrase[:24]}...", missing(phrase), [])

shutil.rmtree(REC, ignore_errors=True)
print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
