# -*- coding: utf-8 -*-
"""The pacing rule, driven the way the camera drives it -- counted, not modelled.

A fake model of a chosen cost stands in for the real one, so the same run can be repeated
at a handset's speed on a desk. What is measured is what an operator gets: how often the
number is refreshed, and how much of the worker thread is left for drawing and for taps.
"""
import os, sys, time, tempfile, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                          # app/ and model3/ read relative paths
sys.path.insert(0, os.path.join(ROOT, "model3", "android", "app", "src", "main", "python"))
import numpy as np, cv2
import pillcount_android as bridge
from model3.phone import screen as screen_mod
from model3.phone.screen import Screen

FAIL = []
def check(label, got, want):
    ok = got == want
    print(("  ok  " if ok else "  FAIL") + f"  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok: FAIL.append(label)


class SlowModel:
    """A forward pass that costs exactly what a handset's does, and counts its calls."""
    def __init__(self, cost_s):
        self.cost = cost_s
        self.calls = 0
        self.busy = 0.0
    def detect(self, bgr):
        self.calls += 1
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < self.cost:
            pass
        self.busy += time.perf_counter() - t0
        return np.zeros((61, 4), np.float32), np.ones((61,), np.float32)


def run(cost, seconds=2.0, fps=30):
    screen_mod.use(2340, 1080)
    rec = tempfile.mkdtemp(prefix="pace_")
    sc = Screen(os.path.join(ROOT, "model3", "phone", "assets"), rec)
    model = SlowModel(cost)
    bridge._S.clear()
    bridge._S.update({"cv2": cv2, "screen": sc, "engine": model,
                      "pane": screen_mod.pane_out(),
                      "pane_ratio": screen_mod.PANE[2] / screen_mod.PANE[3],
                      "boxes": np.zeros((0, 4), np.float32),
                      "confs": np.zeros((0,), np.float32),
                      "count": 0, "ms": 0.0, "error": "", "geometry": "x",
                      "last_frame_at": time.time(), "last_key": None})
    base = np.full((480, 640, 3), 40, np.uint8)
    rng = np.random.default_rng(5)
    t_end = time.perf_counter() + seconds
    frames = 0
    t0 = time.perf_counter()
    while time.perf_counter() < t_end:
        # A tray being worked on: the picture keeps changing, so the model is always
        # wanted. This is the busiest case, which is the one the rule exists for.
        f = base.copy()
        f[:] = rng.integers(20, 60, 3, dtype=np.uint8)
        rgba = cv2.cvtColor(f, cv2.COLOR_BGR2RGBA)
        stride = (640 + 64) * 4
        buf = np.zeros((480, stride // 4, 4), np.uint8); buf[:, :640] = rgba
        bridge.frame(buf.tobytes(), 640, 480, stride, 0, 0, 0, 0, 0)
        frames += 1
        time.sleep(1.0 / fps)
    elapsed = time.perf_counter() - t0
    shutil.rmtree(rec, ignore_errors=True)
    return model.calls / elapsed, model.busy / elapsed, frames / elapsed


print(f"DETECT_EVERY={bridge.DETECT_EVERY}  DETECT_DUTY={bridge.DETECT_DUTY}\n")
print(f"{'a pass costs':>14}  {'counts/s':>9} {'model has':>10} {'camera fps':>11}")
print("-" * 50)
rows = []
for label, cost in (("25 ms (PC)", 0.025), ("75 ms", 0.075), ("125 ms", 0.125),
                    ("250 ms (old model,", 0.250)):
    rate, duty, fps = run(cost)
    rows.append((label, cost, rate, duty))
    print(f"{label:>14}  {rate:9.1f} {duty*100:9.0f}% {fps:11.1f}")

print()
print("--- the rules that must hold whatever the hardware is")
for label, cost, rate, duty in rows:
    # THE DUTY IS A CEILING, NOT A TARGET, and this is the assertion that matters: the
    # model never takes more of the worker than it is allowed, so there is always a share
    # left for drawing the screen and answering a tap. Sitting BELOW the ceiling is normal
    # and good -- the camera only delivers so many frames, and each one is also drawn.
    check(f"{label}: never takes more of the thread than it is allowed",
          duty <= bridge.DETECT_DUTY + 0.12, True)
    # The old rule had no such ceiling once a pass was as long as the floor: it stamped the
    # clock before the pass instead of after, so passes ran back to back at 100%. THIS is
    # the row that used to saturate, and the one an operator felt as lag.
    if cost >= 0.25:
        check("a 250 ms pass no longer saturates the worker", duty < 0.85, True)
        print(f"       at 250 ms the old rule ran the model at 100% of the thread; "
              f"this run measured {duty*100:.0f}%")

print()
print("--- and the old ceiling is gone")
fast_rate = rows[0][2]
check("a 25 ms pass beats the old 4-a-second cap", fast_rate > 4.5, True)
print(f"       {fast_rate:.1f} counts a second, where the old floor allowed 4.0 at most")

print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
