# -*- coding: utf-8 -*-
"""The faster de-duplication counts the same trays, and costs far fewer numpy calls."""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                          # app/ and model3/ read relative paths
import numpy as np, cv2, onnxruntime as ort
from model3.phone import engine as eng

FAIL = []
def check(label, got, want):
    ok = got == want
    print(("  ok  " if ok else "  FAIL") + f"  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok: FAIL.append(label)

HW = (320, 416)
sess = ort.InferenceSession(os.path.join(ROOT, "model3", "weights", "v3-320x416.onnx"),
                            providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name


def py_nms(boxes, scores, iou):
    """The loop exactly as it was, kept here as the thing to agree with."""
    if not len(boxes):
        return np.zeros((0,), np.int32)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size:
        best = order[0]; keep.append(best)
        if order.size == 1: break
        rest = order[1:]
        xx1 = np.maximum(x1[best], x1[rest]); yy1 = np.maximum(y1[best], y1[rest])
        xx2 = np.minimum(x2[best], x2[rest]); yy2 = np.minimum(y2[best], y2[rest])
        ov = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        un = areas[best] + areas[rest] - ov
        ratio = np.where(un > 0, ov / np.maximum(un, 1e-9), 0.0)
        order = rest[ratio <= iou]
    return np.array(keep, np.int32)


def clip(img, n=25, seed=11):
    out, rng = [], np.random.default_rng(seed)
    big = cv2.resize(img, (656, 496))
    for _ in range(n):
        dx, dy = rng.integers(-8, 9, 2)
        f = big[8 + dy:488 + dy, 8 + dx:648 + dx].astype(np.float32)
        f = np.clip(f * rng.uniform(0.9, 1.1) + rng.normal(0, 2, f.shape), 0, 255)
        out.append(f.astype(np.uint8))
    return out


frames = []
for name in ("tray_full.jpg", "tray_sparse.jpg"):
    img = cv2.imread(os.path.join(ROOT, "model3", "samples", name))
    if img is None:
        continue
    frames += clip(img)
    for shrink in (0.7, 0.55, 0.4):
        small = cv2.resize(img, (int(640 * shrink), int(480 * shrink)))
        canvas = np.full((480, 640, 3), 28, np.uint8)
        oy, ox = (480 - small.shape[0]) // 2, (640 - small.shape[1]) // 2
        canvas[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
        frames.append(canvas)

print(f"--- {len(frames)} frames: does it keep the same tablets?")
disagree, counts_py, counts_new = [], [], []
for i, f in enumerate(frames):
    lb, gain, pads = eng.letterbox(f, HW)
    raw = sess.run(None, {iname: eng.blob(lb)})[0]
    pred = np.squeeze(raw, 0)
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    sc = pred[:, 4]
    k = sc >= 0.45
    p, sc = pred[k], sc[k]
    if not len(p):
        counts_py.append(0); counts_new.append(0); continue
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)
    a, b = py_nms(boxes, sc, 0.4), eng.nms(boxes, sc, 0.4)
    counts_py.append(len(a)); counts_new.append(len(b))
    if set(a.tolist()) != set(b.tolist()):
        disagree.append((i, len(a), len(b)))

check("the kept set is identical on every frame", disagree, [])
check("and so is every count", counts_py, counts_new)
print(f"       counts seen: {sorted(set(counts_py))}")
print(f"       cv2 form chosen: {eng._CV_NMS_FORM!r}")
check("it is using cv2, not the loop", eng._CV_NMS_FORM in ("array", "list"), True)

print()
print("--- and the count of Python-to-numpy calls, which is what the board pays for")


class Counter:
    """Counts every numpy call the de-duplication makes."""
    def __init__(self):
        self.n = 0
        self.saved = {}
    def __enter__(self):
        for name in ("maximum", "minimum", "where", "argsort", "array", "asarray", "rint"):
            self.saved[name] = getattr(np, name)
            setattr(np, name, self._wrap(self.saved[name]))
        return self
    def _wrap(self, fn):
        def go(*a, **k):
            self.n += 1
            return fn(*a, **k)
        return go
    def __exit__(self, *e):
        for name, fn in self.saved.items():
            setattr(np, name, fn)


f = frames[0]
lb, gain, pads = eng.letterbox(f, HW)
raw = sess.run(None, {iname: eng.blob(lb)})[0]
pred = np.squeeze(raw, 0)
pred = pred.T if pred.shape[0] < pred.shape[1] else pred
sc = pred[:, 4]; k = sc >= 0.45; p, sc = pred[k], sc[k]
cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
boxes = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)

with Counter() as c:
    kept = py_nms(boxes, sc, 0.4)
old_calls = c.n
with Counter() as c:
    eng.nms(boxes, sc, 0.4)
new_calls = c.n
print(f"       {len(boxes)} candidates, {len(kept)} tablets kept")
print(f"       the loop         {old_calls:5} numpy calls")
print(f"       cv2              {new_calls:5} numpy calls")
check("far fewer calls", new_calls * 10 < old_calls, True)

print()
print("--- the region test builds its polygon once, not once per tablet")
roi = [(50, 50), (600, 50), (600, 430), (50, 430)]
eng._POLY_CACHE.clear()
with Counter() as c:
    n = eng.count_inside(np.zeros((61, 4), np.float32) + 200, roi)
print(f"       61 tablets against a region: {c.n} numpy calls (one array, reused)")
check("one polygon, not sixty-one", c.n <= 3, True)
check("and it still counts them", n, 61)

print()
print("--- no region at all is the common case, and costs nothing")
with Counter() as c:
    n = eng.count_inside(np.zeros((61, 4), np.float32), None)
check("no numpy at all", c.n, 0)
check("everything counts", n, 61)

print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
