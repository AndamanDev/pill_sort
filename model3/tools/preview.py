"""The phone screen, drawn on the PC, from a real frame and real detections.

    python -m model3.tools.preview [ภาพ.jpg]

WHY THIS EXISTS. An APK takes two minutes to build and a phone to look at. The screen is
plain numpy either side of that, so every layout decision -- what is too small, what
collides, where the eye goes -- can be made here in a second, against the same model
output the handset will see. Only the camera, the touch screen and the forward pass are
really Android's; the picture is not.

It writes both pages, and a third image with the region half-dragged, because the states
that look wrong are never the resting one.
"""
from __future__ import annotations

import glob
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL3 = os.path.dirname(HERE)
ROOT = os.path.dirname(MODEL3)
sys.path.insert(0, ROOT)

from model3.phone.engine import Engine, count_inside                # noqa: E402
from model3.phone.screen import Screen                              # noqa: E402

ASSETS = os.path.join(MODEL3, "phone", "assets")
WEIGHTS = os.path.join(MODEL3, "weights", "pillcount-det-v3-480x640.onnx")
OUT = os.path.join(MODEL3, "phone", "preview")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    frames = argv or sorted(glob.glob(os.path.join(MODEL3, "samples", "*.jpg")))
    if not frames:
        raise SystemExit("ไม่มีภาพตัวอย่างให้วาด")
    frame = cv2.imread(frames[0])
    if frame is None:
        raise SystemExit(f"เปิดภาพไม่ได้: {frames[0]}")

    engine = Engine.from_file(WEIGHTS)
    t0 = time.perf_counter()
    boxes, confs = engine.detect(frame)
    ms = (time.perf_counter() - t0) * 1000

    os.makedirs(OUT, exist_ok=True)
    # The records the preview lists are the bench's own, so the list has something real
    # in it -- a layout tested against an empty folder is a layout nobody has seen.
    screen = Screen(ASSETS, os.path.join(ROOT, "app", "records"), target=60)

    count = count_inside(boxes, screen.roi)
    cv2.imwrite(os.path.join(OUT, "1-count.png"),
                screen.compose(frame, boxes, count, ms))

    h, w = frame.shape[:2]
    screen.arming = True
    screen.drag_rect = (w * 0.12, h * 0.12, w * 0.88, h * 0.9)
    screen.say("ลากนิ้วคลุมพื้นที่ถาด")
    cv2.imwrite(os.path.join(OUT, "2-drag.png"),
                screen.compose(frame, boxes, count, ms))

    screen.arming = False
    screen.drag_rect = None
    screen.roi = [(int(w * .12), int(h * .12)), (int(w * .88), int(h * .12)),
                  (int(w * .88), int(h * .9)), (int(w * .12), int(h * .9))]
    screen.saved(count_inside(boxes, screen.roi), "20260917-110500")
    cv2.imwrite(os.path.join(OUT, "3-saved.png"),
                screen.compose(frame, boxes, count_inside(boxes, screen.roi), ms))

    screen.toast = None
    screen.page = "records"
    cv2.imwrite(os.path.join(OUT, "4-records.png"),
                screen.compose(frame, boxes, count, ms))

    screen.target = 10
    screen.page = "count"
    cv2.imwrite(os.path.join(OUT, "5-over.png"),
                screen.compose(frame, boxes, count, ms))

    print(f"นับได้ {count} เม็ด จาก {os.path.basename(frames[0])}  ({ms:.0f} ms)")
    print("เขียนภาพหน้าจอลง", OUT)
    for name in sorted(os.listdir(OUT)):
        print("   ", name)


if __name__ == "__main__":
    main()
