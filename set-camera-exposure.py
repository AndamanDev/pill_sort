"""Set this bench camera so the picture is both LIT and SMOOTH, and prove which it is.

    python set-camera-exposure.py            measure everything, keep the best
    python set-camera-exposure.py --value -3 force a manual step
    python set-camera-exposure.py --auto     force automatic

WHY THIS MEASURES FRAME RATE, having once measured only brightness.

An earlier version of this script ranked exposures by mean brightness alone and picked -1.
That is the correct answer to the question it asked and the wrong setting for the bench,
because on this camera the manual exposure scale is SHUTTER TIME: -1 holds the shutter open
about half a second, and a camera that takes half a second per frame delivers two frames a
second. The counting app was then blamed for being slow while it sat waiting on a camera
this script had slowed down.

Measured here, 640x480, same room, same minute:

    setting                 fps    brightness
    as the driver left it   2.0        23.5      <- what the old script chose
    AUTOMATIC              20.0        53.9      <- brighter AND ten times faster
    manual -3               8.0         4.5
    manual -5              30.1         0.2

Automatic wins both ways because it can raise GAIN, which costs no time, where the manual
scale can only lengthen the exposure, which costs nothing but. So automatic is what this
script now prefers, and manual steps are kept only for the case where a fixed exposure is
wanted on purpose -- a repeatable measurement, say.

WHY A SEPARATE SCRIPT AT ALL. Exposure is a property of the DEVICE, so setting it once here
leaves it set for whatever opens the camera next. That matters because the Flutter app
cannot set it: `camera_windows` throws UnimplementedError from setExposureOffset, so on
Windows a Flutter app gets whatever the driver last left. model3/bench.py and app/ set it
themselves through OpenCV and do not need this.

Whether the setting survives a reboot or a replug is up to the driver. Run this again if
the picture goes dark or the preview turns sluggish.
"""
from __future__ import annotations

import argparse
import time

import cv2

#: Manual steps worth trying. Each is a shutter time, and the cost in frame rate is why
#: they are all measured rather than reasoned about.
CANDIDATES = (-7, -6, -5, -4, -3, -2, -1)

#: What the model was characterised at. Below roughly 60 it starts missing pills.
TARGET = 100.0

#: Under this the preview reads as stuttering to someone working at the bench, whatever
#: the brightness. Ten was chosen because 8 fps looked visibly wrong here and 20 did not.
MIN_FPS = 10.0

#: On this camera, through DirectShow: 1 turns the automatic mode on, 0.75 hands control
#: to CAP_PROP_EXPOSURE. Measured, not taken from the usual 0.25/0.75 convention, which
#: did not match what this backend actually did.
AUTO_ON = 1.0
AUTO_OFF = 0.75


def measure(cap, *, auto: float, value: float | None, frames: int = 30):
    """Apply a setting, let it settle, and return (fps, brightness)."""
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto)
    if value is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, float(value))
    # Ten frames thrown away: the first frames after a change are taken with the old
    # setting, and automatic mode in particular needs a moment to find its level.
    for _ in range(10):
        cap.read()
    started = time.time()
    mean = 0.0
    read = 0
    for _ in range(frames):
        ok, frame = cap.read()
        if ok:
            mean = frame.mean()
            read += 1
    elapsed = time.time() - started
    if read == 0 or elapsed <= 0:
        return None
    return read / elapsed, mean


def usable(result) -> bool:
    """Bright enough for the model AND fast enough to work in front of."""
    fps, mean = result
    return mean >= 60 and fps >= MIN_FPS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--value", type=float, default=None,
                    help="force this manual step instead of measuring")
    ap.add_argument("--auto", action="store_true", help="force automatic")
    a = ap.parse_args(argv)

    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"เปิดกล้อง {a.camera} ไม่ได้ -- มีโปรแกรมอื่นใช้อยู่หรือเปล่า")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if a.auto or a.value is not None:
        auto = AUTO_ON if a.auto else AUTO_OFF
        result = measure(cap, auto=auto, value=None if a.auto else a.value)
        cap.release()
        if result is None:
            raise SystemExit("อ่านภาพจากกล้องไม่ได้")
        fps, mean = result
        print(f"ตั้งเป็น {'อัตโนมัติ' if a.auto else a.value}: "
              f"{fps:.1f} fps  ความสว่าง {mean:.1f}/255")
        return 0

    print("การตั้งค่า          fps    ความสว่าง")

    auto_result = measure(cap, auto=AUTO_ON, value=None)
    if auto_result:
        print(f"  อัตโนมัติ      {auto_result[0]:6.1f}   {auto_result[1]:7.1f}")

    manual = []
    for value in CANDIDATES:
        result = measure(cap, auto=AUTO_OFF, value=value)
        if result is None:
            continue
        manual.append((value, result))
        print(f"  {value:>3}          {result[0]:6.1f}   {result[1]:7.1f}")
    print()

    # AUTOMATIC IS PREFERRED WHEN IT IS GOOD ENOUGH, not when it is strictly best. It
    # tracks the room, so it keeps working when somebody turns a lamp on, where a fixed
    # step is only ever right for the light it was measured in.
    if auto_result and usable(auto_result):
        chosen, result = "อัตโนมัติ", auto_result
        measure(cap, auto=AUTO_ON, value=None)
    else:
        good = [(v, r) for v, r in manual if usable(r)]
        if good:
            # Among the ones that work, the brightest -- there is headroom in the frame
            # rate and none in the light.
            value, result = max(good, key=lambda item: item[1][1])
        elif auto_result:
            value, result = None, auto_result
        elif manual:
            value, result = max(manual, key=lambda item: item[1][1])
        else:
            cap.release()
            raise SystemExit("อ่านภาพจากกล้องไม่ได้เลย")

        if value is None:
            chosen = "อัตโนมัติ"
            measure(cap, auto=AUTO_ON, value=None)
        else:
            chosen = str(value)
            measure(cap, auto=AUTO_OFF, value=value)

    cap.release()
    fps, mean = result
    print(f"กล้อง      : {a.camera}")
    print(f"ตั้งเป็น    : {chosen}")
    print(f"ความเร็ว   : {fps:.1f} เฟรม/วินาที")
    print(f"ความสว่าง  : {mean:.1f} จาก 255 (ต้องการ ~{TARGET:.0f})")
    print()

    if mean < 60:
        print("ยังมืดเกินไป และกล้องทำได้แค่นี้แล้ว -- นี่เป็นเรื่องแสงในห้อง ไม่ใช่การตั้งค่า")
        print("โมเดลถูกวัดไว้ที่ความสว่างราว 100 ถ้าต่ำกว่านี้มากมันจะมองไม่เห็นเม็ดยา")
        print("เปิดไฟที่โต๊ะเพิ่ม หรือย้ายถาดไปที่สว่างกว่านี้")
    elif fps < MIN_FPS:
        print(f"ภาพสว่างพอ แต่ได้แค่ {fps:.1f} เฟรม/วินาที ภาพจะดูกระตุก")
        print("กล้องต้องเปิดชัตเตอร์นานเพราะห้องมืด -- เปิดไฟเพิ่มแล้วรันใหม่")
    else:
        print("ตั้งค่าแล้ว เปิดโปรแกรมนับยาได้เลย")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
