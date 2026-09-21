"""model3's frame loop, for the phone. Three calls in, one picture out.

WHAT KOTLIN KNOWS ABOUT THIS FILE is `start`, `frame` and `touch`. Everything else -- the
letterbox, the thresholds, the counting, the region, the records, every pixel of the
screen -- is Python, and the same Python the PC runs:

    the bench (app/)                     here
    ----------------------------------   ---------------------------------------------
    cv2.VideoCapture.read()              CameraX hands over RGBA bytes
    ultralytics + torch                  model3/phone/engine.py + ONNX Runtime in Kotlin
    PySide6 windows                      model3/phone/screen.py, composed into a Bitmap

STATE LIVES IN A MODULE-LEVEL DICT because Chaquopy keeps one interpreter per process:
the activity starts it once and calls into this same module for every frame after that.

THE FORWARD PASS IS THE BUDGET. On the bench it is about 100 ms; through ONNX Runtime on
a handset expect two to three times that, so the phone runs at a few frames a second and
everything else here is written to stay out of the way -- one resize of the camera frame,
one composed picture, no second pass, no smoothing.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

_S: dict = {}

#: How different two frames have to be, on a 32x24 grey thumbnail scaled 0-255, before
#: the tray counts as having changed. Measured against a still tray under a fluorescent
#: lamp, whose noise sits around 1.5.
MOTION = 3.0

#: After the picture goes still, run this many more passes before trusting it. The last
#: hand leaving the tray and the tray settling are two different moments, and a count
#: taken at the first is taken of a hand.
SETTLE = 2

#: The floor on the gap between forward passes. Not a frame count: a camera that gives
#: 30fps and one that gives 12 would otherwise be counting at very different rates.
DETECT_EVERY = 0.25

#: And the real rule: never spend more than this fraction of the thread on the model.
#: The analyser thread also converts every camera buffer and composes the screen, and a
#: model that takes a second gets a second of quiet afterwards -- otherwise it is always
#: running, the touch handler waits behind it, and the count arrives later than if it had
#: been asked for half as often.
DETECT_DUTY = 0.5


def _report(exc):
    """Put the whole traceback in logcat, where adb can reach it.

    The screen has room for "โมเดลผิดพลาด" and a phone has no console, so without this the
    only evidence of a fault is four words in Thai. Chaquopy routes stdout to logcat:

        adb logcat -s python.stdout python.stderr
    """
    import traceback

    print("PILLCOUNT ERROR", type(exc).__name__, exc)
    traceback.print_exc()


def start(assets_dir: str, records_dir: str, ort,
          view_w: int = 2160, view_h: int = 1080) -> str:
    """Build the engine and the screen. Returns the canvas size as JSON, for the log.

    `ort` is the Kotlin OrtEngine: bytes in, bytes out, and nothing in it knows what a
    YOLO head is. `assets_dir` is where the PNGs were unpacked -- inside an APK there is
    no directory next to the code, so Art.kt writes them to private storage first.
    """
    import cv2                                                      # noqa: F401

    from model3.phone.engine import Engine
    from model3.phone import screen as screen_mod
    from model3.phone.screen import Screen, pane_out

    # Before the Screen is built: it reads the canvas constants as it lays out.
    screen_mod.use(int(view_w), int(view_h))
    OUT_W, OUT_H = screen_mod.OUT_W, screen_mod.OUT_H

    os.makedirs(records_dir, exist_ok=True)
    _S["ort"] = ort
    _S["engine"] = Engine(kotlin=_Forward(ort), conf=0.45, iou=0.4)
    _S["screen"] = Screen(assets_dir, records_dir)
    _S["cv2"] = cv2
    _S["frame"] = None
    _S["boxes"] = np.zeros((0, 4), np.float32)
    _S["confs"] = np.zeros((0,), np.float32)
    _S["count"] = 0
    _S["ms"] = 0.0
    _S["last_frame_at"] = time.time()
    _S["error"] = ""
    _S["pane"] = pane_out()
    _S["last_key"] = None
    # WHICH KERNELS THE MODEL IS ACTUALLY RUNNING ON, on the screen rather than in a log.
    # OrtEngine tries XNNPACK first and falls back to the plain CPU provider if the AAR
    # was built without it -- a difference of two or three times on the forward pass, and
    # the only evidence either way was a line in logcat that needs a cable to read. The
    # footer has room for one word.
    try:
        _S["provider"] = str(ort.provider)
    except Exception:                                               # noqa: BLE001
        try:
            _S["provider"] = str(ort.getProvider())
        except Exception:                                           # noqa: BLE001
            _S["provider"] = "?"
    _S["screen"].provider = _S["provider"]
    # The hole the CameraX preview shows through. Kotlin places its surface there and
    # builds a ViewPort from the same rectangle, which is what keeps the marks on the
    # pills: the analysed image and the displayed one end up being one picture.
    #
    # Imported at the top of this function with the rest, NOT here. A second import of the
    # same name lower down makes it a local for the whole function, so the line that fills
    # _S["pane"] above raised UnboundLocalError before the camera ever opened -- and the
    # only place that showed was a phone, saying "เริ่มระบบไม่สำเร็จ".
    return json.dumps({"width": OUT_W, "height": OUT_H, "pane": pane_out(),
                       "records": records_dir, "flip": bool(_S["screen"].flip)})


class _Forward:
    """Adapts the Kotlin engine to what model3/phone/engine.py expects to call."""

    def __init__(self, ort):
        self.ort = ort

    def run(self, blob_bytes):
        # Chaquopy hands a Java byte[] back as Python bytes, which is what engine.py
        # reads with np.frombuffer. One call, because the first draft managed to invoke
        # the forward pass twice per frame while deciding how to unwrap it.
        return self.ort.forward(blob_bytes)


def frame(rgba: bytes, width: int, height: int, row_stride: int, rotation: int,
          crop_l: int = 0, crop_t: int = 0, crop_r: int = 0, crop_b: int = 0) -> bytes:
    """One camera frame in, the whole screen out, as RGBA bytes for a Bitmap."""
    cv2 = _S["cv2"]
    screen = _S["screen"]

    try:
        bgr = _to_bgr(rgba, width, height, row_stride, rotation,
                      crop_l, crop_t, crop_r, crop_b)
    except Exception as exc:                                        # noqa: BLE001
        # A camera whose buffer is not the shape this expects would otherwise take the
        # whole frame call down, and Kotlin would show its own status over everything.
        _report(exc)
        _S["error"] = f"กล้อง {type(exc).__name__}: {exc}"[:240]
        bgr = _S.get("frame")
        if bgr is None:
            raise
    _S["frame"] = bgr
    _S["last_frame_at"] = time.time()

    # THE MODEL DOES NOT RUN ON EVERY FRAME, and that is what makes the video watchable.
    # The forward pass is most of a second on a handset; tying the picture to it means the
    # camera updates once or twice a second and the whole app feels broken -- which is
    # exactly what the first build on a real phone felt like. Drawing is cheap by
    # comparison, so frames in between are composed from the last boxes: the video runs at
    # the camera's rate and the number refreshes a few times a second, which is the right
    # way round for somebody moving a tray under a lens.
    # THE MODEL RUNS WHEN THE PICTURE CHANGES, not on a clock.
    #
    # A tray that has been counted and left alone does not need counting again, and on a
    # handset where a pass is most of a second that is the difference between an app that
    # is busy all the time and one that is busy while somebody is doing something. The
    # comparison is a 32x24 thumbnail -- microseconds -- and the pass that matters, the
    # one just after the hand comes away, still happens immediately.
    now = time.perf_counter()
    small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (32, 24),
                       interpolation=cv2.INTER_AREA).astype(np.int16)
    previous = _S.get("thumb")
    moved = previous is None or float(np.abs(small - previous).mean()) > MOTION
    _S["thumb"] = small
    if moved:
        _S["settle"] = SETTLE
    screen.moving = moved

    gap = max(DETECT_EVERY, _S.get("detect_cost", 0.0) * (1 - DETECT_DUTY) / DETECT_DUTY)
    due = now - _S.get("last_detect", -9e9) >= gap
    if due and (moved or _S.get("settle", SETTLE) > 0):
        if not moved:
            _S["settle"] = _S.get("settle", SETTLE) - 1
        _S["last_detect"] = now
        try:
            boxes, confs = _S["engine"].detect(bgr)
            _S["boxes"], _S["confs"] = boxes, confs
            _S["error"] = ""
        except Exception as exc:                                    # noqa: BLE001
            # Kotlin's thread is calling this. Without the catch the exception is logged
            # and the next frame tries again, while the screen keeps showing the last
            # count as though it were current -- and on the first handset build that is
            # precisely what happened, for a reason no part of the app admitted to.
            # Kept whole for the snackbar: the class name alone says "something in
            # ONNX Runtime", and the sentence after it is the part that names the fault.
            _S["error"] = f"{type(exc).__name__}: {exc}"[:240]
            _report(exc)
        _S["ms"] = (time.perf_counter() - now) * 1000
        _S["detect_cost"] = _S["ms"] / 1000.0

    _S["count"] = _count_inside()

    return _draw(bgr, 0.0)


def idle() -> bytes:
    """Redraw with NO new frame, so a camera that has stopped looks stopped.

    NOTHING ELSE IN THIS FILE RUNS WHEN THE CAMERA STOPS. Every other path starts with a
    frame arriving: pull a USB lens off the OTG lead, let another app take the camera, or
    let the driver drop the device, and frame() is simply never called again. The last
    picture stays on the screen with its count beside it, looking exactly like a live one
    -- and the save button, which has no idea, still writes it down under the time it is
    pressed. That is a count of a tray that is no longer there, filed as current, which is
    the one failure in this app that produces a WRONG RECORD rather than no record.

    app/worker.py closes that on the bench with `last_frame_at`, and the bench can do it
    alone because its window repaints on a timer of its own. A phone has no such timer:
    the screen is composed by the camera thread. So Kotlin watches the clock and calls
    this, and the same `stale` the bench computes reaches the same _block_check.
    """
    screen = _S.get("screen")
    if screen is None:
        return b""
    last = _S.get("last_frame_at") or 0.0
    return _draw(_S.get("frame"), time.time() - last if last else 0.0)


def _draw(bgr, stale: float) -> bytes:
    """Compose the screen, or b"" when it would be identical to the one already up."""
    cv2 = _S["cv2"]
    screen = _S["screen"]

    # NOTHING CHANGED, NOTHING DRAWN. Kotlin keeps the bitmap it already has when this
    # returns empty, so a still tray costs one comparison a frame instead of a whole
    # screen -- and the touch handler, which shares this thread, stops queueing behind
    # work that would have produced an identical picture.
    key = screen.state_key(_S["count"], _S["ms"], stale, _S["error"])
    if key == _S.get("last_key"):
        return b""
    _S["last_key"] = key

    drew = time.perf_counter()
    picture = screen.compose(bgr, _S["boxes"], _S["count"], _S["ms"],
                             stale=stale, error=_S["error"])
    screen.draw_ms = (time.perf_counter() - drew) * 1000
    rgba = cv2.cvtColor(picture, cv2.COLOR_BGR2RGBA)
    # Punch the hole. The preview surface is behind the ImageView, so alpha 0 here is the
    # whole of what lets it be seen.
    # The video hole. screen.compose worked out which of its pixels are still the
    # sentinel colour -- nothing drew there, so the camera shows through -- and which are
    # ink, whether that ink is a dot, the region, a toast or a fault message.
    #
    # ONLY THE ALPHA CHANNEL IS SET HERE NOW. An ARGB_8888 bitmap is PREMULTIPLIED -- a
    # pixel whose colour exceeds its alpha is undefined, and on the test handset undefined
    # meant opaque, which hid the camera completely -- so the colours used to be blanked
    # here as well. screen._punch does it before the screen is scaled instead, which is
    # the only place the two can be made to agree on a half-covered pixel, and it saves a
    # pass over the pane every frame.
    alpha = screen.hole_alpha
    if alpha is not None:
        px, py, pw, ph = _S["pane"]
        hole = rgba[py:py + ph, px:px + pw]
        if alpha.shape == hole.shape[:2]:
            hole[:, :, 3] = alpha
    return rgba.tobytes()


def touch(phase: str, x: int, y: int) -> str:
    """One touch, in canvas coordinates. Returns JSON of anything Kotlin has to act on."""
    screen = _S["screen"]
    shape = _S["frame"].shape if _S.get("frame") is not None else None
    screen.touch(phase, int(x), int(y), frame_shape=shape,
                 on_save=_save, on_export=_export)
    # Whatever it did, the next frame is drawn: a tap that changed something too subtle
    # for state_key would otherwise leave the screen showing the state before it.
    _S["last_key"] = None
    # `flip` IS HERE BECAUSE KOTLIN OWNS HALF OF IT. Everything else on this screen is
    # drawn by Python and needs no reply at all -- but the video is CameraX's own preview
    # surface showing through the hole in the canvas, and nothing on this side can turn it
    # round. Python flips the frame it analyses, so the marks and the region stay on the
    # tablets; MainActivity flips the surface, so the operator sees the same picture. Send
    # only one of those and they disagree, with every mark landing on the mirror image of
    # the tablet it belongs to -- which is worse than the mirrored picture it set out to fix.
    return json.dumps({"target": screen.target, "page": screen.page,
                       "flip": bool(screen.flip)})


# --------------------------------------------------------------------------- helpers --
def _count_inside():
    from model3.phone.engine import count_inside

    return count_inside(_S["boxes"], _S["screen"].roi)


def _save():
    """Write the record, with the SAME arithmetic the screen did.

    Not the live count on its own. Filing the tray while the screen showed a total would
    put 25 in the record for a prescription of 60 -- wrong in the direction that reads as a
    short dispense, and unfalsifiable afterwards because the other 35 are in a bottle. The
    pours are read once, here, so what is written is what was on screen when it was tapped.
    """
    from model3.phone import records as rec_store

    screen = _S["screen"]
    live = 0 if screen.clearing else int(_S["count"])
    rounds = list(screen.rounds) + ([live] if live or not screen.rounds else [])
    total = screen.banked() + live
    stamp = rec_store.save(
        screen.records_dir, _S.get("frame"), _S["boxes"], _S["confs"],
        total, screen.target, _S["ms"], screen.roi,
        screen.conf, screen.iou, screen.imgsz,
        write_jpeg=lambda path, img: _S["cv2"].imwrite(path, img),
        rounds=rounds, round_frames=list(screen.round_frames))
    screen.saved(total, stamp)


def _export():
    """Write the CSV where Android lets other apps reach it, and say where it went."""
    from model3.phone import records as rec_store

    screen = _S["screen"]
    rows = screen.shown() if screen.page == "records" else screen.rows
    if not rows:
        screen.say("ยังไม่มีรายการที่บันทึก")
        return
    name = f"pillsort_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    path = os.path.join(screen.records_dir, name)
    rec_store.export_csv(rows, path)
    screen.say(f"ส่งออกแล้ว {len(rows)} รายการ")
    _S["last_csv"] = path


def last_csv() -> str:
    """The most recent export, for Kotlin to hand to a share sheet. "" if there is none."""
    return _S.get("last_csv", "")


def _to_bgr(rgba, width, height, row_stride, rotation, crop_l, crop_t, crop_r, crop_b):
    """CameraX's padded, un-rotated, possibly cropped buffer -> an upright BGR frame.

    Three things are wrong with the bytes as they arrive, and all three are the caller's
    business rather than a camera fault: every row is padded out to a stride of CameraX's
    choosing, so the buffer is wider than the picture; `cropRect` says which part of it the
    viewport is actually showing; and the sensor is mounted at whatever angle the
    manufacturer chose, with the rotation reported rather than applied.

    The order matters. The crop is in the buffer's own coordinates, so it has to happen
    before the rotation, or a portrait phone crops the wrong edge entirely.
    """
    cv2 = _S["cv2"]
    rows = len(rgba) // row_stride
    buf = np.frombuffer(rgba, np.uint8, count=rows * row_stride)
    buf = buf.reshape(rows, row_stride // 4, 4)

    x0, y0 = max(0, crop_l), max(0, crop_t)
    x1 = min(width, crop_r) if crop_r > crop_l else width
    y1 = min(rows, crop_b) if crop_b > crop_t else height
    if x1 - x0 < 16 or y1 - y0 < 16:
        x0, y0, x1, y1 = 0, 0, width, height
    bgr = cv2.cvtColor(np.ascontiguousarray(buf[y0:y1, x0:x1]), cv2.COLOR_RGBA2BGR)

    if rotation == 90:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        bgr = cv2.rotate(bgr, cv2.ROTATE_180)
    elif rotation == 270:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # THE MIRROR IS NOT DONE HERE, and it was, for a while. Flipping the frame gives the
    # detector a different image: measured on this bench it moved the count by as much as
    # three on one tray, so a preference about which way round a picture LOOKS could change
    # a number that goes in a record. It lives in two places instead, both of them at the
    # glass -- screen.spot() mirrors the marks inside the video pane, MainActivity mirrors
    # the preview surface underneath them, and screen._to_frame undoes it for every tap.
    # The frame, the model, the region and the saved JPEG never learn the setting exists.
    return bgr
