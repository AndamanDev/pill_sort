"""Run the Android bridge on the PC, exactly as the activity would.

    python -m model3.tools.simulate_phone

WHY THIS EXISTS, and it is not a nice-to-have. pillcount_android.py is the one file in
model3 that no test ever executed: it imports nothing on a desktop that a desktop cannot
provide, but it was only ever compiled, and py_compile does not run a line of it. Two
faults reached a handset that way -- a tensor shape decided in Kotlin, and an import placed
below the line that used it, which Python turns into UnboundLocalError and only at runtime.
Each cost a build, an install, and somebody reading Thai off a phone to tell me what broke.

So this stands in for the parts that are Android's:

    OrtEngine.kt        -> Ort, below: bytes in, bytes out, onnxruntime underneath
    CameraX RGBA_8888   -> a JPEG, padded to a row stride the way CameraX pads
    the touch screen    -> touch() calls in the coordinates Android sends
    Art.kt              -> the assets directory, which on a PC is simply there

Everything else -- start, frame, touch, the engine, the screen, the records -- is the code
that ships. If this runs clean, the phone gets something that has at least been executed.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL3 = os.path.dirname(HERE)
ROOT = os.path.dirname(MODEL3)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(MODEL3, "android", "app", "src", "main", "python"))

ASSETS = os.path.join(MODEL3, "phone", "assets")
WEIGHTS = os.path.join(MODEL3, "weights", "pillcount-det-v3-480x640.onnx")


class Ort:
    """What OrtEngine.kt is, in the only two ways Python can tell.

    It takes the NCHW float32 blob as bytes and gives back every output tensor's bytes,
    which is the whole of the contract -- and it reads the input shape from the model
    rather than assuming a square one, because that assumption is what shipped broken.
    """

    def __init__(self, path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.name = self.session.get_inputs()[0].name
        self.shape = [int(v) for v in self.session.get_inputs()[0].shape]
        self._last = "[]"

    def imgsz(self):
        return self.shape[2]

    def shapes(self):
        return self._last

    def forward(self, chw: bytes) -> bytes:
        x = np.frombuffer(chw, "<f4").reshape(self.shape)
        outs = self.session.run(None, {self.name: x})
        self._last = json.dumps([list(o.shape) for o in outs])
        return b"".join(np.ascontiguousarray(o, "<f4").tobytes() for o in outs)


def camera_bytes(frame, stride_pad=64):
    """A frame as CameraX hands it over: RGBA, and every row padded to a stride."""
    rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    h, w = rgba.shape[:2]
    row_stride = (w + stride_pad) * 4
    buf = np.zeros((h, row_stride // 4, 4), np.uint8)
    buf[:, :w] = rgba
    return buf.tobytes(), w, h, row_stride


def main():
    frames = [cv2.imread(p) for p in
              (os.path.join(MODEL3, "samples", "tray_full.jpg"),
               os.path.join(MODEL3, "samples", "tray_sparse.jpg"))]
    frames = [f for f in frames if f is not None]
    if not frames:
        raise SystemExit("ไม่มีภาพตัวอย่าง")

    records = tempfile.mkdtemp(prefix="sim_records_")
    import pillcount_android as app

    # A real handset's proportions, not the design's: the canvas is laid out from what
    # the activity reports, so a test that always passes 2:1 would never see the shape
    # every actual phone has.
    info = json.loads(app.start(ASSETS, records, Ort(WEIGHTS), 2340, 1080))
    print("start ->", info)
    w, h, pane = info["width"], info["height"], info["pane"]

    ok = 0
    for i, frame in enumerate(frames * 3):
        # Space them out past DETECT_EVERY, or the pacing skips the model and every
        # frame reports the first one's count -- which is correct behaviour and a
        # useless test.
        if i:
            time.sleep(0.3)
        rgba, fw, fh, stride = camera_bytes(cv2.resize(frame, (640, 480)))
        t0 = time.perf_counter()
        out = app.frame(rgba, fw, fh, stride, 0, 0, 0, 0, 0)
        ms = (time.perf_counter() - t0) * 1000
        drawn = len(out) > 0
        if drawn and len(out) != w * h * 4:
            raise SystemExit(f"ภาพที่คืนมาขนาดผิด: {len(out)} ต่าง {w * h * 4}")
        if app._S["error"]:
            raise SystemExit(f"โมเดลพัง: {app._S['error']}")
        ok += 1
        print(f"  frame {i}: นับได้ {app._S['count']:3}  {ms:6.1f} ms  "
              f"{'วาดใหม่' if drawn else 'ข้าม (เหมือนเดิม)'}")

    # The hole has to be transparent, or the preview surface behind it is invisible.
    # Forced, because a frame that changes nothing is deliberately not drawn at all.
    screen = app._S["screen"]
    app._S["last_key"] = None
    rgba, fw, fh, stride = camera_bytes(cv2.resize(frames[0], (640, 480)))
    last = app.frame(rgba, fw, fh, stride, 0, 0, 0, 0, 0)
    pix = np.frombuffer(last, np.uint8).reshape(h, w, 4)
    # A corner of the hole, which no mark reaches, and the hole as a whole -- which must
    # contain some ink, or the dots have gone missing again.
    hole = pix[pane[1] + 10, pane[0] + 10]
    pane_pixels = pix[pane[1]:pane[1] + pane[3], pane[0]:pane[0] + pane[2]]
    ink = int((pane_pixels[:, :, 3] > 0).sum())
    clear = int(pane_pixels[:, :, 3].size - ink)
    print(f"  ในรูกล้อง: ทึบ {ink} px (จุดมาร์คและกรอบ)   โปร่งใส {clear} px (ภาพกล้อง)")
    # Both halves, because each has been wrong on a handset: no ink means the dots were
    # erased with the hole, and no clear pixels means the hole covered the camera.
    assert ink > 0, "ไม่มีจุดมาร์คในรูกล้องเลย"
    assert clear > ink, "รูแทบไม่โปร่งเลย กล้องจะถูกบัง"
    outside = int(pix[10, 10, 3])
    print(f"  พิกเซลในรูกล้อง = {tuple(int(v) for v in hole)} (ต้องเป็น 0 ทั้งสี่ช่อง)"
          f"   alpha นอกรู = {outside}")
    # Every channel, because Android bitmaps are premultiplied and a coloured pixel with
    # alpha 0 is undefined -- which is how the preview came to be covered by its own hole.
    assert not hole.any(), "รูกล้องยังมีสีค้างอยู่ (premultiplied alpha จะพัง)"
    assert outside == 255, "นอกรูต้องทึบ"

    # THE SENTINEL MUST NOT BE VISIBLE ANYWHERE. Scaling the screen down blends the magenta
    # into whatever was drawn against it, and the blend is far enough from magenta to be
    # kept: that put a pink hairline round the video and turned the white rim of every mark
    # pink, which is the rim's whole job gone. Magenta is red and blue high with green well
    # under both -- a white rim is not that, and a test that said it was is how this went
    # unnoticed once already.
    r, g, b = (pane_pixels[:, :, i].astype(int) for i in range(3))
    magenta = int(((r > 140) & (b > 140) & (g < np.minimum(r, b) - 60)
                   & (pane_pixels[:, :, 3] > 0)).sum())
    marks = int((np.abs(pane_pixels[:, :, :3].astype(int)
                        - np.array([16, 134, 68])).sum(2) < 40).sum())
    print(f"  สีรูรั่วออกมา {magenta} px (ต้องเป็น 0)   "
          f"จุดสีเขียว {marks} px จาก {app._S['count']} เม็ด")
    assert magenta == 0, "สีรูกล้อง (ชมพู) โผล่ให้เห็น"
    # One mark's green core is about 80 px at this scale, and the count standing here is
    # the sparse tray's -- so this says "at least one dot", not "as many as the tray has".
    assert marks > 50, "ไม่มีจุดสีเขียวบนยาเลย"

    # A tray nobody is touching must stop costing anything. The same frame over and over
    # is what a counted tray sitting on a bench looks like, and after the settle passes
    # the model should not run again at all -- which is most of why the app feels quick.
    still = camera_bytes(cv2.resize(frames[0], (640, 480)))
    costs = []
    for _ in range(8):
        time.sleep(0.3)
        t0 = time.perf_counter()
        app.frame(still[0], still[1], still[2], still[3], 0, 0, 0, 0, 0)
        costs.append((time.perf_counter() - t0) * 1000)
    print(f"  ถาดนิ่ง: เฟรมแรก ๆ {costs[0]:.0f} ms  ->  เฟรมหลัง ๆ {costs[-1]:.1f} ms")
    assert costs[-1] < costs[0] / 4, "ถาดนิ่งแล้วยังเรียกโมเดลอยู่"

    # Two things that were wrong on a handset and right in every test until they were
    # asked for by name: a message drawn over the video, and the records page.
    screen.saved(7, "20260917-120000")
    app._S["last_key"] = None
    shot = np.frombuffer(app.frame(rgba, fw, fh, stride, 0, 0, 0, 0, 0),
                         np.uint8).reshape(h, w, 4)
    middle = shot[h // 2, w // 2]
    print(f"  toast กลางจอ (ทับพื้นที่กล้อง) = {tuple(int(v) for v in middle)}")
    assert middle[3] == 255, "toast โปร่งใส กล้องจะบังข้อความ"

    screen.toast = None
    screen.page = "records"
    app._S["last_key"] = None
    page = np.frombuffer(app.frame(rgba, fw, fh, stride, 0, 0, 0, 0, 0),
                         np.uint8).reshape(h, w, 4)
    see_through = int((page[pane[1]:pane[1] + pane[3],
                            pane[0]:pane[0] + pane[2], 3] == 0).sum())
    print(f"  หน้ารายการ: พิกเซลโปร่งใสตรงพื้นที่กล้อง = {see_through} (ต้องเป็น 0)")
    assert see_through == 0, "หน้ารายการยังเจาะรู กล้องจะโผล่มาข้างหน้า"
    screen.page = "count"
    app._S["last_key"] = None
    app.frame(rgba, fw, fh, stride, 0, 0, 0, 0, 0)

    # The touches Android sends, aimed at the rectangles the screen actually drew rather
    # than at coordinates typed here -- which is the same mistake as drawing a button in
    # one place and hit-testing it in another.
    #
    # THE IMPORT SITS ABOVE THE FIRST LINE THAT USES THESE NAMES, and it has to: an import
    # anywhere inside a function makes that name local to the WHOLE function, so one
    # placed lower down turns every earlier use into UnboundLocalError. That is the exact
    # fault this file was written to catch on a desktop instead of on a handset -- and it
    # has now caught it in itself.
    from model3.phone.screen import H as DH, OUT_H, OUT_W, W as DW

    for want in ("target+", "preset60", "records", "back"):
        rect = next((r for n, r in screen.hits if n == want), None)
        if rect is None:
            print(f"  ไม่พบปุ่ม {want} บนหน้านี้")
            continue
        x = (rect[0] + rect[2] / 2) * OUT_W / DW
        y = (rect[1] + rect[3] / 2) * OUT_H / DH
        before = (screen.target, screen.page)
        app.touch("down", x, y)
        app.touch("up", x, y)
        print(f"  แตะ {want}: {before} -> {(screen.target, screen.page)}")
        app._S["last_key"] = None
        app.frame(rgba, fw, fh, stride, 0, 0, 0, 0, 0)

    # THE CAMERA STOPS. Nothing else in the bridge runs when it does -- every other path
    # begins with a frame arriving -- so this is the one call that has to work with no
    # camera at all, and the one that stops a count of a tray that is no longer there
    # being filed under the time somebody pressed save.
    app._S["last_frame_at"] = time.time() - 5
    app._S["last_key"] = None
    stalled = app.idle()
    assert stalled, "กล้องหยุดแล้วแต่หน้าจอไม่วาดใหม่"
    pix = np.frombuffer(stalled, np.uint8).reshape(h, w, 4)
    edge = pix[pane[1]:pane[1] + pane[3], pane[0]:pane[0] + pane[2]]
    red = int((np.abs(edge[:, :, :3].astype(int) - np.array([220, 60, 60])).sum(2) < 60).sum())
    print(f"  กล้องเงียบ 5 วิ: ห้ามบันทึกว่า {screen.blocked!r}   ขอบแดง {red} px")
    assert "หยุด" in screen.blocked, "ไม่ได้ห้ามบันทึกตอนกล้องหยุด"
    assert red > 500, "ไม่มีขอบแดงรอบภาพ"

    # And back: a camera that returns has to clear it, or the app is stuck refusing to
    # save on a bench where nothing is wrong any more.
    app._S["last_key"] = None
    app.frame(rgba, fw, fh, stride, 0, 0, 0, 0, 0)
    print(f"  กล้องกลับมา:    ห้ามบันทึกว่า {screen.blocked!r}")
    assert "หยุด" not in screen.blocked, "กล้องกลับมาแล้วยังบันทึกไม่ได้"

    # A PRESS THAT WOBBLES IS STILL A PRESS. Every tap in the loop above goes down and
    # up on the same pixel, which no thumb has ever done; the screen used to throw away
    # any press whose finger had travelled more than 24 design pixels -- a millimetre and
    # a half -- on every page, because a drag down the records list has to scroll it
    # instead. On the counting screen there is nothing to scroll and nothing to keep it
    # from swallowing presses, which is what "กดจำนวนไม่ค่อยติด" was.
    screen.page = "count"
    before_target = screen.target
    rect = next(r for n, r in screen.hits if n == "target+")
    tx = (rect[0] + rect[2] / 2) * OUT_W / DW
    ty = (rect[1] + rect[3] / 2) * OUT_H / DH
    app.touch("down", tx, ty)
    app.touch("move", tx + 9, ty + 26)          # a thumb rolling on the glass
    app.touch("up", tx + 9, ty + 26)
    print(f"  กดแบบนิ้วขยับ: จำนวน {before_target} -> {screen.target}")
    assert screen.target == before_target + 1, "นิ้วขยับนิดเดียวแล้วปุ่มไม่ติด"
    app._S["last_key"] = None
    app.frame(rgba, fw, fh, stride, 0, 0, 0, 0, 0)

    # THE NUMBER PAD IS DRAWN OVER THE HOLE, and the hole is what the camera shows
    # through. A three-pixel band of the pane's edge used to be forced transparent no
    # matter what was drawn on it, so the pad -- which is centred, and whose edge lands
    # exactly there -- had a slot of live video cut through it. Tested a little inside the
    # pad's own edge, because that edge is SUPPOSED to blend into the picture now.
    from model3.phone.screen import H as DH2, OUT_H as OH2, OUT_W as OW2, W as DW2

    screen.typing = ""
    app._S["last_key"] = None
    pad = np.frombuffer(app.frame(rgba, fw, fh, stride, 0, 0, 0, 0, 0),
                        np.uint8).reshape(h, w, 4)
    kx = int(((DW2 - 560) // 2 + 70) * OW2 / DW2)
    ky = int(((DH2 - 700) // 2 + 70) * OH2 / DH2)
    kw = int((560 - 140) * OW2 / DW2)
    kh = int((700 - 140) * OH2 / DH2)
    through = int((pad[ky:ky + kh, kx:kx + kw, 3] < 255).sum())
    print(f"  แป้นตัวเลข: พิกเซลที่กล้องทะลุขึ้นมา = {through} (ต้องเป็น 0)")
    assert through == 0, "มีภาพกล้องทะลุแป้นตัวเลข"
    screen.typing = None
    app._S["last_key"] = None
    app.frame(rgba, fw, fh, stride, 0, 0, 0, 0, 0)

    # A DRAG ON THE RECORDS LIST, IN THE COORDINATES ANDROID ACTUALLY SENDS: fractions
    # of a pixel. The scroll was assigned straight from them, subtracted from each row's
    # y, and that y indexes the canvas -- numpy will not take a float as a slice, so the
    # first drag on the list threw TypeError inside compose, Kotlin caught it, and the
    # phone showed "ประมวลผลภาพไม่สำเร็จ" over the page until it was scrolled back.
    # Every tap in the loop above happens to land on whole pixels, which is why none of
    # them found it.
    screen.page = "records"
    app._S["last_key"] = None
    app.frame(rgba, fw, fh, stride, 0, 0, 0, 0, 0)      # so the list has been laid out
    lx, ly, lw, lh = screen.list_box
    # In the middle of the list, in the fractional coordinates Android really sends, and
    # far enough to be a drag rather than a press.
    dx = (lx + lw / 2) * OUT_W / DW + 0.5
    dy0 = (ly + lh * 0.75) * OUT_H / DH + 0.25
    dy1 = (ly + lh * 0.25) * OUT_H / DH + 0.75
    app.touch("down", dx, dy0)
    app.touch("move", dx, dy1)
    app.touch("up", dx, dy1)
    print(f"  ลากรายการ: scroll = {screen.scroll!r}")
    assert isinstance(screen.scroll, int), "scroll ไม่ใช่จำนวนเต็ม จะใช้ index ภาพไม่ได้"
    assert screen.scroll > 0, "ลากกลางตารางแล้วรายการไม่เลื่อน"
    app._S["last_key"] = None
    app.frame(rgba, fw, fh, stride, 0, 0, 0, 0, 0)       # composes the records page
    screen.page = "count"
    screen.scroll = 0
    app._S["last_key"] = None
    app.frame(rgba, fw, fh, stride, 0, 0, 0, 0, 0)

    # THE COUNTING FRAME OUTLIVES THE APP. Tapped in through the screen's own touch path,
    # then read back by a second Screen on the same folder -- which is what the next
    # launch is, and what the phone had no answer for: every launch started with no
    # region and a count of the whole picture, on a bench whose tray has not moved.
    #
    # FOUR CORNERS, AND THE LAST TWO GO IN THE WRONG ORDER ON PURPOSE. Tapped in that
    # order they would make a bow tie, and quad() is what turns any four taps into a shape
    # with no crossing edges; a test that only ever tapped them round the rim would never
    # execute the line that matters.
    from model3.phone.screen import pane_out, Screen as Fresh

    screen.page = "count"
    screen.arming = True
    screen.pending = []
    # pane_out(), not PANE. Android sends touches in the coordinates of the bitmap it was
    # given, which is the SCALED screen; PANE is the design-size rectangle, half as big
    # again. The earlier version of this test passed design coordinates and still got a
    # region, because the drag it was testing clamped whatever it was handed back into the
    # frame -- so the test passed while aiming a third of the way off the tray.
    fx, fy, fw_, fh_ = pane_out()
    corners = [(fx + 100, fy + 90), (fx + fw_ - 120, fy + fh_ - 110),
               (fx + fw_ - 120, fy + 90), (fx + 100, fy + fh_ - 110)]
    for tx, ty in corners:
        screen.touch("down", tx, ty, (480, 640, 3))
        screen.touch("up", tx, ty, (480, 640, 3))
    kept = Fresh(ASSETS, records)
    kept.compose(cv2.resize(frames[0], (640, 480)), np.zeros((0, 4), np.float32), 0, 0.0)
    print(f"  กรอบนับ: แตะ 4 มุมแล้วได้ {screen.roi is not None}   "
          f"{screen.roi}   เปิดใหม่ยังอยู่ {kept.roi == screen.roi}")
    assert screen.roi, "แตะครบสี่มุมแล้วไม่ได้กรอบ"
    assert not screen.arming, "มุมที่สี่แล้วยังไม่ปิดกรอบ"
    assert len(screen.roi) == 4 and len(set(map(tuple, screen.roi))) == 4, "มุมซ้ำกัน"
    assert kept.roi == screen.roi, "เปิดโปรแกรมใหม่แล้วกรอบนับหาย"
    # THE BOW TIE THE TAP ORDER WOULD OTHERWISE HAVE MADE. Four corners of a rectangle
    # joined 1-2-3-4 in the order tapped above cross in the middle and enclose ZERO area;
    # joined round the rim they enclose the whole rectangle. So the shoelace area of what
    # was saved, against the area of its own bounding box, is the test -- and it is the
    # difference between counting the tray and counting nothing.
    r = screen.roi
    area = abs(sum(r[i][0] * r[(i + 1) % 4][1] - r[(i + 1) % 4][0] * r[i][1]
                   for i in range(4))) / 2.0
    box = ((max(q[0] for q in r) - min(q[0] for q in r))
           * (max(q[1] for q in r) - min(q[1] for q in r)))
    print(f"  กรอบนับ: พื้นที่ {area:.0f} จาก {box} (โบว์ไทจะได้ 0)")
    assert area > 0.9 * box, "มุมเรียงไขว้กัน กรอบเป็นโบว์ไท"

    print(f"\nรันครบ {ok} เฟรม ไม่มี exception")
    print("records:", records)


if __name__ == "__main__":
    main()
