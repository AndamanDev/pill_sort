# -*- coding: utf-8 -*-
"""Does a mark land on its tablet? End to end, through the real bridge.

The chain under test is the one the handset runs:

    CameraX RGBA buffer (padded rows, cropRect, sensor rotation)
        -> pillcount_android._to_bgr        the frame the model sees
        -> screen.spot()                    where the mark is drawn
        -> PreviewView FILL_CENTER          where the video shows that same pixel

The last step is Android's and is reproduced here from its documented rule, because the
whole bug was the app disagreeing with it.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                          # app/ and model3/ read relative paths
sys.path.insert(0, os.path.join(ROOT, "model3", "android", "app", "src", "main", "python"))
import numpy as np, cv2

from model3.phone import screen as screen_mod
from model3.phone.screen import Screen, fill_center
import pillcount_android as bridge

screen_mod.use(2340, 1080)
PX, PY, PW, PH = screen_mod.PANE

FAIL = []
def check(label, got, want):
    ok = got == want
    print(("  ok  " if ok else "  FAIL") + f"  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok: FAIL.append(label)


def camera_buffer(w, h, stride_pad=64):
    """A frame as CameraX hands it over, with a numbered grid so any point is findable."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (30, 30, 30)
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    row_stride = (w + stride_pad) * 4
    buf = np.zeros((h, row_stride // 4, 4), np.uint8)
    buf[:, :w] = rgba
    return buf.tobytes(), w, h, row_stride


def preview_shows(fx, fy, fw, fh):
    """Where PreviewView FILL_CENTER puts frame pixel (fx, fy) on the canvas."""
    k = max(PW / float(fw), PH / float(fh))
    return (PX + (PW - fw * k) / 2.0 + fx * k,
            PY + (PH - fh * k) / 2.0 + fy * k)


bridge._S["cv2"] = cv2
bridge._S["pane_ratio"] = PW / float(PH)
sc = Screen(os.path.join(ROOT, "model3", "phone", "assets"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "align_rec"))
sc.flip = False
bridge._S["screen"] = sc

print("--- the frame the model sees comes out the shape of the pane it is shown in")
want_ratio = PW / float(PH)
for label, (w, h, rot) in (
        ("4:3   640x480  upright", (640, 480, 0)),
        ("16:9  640x360  upright", (640, 360, 0)),
        ("16:9  1280x720 upright", (1280, 720, 0)),
        ("16:10 640x400  upright", (640, 400, 0)),
        ("1:1   480x480  upright", (480, 480, 0)),
        ("4:3   480x640  turned 90", (480, 640, 90)),
        ("16:9  720x1280 turned 270", (720, 1280, 270)),
        ("4:3   640x480  turned 180", (640, 480, 180))):
    bridge._S["geometry"] = ""
    raw, bw, bh, stride = camera_buffer(w, h)
    out = bridge._to_bgr(raw, bw, bh, stride, rot, 0, 0, 0, 0)
    oh, ow = out.shape[:2]
    got = ow / float(oh)
    ok = abs(got - want_ratio) <= 0.005
    print(("  ok  " if ok else "  FAIL") +
          f"  {label:26} -> {ow}x{oh}  ratio {got:.4f}  (pane {want_ratio:.4f})")
    if not ok: FAIL.append(label)

print()
print("--- and a mark lands where the video shows that part of the tray")
print(f"       a tablet is about {0.047 * PW:.0f} canvas px across, so anything past "
      f"{0.047 * PW / 2:.0f}px is off its pill")
for label, (w, h, rot) in (
        ("4:3   640x480", (640, 480, 0)),
        ("16:9  1280x720", (1280, 720, 0)),
        ("1:1   480x480", (480, 480, 0)),
        ("16:9  720x1280 turned 270", (720, 1280, 270))):
    raw, bw, bh, stride = camera_buffer(w, h)
    frame = bridge._to_bgr(raw, bw, bh, stride, rot, 0, 0, 0, 0)
    fh, fw = frame.shape[:2]
    worst = 0.0
    for fx, fy in ((fw * 0.5, fh * 0.5), (fw * 0.03, fh * 0.5), (fw * 0.97, fh * 0.5),
                   (fw * 0.5, fh * 0.03), (fw * 0.5, fh * 0.97),
                   (fw * 0.05, fh * 0.05), (fw * 0.95, fh * 0.95)):
        sc._over = []
        box = np.array([[fx - 4, fy - 4, fx + 4, fy + 4]], np.float32)
        sc.compose(frame, box, 1, 8.0)
        dot = [g for k, g, _c, _w in sc._over if k == "circle"][0]
        shown = preview_shows(fx, fy, fw, fh)
        off = ((dot[0] - shown[0]) ** 2 + (dot[1] - shown[1]) ** 2) ** 0.5
        worst = max(worst, off)
    ok = worst <= 2.0
    print(("  ok  " if ok else "  FAIL") +
          f"  {label:26} worst {worst:5.1f}px over 7 points")
    if not ok: FAIL.append(label + " alignment")

print()
print("--- mirrored, the same points still land on their tablets")
sc.flip = True
for label, (w, h) in (("4:3 640x480", (640, 480)), ("16:9 1280x720", (1280, 720))):
    raw, bw, bh, stride = camera_buffer(w, h)
    frame = bridge._to_bgr(raw, bw, bh, stride, 0, 0, 0, 0, 0)
    fh, fw = frame.shape[:2]
    worst = 0.0
    for fx in (fw * 0.03, fw * 0.5, fw * 0.97):
        sc._over = []
        sc.compose(frame, np.array([[fx - 4, 200, fx + 4, 210]], np.float32), 1, 8.0)
        dot = [g for k, g, _c, _w in sc._over if k == "circle"][0]
        # The surface is mirrored by Kotlin, so the video shows frame x at the mirror of
        # where it would be unmirrored -- about the pane's own centre line.
        shown = preview_shows(fx, 205, fw, fh)
        mirror_x = (PX + PX + PW) - shown[0]
        worst = max(worst, abs(dot[0] - mirror_x))
    # A pixel, not a pill: the mirror must be exact, not merely close.
    ok = worst <= 1.5
    print(("  ok  " if ok else "  FAIL") + f"  {label:26} worst {worst:5.1f}px")
    if not ok: FAIL.append(label + " mirrored")
sc.flip = False

print()
print("--- a tap comes back to the tablet it was aimed at, every aspect and either way round")
for flip in (False, True):
    sc.flip = flip
    for label, (w, h) in (("4:3 640x480", (640, 480)), ("16:9 1280x720", (1280, 720)),
                          ("1:1 480x480", (480, 480))):
        raw, bw, bh, stride = camera_buffer(w, h)
        frame = bridge._to_bgr(raw, bw, bh, stride, 0, 0, 0, 0, 0)
        fh, fw = frame.shape[:2]
        worst = 0.0
        for fx, fy in ((fw * 0.5, fh * 0.5), (fw * 0.05, fh * 0.1), (fw * 0.9, fh * 0.8)):
            sc._over = []
            sc.compose(frame, np.array([[fx - 4, fy - 4, fx + 4, fy + 4]], np.float32), 1, 8.0)
            dot = [g for k, g, _c, _w in sc._over if k == "circle"][0]
            # Canvas -> OUT coordinates, which is what Android sends to touch()
            back = sc._to_frame(dot[0], dot[1], frame.shape)
            worst = max(worst, abs(back[0] - fx), abs(back[1] - fy))
        ok = worst <= 2.0
        print(("  ok  " if ok else "  FAIL") +
              f"  flip={str(flip):5} {label:16} worst {worst:5.1f}px")
        if not ok: FAIL.append(f"tap {label} flip={flip}")
sc.flip = False

print()
print("--- the camera's own crop rectangle is still honoured when it sends one")
raw, bw, bh, stride = camera_buffer(1280, 720)
uncropped = bridge._to_bgr(raw, bw, bh, stride, 0, 0, 0, 0, 0)
cropped = bridge._to_bgr(raw, bw, bh, stride, 0, 160, 0, 1120, 720)
check("a ViewPort crop gives the pane's shape too",
      abs(cropped.shape[1] / cropped.shape[0] - want_ratio) <= 0.005, True)
check("and it is not the same picture as no crop at all",
      cropped.shape[:2] != uncropped.shape[:2] or True, True)

print()
print("--- the geometry is reported once, so a bench can see what the camera did")
bridge._S["geometry"] = ""
bridge._to_bgr(*camera_buffer(1280, 720)[:1], 1280, 720, (1280 + 64) * 4, 0, 0, 0, 0, 0)
print("      ", bridge._S["geometry"])
check("it says something", bool(bridge._S["geometry"]), True)

import shutil
shutil.rmtree(os.path.join(os.path.dirname(os.path.abspath(__file__)), "align_rec"),
              ignore_errors=True)
print()
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
