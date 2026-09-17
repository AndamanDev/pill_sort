"""Count the tablets inside a region you draw, live. The instrument, not the product.

    1-ทดสอบโมเดล.bat                                        the easy way
    python model3/bench.py                                  the camera
    python model3/bench.py --source clip.mp4 --save out.mp4  a recording
    python model3/bench.py --model model3/weights/pillcount-det-v2.pt   an older one

Run it with the interpreter that has torch:

    D:\\pill-counter-lite\\pillcount-v12\\.venv-train\\Scripts\\python.exe model3/bench.py

MOVED HERE FROM model2/count.py and repointed at model3's own weights. model1 and model2
are kept as they were, and nothing that runs day to day reaches into them any more. Their
names still appear below wherever a measurement was taken there -- that is history, not a
dependency.

WHAT IT IS FOR, and why app/ does not replace it: this prints the SPREAD of the count
over a session and lets nothing else compete for the screen. No target, no buttons, no
smoothing. "How good is this model" is asked here; "how many pills are on this tray" is
asked in the window.

This is section 5 of the Labellerr cookbook -- the RealTimePillCounter -- against a live
camera rather than a file, which is what a bench has. Click a polygon around the tray,
and the number is how many tracked tablets are inside it RIGHT NOW. Not cumulative: a
tray that has had ninety tablets pass over it still reads sixty when sixty are on it.

TRACKING IS OFF BY DEFAULT, WHICH IS NOT WHAT THE COOKBOOK SAYS. Its counter turns
ByteTrack on for "stable counting", and the argument is a good one: an identity carried
between frames should hold a tablet the detector is briefly unsure about. Measured on
forty real frames of a tray nobody was touching, with this model:

    448px  conf 0.15  detection only     median 60   spread  6
    448px  conf 0.15  with ByteTrack     median 32   spread 13

It loses half the tray. A tracker confirms a new track over several frames before it
reports it, and with sixty near-identical tablets packed together and none of them
moving there is nothing for the association step to work with -- so most never graduate
from tentative. Tracking is for a few objects moving THROUGH a scene, which is what the
cookbook's other examples are. Pass `--track` to watch it happen.

The same measurement fixes two more defaults. conf 0.15, not the cookbook's 0.6: at 0.6
this model finds 33 tablets of about 61. And imgsz 448, not 640 or 960 -- it was TRAINED
at 448, and at 960 the median is still 60 but the spread is 57.

WHAT THE MARKERS ARE. --marker dot is the default and stays out of the way on a full
tray. --marker number writes 1..N across the pills IN READING ORDER, which makes the
count checkable by eye: the highest number on screen must equal the number in the corner,
and a repeat or a gap is a fault you can point at. --marker box draws what the detector
actually outlined, which is the view to use when a number looks wrong.

The dot is black inside a white rim, and the numbers are white with a black rim, because
those are what stay legible on everything in shot at once. Rendered side by side on a
frame holding white tablets, a cyan capsule, a black capsule, a tan oblong and the blue
tray: a plain black dot DISAPPEARS on the dark capsule, and the random palette this file
used to draw threw pale greens and tans that sank into the tablets. The rim is what does
the work -- take it off and the dark capsules lose their markers.

Outlines and boxes keep a per-position colour, since two adjacent outlines in one colour
read as a single shape. That colour comes from POSITION, never from the detector's output
order -- that order is by confidence and reshuffles every frame, and on a tray nobody was
touching one tablet took sixteen different colours across forty frames. The count was
steady the whole time; only the screen was lying about it.

WHAT THE ROI IS FOR. The same job the counting frame does in model1: the bench, the
operator's hands and the tray next to this one are all in view, and none of them should
be counted. A polygon rather than a rectangle because a tray photographed at an angle is
not a rectangle.

WHAT IT PRINTS ON EXIT is the point of running it: min, max and the spread of the count
over the session. On a still tray that spread is the model's steadiness, and it is the
number to compare before and after a retrain.

THERE IS NOW A BETTER MODEL TO POINT THIS AT, and it wants different flags:

    python model3/bench.py

All four of those are the DEFAULTS here, because pillcount-det-v3 is the
default model. Pass them only to override.


Every one of those four numbers was measured, and the three sections below say how. That
is a DETECTION model trained on model3/dataset (21,022 labelled tablets from
Roboflow Universe, CC BY 4.0) -- so boxes, no masks, and this file draws dots for it
rather than outlines. Measured on the forty frames of model2/frames/tray_full.mp4, the
same clip the numbers at the top of this docstring come from:

    448px  conf 0.25  pillcount-det     median 61   spread  2    75 ms
    640px  conf 0.25  pillcount-det     median 61   spread  1    42 ms   <- use this
    960px  conf 0.25  pillcount-det     median 63   spread 11    78 ms
    448px  conf 0.15  pillsort-seg      median 60   spread  6            (shipped)

640, not this file's default 448, because that is what it was trained at -- the same
argument that put the seg model at 448, pointing the other way. It is steadier by six
times and about twice as fast, and its confidences spread 0.72-0.93 instead of saturating.

AND IT WANTS --iou 0.4, which took a real tray to find. On a live frame it put TWO boxes
on each of two tablets -- 56% and 58% of one box covered by the other, centres 13 px
apart. NMS drops a box that overlaps a better one by MORE than the threshold, and these
landed at 0.50 against a threshold of 0.50, so they survived by a rounding error. The
same frame, model unchanged, only --iou moved:

    iou 0.70    35 boxes    5 duplicated
    iou 0.50    34 boxes    4 duplicated      <- this file's default
    iou 0.45    32 boxes    0 duplicated
    iou 0.20    32 boxes    0 duplicated

One frame is not enough to pick a number, and the second one saved off the same bench
proves it -- 0.45 was NOT sufficient there:

    iou 0.50    41 boxes    2 duplicated
    iou 0.45    41 boxes    2 duplicated      <- still doubled
    iou 0.40    40 boxes    0 duplicated by the >50% test
    iou 0.30    40 boxes    0
    iou 0.20    39 boxes    0

So the band is bounded on BOTH sides: above 0.45 tablets get counted twice, and by 0.20
two touching tablets are being counted once. 0.4 sits inside it on both frames.

AND --conf 0.45, for a reason iou cannot fix. Zooming into what survived at 0.4 on that
second frame showed one tablet LYING ON ITS EDGE with two boxes straddling it, each
covering about half. They overlap 44% -- under the >50% test above, and IoU 0.29, under
any workable NMS threshold -- so neither the check nor the suppression could see them.
What separates them is confidence: 0.41 and 0.51 against a median of 0.85 for tablets
lying flat. The true count there is 39, not the 40 it first looked like.

    conf        frame A (39 true)   frame B (32 true)
    0.25            40  (+1)            32
    0.45            39                  32            <- both correct
    0.55            38  (-1)            32
    0.65            38  (-1)            31  (-1)

An edge-on tablet is the weak spot of this model, and the training set is why: the images
in model3/dataset are trays of tablets lying flat. Raising conf hides the symptom. The
cure is frames of tablets on their edge, labelled, in a retrain.

Worth knowing HOW this was found, because the measurement nearly missed it: forty frames
of tray_full.mp4 show zero duplicates at every threshold from 0.2 to 0.7. The clip simply
never arranges two tablets that way. It took pressing `s` on a live bench at the moment
the number looked wrong -- which is what that key is for.

Its defaults are NOT this file's defaults, and they are not changed for it: --conf 0.15,
--imgsz 448 and --iou 0.5 are measured facts about pillsort-seg and stay true. Pass the
flags:

    python model3/bench.py

All four of those are the DEFAULTS here, because pillcount-det-v3 is the
default model. Pass them only to override.

"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from collections import Counter as Tally

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_MODEL = os.path.join(HERE, "weights", "pillcount-det-v3.pt")

ROI_LINE = (90, 220, 110)
INK = (245, 245, 245)

#: The marker, in BGR: the app's green inside a white rim, the same pair app/window.py
#: and the phone draw -- this is where an operator checks the model, and a tool that
#: marks pills differently from the app is a tool that has to be translated.
#:
#: The rim is not decoration. A plain dot VANISHES on the dark capsules on this tray,
#: which is visible the moment two styles are rendered side by side on one real frame;
#: the random palette this file drew before threw pale greens and tans that sank into
#: the tablets. The colour inside it was black until it was not: black is what the
#: shadow between two touching pills already looks like.
MARK = (68, 134, 16)                        # #108644, as in app/theme.py
MARK_RIM = (255, 255, 255)


def _open(source):
    """A camera index or a path, opened the way the rest of this project opens one."""
    if isinstance(source, int) or str(source).isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(source))
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:                                      # noqa: BLE001
            pass
        return cap
    return cv2.VideoCapture(str(source))


def select_roi(frame, title="draw the tray -- click corners, SPACE to accept", cap=None):
    """Click the corners of the region to count inside. Returns a list of points.

    SPACE accepts, r starts over, Esc accepts what is there or gives up on it. Three
    points minimum, because two do not enclose anything.

    PASS `cap` AND THE VIEW IS LIVE. Without it this draws on the single frame it was
    handed, which is what it used to do always, and an operator who moves the tray and
    sees nothing move cannot tell a frozen window from a dead camera. That is the whole
    bug: the picture was never updating.
    """
    points = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)
    print("  click the corners of the tray   right-click undoes one")
    print("  SPACE accept    r restart    Esc count the whole view")
    while True:
        if cap is not None:
            ok, fresh = cap.read()
            if ok:
                frame = fresh          # keep the last good one if a read drops
        shown = frame.copy()
        if len(points) >= 3:
            pts = np.array(points, np.int32).reshape(-1, 1, 2)
            wash = shown.copy()
            cv2.fillPoly(wash, [pts], ROI_LINE)
            cv2.addWeighted(wash, 0.18, shown, 0.82, 0, shown)
            cv2.polylines(shown, [pts], True, ROI_LINE, 2, cv2.LINE_AA)
        elif len(points) == 2:
            cv2.line(shown, points[0], points[1], ROI_LINE, 2, cv2.LINE_AA)
        for i, p in enumerate(points):
            cv2.circle(shown, p, 6, ROI_LINE, -1, cv2.LINE_AA)
            cv2.putText(shown, str(i + 1), (p[0] + 9, p[1] - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ROI_LINE, 1, cv2.LINE_AA)
        cv2.putText(shown, f"{len(points)} corners   SPACE accept   r restart   Esc skip",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(shown, f"{len(points)} corners   SPACE accept   r restart   Esc skip",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, INK, 1, cv2.LINE_AA)
        cv2.imshow(title, shown)
        key = cv2.waitKey(20) & 0xFF
        if key == ord(" ") and len(points) >= 3:
            break
        if key == ord("r"):
            points.clear()
        if key == 27:
            break
    cv2.destroyWindow(title)
    return points if len(points) >= 3 else None


def _inside(roi, point) -> bool:
    if not roi:
        return True
    return cv2.pointPolygonTest(np.array(roi, np.int32), point, False) >= 0


#: A split capsule shows up as two boxes inside one bright blob. These three numbers are
#: what separate it from a pile of tablets that merely touch -- see merge_split().
MERGE_ELONGATION = 1.8
MERGE_MAX_FILL = 0.65
MERGE_MIN_BLOB = 250


def merge_split(frame, boxes, roi, elongation=MERGE_ELONGATION,
                max_fill=MERGE_MAX_FILL):
    """Group boxes that are two halves of one pill. Returns a list of index groups.

    THE PROBLEM THIS SOLVES. A two-tone capsule -- white one end, cyan the other -- comes
    back as TWO boxes, one per colour. Measured on this bench: four capsules, six boxes,
    and the two solid-coloured ones were fine. The colour join inside the capsule is as
    hard an edge as the capsule's own outline, and nothing in the detector says an object
    may not contain one. Neither threshold helps: the two halves overlap 13-44%, so
    suppressing them means an --iou low enough to merge tablets that are merely touching.

    THE FIX IS NOT THE DETECTOR, IT IS THE TRAY. Against dark blue plastic a capsule is
    one connected bright region, colour join and all. So: segment, and if two boxes land
    in one region, ask whether that region looks like a capsule.

    "LOOKS LIKE A CAPSULE" IS THREE TESTS, AND ALL THREE ARE LOAD-BEARING. The naive rule
    -- one blob is one pill -- turns a packed tray of twenty-five tablets into one. So:

      exactly two boxes in the blob   a pile has twenty-five, and is skipped outright
      elongation >= 1.8              measured with minAreaRect, NOT the bounding box: a
                                     capsule lying diagonally has a near-square bounding
                                     box and would fail the obvious test
      FILL < 0.65                    blob area over the two boxes' area, and the only
                                     test that actually separates the two cases

    FILL IS THE ONE THAT MATTERS, and it took a false merge on the bench to find it. The
    first version of this rule used elongation and a loose area bound, and on a live tray
    it merged three pairs of ROUND TABLETS that were merely touching -- undercounting by
    three. Two touching tablets are elongated (≈2.0) and convex, so neither elongation,
    solidity nor a waist measurement tells them from a capsule. Measured:

                          elongation   solidity   waist    fill
        split capsules      2.92 2.38    0.93     .75 .61   0.55  0.55
        touching tablets  1.62 - 2.02  0.89-0.95  .21-.61   0.75 - 0.86

    Only fill separates them, and it separates them by geometry rather than luck: a round
    tablet fills pi/4 = 0.785 of its own bounding box, always. A capsule HALF wearing a
    box this detector sized for a round tablet fills about 0.55. The gap is the model's
    round-pill prior showing through, so it holds as long as the detector keeps that
    prior -- if a future model boxes capsule halves tightly, this number must be
    re-measured.

    MEASURED, on frames saved off this bench:

        capsules only        6 boxes -> 4   (true 4)    both two-tone capsules merged
        packed tablets      69 boxes -> 69              nothing touched
        packed tablets      69 boxes -> 69              nothing touched
        touching tablets    19 boxes -> 19              the frame that caught the bug
        edge-on tablet      34 boxes -> 34  (true 32)   NOT repaired -- see below

    WHAT THIS DELIBERATELY GIVES UP. The loose first version did fix that last frame, an
    edge-on tablet cut in two, taking 34 to a correct 32. The fill test gives that back:
    the halves of a tablet standing on edge are round-ish and fill their boxes like any
    tablet, so they read as two touching tablets and are left alone.

    That trade is taken on purpose. Merging two real tablets into one makes the counter
    say 29 when the tray holds 30, and a pharmacy that trusts it dispenses short. Leaving
    an edge-on tablet counted twice says 31 when it holds 30 -- wrong, but wrong in the
    direction an operator notices, and visible on screen as two markers on one tablet.
    When the rule is uncertain it must decline to merge.

    THE OTHER THING IT GIVES UP, and the one an operator will notice first: a capsule
    only merges when it is lying CLEAR of the other pills. Touching a tablet puts it in
    the pile's blob -- twenty-two boxes in one measured case -- and "exactly two" rejects
    it. So on a full tray some capsules still read as two. That is the rule declining,
    not failing.

    DO NOT "FIX" THIS BY MEASURING THE PAIR INSTEAD OF THE BLOB. It was tried: take the
    two boxes' union, count foreground pixels inside it, divide by the two boxes' area,
    and the blob's box count never enters. Capsule pairs came back 0.14-0.57, which looks
    decisive until the frames that must not merge are measured too -- touching tablets
    ran 0.37-0.91 and a packed tray 0.24-1.70, putting SEVEN pairs under the 0.65 line
    across two frames that should yield none. The "exactly two boxes in the blob" test is
    what was holding the pile out, not fill. Removing it trades a handful of capsules for
    a risk of undercounting a full tray.

    The real fix for a capsule in a pile is the detector not splitting it -- training
    data, not geometry. See model3/README.md.

    AND THAT FIX ARRIVED, WHICH IS WHY THIS IS OFF BY DEFAULT. pillcount-det-v3, trained
    on two more datasets including two-tone capsules, does not split them at all. On the
    three frames with known answers it is right on the raw boxes:

                        true   v2 raw   v3 raw   v3 + this rule
        capsules only      4        6        4         4
        capsule in pile   23       24       23        23
        round tablets     61       67       61        60   <- the rule COSTS one

    With nothing left to repair the rule can only make mistakes, and on a full tray it
    makes one. Pass --merge to turn it on, which is worth doing only with v2 or older.
    """
    groups = [[i] for i in range(len(boxes))]
    if len(boxes) < 2:
        return groups

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(grey.shape, np.uint8)
    if roi:
        cv2.fillPoly(mask, [np.array(roi, np.int32)], 255)
    else:
        mask[:] = 255
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_and(binary, mask)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    h, w = labels.shape
    by_blob = {}
    for i, b in enumerate(boxes):
        cx = int((b[0] + b[2]) / 2)
        cy = int((b[1] + b[3]) / 2)
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        lab = int(labels[cy, cx])
        if lab:
            by_blob.setdefault(lab, []).append(i)

    merged = []
    taken = set()
    for lab, members in by_blob.items():
        if len(members) != 2 or stats[lab, cv2.CC_STAT_AREA] < MERGE_MIN_BLOB:
            continue
        comp = (labels == lab).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        (_, _), (bw, bh), _ = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        if max(bw, bh) / max(min(bw, bh), 1e-6) < elongation:
            continue
        box_area = sum((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                       for i in members)
        if not box_area or stats[lab, cv2.CC_STAT_AREA] / box_area >= max_fill:
            continue
        merged.append(members)
        taken.update(members)

    return merged + [[i] for i in range(len(boxes)) if i not in taken]


def _colour(track_id: int):
    """A stable colour per identity, so a tablet keeps its colour while it is tracked."""
    rng = np.random.default_rng(int(track_id) * 2654435761 % (2 ** 32))
    return tuple(int(v) for v in rng.integers(70, 255, 3))


def reading_order(units, boxes, row=28):
    """Sort units top-to-bottom, left-to-right, so the numbers drawn on them hold still.

    NUMBERING BY DETECTION INDEX WOULD BE WORSE THAN THE COLOURS WERE. The list comes
    back ordered by confidence, which reshuffles every frame; a tablet would be 3, then
    17, then 8. A colour that flickers is noise, but a NUMBER that flickers reads as
    meaning -- the operator counts along it and it lies.

    Position is stable when the tray is. Rows first, because a pure top-to-bottom sort
    puts two tablets side by side in an order that flips whenever one box shifts a pixel;
    quantising y into bands about two-thirds of a tablet high keeps a row together and
    then sorts it left to right, which is how a person reads a tray anyway.
    """
    def key(unit):
        b = boxes[unit[0]]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        return (int(cy) // row, cx)
    return sorted(units, key=key)


def _colour_at(cx, cy, cell=24):
    """A colour from WHERE a pill is, not from what number it came back as.

    WHY THE DOTS USED TO STROBE. Without --track the ids are just `arange(len(boxes))` --
    a pill's "identity" is its position in the detector's output list, and that list is
    ordered by confidence, which reshuffles every frame. A tray nobody is touching would
    repaint every dot a new colour thirty times a second. The count was steady; only the
    colours were noise, and the screen looked far less trustworthy than the number it was
    showing.

    Position is the stable thing when nothing is moving. Snapping to a 24px grid -- about
    half a tablet -- keeps the colour constant while the box jitters a pixel or two, and
    the odd pill that sits on a grid line and flickers between two colours is a much
    smaller lie than all of them flickering at once.

    This is cosmetic on purpose. If identities that survive real movement are ever needed,
    that is --track, and the docstring at the top says what it costs.
    """
    gx, gy = int(cx) // cell, int(cy) // cell
    return _colour((gx * 73856093) ^ (gy * 19349663))


def set_exposure(cap, value) -> bool:
    """Set exposure. `value` is a number, or "auto". Returns whether the camera took it.

    WHY THERE IS A KNOB FOR THIS. Camera 0 on this bench sits wherever the driver left it,
    and a dark frame is dangerous rather than obviously broken: the model still finds
    tablets in it and reports a plausible number while every confidence sags and the count
    wanders. Measured over twelve frames a setting, same tray, model unchanged:

        exposure -7   brightness   6.6   median 54   spread 4   conf median 0.69
        exposure -5   brightness  36.0   median 62   spread 4   conf median 0.82
        exposure -3   brightness  97.6   median 59   spread 1   conf median 0.88
        exposure -2   brightness 134.3   median 58   spread 2   conf median 0.87

    THE DEFAULT IS "auto" AND -3 IS NOT A FACT ABOUT THIS CAMERA. It was the default here
    for a long time on the strength of that table, and the table is real -- but it was
    measured in a bright room, and what it actually records is how much light -3 needed.
    Re-measured on the same camera in the same spot one evening:

        exposure -3    8.0 fps   brightness   4.5    <- the same setting, unusable
        AUTOMATIC     20.0 fps   brightness  53.9

    Two things went wrong with a fixed number. The picture went black when the room did,
    and -- because this scale is SHUTTER TIME -- the brighter manual steps collapse the
    frame rate: -1 is about half a second a frame, which is two frames a second. A whole
    evening was spent blaming a counting app for lag that was this setting.

    Automatic can raise GAIN, which costs no time, so it wins on both counts and keeps
    winning when somebody turns a lamp on. Pass a number when a FIXED exposure is the
    point -- a repeatable measurement -- and check the frame rate when you do.

    See set-camera-exposure.py in the repository root, which measures both.
    """
    if value is None:
        return False
    if isinstance(value, str) and value.lower() == "auto":
        # 1 is automatic on this camera through DSHOW; 0.75 hands control to the EXPOSURE
        # property. Measured on this backend rather than taken from the usual 0.25/0.75
        # convention, which did not match what it did.
        return bool(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1))
    if value == 0:
        return False
    # Manual first or DSHOW ignores the EXPOSURE write.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    return bool(cap.set(cv2.CAP_PROP_EXPOSURE, float(value)))


def exposure_arg(text):
    """"auto", or a number. Used as an argparse type so a typo fails at the command line."""
    if text.lower() == "auto":
        return "auto"
    return float(text)


def run(source=0, model_path=DEFAULT_MODEL, conf=0.15, iou=0.5, imgsz=448, save=None,
        masks=True, ids=False, track=False, exposure="auto",
        merge=False, marker="dot") -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics is not installed in this interpreter -- tracking needs it.\n"
            "    D:\\pill-counter-lite\\pillcount-v12\\.venv-train\\Scripts\\python.exe "
            "model3/bench.py")
    if not os.path.isfile(model_path):
        raise SystemExit(f"model not found: {model_path}")

    model = YOLO(model_path)
    cap = _open(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {source!r}")
    if set_exposure(cap, exposure):
        print(f"exposure    : {exposure}"
              f"{'' if exposure == 'auto' else ' (manual -- check the frame rate)'}")

    # Throw away the first frames. Cheap, and a camera that IS ramping its exposure
    # should not hand its darkest frame to the size check below. This bench's camera 0
    # does not ramp -- see --exposure.
    first = None
    for _ in range(15):
        ok, frame = cap.read()
        if ok:
            first = frame
    if first is None:
        cap.release()
        raise SystemExit(f"{source!r} opened but gave no frames -- another window may "
                         f"have the camera")
    h, w = first.shape[:2]
    print(f"model       : {os.path.basename(model_path)}")
    print(f"source      : {source}   {w}x{h}")
    print(f"conf        : {conf}   iou {iou}   imgsz {imgsz}   "
          f"tracking {'ON -- see the docstring' if track else 'off'}")
    print()

    roi = select_roi(first, cap=cap)
    print(f"region      : {'the whole view' if roi is None else f'{len(roi)} corners'}")
    print("  q / Esc  quit      r  redraw the region      s  save this frame + its boxes")

    writer = None
    if save:
        writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))

    title = "model2 count -- q quits, s saves this frame"
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    counts, frames, started = [], 0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1

            if track:
                # persist=True is what makes the ids survive between calls; without it
                # every frame starts a new set and tracking buys nothing.
                res = model.track(frame, persist=True, conf=conf, iou=iou, imgsz=imgsz,
                                  verbose=False)
            else:
                res = model(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
            r = res[0]

            shown = frame.copy()
            count = 0
            boxes = r.boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                tids = (boxes.id.cpu().numpy().astype(int)
                        if (track and boxes.id is not None) else np.arange(len(xyxy)))
                mm = r.masks
                # Only the boxes being counted go into the merge -- an off-tray object
                # must not be able to pair with a tablet on it.
                keep = [i for i in range(len(xyxy))
                        if _inside(roi, (int((xyxy[i][0] + xyxy[i][2]) / 2),
                                         int((xyxy[i][1] + xyxy[i][3]) / 2)))]
                if merge and mm is None and len(keep) > 1:
                    units = merge_split(frame, [xyxy[i] for i in keep], roi)
                    units = [[keep[j] for j in u] for u in units]
                else:
                    units = [[i] for i in keep]
                count = len(units)
                # Numbered markers are only worth anything in a stable order.
                if marker == "number":
                    units = reading_order(units, xyxy)
                # A merged pair draws as ONE marker at the joined centre, so the screen
                # and the number can never disagree about what was counted.
                for seq, unit in enumerate(units, start=1):
                    i = unit[0]
                    box, tid = xyxy[i], tids[i]
                    if len(unit) > 1:
                        pts = np.array([xyxy[k] for k in unit])
                        box = [pts[:, 0].min(), pts[:, 1].min(),
                               pts[:, 2].max(), pts[:, 3].max()]
                    cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                    # Masks and boxes still get a per-position colour, because their
                    # job is to show WHICH pixels belong to which pill and two adjacent
                    # outlines in one colour read as one shape. A dot has no extent to
                    # confuse, so it takes the fixed high-contrast marker instead.
                    colour = _colour(tid) if track else _colour_at(cx, cy)
                    if len(unit) > 1:
                        cv2.rectangle(shown, (int(box[0]), int(box[1])),
                                      (int(box[2]), int(box[3])), colour, 1, cv2.LINE_AA)
                    if masks and mm is not None and i < len(mm.data):
                        m = mm.data[i].cpu().numpy()
                        m = (cv2.resize(m, (w, h)) > 0.5).astype(np.uint8)
                        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(shown, cs, -1, colour, 2, cv2.LINE_AA)
                    elif marker == "box":
                        cv2.rectangle(shown, (int(box[0]), int(box[1])),
                                      (int(box[2]), int(box[3])), colour, 2, cv2.LINE_AA)
                    elif marker == "number":
                        # Black underneath, white on top: a white tablet and a dark
                        # capsule are both in shot, and neither must swallow the digits.
                        tag = str(seq)
                        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX,
                                                      0.45, 1)
                        org = (cx - tw // 2, cy + th // 2)
                        # Digits stay white-on-black -- the dot's colours the other way
                        # round -- because a thin black glyph on a white tablet is much
                        # harder to read at this size than a white one with a dark edge.
                        cv2.putText(shown, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(shown, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    INK, 1, cv2.LINE_AA)
                    else:
                        cv2.circle(shown, (cx, cy), 7, MARK_RIM, -1, cv2.LINE_AA)
                        cv2.circle(shown, (cx, cy), 5, MARK, -1, cv2.LINE_AA)
                    if ids:
                        cv2.putText(shown, str(tid), (cx - 8, cy + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3,
                                    cv2.LINE_AA)
                        cv2.putText(shown, str(tid), (cx - 8, cy + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, INK, 1, cv2.LINE_AA)
            counts.append(count)

            if roi:
                cv2.polylines(shown, [np.array(roi, np.int32).reshape(-1, 1, 2)], True,
                              ROI_LINE, 2, cv2.LINE_AA)

            n = str(count)
            cv2.putText(shown, n, (16, 74), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 0), 9,
                        cv2.LINE_AA)
            cv2.putText(shown, n, (16, 74), cv2.FONT_HERSHEY_SIMPLEX, 2.4, INK, 3,
                        cv2.LINE_AA)
            secs = time.perf_counter() - started
            line = f"{frames / max(secs, 1e-6):4.1f} fps   {'tracked' if track else 'per frame'}"
            cv2.putText(shown, line, (16, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                        3, cv2.LINE_AA)
            cv2.putText(shown, line, (16, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1,
                        cv2.LINE_AA)

            if writer is not None:
                writer.write(shown)
            cv2.imshow(title, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                roi = select_roi(frame, cap=cap)
            if key == ord("s"):
                # The frame AND the boxes it produced. A count that looks wrong on screen
                # is gone by the time anyone can describe it, and "one tablet counted
                # twice" cannot be diagnosed from a description -- it needs the pixels and
                # the coordinates that were drawn over them. Same key app.py uses.
                shots = os.path.join(HERE, "shots")
                os.makedirs(shots, exist_ok=True)
                stamp = time.strftime("%m%d-%H%M%S")
                cv2.imwrite(os.path.join(shots, f"frame_{stamp}.jpg"), frame)
                cv2.imwrite(os.path.join(shots, f"drawn_{stamp}.jpg"), shown)
                with open(os.path.join(shots, f"boxes_{stamp}.txt"), "w") as fh:
                    fh.write(f"count_in_roi {count}\n")
                    # THE REGION GOES IN TOO. Without it a reader has every box but no
                    # way to tell which ones the count included, and anything measured
                    # off the file -- grouping, odd-one-out -- silently takes in whatever
                    # was sitting on the bench outside the tray. That happened: six
                    # off-tray objects turned one uniform tray into six "kinds".
                    fh.write("roi " + (" ".join(f"{x},{y}" for x, y in roi)
                                       if roi else "whole_view") + "\n")
                    if boxes is not None and len(boxes):
                        for box in boxes.xyxy.cpu().numpy():
                            cx = (box[0] + box[2]) / 2
                            cy = (box[1] + box[3]) / 2
                            mark = "in" if _inside(roi, (cx, cy)) else "out"
                            fh.write(" ".join(f"{v:.1f}" for v in box) + f" {mark}\n")
                print(f"saved       : model2/shots/*_{stamp}.*   count {count}")
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    secs = time.perf_counter() - started
    print(f"window      : {frames} frames in {secs:.1f}s   {frames / max(secs, 1e-6):.1f} fps")
    if counts:
        print(f"count       : min {min(counts)}  max {max(counts)}  "
              f"median {statistics.median(counts):.0f}")
        print(f"spread      : {max(counts) - min(counts)} tablets   "
              + "  ".join(f"{v}x{n}" for v, n in Tally(counts).most_common(3)))
    if save:
        print(f"saved       : {save}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="0", help="camera index, or a video path")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    # THE DEFAULTS BELONG TO THE DEFAULT MODEL. They were 0.15 / 448 / 0.5 while this
    # file opened pillsort-seg, and each of those was measured -- see the docstring. The
    # default model is pillcount-det-v3 now, so the defaults are ITS measured values.
    # Changing one without the other is how a tool starts quietly lying.
    ap.add_argument("--conf", type=float, default=0.45,
                    help="0.45, measured on two bench frames. Lower double-counts an "
                         "edge-on tablet; higher loses a real one.")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="the size v3 was trained at. 448 raises the spread, 960 wrecks it")
    ap.add_argument("--iou", type=float, default=0.4,
                    help="0.4. Above 0.45 tablets count twice, by 0.20 touching ones "
                         "merge into one")
    ap.add_argument("--save", default=None, help="also write an annotated mp4 here")
    ap.add_argument("--no-masks", action="store_true", help="dots instead of outlines")
    ap.add_argument("--ids", action="store_true", help="draw the tracking id on each")
    ap.add_argument("--track", action="store_true",
                     help="ByteTrack. Measured as WORSE here -- see the docstring.")
    ap.add_argument("--marker", choices=("dot", "number", "box"), default="dot",
                    help="how each counted pill is drawn. dot: least clutter on a full "
                         "tray. number: 1..N in reading order, so the last number can be "
                         "checked against the count. box: what the detector actually "
                         "outlined, for when a number looks wrong.")
    ap.add_argument("--merge", action="store_true",
                    help="repair capsules the detector split in two. Needed by "
                         "pillcount-det-v2 and EARLIER; v3 does not split them and the "
                         "rule costs it a tablet. See merge_split().")
    ap.add_argument("--exposure", type=exposure_arg, default="auto",
                    help="auto (default) or a number. A fixed number is shutter time: the "
                         "bright end of the scale costs frame rate, and -3 was measured in "
                         "a brighter room than tonight's. See set_exposure.")
    a = ap.parse_args(argv)
    return run(a.source, model_path=a.model, conf=a.conf, iou=a.iou, imgsz=a.imgsz,
               save=a.save, masks=not a.no_masks, ids=a.ids, track=a.track,
               exposure=a.exposure, merge=a.merge, marker=a.marker)


if __name__ == "__main__":
    raise SystemExit(main())
