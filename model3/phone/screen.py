"""The whole phone screen, composed into one picture, and the taps that drive it.

WHY ONE PICTURE. Android draws it as a single Bitmap in an ImageView, and the Kotlin side
owns nothing but the camera, the model's forward pass and that view. Every button, card
and number on the handset is drawn here -- which means the layout can be changed, looked
at and argued about on a PC without an Android build at all:

    python -m model3.tools.preview

TWO PAGES, ONE STATE. The counting page and the records page are the same object with a
flag, because they share the target, the region and the folder of records, and because a
phone has no room for two windows. The desktop opens a second window for the list; here
the list replaces the screen and a back arrow returns.

TAP TARGETS ARE THE RECTANGLES THAT WERE DRAWN. Every control appends the rect it drew to
`self.hits`, and a tap is tested against that list in reverse order, newest on top. A
control cannot be drawn in one place and hit in another because there is only one set of
coordinates, and a control that is not drawn this frame cannot be hit at all -- which is
what makes a disabled or hidden button safe.
"""
from __future__ import annotations

import datetime as dt
import os
import time

import numpy as np

try:
    import cv2
except ImportError:                                             # pragma: no cover
    cv2 = None

from . import records as rec_store
from . import ui
from .strings import WEEKDAYS
from .text import Text

#: The design canvas, LANDSCAPE. Android scales it to whatever the view is.
#:
#: Sideways for the same reason the bench window is: a tray is wider than it is deep, and
#: so is the picture of it. Turned upright, a 4:3 frame can only use the top third of a
#: phone and the count has to go underneath it; turned sideways the picture takes the right
#: two thirds at full height and every control sits in a column beside it -- which is the
#: desktop's arrangement, on a device you can hold over the bench.
W, H = 2160, 1080

#: What actually goes to Android. The screen is DRAWN at the design size above and handed
#: over smaller, because every pixel of it is copied into a Java Bitmap on the way and
#: 2160x1080 is 9.3 MB a frame -- measured at 28 ms on a desktop, so a good deal worse on
#: a handset. At 1440x720 it is 4.1 MB, the phone stretches it back up, and the type
#: survives because it was rendered at 96px and has been shrinking ever since.
OUT_W, OUT_H = 1440, 720

#: Where the live picture goes, in canvas coordinates. ANDROID DRAWS IT, NOT THIS FILE.
#: See the note in _camera: the canvas leaves a hole and CameraX's own preview surface
#: shows through it, which is the difference between a video at the camera's rate and one
#: at the model's. The rectangle is 4:3, the shape the analysis stream is asked for, so
#: the ViewPort in MainActivity can make the analysed image and the displayed one the
#: same picture.
PANE = (844, 166, 1088, 816)

#: The colour that means "nothing has been drawn here", used only inside PANE.
#:
#: WHY A SENTINEL AND NOT A LIST. The hole has to be transparent where the camera should
#: show through and opaque wherever the screen drew something over it -- the dots, the
#: region, a toast, a fault message. Tracking that by hand meant every drawing call needed
#: a second call onto a mask, and the two went out of step immediately: the marks were
#: masked and the toast was not, so saving a count showed a message that the camera
#: appeared to swallow. Painting the hole with a colour nothing else uses and asking
#: afterwards what is still that colour cannot go out of step, because it is not a record
#: of what was drawn -- it is the drawing.
#:
#: Magenta: not in this app's palette, not a colour a pill or a tray is, and far from every
#: green and grey here, so the tolerance below can be generous.
HOLE = (255, 0, 255)


def use(view_w, view_h):
    """Lay the canvas out for THIS screen's shape. Called once, before the Screen exists.

    The design was a flat 2160x1080, and a phone is rarely 2:1: on a 16:9 handset the
    picture was letterboxed with a grey band across the top and another across the bottom,
    which is what an app that has not been told the shape of its own window looks like.
    Everything here is laid out from W and H, so the fix is to set them from the display
    and derive the video pane again -- the control column keeps its width and the camera
    card takes whatever is left, which is exactly what it does on the bench when the window
    is resized.
    """
    global W, H, OUT_W, OUT_H, PANE
    if view_w < view_h:                     # portrait: the activity is locked landscape,
        view_w, view_h = view_h, view_w     # but the size can arrive either way round
    H = 1080
    W = int(round(H * max(1.3, min(2.6, view_w / max(1, view_h)))))
    OUT_H = 720
    OUT_W = int(round(W * OUT_H / H))

    # The video hole: the largest 4:3 rectangle that fits the camera card, centred in it.
    card = (828, 104, W - 856, 896)
    inner = (card[0] + 16, card[1] + 62, card[2] - 32, card[3] - 78)
    pane_w = min(inner[2], int(inner[3] * 4 / 3))
    pane_h = int(pane_w * 3 / 4)
    PANE = (inner[0] + (inner[2] - pane_w) // 2, inner[1] + (inner[3] - pane_h) // 2,
            pane_w, pane_h)
    return W, H


def pane_out():
    """PANE in the coordinates Android places the preview with: the output size."""
    x, y, w, h = PANE
    k_x, k_y = OUT_W / W, OUT_H / H
    return [int(x * k_x), int(y * k_y), int(w * k_x), int(h * k_y)]


#: How far a finger may travel between down and up and still count as a press, in design
#: pixels.
#:
#: IT USED TO BE 24 AND IT APPLIED TO EVERY PAGE, which is why pressing a number on the
#: counting screen so often did nothing at all. The rule exists for one thing: a drag down
#: the records list must scroll it rather than open whatever the finger happened to lift
#: over. On the counting screen there is nothing to scroll, so all it did was throw away
#: presses -- and 24 design pixels is about a millimetre and a half of glass, which is less
#: than a thumb rolls on any button it means to press. Now it is checked only where there
#: is a list, only when the press began ON that list, and at a distance somebody has to
#: mean.
TAP_SLOP = 48

PRESETS = (10, 20, 30, 60, 90, 100)
MIN_ROI = 40               # frame pixels, as on the bench

#: How many corners a region takes. FOUR, and the shape closes itself on the fourth.
#:
#: A tray seen from above is a quadrilateral -- and only a RECTANGLE when the lens is
#: exactly square to the bench, which a phone propped over one never is. The rubber band
#: this replaced could only make rectangles, so an angled view left a choice between a box
#: that took in the bench beside the tray and one that cut its far corners off. Pills sit
#: in corners.
ROI_POINTS = 4
NOTE_SECONDS = 4.0
TOAST_SECONDS = 1.8
CONFIRM_SECONDS = 4.0      # how long a "tap again to confirm" stays armed
THUMB_CACHE = 40

#: Seconds the number must hold STILL before a pour may be taken from it, and the same
#: seconds the tray must read EMPTY before the next pour may begin.
#:
#: ONE CONSTANT WHERE THE BENCH HAS TWO, and deliberately. app/window.py separates them
#: because it has a 60fps repaint to hang each on; here the screen is only composed when
#: something changes, so both are answered by the same question -- has this number held
#: for long enough -- and one threshold means one mechanism to get right.
#:
#: Why it exists at all: pillcount-det-v3 reads a motionless tray with a spread of 1, so
#: the figure flickers between 34 and 35 as a box comes and goes on one tablet at the edge
#: of the region. A pour taken on whichever frame the finger landed on is a coin toss
#: between two numbers, and the one it picks is then frozen into the total and swept into a
#: bottle where nobody can re-count it.
STILL_S = 0.5

#: Said while a pour has been taken but the tray it was taken from is still full.
CLEAR_NOTE = "กวาดเม็ดในถาดออกให้หมด แล้วจึงเทรอบต่อไป"

MONTHS = ("", "ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
          "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค.")


def quad(points, size, min_side=MIN_ROI):
    """Four taps -> a simple quadrilateral, clamped to the frame, or None if it is unusable.

    THE ORDER THEY WERE TAPPED IN IS THROWN AWAY. Four corners joined in tap order can
    cross: top-left, bottom-right, top-right, bottom-left makes a bow tie, and
    pointPolygonTest on a bow tie counts the pills in one lobe and not the other -- a wrong
    count wearing a drawn region, which is the worst way to be wrong. So the corners can be
    tapped in whatever order the hand reaches them.

    THE ORDERING IS CHOSEN BY AREA, and that is exact rather than a heuristic. Four points
    have only three distinct cyclic orders, and a crossed one's shoelace is the DIFFERENCE
    of its two lobes -- so it can never total more than the same four corners joined
    without a crossing. Taking the largest therefore picks the simple shape every time,
    including the concave case where one corner was tapped inside the triangle of the other
    three, which sorting the corners by angle round their centre does NOT reliably handle.

    The ring is then turned to start at its lowest corner and wound in one direction, so
    that the same four corners tapped in any of the twenty-four orders are SAVED as the
    same four numbers -- a region that survives a restart has to compare equal to itself.

    Clamped, because tapping past the edge of the picture is the ordinary way to say "all
    of it" and a polygon outside the frame would quietly drop the pills nearest that edge.

    REFUSED ON TWO COUNTS, because one is not enough. Area alone lets through a sliver six
    hundred pixels long and three deep -- 1800 square pixels, over any floor worth setting,
    and no tray. A bounding box alone lets through a thin diagonal kite that fills a corner
    of a large box with nothing. A region has to be big BOTH ways and actually enclose
    something.

    app/window.py has this function too, word for word, and the two are kept in step by a test
    that runs both over the same inputs rather than by this sentence. Neither imports the
    other: the bench must not depend on the phone's package, and the phone cannot have Qt
    anywhere near it.
    """
    w, h = int(size[0]), int(size[1])
    if len(points) != ROI_POINTS or w < 2 or h < 2:
        return None
    pts = [(min(max(int(x), 0), w - 1), min(max(int(y), 0), h - 1)) for x, y in points]

    def enclosed(ring):
        return abs(sum(ring[i][0] * ring[(i + 1) % 4][1] - ring[(i + 1) % 4][0] * ring[i][1]
                       for i in range(4))) / 2.0

    area, ring = -1.0, pts
    for order in ((0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3)):
        loop = [pts[i] for i in order]
        got = enclosed(loop)
        if got > area:
            area, ring = got, loop
    start = min(range(4), key=lambda i: ring[i])
    ring = ring[start:] + ring[:start]
    if ring[1] > ring[3]:
        ring = [ring[0]] + ring[:0:-1]

    wide = max(p[0] for p in ring) - min(p[0] for p in ring)
    tall = max(p[1] for p in ring) - min(p[1] for p in ring)
    if wide < min_side or tall < min_side or area < min_side * min_side:
        return None
    return ring


def short_when(rec):
    """"17 ก.ย. 10:24" -- the same compact stamp the desktop list shows."""
    when = rec_store.when(rec)
    if not when:
        return rec.get("time", "")
    _, month, day, hh, mm = when
    return f"{day} {MONTHS[month]} {hh:02d}:{mm:02d}"


class Screen:
    def __init__(self, assets_dir, records_dir, target=0, conf=0.45, iou=0.4, imgsz=640):
        self.text = Text(os.path.join(assets_dir, "text"))
        # The same wordmark the desktop window puts in its header, staged into the APK.
        # Missing is not a fault: the name is drawn in type instead.
        logo = os.path.join(assets_dir, "logo.png")
        self._logo_img = cv2.imread(logo, cv2.IMREAD_UNCHANGED) if os.path.isfile(logo) else None
        if self._logo_img is not None:
            if self._logo_img.shape[2] == 3:
                self._logo_img = np.dstack(
                    [self._logo_img, np.full(self._logo_img.shape[:2], 255, np.uint8)])
            scale = 56 / self._logo_img.shape[0]
            self._logo_img = cv2.resize(
                self._logo_img, (int(self._logo_img.shape[1] * scale), 56),
                interpolation=cv2.INTER_AREA)
        self.records_dir = records_dir
        self.target = int(target)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz

        self.page = "count"
        self.roi = None                     # [(x, y)] in FRAME pixels
        #: The frame from the last run is read on the FIRST picture, not here: it is only
        #: valid at the resolution it was drawn at, and nothing knows that resolution
        #: until a camera frame arrives. See compose().
        self._roi_loaded = False
        self.arming = False
        self.pending = []                   # corners tapped so far, in frame pixels
        self.note = ""
        self.note_at = 0.0
        self.toast = None                   # (title, detail, when)
        self.blocked = ""                   # why saving is refused, "" when it is allowed
        self.fault = ""                     # the last message from something that threw
        self.fault_seen = ""                # the one the operator has already dismissed

        self.rows = []                      # records, newest first
        self.filter = 0                     # 0 all, 1 today, 2 off-target, 3 date range
        #: The range the fourth filter uses. Whole DAYS, both ends inclusive -- the phone
        #: has no keyboard and a calendar can only offer days, so the boxes say what they
        #: can deliver rather than pretending to a time nobody can enter.
        self.from_date = dt.date.today()
        self.to_date = dt.date.today()
        self.picking = ""                   # "", "from" or "to" while the calendar is up
        self.typing = None                  # the digits typed so far, or None when closed
        self.provider = ""                  # which ONNX Runtime kernels are in use
        self.draw_ms = 0.0                  # what composing the last screen cost
        self._over = []                     # marks to paint after the screen is scaled
        self.moving = False                 # is the tray being handled right now
        self.pick_month = dt.date.today().replace(day=1)
        self.scroll = 0
        #: Which row the preview panel is showing. The bench selects the newest record
        #: when the window opens and shows it beside the list; a phone that showed
        #: nothing there would be a page with a hole in it.
        self.sel = 0
        self.list_box = None                # the records list's rect, for the scroll test
        self.confirm_at = 0.0               # when the wipe was armed
        #: POURS ALREADY COUNTED AND ALREADY TIPPED AWAY, in the order they were taken.
        #:
        #: The tray holds about sixty tablets before they start lying on one another, and a
        #: detector cannot count what it cannot see -- so a prescription for a hundred is
        #: physically two pours, and was unanswerable here until this list existed. What is
        #: on screen is this list plus whatever is on the tray NOW; an empty list makes that
        #: the plain live count, which is what this screen did before and still does for one
        #: pour.
        self.rounds = []
        self.round_frames = []              # one picture each, so a total can be re-counted
        self.clearing = False               # taken a pour, waiting to see the tray empty
        self._steady_n = None               # the count the stillness timer is timing
        self._steady_at = 0.0               # when it last changed
        self.save_armed_at = 0.0            # the two-tap guard on a mid-pour save
        self.reset_armed_at = 0.0           # and on starting the prescription over
        #: Show the picture left-to-right reversed. A phone camera is built to be pointed
        #: at a face and hands over a mirror image because that is what a face expects; a
        #: tray does not, and the operator reaches left for a tablet the screen shows on the
        #: right. On until somebody turns it off, because that is what the hardware does.
        #:
        #: IT TOUCHES THE DRAWING AND NOTHING ELSE. spot() mirrors the marks inside the
        #: video pane, MainActivity mirrors the preview surface under them, and _to_frame
        #: undoes it for every tap. The frame handed to the model, the counting region and
        #: the saved JPEG all go on living in the camera's own coordinates -- because
        #: flipping the frame gives the detector a different image, and measured on this
        #: bench that moved the count by as much as three on one tray.
        self.flip = rec_store.load_flip(records_dir)
        #: The last frame composed, so the round button has a picture to bank. The touch
        #: handler is given one by the bridge for the region maths, but a tap that lands
        #: between two camera frames would have nothing; this always has the last one.
        self._last_frame = None
        self._thumbs = {}
        self._view = None                   # (x, y, scale) of the camera picture, for taps
        self.hole_alpha = None              # opacity of the video hole, worked out per frame
        self._chrome = None                 # the screen with nothing moving on it
        self._chrome_key = None             # what that picture was drawn for
        self._chrome_hits = []              # and the tap targets it put there
        self.hits = []                      # [(name, rect)] from the last compose
        self._touch_start = None            # where the finger went down, in canvas pixels
        self._scroll_start = 0
        self.reload()

    # ------------------------------------------------------------------ the records --
    def reload(self):
        self.rows = rec_store.load(self.records_dir)
        self.scroll = 0
        self.sel = 0                        # the newest record, as the bench opens on

    def shown(self):
        today = time.strftime("%Y%m%d")
        out = []
        for rec in self.rows:
            kind = rec_store.verdict(rec)[0]
            if self.filter == 1 and not rec.get("stamp", "").startswith(today):
                continue
            if self.filter == 2 and kind in ("ok", "none"):
                continue
            if self.filter == 3:
                when = rec_store.when(rec)
                if when is None:
                    continue
                day = dt.date(when[0], when[1], when[2])
                if not self.from_date <= day <= self.to_date:
                    continue
            out.append(rec)
        return out

    def _thumb(self, path, size=(200, 150)):
        """A record's picture, shrunk to fit inside `size` and kept.

        KEYED BY THE SIZE AS WELL AS THE PATH. The list wants a small one and the preview
        panel beside it wants a large one, of the same file; a cache that ignored that
        would hand whichever asked second the other one's copy.

        Fitted, not stretched: these are photographs of a tray, and a tray that is the
        wrong shape is harder to recognise than a smaller one.
        """
        key = (path, size)
        got = self._thumbs.get(key)
        if got is None:
            img = cv2.imread(path)
            if img is None:
                return None
            scale = min(size[0] / img.shape[1], size[1] / img.shape[0])
            got = cv2.resize(img, (max(1, int(img.shape[1] * scale)),
                                   max(1, int(img.shape[0] * scale))),
                             interpolation=cv2.INTER_AREA)
            if len(self._thumbs) > THUMB_CACHE:
                # A phone has a few hundred megabytes for everything, and a folder of
                # records grows for ever. Oldest out, rather than no cache at all.
                self._thumbs.pop(next(iter(self._thumbs)))
            self._thumbs[key] = got
        return got

    # -------------------------------------------------------------------- messaging --
    def say(self, note):
        self.note, self.note_at = note, time.time()

    def _note_now(self):
        # The refusal is about the save button, which only the counting page has. Carrying
        # it onto the list made the records page say "take some out" over a screen with
        # nothing to take anything out of.
        if self.blocked and self.page == "count":
            return self.blocked, ui.DANGER if "ภาพ" in self.blocked or "โมเดล" in self.blocked else ui.WARN
        if self.note and time.time() - self.note_at < NOTE_SECONDS:
            return self.note, ui.GREEN_700
        return "", ui.GREEN_700

    def state_key(self, count, ms, stale, error):
        """Everything that can change the picture, in one comparable value.

        WHY: the screen was recomposed for every camera frame whether or not it differed
        from the last one -- a whole canvas filled, forty pieces of type blitted and four
        megabytes handed to Java, thirty times a second, to show the same number. The work
        went on the one worker thread, so a tap waited behind it and the entire app felt
        like treacle. Now a frame that changes nothing costs one tuple comparison.

        `ms` is deliberately absent: it moves every single pass and means nothing to
        anybody watching. The clock is in here as a minute, not a time, for the same
        reason -- a redraw a minute is not a cost.
        """
        return (self.page, count, self.target, self.blocked, self.arming,
                # The pours, and whether the number has held long enough to be taken. The
                # steadiness is in here so that the moment it flips -- half a second after
                # the tray stopped moving, with no camera frame needed to say so -- the
                # round button comes to life and the screen is composed to show it.
                tuple(self.rounds), self.clearing, self.steady(),
                round(self.save_armed_at, 1)
                if time.time() - self.save_armed_at < CONFIRM_SECONDS else 0,
                tuple(map(tuple, self.pending)),
                None if self.roi is None else tuple(map(tuple, self.roi)),
                self.filter, self.scroll, len(self.rows), stale > 1.5, error,
                self.picking, self.pick_month, self.from_date, self.to_date, self.typing,
                self.fault if self.fault != self.fault_seen else "",
                self.note if time.time() - self.note_at < NOTE_SECONDS else "",
                bool(self.toast), time.strftime("%H:%M"),
                round(self.confirm_at, 1) if time.time() - self.confirm_at < CONFIRM_SECONDS else 0)

    # ---------------------------------------------------------------------- drawing --
    def chrome_key(self):
        """What the unchanging half of the screen depends on.

        EVERYTHING ELSE IS REDRAWN EVERY TIME, and this is the line between them. The cards,
        the buttons, the preset chips, the labels and the list are the same picture from one
        frame to the next unless one of these changes -- so they are drawn once into a
        template and copied. A copy of the canvas is about a twentieth of the cost of
        composing it, which is the difference between a number that updates while somebody
        watches and one that arrives after they have looked away.
        """
        return (self.page, self.target, self.arming, self.roi is not None,
                len(self.pending),      # the row's second button counts them
                self.flip,
                bool(self.blocked), self.filter, self.scroll, len(self.rows),
                # The round row and the save button are both furniture, and both change
                # with these: how many pours are banked, whether one may be taken now, and
                # whether the save is about to file a short count or is armed to.
                len(self.rounds), self.clearing, not self.round_block(),
                self._save_look(),
                round(self.reset_armed_at, 1)
                if time.time() - self.reset_armed_at < CONFIRM_SECONDS else 0,
                round(self.save_armed_at, 1)
                if time.time() - self.save_armed_at < CONFIRM_SECONDS else 0,
                # The selected row tints one row of the list and fills the whole preview
                # panel beside it, and both are part of the template -- so a tap that
                # moves the selection has to throw the template away.
                self.sel, self.from_date, self.to_date,
                round(self.confirm_at, 1) if time.time() - self.confirm_at < CONFIRM_SECONDS
                else 0)

    def compose(self, frame, boxes, count, ms, stale=0.0, error=""):
        """Everything on screen this frame. Returns the picture Android will show."""
        if error and error != self.fault:
            self.fault_seen = ""            # a NEW message is worth showing again
        # FIRST, because everything below reads the total and the total depends on whether
        # the tray has been seen empty yet.
        self._last_frame = frame
        self._tick_pours(count)
        self._block_check(count, stale, error)

        # THE FRAME SURVIVES A RESTART. A bench counts the same tray under the same lens
        # all day, and asking somebody to redraw the region every time the app is opened
        # is asking them to redraw it slightly differently every time. Read on the first
        # picture, because only then is the resolution it was drawn at known.
        if frame is not None and not self._roi_loaded:
            self._roi_loaded = True
            pts, why = rec_store.load_roi(self.records_dir,
                                          (frame.shape[1], frame.shape[0]))
            if pts:
                self.roi = pts
            elif why:
                self.say(why)

        key = self.chrome_key()
        if self._chrome is None or key != self._chrome_key:
            base = np.full((H, W, 3), ui.BG, np.uint8)
            self.hits = []
            self._header(base)
            if self.page == "count":
                self._chrome_count(base)
            else:
                self._page_records(base)
            self._chrome, self._chrome_key = base, key
            self._chrome_hits = list(self.hits)

        img = self._chrome.copy()
        self.hits = list(self._chrome_hits)
        self._clock(img)
        if self.page == "count":
            self._live_count(img, frame, boxes, count, ms, stale)
        self._footer(img)
        self._calendar(img)
        self._keypad(img)
        self._snackbar(img)
        self._toast(img)
        # PUNCHED BEFORE THE SCALE, not after. See _punch.
        self.hole_alpha = self._punch(img)
        if (OUT_W, OUT_H) != (W, H):
            img = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        self._paint_over_video(img)
        return img

    def _paint_over_video(self, out):
        """The dots and the region, drawn onto the SCALED screen. Marks, not chrome.

        WHY NOT WITH EVERYTHING ELSE, which is where they used to be. The hole is painted
        in a sentinel colour and the whole screen is then scaled down, and INTER_AREA
        blends every stroke inside the hole with the magenta around it. The dot's white
        rim is four pixels at the design size -- under three after the scale, magenta
        creeping in from both sides -- so what reached the handset was a pink ring, and
        the rim is the entire reason a mark stays visible on a dark capsule. Drawn here,
        after the scale and after the hole has been measured, a mark is never next to the
        sentinel and comes out the colour it was asked for, at the size Android shows.

        ONLY WHERE THE HOLE IS STILL OPEN. A toast, a fault message or the number pad is
        drawn over the video and is already opaque in hole_alpha; painting only into the
        transparent part keeps the marks under them, exactly as they were when they were
        drawn before the toast.
        """
        if self.hole_alpha is None or not self._over:
            return
        x, y, w, h = pane_out()
        k = OUT_W / W
        patch = np.zeros((h, w, 3), np.uint8)
        mask = np.zeros((h, w), np.uint8)

        def at(v, origin):
            return int(round(v * k)) - origin

        for kind, geom, colour, weight in self._over:
            # No LINE_AA anywhere: an antialiased edge would blend into the black patch
            # and into a partial alpha, and neither survives a premultiplied bitmap.
            for target, ink in ((patch, colour), (mask, 255)):
                if kind == "circle":
                    cx, cy, r = geom
                    cv2.circle(target, (at(cx, x), at(cy, y)),
                               max(1, int(round(r * k))), ink, -1)
                elif kind == "rect":
                    x0, y0, x1, y1 = geom
                    cv2.rectangle(target, (at(x0, x), at(y0, y)), (at(x1, x), at(y1, y)),
                                  ink, max(1, int(round(weight * k))))
                elif kind in ("poly", "path"):
                    # The region as the shape it IS. It used to be drawn as the bounding
                    # box of its own corners, which was honest only while a rubber band
                    # could make nothing but rectangles -- on a tray seen at an angle the
                    # box is visibly bigger than the area being counted, so the screen
                    # would show pills inside the outline that the count left out.
                    pts = np.array([[at(vx, x), at(vy, y)] for vx, vy in geom], np.int32)
                    cv2.polylines(target, [pts.reshape(-1, 1, 2)], kind == "poly",
                                  ink, max(1, int(round(weight * k))))
                elif kind == "ring":
                    cx, cy, r = geom
                    cv2.circle(target, (at(cx, x), at(cy, y)),
                               max(1, int(round(r * k))), ink,
                               max(1, int(round(weight * k))))
                else:
                    rx, ry, rw, rh = geom
                    length, width = weight
                    ui.corner_ticks(target, (at(rx, x), at(ry, y),
                                             int(round(rw * k)), int(round(rh * k))),
                                    ink, int(round(length * k)),
                                    max(1, int(round(width * k))))

        # Where the hole is still open AND something was drawn. cv2 again rather than
        # `(alpha == 0) & (mask > 0)` and a boolean index: same reason as _punch, and
        # copyTo does the masked write in one pass instead of building an index array.
        free = cv2.bitwise_and(cv2.bitwise_not(self.hole_alpha), mask)
        cv2.copyTo(patch, free, out[y:y + h, x:x + w])
        cv2.bitwise_or(self.hole_alpha, free, self.hole_alpha)

    def _punch(self, img):
        """Empty the camera hole and measure it, at the DESIGN size, before any blending.

        WHY BEFORE THE SCALE, when it used to be after. Down here the picture is exact:
        the hole is the sentinel colour and nothing else is, so one comparison finds it
        with no tolerance at all. Measured on the scaled picture instead, every edge of
        the hole has been averaged with whatever sits against it, and no threshold can
        separate "half sentinel, half card" from "a colour somebody drew". The last
        attempt papered over that by forcing the outermost three pixels of the hole
        transparent no matter what -- which worked for the card, and cut a three-pixel
        slot of live video through the number pad, because the pad is drawn over the hole
        and its edge lands exactly there. That slot is the bug this replaces.

        WHY THE HOLE IS FILLED WITH BLACK. An Android ARGB_8888 bitmap is PREMULTIPLIED:
        a pixel's colour is already scaled by its own alpha. Zero the hole, and the scale
        that follows averages each edge pixel between what was drawn and nothing -- which
        is precisely colour x coverage. Take the alpha from the same average of the same
        mask and the two agree by construction: the edge of the pad, the edge of the card
        and the corner of the region all blend smoothly into the live picture instead of
        wearing a fringe of leftover magenta or a hairline of video.
        """
        if self.page != "count":
            return None
        px, py, pw, ph = PANE
        pane = img[py:py + ph, px:px + pw]
        sentinel = np.array(HOLE, np.uint8)
        hole = cv2.inRange(pane, sentinel, sentinel)

        # dst = src - src where the mask says so, untouched everywhere else: one pass in
        # C++, no temporary the size of the pane.
        cv2.subtract(pane, pane, pane, mask=hole)

        x, y, w, h = pane_out()
        if (w, h) != (pw, ph):
            hole = cv2.resize(hole, (w, h), interpolation=cv2.INTER_AREA)
        return cv2.bitwise_not(hole)

    def _wrap(self, words, px, width):
        """Break a line to fit, on spaces where there are any and mid-word where not.

        A fault message is somebody else's sentence -- it can be two words or a hundred
        characters of Java class name with no space in it at all -- so neither rule alone
        is enough.
        """
        lines, line = [], ""
        for word in words.split(" "):
            trial = f"{line} {word}".strip()
            if line and self.text.measure(trial, px) > width:
                lines.append(line)
                line = word
            else:
                line = trial
            while self.text.measure(line, px) > width and len(line) > 1:
                cut = len(line)
                while cut > 1 and self.text.measure(line[:cut], px) > width:
                    cut -= 1
                lines.append(line[:cut])
                line = line[cut:]
        if line:
            lines.append(line)
        return lines

    def _snackbar(self, img):
        """What went wrong, spelled out, along the bottom. Tap it to put it away.

        THE SCREEN SAYS THE MESSAGE, not a category. "โมเดลผิดพลาด" is a category: it told
        an operator something was wrong and told a developer nothing at all, and a wrong
        tensor shape lived through two builds behind it because the only place the reason
        existed was a logcat nobody had a cable for. The exception's own words go here.

        It is DARK, not red, and it does not block anything: the count and the camera stay
        where they are, and a tap dismisses it until the message changes. A fault that
        repeats identically is not worth interrupting the same person twice.
        """
        if not self.fault or self.fault == self.fault_seen:
            return
        px = 24
        margin = 40
        width = W - margin * 2 - 48
        lines = self._wrap(self.fault, px, width)[:3]
        line_h = self.text.height(px)
        h = line_h * len(lines) + 44
        rect = (margin, H - h - 24, W - margin * 2, h)
        ui.rounded(img, rect, 20, (36, 42, 38), -1)
        for i, line in enumerate(lines):
            self.text.draw(img, line, rect[0] + 24, rect[1] + 22 + i * line_h, px,
                           (236, 240, 238))
        self.hits.append(("dismiss-fault", rect))

    def _block_check(self, count, stale, error):
        if stale > 1.5:
            self.blocked = f"ภาพจากกล้องหยุด {stale:.0f} วินาที ตรวจกล้อง"
        elif error:
            self.blocked = "โมเดลผิดพลาด"
            self.fault = error
        elif self.target and self.total(count) > self.target:
            self.blocked = "เกินจำนวนที่ต้องการ นำออกก่อนจึงบันทึกได้"
        else:
            self.blocked = ""

    def _header(self, img):
        cv2.rectangle(img, (0, 0), (W, 88), ui.SURFACE, -1)
        cv2.line(img, (0, 88), (W, 88), ui.LINE, 2)
        self._logo(img)
        if self.page == "count":
            # Both of them up here, as on the bench: neither is a counting control, and
            # the column beside the camera has no room to spare for either.
            self.hits.append(("records", ui.button(
                img, self.text, (W - 800, 12, 380, 64),
                "ดูรายการที่บันทึก", 26, "ghost")))
            self.hits.append(("export", ui.button(
                img, self.text, (W - 404, 12, 240, 64),
                "ส่งออก CSV", 26, "ghost")))

    def _logo(self, img):
        """The wordmark, if it was staged into the assets; the name in type if not."""
        if self._logo_img is None:
            self.text.draw(img, "DrugCount", 28, 20, 36, ui.GREEN_700)
            return
        lh, lw = self._logo_img.shape[:2]
        piece = self._logo_img
        area = img[16:16 + lh, 28:28 + lw]
        alpha = piece[:, :, 3:4].astype(np.float32) / 255.0
        area[:] = (area * (1 - alpha) + piece[:, :, :3] * alpha).astype(np.uint8)

    def _clock(self, img):
        """Drawn over the template every frame, on a patch of the header it owns."""
        cv2.rectangle(img, (W - 170, 14), (W - 20, 72), ui.SURFACE, -1)
        self.text.draw(img, time.strftime("%H:%M"), W - 28, 22, 30, ui.INK_MUTED,
                       align="right")

    def _footer(self, img):
        """The message on the left, the settings on the right -- the bench's own footer.

        The right-hand half is the same four facts the desktop shows and for the same
        reason: they are asked about once an hour and never while counting, which is what
        a footer is for. The saved count comes from the list this screen already holds, so
        it costs nothing to say.
        """
        note, colour = self._note_now()
        if note:
            self.text.draw(img, note, 34, H - 62, 28, colour)
        provider = f"   {self.provider}" if self.provider else ""
        self.text.draw(img,
                       f"conf {self.conf}   iou {self.iou}   imgsz {self.imgsz}{provider}"
                       f"   บันทึกไว้ {len(self.rows)} รายการ",
                       W - 34, H - 58, 24, ui.INK_MUTED, align="right")

    def _toast(self, img):
        if not self.toast:
            return
        title, detail, at = self.toast
        if time.time() - at > TOAST_SECONDS:
            self.toast = None
            return
        w, h = 720, 236
        rect = ((W - w) // 2, (H - h) // 2, w, h)
        ui.rounded(img, rect, 34, ui.GREEN_700, -1)
        cx = rect[0] + w // 2
        cv2.circle(img, (cx, rect[1] + 74), 34, ui.SURFACE, 5, cv2.LINE_AA)
        cv2.line(img, (cx - 16, rect[1] + 74), (cx - 4, rect[1] + 88), ui.SURFACE, 7,
                 cv2.LINE_AA)
        cv2.line(img, (cx - 4, rect[1] + 88), (cx + 18, rect[1] + 60), ui.SURFACE, 7,
                 cv2.LINE_AA)
        self.text.draw(img, title, cx, rect[1] + 124, 46, ui.SURFACE, align="centre")
        self.text.draw(img, detail, cx, rect[1] + 192, 28, (217, 242, 226), align="centre")

    # ------------------------------------------------------------------ counting page --
    def _chrome_count(self, img):
        """The parts of the counting page that only change when a control does."""
        panel = (28, 104, 772, 896)
        ui.card(img, panel)
        self.cam_box = (828, 104, W - 856, 896)
        ui.card(img, self.cam_box)
        self.text.draw(img, "ภาพจากกล้อง", self.cam_box[0] + 24, self.cam_box[1] + 16,
                       26, ui.INK_SOFT)
        # THE FLIP LIVES ON THE PICTURE'S CARD, not in the counting panel. It is a property
        # of the picture rather than of the count, it is set once when the phone is aimed
        # and then never again, and the panel has no row to spare -- the header of this card
        # is the only place on the screen with room that is also the right place.
        self.hits.append(("flip", ui.button(
            img, self.text, (self.cam_box[0] + 240, self.cam_box[1] + 8, 220, 60),
            "พลิกภาพ", 26, "chip", "on" if self.flip else "off")))
        # The region and the model's time are CHIPS, drawn live in _camera where the bench
        # draws them: top right of this card, not grey words beside the title.

        # THE PANEL PAYS FOR THE ROUND ROW OUT OF THE COUNT, which is the same trade
        # app/window.py's PANEL_FITS makes and for the same reason: a row of buttons is
        # about ninety pixels, the panel had none spare, and the count is the one thing
        # here that can give some up safely. It is a NUMERAL -- no tone marks -- so
        # shrinking it cannot break the way shrinking Thai does, and at 134px on a canvas
        # this size it is still the figure you read from across a bench.
        self.text.draw(img, "เม็ด", 414, 318, 28, ui.INK_MUTED, align="centre")
        self.text.draw(img, "จำนวนที่ต้องการ", 60, 460, 28, ui.INK_SOFT)
        row_y = 500
        self.hits.append(("target-", ui.button(img, self.text, (60, row_y, 110, 90),
                                               "-", 46, "ghost")))
        # The number itself is a button: press it and a pad comes up. A phone has no
        # keyboard to reach for, and stepping from 0 to 120 with the + key is not a
        # control, it is a punishment.
        box = (186, row_y, 456, 90)
        ui.card(img, box, 18)
        self.text.draw(img, str(self.target), 414, row_y + 16, 48, ui.INK, align="centre")
        self.hits.append(("type-target", box))
        self.hits.append(("target+", ui.button(img, self.text, (658, row_y, 110, 90),
                                               "+", 46, "ghost")))

        chip_w = (708 - 5 * 10) // 6
        for i, n in enumerate(PRESETS):
            rect = (60 + i * (chip_w + 10), 602, chip_w, 68)
            ui.button(img, self.text, rect, str(n), 26, "chip",
                      "on" if n == self.target else "off")
            self.hits.append((f"preset{n}", rect))

        # ONE ROW FOR THE TWO FRAME CONTROLS. They are a pair -- set it, clear it -- and
        # a row each would say they were two unrelated things while spending twice the
        # panel on saying it. app/window.py stacked them and now does the same as this.
        # THE ROUND CONTROLS SIT ABOVE THE FRAME CONTROLS, and the order is the order of
        # the job: the frame is set once when the phone is propped over the bench and then
        # never touched, while these two are tapped once per pour with a tray in the other
        # hand. The thing used every minute belongs nearer the thumb.
        self.hits.append(("round", ui.button(
            img, self.text, (60, 682, 468, 88),
            f"เก็บรอบที่ {len(self.rounds) + 1}", 28, "ghost",
            enabled=not self.round_block())))
        # Live while there is a total to discard, INCLUDING mid-sweep: the tray that has
        # just been banked is exactly the moment somebody notices it was the wrong tray.
        self.hits.append(("reset", ui.button(
            img, self.text, (540, 682, 228, 88),
            "กดอีกครั้ง"
            if time.time() - self.reset_armed_at < CONFIRM_SECONDS else "นับใหม่",
            28, "ghost", enabled=bool(self.rounds or self.clearing))))

        half = (708 - 12) // 2
        self.hits.append(("roi", ui.button(
            img, self.text, (60, 782, half, 84),
            "ยกเลิก" if self.arming else ("กำหนดกรอบใหม่" if self.roi else "กำหนดกรอบนับ"),
            28, "ghost")))
        self.hits.append(("roi-clear", ui.button(
            img, self.text, (72 + half, 782, half, 84),
            "ถอยจุด" if self.arming else "ล้างกรอบ", 28, "ghost",
            enabled=bool(self.pending) if self.arming else bool(self.roi))))
        # The save button says what it is about to file. Armed mid-pour it says what the
        # next tap will cost, because that is the press worth hesitating over -- see _act.
        armed = time.time() - self.save_armed_at < CONFIRM_SECONDS
        if armed:
            words, px = f"แตะอีกครั้งเพื่อบันทึก {self.live_total()}", 30
        elif self._save_look() == "warn":
            words, px = f"บันทึกว่าไม่ครบ ขาด {self.target - self.live_total()}", 32
        else:
            words, px = "บันทึกผล", 38
        self.hits.append(("save", ui.button(
            img, self.text, (60, 878, 708, 92), words, px,
            "warn" if armed or self._save_look() == "warn" else "primary",
            enabled=not self.blocked)))

    def _live_count(self, img, frame, boxes, count, ms, stale):
        """The number, the verdict, the bar and the video overlay: redrawn every frame."""
        total = self.total(count)
        kind, words, colour = self._verdict(total)
        # WHERE THE BIG NUMBER CAME FROM, and only when that is a question. On one pour it
        # says nothing and takes no room, because on one pour the figure below is simply
        # what the camera can see. From the second pour on the figure is PARTLY MEMORY --
        # 35 of it is in a bottle and cannot be checked against the picture -- and a number
        # the picture does not corroborate has to say so, or there is no way to tell a
        # working total from a stuck one.
        if self.clearing:
            strip = (f"เก็บแล้ว {len(self.rounds)} รอบ รวม {self.banked()} เม็ด   "
                     f"รอกวาดถาด")
        elif self.rounds:
            strip = (f"เก็บแล้ว {len(self.rounds)} รอบ รวม {self.banked()} "
                     f"+ ในถาด {count}")
        else:
            strip = ""
        if strip:
            self.text.draw(img, strip, 414, 112, 24, ui.INK_SOFT, align="centre")
        self.text.draw(img, str(total), 414, 146, 134, colour, align="centre")
        ui.badge(img, self.text, 414, 362, words, 30, kind)
        if self.target:
            ui.progress(img, (68, 428, 692, 16), total / max(1, self.target), colour)
        self._camera(img, frame, boxes, getattr(self, "cam_box", (828, 104, W - 856, 896)),
                     stale, ms)

    def _camera(self, img, frame, boxes, box, stale, ms=0.0):
        """The card around the live picture, and the marks drawn over it.

        The picture is CameraX's preview surface showing through the hole; this paints the
        hole in HOLE and then draws on top of it exactly as it would anywhere else. What is
        still HOLE at the end of compose() becomes transparent, so a dot, a region, a toast
        or a fault message over the video all survive without any of them knowing about it.
        """
        # The bench's two chips, in the bench's corner and the bench's words. Live rather
        # than chrome because the time changes every pass, and the region chip has to sit
        # beside whatever width that one turns out to be.
        right = box[0] + box[2] - 24
        if ms:
            right = ui.tag(img, self.text, right, box[1] + 18, f"{ms:.0f} ms")[0] - 28
        ui.tag(img, self.text, right, box[1] + 18,
               "ภาพค้าง" if stale > 1.5 else
               ("เฉพาะในกรอบ" if self.roi else "นับทั้งภาพ"))

        self._over = []
        px, py, pw, ph = PANE
        if frame is None:
            ui.rounded(img, PANE, 20, (20, 23, 15), -1)
            self.text.draw(img, "กำลังเปิดกล้อง", px + pw // 2, py + ph // 2, 30,
                           ui.INK_MUTED, align="centre")
            return

        img[py:py + ph, px:px + pw] = HOLE
        fh, fw = frame.shape[:2]
        kx, ky = pw / fw, ph / fh

        def spot(x, y):
            """Frame point -> canvas point, THROUGH THE MIRROR when one is being shown.

            Every mark on the video goes through here -- the dots, the region, the corners
            going down -- so mirroring inside this one function turns all of them together
            and none of them has to know. Kotlin turns the preview surface round by the
            same axis, which is what keeps a dot on its tablet.
            """
            sx = (fw - 1 - x) if self.flip else x
            return int(px + sx * kx), int(py + y * ky)

        # NOTHING IS DRAWN INTO THE HOLE HERE -- it is written down and drawn by
        # _paint_over_video once the screen has been scaled. See that method.
        for b in boxes:
            if self.roi is not None and not self._inside(b):
                continue
            cx, cy = spot((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            self._over.append(("circle", (cx, cy, 11), ui.MARK_RIM, -1))
            self._over.append(("circle", (cx, cy, 7), ui.MARK, -1))

        if self.arming:
            # RINGS, NOT FILLED DOTS. A corner of a full tray has a pill under it, and a
            # solid mark would hide the very thing the corner is being placed around.
            corners = [spot(cx, cy) for cx, cy in self.pending]
            if len(corners) > 1:
                self._over.append(("path", corners, ui.ROI_BAND, 5))
            for cx, cy in corners:
                self._over.append(("ring", (cx, cy, 20), ui.ROI_BAND, 5))
        elif self.roi:
            corners = [spot(cx, cy) for cx, cy in self.roi]
            self._over.append(("poly", corners, ui.ROI_LINE, 4))
            # ELBOWS AT THE REGION'S OWN CORNERS, not at the corners of a box drawn round
            # it. They used to be the latter, which was the same thing back when a rubber
            # band could only make rectangles; on a tray seen at an angle the box stands
            # away from the shape and the ticks float off the outline they belong to.
            # Each elbow points INWARD, so it marks its corner from outside the tray
            # rather than sitting on the pill in it -- app/window.py draws the same.
            mx = sum(q[0] for q in corners) / len(corners)
            my = sum(q[1] for q in corners) / len(corners)
            for qx, qy in corners:
                arm = 34
                ax = arm if qx < mx else -arm
                ay = arm if qy < my else -arm
                self._over.append(("path", [(qx + ax, qy), (qx, qy), (qx, qy + ay)],
                                   ui.ROI_LINE, 8))

        if stale > 1.5:
            self._over.append(("rect", (px, py, px + pw - 1, py + ph - 1),
                               ui.STALE_EDGE, 12))
        self.hits.append(("view", PANE))

    # ------------------------------------------------------------------ records page --
    def _page_records(self, img):
        """The bench's records window, in the bench's order.

        WHAT IT LOOKED LIKE BEFORE AND WHY IT CHANGED. The three totals were stacked down
        the right-hand side with the export and wipe buttons under them, and the list had
        the rest -- no column headings, and no preview at all. It worked, and it was not
        the same page: the bench puts the totals in a row across the top, heads its
        columns, and gives half the width to the record it has selected, with its picture
        and its numbers. Somebody who has learnt one of these should not have to learn the
        other, so this is that layout, at this screen's size.
        """
        rows = self.shown()
        self.text.draw(img, "รายการที่บันทึก", 28, 104, 36, ui.INK)

        # THE THREE TOTALS, ACROSS THE TOP. Same three, same order, same colours.
        kinds = [rec_store.verdict(r)[0] for r in self.rows]
        tiles = (("บันทึกทั้งหมด", len(self.rows), ui.INK),
                 ("ครบตามจำนวน", kinds.count("ok"), ui.GREEN_700),
                 ("ไม่ตรงจำนวน", kinds.count("over") + kinds.count("short"), ui.DANGER))
        tile_w = (W - 56 - 32) // 3
        for i, (caption, value, colour) in enumerate(tiles):
            rect = (28 + i * (tile_w + 16), 160, tile_w, 116)
            ui.card(img, rect, 24)
            self.text.draw(img, str(value), rect[0] + 28, rect[1] + 10, 46, colour)
            self.text.draw(img, caption, rect[0] + 28, rect[1] + 70, 24, ui.INK_MUTED)

        for i, name in enumerate(("ทั้งหมด", "วันนี้", "ไม่ตรงจำนวน", "ช่วงวันที่")):
            rect = (28 + i * 236, 292, 222, 72)
            ui.button(img, self.text, rect, name, 26, "chip",
                      "on" if self.filter == i else "off")
            self.hits.append((f"filter{i}", rect))
        self.text.draw(img, f"ทั้งหมด {len(rows)} รายการ", W - 28, 306, 24, ui.INK_MUTED,
                       align="right")

        # The two ends of the range, always live. They were greyed until the range chip
        # was chosen on the desktop and that was simply wrong there too: the first thing
        # anybody does is press the date.
        self.text.draw(img, "ตั้งแต่", 34, 392, 24, ui.INK_SOFT)
        self.hits.append(("pick-from", ui.button(
            img, self.text, (130, 380, 250, 72), self._date_label(self.from_date), 26,
            "ghost")))
        self.text.draw(img, "ถึง", 400, 392, 24, ui.INK_SOFT)
        self.hits.append(("pick-to", ui.button(
            img, self.text, (452, 380, 250, 72), self._date_label(self.to_date), 26,
            "ghost")))
        self.text.draw(img, "(นับทั้งวัน)", 722, 392, 24, ui.INK_MUTED)

        # The list and the record it has selected, side by side, as on the bench.
        body_y, body_h = 480, 420
        list_w = int((W - 76) * 0.60)
        list_box = self.list_box = (28, body_y, list_w, body_h)
        ui.card(img, list_box, 28)
        self._list(img, list_box, rows)

        pv_x = 28 + list_w + 20
        self._preview(img, (pv_x, body_y, W - 28 - pv_x, body_h), rows)

        # The bench's footer buttons, in the bench's corners: what can be done TO the
        # folder on the left, what can be done WITH it on the right.
        self.hits.append(("refresh", ui.button(img, self.text, (28, 916, 200, 84),
                                               "รีเฟรช", 28, "ghost")))
        armed = time.time() - self.confirm_at < CONFIRM_SECONDS
        self.hits.append(("wipe", ui.button(
            img, self.text, (244, 916, 330, 84),
            "ยืนยันลบทั้งหมด" if armed else "ล้างข้อมูลทั้งหมด", 26, "danger",
            enabled=bool(self.rows))))
        self.hits.append(("back", ui.button(img, self.text, (W - 208, 916, 180, 84),
                                            "ปิด", 28, "ghost")))
        self.hits.append(("export", ui.button(img, self.text, (W - 508, 916, 280, 84),
                                              "ส่งออก CSV", 28, "primary",
                                              enabled=bool(rows))))

    def _preview(self, img, box, rows):
        """The selected record: its picture, its numbers, and what it was counted with.

        THE PICTURE IS THE POINT. A row in a list says 30 of 30; this says what the tray
        looked like when somebody wrote that down, which is the only thing that can settle
        an argument about it months later. The bench gives this half its window and so
        does this.
        """
        x, y, w, h = box
        ui.card(img, box, 28)
        if not rows:
            self.text.draw(img, "เลือกรายการเพื่อดูภาพ", x + w // 2, y + h // 2 - 16, 26,
                           ui.INK_MUTED, align="centre")
            return
        rec = rows[max(0, min(self.sel, len(rows) - 1))]

        self.text.draw(img, "ภาพขณะบันทึก", x + 24, y + 18, 24, ui.INK_SOFT)
        self.text.draw(img, str(rec.get("time", "")), x + w - 24, y + 18, 24, ui.INK,
                       align="right")

        shot_box = (x + 24, y + 62, w - 48, 200)
        thumb = self._thumb(rec.get("image"), (shot_box[2], shot_box[3])) \
            if rec.get("image") else None
        if thumb is None:
            ui.rounded(img, shot_box, 16, ui.BG, -1)
            self.text.draw(img, "ไม่มีภาพ", x + w // 2, y + 148, 24, ui.INK_MUTED,
                           align="centre")
        else:
            ui.rounded(img, shot_box, 16, (20, 23, 15), -1)
            ui.paste(img, thumb, shot_box[0] + (shot_box[2] - thumb.shape[1]) // 2,
                     shot_box[1] + (shot_box[3] - thumb.shape[0]) // 2)

        kind, words = rec_store.verdict(rec)
        target = int(rec.get("target") or 0)
        facts = (("นับได้", str(rec.get("count", 0)), ui.INK),
                 ("ต้องการ", str(target) if target else "—", ui.INK),
                 ("ผล", words, ui.GREEN_700 if kind == "ok" else
                  ui.DANGER if kind in ("over", "short") else ui.INK_SOFT),
                 ("เวลาโมเดล (ms)", f"{float(rec.get('model_ms') or 0):.1f}", ui.INK))
        col = (w - 48) // 4
        for i, (caption, value, colour) in enumerate(facts):
            fx = x + 24 + i * col
            self.text.draw(img, value, fx, y + 286, 32, colour)
            self.text.draw(img, caption, fx, y + 332, 22, ui.INK_MUTED)

        cv2.line(img, (x + 24, y + 372), (x + w - 24, y + 372), ui.LINE, 2)
        roi = "เฉพาะในกรอบ" if rec.get("roi") else "นับทั้งภาพ"
        self.text.draw(img,
                       f"conf {rec.get('conf', '')}   iou {rec.get('iou', '')}   "
                       f"imgsz {rec.get('imgsz', '')}   {roi}",
                       x + 24, y + 386, 22, ui.INK_MUTED)

    def _date_label(self, day):
        return f"{day.day}/{day.month}/{day.year}"

    def _calendar(self, img):
        """A month, as a grid of days to tap. Drawn over whatever is behind it.

        The desktop opens a native date picker; there is no such thing to open here, so
        this is one -- a header with arrows, the weekday initials, and the days of the
        month. Days outside the month are left off rather than greyed: a phone tap is
        fat, and a neighbouring month's 31st sitting next to this month's 1st is a trap.
        """
        if not self.picking:
            return
        w, h = 760, 720
        rect = ((W - w) // 2, (H - h) // 2, w, h)
        ui.rounded(img, rect, 28, ui.SURFACE, -1)
        ui.rounded(img, rect, 28, ui.LINE_STRONG, 2)
        x, y = rect[0], rect[1]

        self.text.draw(img, "เลือกวันที่", x + w // 2, y + 24, 30, ui.INK, align="centre")
        self.hits.append(("month-", ui.button(img, self.text, (x + 24, y + 86, 90, 72),
                                              "-", 34, "ghost")))
        head = f"{MONTHS[self.pick_month.month]} {self.pick_month.year}"
        self.text.draw(img, head, x + w // 2, y + 104, 30, ui.INK, align="centre")
        self.hits.append(("month+", ui.button(img, self.text, (x + w - 114, y + 86, 90, 72),
                                              "+", 34, "ghost")))

        cell = (w - 48) // 7
        for i, name in enumerate(WEEKDAYS):
            self.text.draw(img, name, x + 24 + i * cell + cell // 2, y + 182, 24,
                           ui.INK_MUTED, align="centre")

        first = self.pick_month
        start = first.weekday()             # Monday is 0, which is how WEEKDAYS is ordered
        days = (first.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - first
        chosen = self.from_date if self.picking == "from" else self.to_date
        for n in range(days.days):
            day = first + dt.timedelta(days=n)
            col = (start + n) % 7
            row = (start + n) // 7
            cx = x + 24 + col * cell
            cy = y + 226 + row * 76
            box = (cx, cy, cell - 6, 70)
            if day == chosen:
                ui.rounded(img, box, 16, ui.GREEN_700, -1)
                ink = ui.SURFACE
            elif day == dt.date.today():
                ui.rounded(img, box, 16, ui.GREEN_TINT, -1)
                ink = ui.GREEN_700
            else:
                ink = ui.INK
            self.text.draw(img, str(day.day), cx + (cell - 6) // 2, cy + 18, 28, ink,
                           align="centre")
            self.hits.append((f"day{day.isoformat()}", box))

        self.hits.append(("pick-today", ui.button(
            img, self.text, (x + 24, y + h - 96, (w - 72) // 2, 72), "วันนี้", 26,
            "ghost")))
        self.hits.append(("pick-close", ui.button(
            img, self.text, (x + w // 2 + 12, y + h - 96, (w - 72) // 2, 72), "ปิด", 26,
            "ghost")))

    def _keypad(self, img):
        """Digits, a backspace and an OK. Drawn over the screen, dismissed by either.

        NOT THE SYSTEM KEYBOARD. Android's would cover half a landscape screen, arrive
        with autocorrect and a language the operator did not ask for, and need an EditText
        and an InputConnection to exist at all -- in an app whose entire UI is one drawn
        image. Ten buttons is less code than the plumbing for a keyboard nobody wants, and
        it keeps the hit-testing where every other control's already is.
        """
        if self.typing is None:
            return
        w, h = 560, 700
        rect = ((W - w) // 2, (H - h) // 2, w, h)
        ui.rounded(img, rect, 28, ui.SURFACE, -1)
        ui.rounded(img, rect, 28, ui.LINE_STRONG, 2)
        x, y = rect[0], rect[1]

        self.text.draw(img, "ใส่จำนวน", x + w // 2, y + 22, 28, ui.INK_SOFT,
                       align="centre")
        ui.card(img, (x + 32, y + 74, w - 64, 96), 18)
        shown = self.typing or "0"
        self.text.draw(img, shown, x + w // 2, y + 92, 48,
                       ui.INK if self.typing else ui.INK_MUTED, align="centre")

        keys = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "ล้าง", "0", "ลบ")
        kw, kh = (w - 80) // 3, 88
        for i, key in enumerate(keys):
            col, row = i % 3, i // 3
            box = (x + 32 + col * (kw + 8), y + 196 + row * (kh + 8), kw, kh)
            ui.button(img, self.text, box, key, 34 if key.isdigit() else 26, "ghost")
            self.hits.append((f"key-{key}", box))

        self.hits.append(("key-ok", ui.button(
            img, self.text, (x + 32, y + h - 104, w - 64, 88), "ตกลง", 32, "primary")))

    def _list(self, img, box, rows):
        """The records, in columns with headings, as the bench's table has them."""
        x, y, w, h = box
        head_h = 56
        cols = (x + int(w * 0.60), x + int(w * 0.76), x + int(w * 0.90))
        self.text.draw(img, "เวลา", x + 188, y + 18, 24, ui.INK_MUTED)
        for cx, name in zip(cols, ("นับได้", "ต้องการ", "ผล")):
            self.text.draw(img, name, cx, y + 18, 24, ui.INK_MUTED, align="centre")
        cv2.line(img, (x + 16, y + head_h), (x + w - 16, y + head_h), ui.LINE, 2)

        if not rows:
            self.text.draw(img, "ยังไม่มีรายการที่บันทึก" if not self.rows
                           else "ไม่มีรายการที่ตรงกับตัวกรอง",
                           x + w // 2, y + h // 2 - 20, 28, ui.INK_MUTED, align="centre")
            return

        # THE ROW HEIGHT DIVIDES THE CARD, and the scroll moves in whole rows.
        #
        # Nothing here clips: every draw goes straight onto the canvas, so a row that
        # starts inside the card and ends below it is drawn below it -- over the buttons,
        # which is exactly what the first version of this page did. Rows that always
        # divide the space, and a scroll that always lands on one, mean no row is ever
        # half in and half out, and no clipping is needed to say so.
        top = y + head_h + 8
        view_h = h - head_h - 16
        rows_fit = max(1, view_h // 112)
        row_h = view_h // rows_fit
        # Clamped here rather than at the tap: the list also gets shorter underneath a
        # scroll position, when a filter changes or a record is deleted.
        max_scroll = max(0, len(rows) * row_h - view_h)
        self.scroll = max(0, min(self.scroll, max_scroll))
        self.scroll -= self.scroll % row_h
        self.sel = max(0, min(self.sel, len(rows) - 1))

        for i, rec in enumerate(rows):
            ry = int(top + i * row_h - self.scroll)      # indexes the canvas: must be int
            if ry < top - 1 or ry + row_h > y + h + 1:
                continue                    # off the card: not drawn, so not tappable
            if i == self.sel:
                # The selected row is tinted, not outlined -- the bench's own choice, and
                # the reason the preview beside it is not a mystery.
                ui.rounded(img, (x + 8, ry + 2, w - 16, row_h - 4), 14, ui.GREEN_TINT, -1)
            elif i:
                cv2.line(img, (x + 24, ry), (x + w - 24, ry), ui.LINE, 2)
            thumb = self._thumb(rec["image"], (140, 96)) if rec.get("image") else None
            if thumb is not None:
                ui.paste(img, thumb, x + 24, ry + (row_h - thumb.shape[0]) // 2)
            else:
                ui.rounded(img, (x + 24, ry + 14, 140, 84), 12, ui.BG, -1)
            self.text.draw(img, short_when(rec), x + 188, ry + 40, 28, ui.INK)
            self.text.draw(img, str(rec.get("count", 0)), cols[0], ry + 40, 28, ui.INK,
                           align="centre")
            target = int(rec.get("target") or 0)
            self.text.draw(img, str(target) if target else "—", cols[1], ry + 40, 28,
                           ui.INK_SOFT, align="centre")
            kind, words = rec_store.verdict(rec)
            ui.badge(img, self.text, cols[2], ry + 34, words, 24, kind)
            self.hits.append((f"row{i}", (x, ry, w, row_h)))

    # --------------------------------------------------------------------- the pours --
    def banked(self) -> int:
        """Tablets already counted and already tipped out of the tray."""
        return sum(self.rounds)

    def total(self, count) -> int:
        """What the big number is:

            total = what is already in the bottle + what is on the tray right now

        and the second term is dropped while `clearing`, which is the whole of the safety.
        Between taking a pour and the tray being seen empty those tablets are counted in
        BOTH terms -- the list remembers them and the camera can still see them -- so
        adding the two would report a tray of 35 as 70. Dropping the live term until the
        tray has been observed empty means the machine has watched the tablets leave before
        it agrees to count anything else, and a pour cannot be counted twice without
        physically pouring it twice.
        """
        return self.banked() + (0 if self.clearing else int(count))

    def live_total(self) -> int:
        """The total as of the last composed frame, for the parts that have no `count`."""
        return self.total(self._steady_n or 0)

    def _save_look(self) -> str:
        """"warn" when the save would file a SHORT count, "primary" when it would not.

        Saving under the target stays possible on purpose -- the stock runs out, and a
        screen that refuses to record 47 of 60 does not create the missing thirteen, it
        sends the number onto a scrap of paper where nothing can audit it. What was wrong
        was that it looked identical to filing a complete one: same green, same word, so
        the operator learns the gesture rather than the state. The button keeps the job and
        loses the disguise.
        """
        return "warn" if self.target and self.live_total() < self.target else "primary"

    def steady(self) -> bool:
        """Has the live count held the same value for long enough to be acted on."""
        return (self._steady_n is not None
                and time.time() - self._steady_at >= STILL_S)

    def round_block(self) -> str:
        """Why a pour may not be taken right now, or "" when it may.

        The two transient reasons are named rather than left to a grey button: an empty
        tray and a figure still moving both look like the app ignoring a press, and
        somebody who thinks a control is broken presses it harder rather than waiting the
        half second it is asking for.
        """
        if self.blocked:
            return self.blocked
        if self.clearing:
            return "กวาดเม็ดในถาดออกให้หมด แล้วจึงเทรอบต่อไป"
        if not self._steady_n:
            return "ถาดว่าง ยังไม่มีอะไรให้เก็บ"
        if not self.steady():
            return "ตัวเลขยังไม่นิ่ง รอสักครู่"
        return ""

    def _tick_pours(self, count):
        """Advance the stillness timer, and let the tray out of `clearing` once it is empty.

        Called from compose, which runs only when state_key changed -- and `count` and
        `steady()` are both in that key, so every change this needs to see composes a
        frame. A timer of its own would be a second clock to keep in step with the first.
        """
        if count != self._steady_n:
            self._steady_n, self._steady_at = int(count), time.time()
        if self.clearing and not count and self.steady():
            self.clearing = False
            if self.note == CLEAR_NOTE:
                self.say("ถาดว่างแล้ว เทรอบต่อไปได้")

    def take_round(self, frame):
        """Freeze what is on the tray into the total, and refuse the tray until it is empty.

        The guard is not the greyed button repeated for neatness. A disabled button is a
        drawing; the rule that a pour counts only when the figure has settled, the picture
        is alive and the tray has been seen empty since the last pour is what stops the
        same tablets being counted twice, and it belongs where the addition happens.
        """
        why = self.round_block()
        if why:
            self.say(why)
            return False
        self.rounds.append(int(self._steady_n))
        self.round_frames.append(None if frame is None else frame.copy())
        self.clearing = True
        self._steady_n = None               # the stillness timer restarts on the new state
        self.toast = (f"เก็บรอบที่ {len(self.rounds)} {self.rounds[-1]} เม็ด",
                      f"สะสมแล้ว {self.banked()} เม็ด", time.time())
        self.say(CLEAR_NOTE)
        return True

    def reset_rounds(self):
        """Start the prescription again. TWICE, because the total cannot be got back.

        IT DISCARDS EVERYTHING, not the last pour. Stepping back a single pour sounds
        gentler and is false precision: when something has gone wrong mid-prescription --
        a tray tipped twice, a pour banked off the wrong tray, a number nobody is sure of
        -- the operator does not know WHICH pour is wrong, and a button that removes the
        last one invites them to guess. There is no evidence left to check against either,
        because those tablets are in a bottle. Tipping the bottle back out and counting
        again is what restores certainty, so that is what this does and what it says.

        The two taps are the guard the mid-pour save carries, for the same reason: what is
        being discarded cannot be recovered by looking at anything, so arming the button
        and saying on it what the next tap costs turns an accident into two in a row.
        """
        if not (self.rounds or self.clearing):
            return False
        if time.time() - self.reset_armed_at < CONFIRM_SECONDS:
            banked = self.banked()
            self.clear_rounds()
            self.say(f"เริ่มนับใหม่ ทิ้งยอดสะสม {banked} เม็ดแล้ว")
            return True
        self.reset_armed_at = time.time()
        self.say(f"จะทิ้งยอดสะสม {self.banked()} เม็ด กดอีกครั้งเพื่อเริ่มนับใหม่")
        return True

    def clear_rounds(self):
        """Back to a single-pour screen. Called after a save, and only after a save."""
        self.rounds = []
        self.round_frames = []
        self.clearing = False
        self._steady_n = None
        self.save_armed_at = 0.0
        self.reset_armed_at = 0.0

    # ------------------------------------------------------------------- the verdict --
    def _verdict(self, count):
        if not self.target:
            return "none", "ยังไม่กำหนดจำนวน", ui.GREEN_700
        diff = count - self.target
        if diff == 0:
            return "ok", "ครบตามจำนวน", ui.GREEN_700
        if diff > 0:
            return "over", f"เกิน {diff} เม็ด", ui.WARN
        return "short", f"ขาด {-diff} เม็ด", ui.DANGER

    def _inside(self, box):
        if not self.roi:
            return True
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return cv2.pointPolygonTest(np.array(self.roi, np.int32),
                                    (float(cx), float(cy)), False) >= 0

    # ----------------------------------------------------------------------- input --
    def _to_frame(self, x, y, frame_shape):
        """Canvas point -> frame point, through the pane the preview fills.

        The ViewPort in MainActivity guarantees the analysed frame and the displayed one
        are the same picture, so this is a straight scale; without it the two would differ
        by a crop and every region drawn by finger would be wrong at the edges.
        """
        if frame_shape is None:
            return None
        px, py, pw, ph = PANE
        fh, fw = frame_shape[:2]
        if pw <= 0 or ph <= 0:
            return None
        fx = (x - px) * fw / pw
        if self.flip:
            fx = fw - 1 - fx                # spot()'s mirror, undone
        return fx, (y - py) * fh / ph

    def touch(self, phase, x, y, frame_shape=None, on_save=None, on_export=None):
        """One touch event in CANVAS coordinates. Returns True if something changed.

        The phases are the three Android gives: down, move, up. A drag over the picture
        while the region is being set is the only gesture that uses move; everything else
        acts on the release, so a finger that lands on the wrong button can be slid off it.
        """
        # Taps arrive in the coordinates of the picture Android was given, which is the
        # output size; everything drawn here is in the design size. One multiplication,
        # in one place, or every hit test would be wrong by a third.
        #
        # AND THEY ARRIVE AS FLOATS, which is the other half of this line. Android reports
        # a touch in fractions of a pixel, the scale above keeps the fraction, and a drag
        # on the records list assigned that straight into `scroll` -- which is subtracted
        # from a row's y and then used to index the canvas. numpy will not take a float as
        # a slice, so the frame threw TypeError, Kotlin caught it, and the phone showed
        # "ประมวลผลภาพไม่สำเร็จ" until the list was scrolled back to nothing. Rounded here,
        # at the one door they come in through, rather than at each of the places a
        # coordinate eventually lands.
        x = int(x * W / OUT_W)
        y = int(y * H / OUT_H)
        if phase == "down":
            return self._down(x, y, frame_shape)
        if phase == "move":
            return self._move(x, y, frame_shape)
        return self._up(x, y, frame_shape, on_save, on_export)

    def _down(self, x, y, frame_shape):
        self._touch_start = (x, y)
        self._scroll_start = self.scroll
        return False

    def _move(self, x, y, frame_shape):
        if self._on_list(self._touch_start):
            self.scroll = int(self._scroll_start + (self._touch_start[1] - y))
            return True
        return False

    def _on_list(self, point):
        """Did this touch begin on the records list? Only that list scrolls."""
        return bool(point and self.page == "records" and self.list_box
                    and ui.hit(self.list_box, point[0], point[1]))

    def _up(self, x, y, frame_shape, on_save, on_export):
        # A drag on the list is a scroll, not a tap on whatever it ended over. Anywhere
        # else, a finger that moved is a finger. See TAP_SLOP.
        start = self._touch_start
        if (start and self._on_list(start) and abs(start[1] - y) > TAP_SLOP):
            return False
        # A CORNER GOES DOWN ONLY IF THE FINGER STAYED PUT, which is the affordance every
        # button already has: start the touch, think better of it, slide off before letting
        # go. What is being placed decides which pills are counted, so it deserves the way
        # out at least as much as a button does.
        #
        # AND ONLY ON THE PICTURE, never over a control. The corner has to be placed on the
        # tray, so a tap anywhere else is a tap on what it landed on -- including the very
        # button that armed this mode, which sits a thumb's width from the picture and
        # whose whole job right now is to say "ยกเลิก". `hits` carries ("view", PANE) as
        # its last entry, so "did this land on a control" has to exclude the picture
        # itself; without that, every corner was read as a tap on the video and thrown away.
        on_control = any(ui.hit(rect, x, y) for name, rect in self.hits if name != "view")
        if (self.arming and self.page == "count" and start and not on_control
                and ui.hit(PANE, x, y)
                and abs(start[0] - x) <= TAP_SLOP and abs(start[1] - y) <= TAP_SLOP):
            return self._corner(x, y, frame_shape)
        return self._press(x, y, on_save, on_export)

    def _corner(self, x, y, frame_shape):
        """One corner placed. The fourth closes the shape and sets the region."""
        point = self._to_frame(x, y, frame_shape)
        if point is None:
            return False
        self._chrome_key = None             # the row's second button counts the corners
        self.pending.append((point[0], point[1]))
        if len(self.pending) < ROI_POINTS:
            self._roi_prompt()
            return True
        size = (frame_shape[1], frame_shape[0]) if frame_shape else (0, 0)
        pts = quad(self.pending, size)
        if pts is None:
            # The four corners enclose nothing worth counting -- tapped on one spot, or
            # strung out in a line. Only the LAST one is dropped: three good corners and a
            # slip is the likely case, and throwing all four away would charge the operator
            # four times for one slip.
            self.pending.pop()
            self.say("มุมนี้แคบเกินไป แตะให้ห่างจากมุมอื่น")
            return True
        self.roi = pts
        self.arming = False
        self.pending = []
        if frame_shape:
            rec_store.save_roi(self.records_dir, size, self.roi)
        self.say("กำหนดกรอบแล้ว")
        return True

    def _flip(self, frame_shape=None):
        """Turn the picture round. NOTHING ELSE MOVES, and that is the whole design.

        An earlier version flipped the camera frame and then had to mirror the counting
        region to match, because the region is stored in frame pixels and would otherwise
        have jumped to the other side of the bench. It worked, and it was wrong twice over:
        the detector was handed a different image, and the record it saved was of a picture
        that only existed because of a display preference.

        Now the frame is never touched. Only the pane's drawing and the tap that comes back
        through it know, so the region, the model, the marks and the saved JPEG all go on
        living in the camera's own coordinates.
        """
        self.flip = not self.flip
        rec_store.save_flip(self.records_dir, self.flip)
        self.arming = False
        self.pending = []
        self.say("พลิกภาพซ้าย-ขวาแล้ว" if self.flip else "เลิกพลิกภาพแล้ว")

    def _corner_undo(self):
        """The row's second button while corners are going down: the last one back."""
        if self.pending:
            self.pending.pop()
            self._roi_prompt()
            return
        self.arming = False
        self.say("")

    def _roi_prompt(self):
        left = ROI_POINTS - len(self.pending)
        self.say(f"แตะมุมถาดทีละมุม อีก {left} จุด")

    def _press(self, x, y, on_save, on_export):
        for name, rect in reversed(self.hits):
            if not ui.hit(rect, x, y):
                continue
            return self._act(name, on_save, on_export)

        # Nothing was hit. With a pad or a calendar up that is a press beside it, which
        # everywhere else in the world means "put it away"; without one it is a press on
        # the background and means nothing at all.
        if self.typing is not None or self.picking:
            self.typing, self.picking = None, ""
            self._chrome_key = None
            return True
        return False

    def _act(self, name, on_save, on_export):
        self._chrome_key = None             # whatever it changes, the furniture is redrawn

        # ANY press that is not part of an overlay closes it. The keypad used to stay up
        # across a page change -- press the target, change your mind, open the records and
        # it was still sitting there over the list, with no way back to the screen that
        # owned it. An overlay belongs to the moment it was opened in, so leaving that
        # moment by any door closes it.
        if not name.startswith(("key-", "day", "month", "pick-")):
            self.typing = None
            self.picking = ""
        if name == "target-":
            self.target = max(0, self.target - 1)
        elif name == "target+":
            self.target += 1
        elif name.startswith("preset"):
            self.target = int(name[6:])
        elif name == "flip":
            self._flip()
        elif name == "roi":
            self.arming = not self.arming
            self.pending = []
            if self.arming:
                self._roi_prompt()
            else:
                self.say("")
        elif name.startswith("row"):
            self.sel = int(name[3:])
        elif name == "refresh":
            self.reload()
            self.say("อ่านรายการใหม่แล้ว")
        elif name == "roi-clear":
            # ONE BUTTON, TWO JOBS, and they are the same job at two moments. While corners
            # are going down the only thing anybody wants from it is the last one back;
            # once the region is set the only thing anybody wants is it gone. A separate
            # undo would sit dead and unexplained for every hour nobody is drawing a region
            # -- and this panel has no row to spare for it either.
            if self.arming:
                self._corner_undo()
            else:
                self.roi = None
                self.arming = False
                self.pending = []
                rec_store.save_roi(self.records_dir, (0, 0), None)
        elif name == "round":
            self.take_round(self._last_frame)
        elif name == "reset":
            self.reset_rounds()
        elif name == "save":
            if self.blocked:
                self.say(self.blocked)
            elif not self._save_armed():
                pass                        # armed, and said so; the next tap files it
            elif on_save is not None:
                on_save()
        elif name == "records":
            self.page = "records"
            self.reload()
        elif name == "back":
            self.page = "count"
        elif name.startswith("filter"):
            self.filter = int(name[6:])
            self.scroll = 0
        elif name == "export" and on_export is not None:
            on_export()
        elif name == "wipe":
            self._wipe()
        elif name == "dismiss-fault":
            self.fault_seen = self.fault
        elif name == "type-target":
            self.typing = ""
        elif name.startswith("key-"):
            self._typed(name[4:])
        elif name in ("pick-from", "pick-to"):
            self.picking = name[5:]
            base = self.from_date if self.picking == "from" else self.to_date
            self.pick_month = base.replace(day=1)
        elif name == "pick-close":
            self.picking = ""
        elif name == "pick-today":
            self._set_picked(dt.date.today())
        elif name == "month-":
            self.pick_month = (self.pick_month - dt.timedelta(days=1)).replace(day=1)
        elif name == "month+":
            self.pick_month = (self.pick_month.replace(day=28)
                               + dt.timedelta(days=7)).replace(day=1)
        elif name.startswith("day"):
            self._set_picked(dt.date.fromisoformat(name[3:]))
        else:
            return False
        return True

    def _typed(self, key):
        """One press on the pad. Four digits is as far as it goes -- see the comment.

        A tray holds hundreds, not thousands, and a target of 99999 is a slip rather than
        an instruction; capping it means the number can never be wider than the box that
        shows it, which is a layout that cannot break rather than one that is watched.
        """
        if key == "ตกลง" or key == "ok":
            self.target = int(self.typing) if self.typing else 0
            self.typing = None
        elif key == "ล้าง":
            self.typing = ""
        elif key == "ลบ":
            self.typing = (self.typing or "")[:-1]
        elif key.isdigit() and len(self.typing or "") < 4:
            self.typing = (self.typing or "") + key

    def _set_picked(self, day):
        """Take the date, keep the two ends in order, and switch to the range filter.

        Swapping them rather than refusing: somebody picking "to" before "from" has said
        what they mean perfectly clearly, and an error message would be pedantry.
        """
        if self.picking == "from":
            self.from_date = day
            if self.to_date < day:
                self.to_date = day
        else:
            self.to_date = day
            if self.from_date > day:
                self.from_date = day
        self.picking = ""
        self.filter = 3
        self.scroll = 0

    def _save_armed(self) -> bool:
        """Two taps to file a total with pours banked in it that has not reached target.

        ONLY THAT CASE. A short count on ONE tray is the stock running out, and the button
        already says so in its words and its colour; making that errand take two taps would
        put a guard in front of something done all day, and a guard that fires every day is
        one nobody reads by the end of the week.

        What is different mid-pour is what a stray tap DESTROYS. The tablets in `rounds`
        are in the bottle: they cannot be re-counted, they are not on the tray, and the
        only record of them is the list this stands in front of. Saving clears it. So the
        operator who meant to tap after the next pour and tapped now does not merely file a
        wrong number -- they lose the count and have to tip the bottle out and start the
        prescription again.

        Two taps rather than a dialog because that is what this screen already does for the
        other irreversible thing on it, and because a phone tap goes through in one motion:
        arming the button and saying on it what the next tap costs turns an accident into
        two accidents in a row.
        """
        if not (self.rounds and self.target and self.live_total() < self.target):
            return True
        if time.time() - self.save_armed_at < CONFIRM_SECONDS:
            self.save_armed_at = 0.0
            return True
        self.save_armed_at = time.time()
        self.say(f"ยังเทค้างอยู่ ยอดสะสมจะถูกล้าง "
                 f"แตะอีกครั้งเพื่อบันทึก {self.live_total()}")
        return False

    def _wipe(self):
        """Two taps, not one, and the second only counts within a few seconds.

        There is no dialog here and no Recycle Bin underneath: a phone tap goes through in
        one motion and the files are gone for good. Arming the button and saying so on it
        turns an accident into two accidents in a row.
        """
        if time.time() - self.confirm_at < CONFIRM_SECONDS:
            gone, failed = rec_store.delete(self.rows)
            self.confirm_at = 0.0
            self._thumbs.clear()
            self.reload()
            self.say(f"ลบแล้ว {gone} รายการ" if not failed
                     else f"ลบแล้ว {gone} รายการ เหลือ {len(failed)}")
            return
        self.confirm_at = time.time()
        self.say("แตะอีกครั้งเพื่อยืนยัน")

    def saved(self, count, stamp):
        """Called by the bridge once a record is on disk.

        THE POURS ARE FORGOTTEN HERE AND NOWHERE EARLIER, because here is the first moment
        they are safe to forget: the bridge calls this after the JSON is written. A phone
        that ran out of storage mid-save leaves the total on screen, where it can be filed
        again, rather than clearing it on the strength of a tap.
        """
        self.toast = (f"บันทึกแล้ว {count} เม็ด",
                      f"{time.strftime('%H:%M:%S')} {stamp}", time.time())
        self.say(f"บันทึกแล้ว {count} เม็ด")
        self.clear_rounds()
        self.reload()
