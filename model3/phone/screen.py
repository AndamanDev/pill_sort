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
NOTE_SECONDS = 4.0
TOAST_SECONDS = 1.8
CONFIRM_SECONDS = 4.0      # how long a "tap again to confirm" stays armed
THUMB_CACHE = 40

MONTHS = ("", "ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
          "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค.")


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
        self.drag_rect = None               # (x0, y0, x1, y1) in frame pixels, mid-drag
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
                None if self.drag_rect is None else tuple(int(v) for v in self.drag_rect),
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
                bool(self.blocked), self.filter, self.scroll, len(self.rows),
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
        elif self.target and count > self.target:
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
        # The region and the model's time are CHIPS, drawn live in _camera where the bench
        # draws them: top right of this card, not grey words beside the title.

        self.text.draw(img, "เม็ด", 414, 348, 28, ui.INK_MUTED, align="centre")
        self.text.draw(img, "จำนวนที่ต้องการ", 60, 506, 28, ui.INK_SOFT)
        row_y = 552
        self.hits.append(("target-", ui.button(img, self.text, (60, row_y, 110, 104),
                                               "-", 46, "ghost")))
        # The number itself is a button: press it and a pad comes up. A phone has no
        # keyboard to reach for, and stepping from 0 to 120 with the + key is not a
        # control, it is a punishment.
        box = (186, row_y, 456, 104)
        ui.card(img, box, 18)
        self.text.draw(img, str(self.target), 414, row_y + 22, 48, ui.INK, align="centre")
        self.hits.append(("type-target", box))
        self.hits.append(("target+", ui.button(img, self.text, (658, row_y, 110, 104),
                                               "+", 46, "ghost")))

        chip_w = (708 - 5 * 10) // 6
        for i, n in enumerate(PRESETS):
            rect = (60 + i * (chip_w + 10), 676, chip_w, 76)
            ui.button(img, self.text, rect, str(n), 26, "chip",
                      "on" if n == self.target else "off")
            self.hits.append((f"preset{n}", rect))

        # ONE ROW FOR THE TWO FRAME CONTROLS. They are a pair -- set it, clear it -- and
        # a row each would say they were two unrelated things while spending twice the
        # panel on saying it. app/window.py stacked them and now does the same as this.
        half = (708 - 12) // 2
        self.hits.append(("roi", ui.button(
            img, self.text, (60, 772, half, 96),
            "ยกเลิก" if self.arming else ("กำหนดกรอบใหม่" if self.roi else "กำหนดกรอบนับ"),
            28, "ghost")))
        self.hits.append(("roi-clear", ui.button(img, self.text, (72 + half, 772, half, 96),
                                                 "ล้างกรอบ", 28, "ghost",
                                                 enabled=bool(self.roi))))
        self.hits.append(("save", ui.button(img, self.text, (60, 884, 708, 104),
                                            "บันทึกผล", 38, "primary",
                                            enabled=not self.blocked)))

    def _live_count(self, img, frame, boxes, count, ms, stale):
        """The number, the verdict, the bar and the video overlay: redrawn every frame."""
        kind, words, colour = self._verdict(count)
        self.text.draw(img, str(count), 414, 132, 168, colour, align="centre")
        ui.badge(img, self.text, 414, 396, words, 30, kind)
        if self.target:
            ui.progress(img, (68, 470, 692, 16), count / max(1, self.target), colour)
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
            return int(px + x * kx), int(py + y * ky)

        # NOTHING IS DRAWN INTO THE HOLE HERE -- it is written down and drawn by
        # _paint_over_video once the screen has been scaled. See that method.
        for b in boxes:
            if self.roi is not None and not self._inside(b):
                continue
            cx, cy = spot((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            self._over.append(("circle", (cx, cy, 11), ui.MARK_RIM, -1))
            self._over.append(("circle", (cx, cy, 7), ui.MARK, -1))

        if self.drag_rect:
            band = self._rect_from_drag(self.drag_rect, frame.shape, floor=0)
            if band:
                x0, y0 = spot(band[0], band[1])
                x1, y1 = spot(band[0] + band[2], band[1] + band[3])
                self._over.append(("rect", (x0, y0, x1, y1), ui.ROI_BAND, 5))
                self._over.append(("ticks", (x0, y0, x1 - x0, y1 - y0), ui.ROI_BAND,
                                   (44, 8)))
        elif self.roi:
            pts = np.array(self.roi, np.int32)
            x0, y0 = spot(pts[:, 0].min(), pts[:, 1].min())
            x1, y1 = spot(pts[:, 0].max(), pts[:, 1].max())
            self._over.append(("rect", (x0, y0, x1, y1), ui.ROI_LINE, 4))
            self._over.append(("ticks", (x0, y0, x1 - x0, y1 - y0), ui.ROI_LINE, (48, 8)))

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
        return (x - px) * fw / pw, (y - py) * fh / ph

    def _rect_from_drag(self, drag, frame_shape, floor=MIN_ROI):
        h, w = frame_shape[:2]
        x0, x1 = sorted((drag[0], drag[2]))
        y0, y1 = sorted((drag[1], drag[3]))
        x0, x1 = max(0, int(x0)), min(w - 1, int(x1))
        y0, y1 = max(0, int(y0)), min(h - 1, int(y1))
        if x1 - x0 < floor or y1 - y0 < floor:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

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
        if self.arming and self.page == "count":
            point = self._to_frame(x, y, frame_shape)
            if point is not None:
                self.drag_rect = (point[0], point[1], point[0], point[1])
                return True
        return False

    def _move(self, x, y, frame_shape):
        if self.drag_rect is not None:
            point = self._to_frame(x, y, frame_shape)
            if point is not None:
                self.drag_rect = (self.drag_rect[0], self.drag_rect[1], point[0], point[1])
                return True
            return False
        if self._on_list(self._touch_start):
            self.scroll = int(self._scroll_start + (self._touch_start[1] - y))
            return True
        return False

    def _on_list(self, point):
        """Did this touch begin on the records list? Only that list scrolls."""
        return bool(point and self.page == "records" and self.list_box
                    and ui.hit(self.list_box, point[0], point[1]))

    def _up(self, x, y, frame_shape, on_save, on_export):
        if self.drag_rect is not None:
            rect = self._rect_from_drag(self.drag_rect, frame_shape or (1, 1))
            self.drag_rect = None
            self.arming = False
            if rect is None:
                self.say("กรอบเล็กเกินไป")
                return True
            rx, ry, rw, rh = rect
            self.roi = [(rx, ry), (rx + rw, ry), (rx + rw, ry + rh), (rx, ry + rh)]
            if frame_shape:
                rec_store.save_roi(self.records_dir,
                                   (frame_shape[1], frame_shape[0]), self.roi)
            self.say("กำหนดกรอบแล้ว")
            return True
        # A drag on the list is a scroll, not a tap on whatever it ended over. Anywhere
        # else, a finger that moved is a finger. See TAP_SLOP.
        start = self._touch_start
        if (start and self._on_list(start) and abs(start[1] - y) > TAP_SLOP):
            return False
        return self._press(x, y, on_save, on_export)

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
        elif name == "roi":
            self.arming = not self.arming
            self.drag_rect = None
            self.say("ลากนิ้วคลุมพื้นที่ถาด" if self.arming else "")
        elif name.startswith("row"):
            self.sel = int(name[3:])
        elif name == "refresh":
            self.reload()
            self.say("อ่านรายการใหม่แล้ว")
        elif name == "roi-clear":
            self.roi = None
            self.arming = False
            rec_store.save_roi(self.records_dir, (0, 0), None)
        elif name == "save":
            if self.blocked:
                self.say(self.blocked)
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
        """Called by the bridge once a record is on disk."""
        self.toast = (f"บันทึกแล้ว {count} เม็ด",
                      f"{time.strftime('%H:%M:%S')} {stamp}", time.time())
        self.say(f"บันทึกแล้ว {count} เม็ด")
        self.reload()
