"""The screen: header, the number and its controls on the left, the camera right, footer.

WRITTEN FOR A HOSPITAL DISPENSARY, which sets most of what follows. The person reading it
is standing, their hands are busy, the room is bright, and they are checking a number they
will act on. So: one figure large enough to read across a bench, a verdict in words beside
it rather than a colour alone, controls big enough to hit without looking, and every piece
of small print moved out of the working area into a footer.

NOTHING HERE IS SMALLER THAN 18px, and that is measured, not taste. On this bench every
label at 18px and above rendered cleanly while three at 16-17px came out as an unreadable
smear -- and one of those was a single line, so it was never lines colliding. Below 18 this
font stops placing Thai tone marks correctly at this DPI.
"""
from __future__ import annotations

import glob
import json
import os
import time

import cv2
import numpy as np
from PySide6.QtCore import (QPropertyAnimation, QRectF, QSize, Qt, QTimer,
                            Signal)
from PySide6.QtGui import (QColor, QFont, QImage, QPainter, QPainterPath, QPen,
                           QPixmap)
from PySide6.QtWidgets import (QFrame, QGraphicsOpacityEffect, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QProgressBar,
                               QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from . import LOGO, RECORDS, SETTINGS
from . import sound
from . import theme as T

#: The marker, BGR. THE APP'S OWN GREEN, with a white rim around it.
#:
#: It was black, which is the obvious choice and a slightly wrong one: black is what a
#: shadow between two pills already looks like, so at a glance the marks and the gaps read
#: as the same thing. The green belongs to this app -- the count above the picture is the
#: same colour -- so a marked pill and the number it contributes to are visibly one claim.
#: The white rim stays either way: on the dark capsules this tray also carries, a dot of
#: any colour needs an edge to sit against.
MARK = (68, 134, 16)                        # #108644, the green everything else uses
MARK_RIM = (255, 255, 255)
#: Behind the letterboxed picture, and the same colour the card's #view rule paints, so
#: the bars read as part of the card rather than as a hole in it.
VIEW_BG = "#0f1714"
ROI_LINE = (90, 220, 110)
#: The rubber band while a region is being dragged. Brighter than the settled outline
#: because for those two seconds it IS the thing being looked at.
ROI_BAND = (140, 245, 160)
#: The border drawn round a picture that has stopped arriving. BGR, so this is red.
STALE_EDGE = (60, 60, 220)

DRAW_MS = 16                # ~60 fps repaint, independent of the model
MIN_ROI = 40                # frame pixels; the shortest side a usable region can have

#: Widget pixels a press may travel and still count as a tap on one spot.
#:
#: A corner is placed on RELEASE, not on press, so that sliding off before letting go
#: takes the tap back -- the affordance every button on every screen already has, and the
#: one that matters most when what is being placed is a corner of the counting region.
TAP_SLOP = 8

#: How many corners a region takes. FOUR, and the shape closes itself on the fourth.
#:
#: A tray seen from above is a quadrilateral -- and only a RECTANGLE when the lens is
#: exactly square to the bench, which it never quite is. The rubber band this replaced
#: could only make rectangles, so an angled camera left the operator choosing between a
#: box that ate the bench beside the tray and one that cut its far corners off. Four
#: corners cost four taps and buy the shape the tray actually has.
ROI_POINTS = 4

#: Said in the footer while the save button is dead, so the grey button is never a mystery.
OVER_NOTE = "เกินจำนวนที่ต้องการ  นำออกก่อนจึงบันทึกได้"

#: Said while a round has been taken but the tray it was taken from is still full.
CLEAR_NOTE = "กวาดเม็ดในถาดออกให้หมด  แล้วจึงเทรอบต่อไป"

#: Seconds the number must hold STILL before a round may be taken from it.
#:
#: NOT COSMETIC. pillcount-det-v3 reads a motionless tray with a spread of 1 -- the figure
#: flickers between 34 and 35 as a box appears and disappears on one tablet at the edge of
#: the region. A count taken on whichever frame the button happened to land on is therefore
#: a coin toss between two numbers, and the one it picks is then FROZEN into the total and
#: swept into a bottle where nobody can re-count it. Half a second of the same figure is
#: cheap to wait for and turns the coin toss into a reading.
SETTLE_S = 0.5

#: How long a "press again to confirm" stays armed. Long enough to reach for a second
#: time, short enough that a press a minute later is a fresh first press rather than the
#: back half of a pair nobody remembers making.
CONFIRM_S = 4.0

#: Seconds the tray must read EMPTY before the next round is allowed to begin.
#:
#: This is the whole safety of multi-round counting. After a round is taken the tablets are
#: still lying on the tray, and if the live count were added to the total again the moment
#: the button came back up, one press of it would count the same 35 tablets twice -- 70 for
#: a tray that holds 35, filed as if it were fact. So the total IGNORES the tray until the
#: tray has been seen empty, which is the physical act (sweeping it into the bottle) that
#: makes the next pour a different pour. A number cannot be counted twice if the machine
#: has watched it leave.
CLEAR_S = 0.4

#: Seconds without a new frame before the picture is called dead. Above a dropped frame or
#: two (this camera runs about 20fps), below the time it takes somebody to look up, press
#: save, and file a count of the tray that was there a moment ago.
STALE_S = 1.5
PRESETS = (10, 20, 30, 60, 90, 100)

#: HOW THE PANEL SHRINKS, roomiest first: (count px, spacing, margin, keep the presets).
#:
#: A Qt layout that cannot fit does not scroll, it SQUEEZES -- every widget is handed less
#: than its size hint at once. Measured at 1366x768: the panel wanted 721px and had 600, so
#: the count was clipped through the middle AND the buttons lost the tone marks off their
#: Thai, because the text rect each one got was shorter than the type it paints. It reads
#: as a broken font and is nothing of the kind. So the panel is fitted deliberately: give
#: up size on the count first, then the rhythm, and only at the end the preset row -- the
#: one thing here that is a shortcut rather than a control, since the target can still be
#: typed or stepped.
#:
#: The last two entries were added when the round row arrived below the presets. A row of
#: buttons is about sixty pixels and the tightest fit had no sixty pixels spare, so it is
#: paid for out of the count -- which is the right pocket: 44px of ExtraBold NUMERAL is
#: still legible across a bench, and a numeral carries no tone marks, so shrinking it
#: cannot break the way shrinking Thai text does. That asymmetry is why this list gives up
#: the figure first and the words last.
PANEL_FITS = ((104, 14, 24, True), (92, 14, 24, True), (80, 12, 20, True),
              (68, 10, 16, True), (68, 8, 14, False), (56, 8, 12, False),
              (48, 6, 10, False), (44, 4, 8, False))

#: Header + footer + the body's top and bottom margins: everything between the window and
#: the height the two cards get to share.
CHROME_H = 72 + 56 + 40


# --------------------------------------------------------------------------- settings
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

    model3/phone/screen.py has this function too, word for word, and the two are kept in step by a test
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


def roi_path(camera):
    return os.path.join(SETTINGS, f"roi-cam{camera}.json")


def view_path(camera):
    return os.path.join(SETTINGS, f"view-cam{camera}.json")


def load_flip(camera) -> bool:
    """Whether this camera's picture is mirrored. A FILE OF ITS OWN, beside the region.

    ON UNTIL SOMEBODY TURNS IT OFF. A webcam and a phone camera are both built to be
    pointed at a face, and both hand over a mirror image because that is what a face
    expects to see; a tray does not. Measured on this bench, both screens showed the tray
    the wrong way round, so the setting that matches the hardware is the one that should
    need no press -- and the press is there for a camera that does not mirror.

    Not a field in the region file, which is where it nearly went. Clearing the region
    deletes that file, and an operator who redraws the tray has not asked for the picture
    to turn round -- they would find it mirrored again with nothing on screen to explain
    why. The two settings have different lifetimes, so they get different files.
    """
    try:
        with open(view_path(camera), encoding="utf-8") as fh:
            return bool(json.load(fh).get("flip", True))
    except Exception:                                           # noqa: BLE001
        return True


def save_flip(camera, flip):
    os.makedirs(SETTINGS, exist_ok=True)
    with open(view_path(camera), "w", encoding="utf-8") as fh:
        json.dump({"camera": camera, "flip": bool(flip),
                   "saved": time.strftime("%Y-%m-%d %H:%M")}, fh,
                  ensure_ascii=False, indent=2)


def load_roi(camera, size):
    """The region from last time, if it can still mean what it meant.

    A polygon is raw pixel coordinates, valid only for the camera that drew it and at the
    resolution it was drawn at. Restoring a 640x480 polygon onto a 1280x720 frame would
    cover a quarter of the tray, and the count would be wrong while looking perfectly
    reasonable -- the worst kind of wrong. So the frame size is stored alongside it and a
    mismatch drops the region rather than scaling it.
    """
    path = roi_path(camera)
    if not os.path.isfile(path):
        return None, ""
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        pts = [(int(x), int(y)) for x, y in d["roi"]]
        if len(pts) < 3:
            return None, ""
        if tuple(d.get("frame", ())) != tuple(size):
            return None, "ขนาดภาพเปลี่ยน ต้องกำหนดกรอบใหม่"
        return pts, ""
    except Exception:                                           # noqa: BLE001
        return None, ""


def save_roi(camera, size, pts):
    os.makedirs(SETTINGS, exist_ok=True)
    path = roi_path(camera)
    if not pts:
        if os.path.isfile(path):
            os.remove(path)
        return
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"camera": camera, "frame": list(size),
                   "saved": time.strftime("%Y-%m-%d %H:%M"),
                   "roi": [[int(x), int(y)] for x, y in pts]}, fh,
                  ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------- widgets
def _card() -> QWidget:
    w = QWidget()
    w.setObjectName("card")
    return w


def styled(label, size, weight=QFont.Normal, colour=None):
    """Set the font BOTH ways, because either alone is wrong in a different direction.

    setFont() alone loses: the theme carries `QWidget {{ font-size: 16px }}` and a Qt
    stylesheet beats setFont on the widgets it matches, so a label asked for 104px drew
    at 16 and the count came out the size of a caption.

    A stylesheet alone loses too: sizeHint is computed from the widget's FONT, which the
    stylesheet does not change, so the label reports the width its 9pt default needs and
    the layout hands it too little room. The glyphs then crush together and it reads as
    broken Thai shaping -- which sent three rounds of fixes at the font, the size and the
    layout, all of them innocent.

    So: the QFont makes the measurement honest, the stylesheet makes the painting honest,
    and they are given the same numbers. `base` is kept on the widget so a later recolour
    can rebuild the whole rule instead of replacing it with a bare `color:`.
    """
    css_weight = {QFont.Normal: 400, QFont.DemiBold: 600,
                  QFont.Bold: 700, QFont.ExtraBold: 800}.get(weight, 400)
    font = QFont()
    font.setFamilies(T.FONT_FAMILIES)
    font.setPixelSize(size)
    font.setWeight(weight)
    label.setFont(font)
    label.base = (f"font-family: {T.FONT_STACK}; "
                  f"font-size: {size}px; font-weight: {css_weight};")
    label.setStyleSheet(label.base + (f" color: {colour};" if colour else ""))
    return label


def recolour(label, colour):
    """Change the colour without throwing the size away with it."""
    label.setStyleSheet(getattr(label, "base", "") + f" color: {colour};")


def badge_css(pair) -> str:
    """A tint behind dark text, rebuilt from the label's own font rule.

    theme.badge_css exists and is not used here: it fixes the size at 16px, which is below
    the 18px this screen holds to.
    """
    bg, fg = pair
    return (f" background: {bg}; color: {fg}; border-radius: 999px; "
            f"padding: 8px 22px;")


def sized(widget, size, weight=QFont.DemiBold):
    """Give a BUTTON the font it will be painted with. The stylesheet alone is not enough.

    Same trap styled() describes for labels, arriving from the other side. A button styled
    only through the sheet reports a sizeHint computed from the default 9pt font, so the
    layout hands it a box sized for 9pt text and Qt then paints 22px Thai into it: the
    letters fit, the tone marks do not, and "บันทึกผล" comes out as บนทกผล. It only showed
    once the panel was full enough that the button sat at exactly its hint with no spare
    height to hide the shortfall -- so it looked like a broken font, twice, and was not.
    """
    font = QFont()
    font.setFamilies(T.FONT_FAMILIES)
    font.setPixelSize(size)
    font.setWeight(weight)
    widget.setFont(font)
    return widget


def _rule() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setObjectName("rule")
    return line


class CameraView(QLabel):
    """The live picture, and the corners of the counting region tapped onto it.

    A CORNER AT A TIME, NOT A RUBBER BAND. The band was here first and was chosen for good
    reasons -- one gesture, no mode, and everybody already owns it from every photo tool
    they have used. What it could not do is the shape: a band draws a rectangle, and a tray
    is a rectangle only when the lens is exactly square to the bench. It never quite is. On
    this bench the camera looks down at an angle and the tray arrives as a trapezoid, so
    the band left a choice between a box that took in the bench beside the tray and one
    that cut the far corners off -- and pills sit in corners.

    Four taps cost three more gestures than a drag and buy the shape the tray actually has.
    The order they are tapped in does not matter; see quad().

    Every coordinate leaving this class is in FRAME pixels, the only space the model and the
    saved region agree on. The widget's own pixels stop meaning anything the moment somebody
    resizes the window.
    """

    tapped = Signal(float, float)           # a corner, in frame coordinates
    cancelled = Signal()                    # right click: take the last corner back

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignCenter)
        self.setObjectName("view")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._geom = None                   # (x0, y0, scale) of the last drawn pixmap
        self._pix = None                    # the frame as painted, letterboxed
        self._fw = 0                        # the frame's width, for the mirror
        #: Is the picture being shown left-to-right reversed. Set by the window, and the
        #: ONLY thing this class does with it is undo it on the way in -- see _to_frame.
        self.flip = False
        self.arming = False                 # set by the window; drives the cursor
        self._press_at = None               # where the button went down, for TAP_SLOP

    def set_arming(self, on):
        self.arming = on
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    def set_flip(self, on):
        self.flip = bool(on)

    def minimumSizeHint(self):
        """A picture must never set the floor for the window. Found while measuring.

        QLabel reports the size of its PIXMAP as its minimum hint, and this label is handed
        a pixmap scaled to its own size sixty times a second: the hint follows the widget
        up, the layout's minimum follows the hint, and the window's minimum follows that.
        It ratchets -- after a minute on a big screen the window cannot be made small again,
        and the minimum had climbed to 900px tall while the label's own explicit minimum
        said 480. Maximised, nobody sees it; the first person to un-maximise does.
        """
        return QSize(320, 240)

    def show_frame(self, bgr):
        h, w = bgr.shape[:2]
        self._fw = w
        img = QImage(bgr.data, w, h, 3 * w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(img).scaled(self.size(), Qt.KeepAspectRatio,
                                            Qt.SmoothTransformation)
        self._geom = ((self.width() - pix.width()) / 2,
                      (self.height() - pix.height()) / 2, pix.width() / w)
        self._pix = pix
        self.update()

    def paintEvent(self, ev):
        """Draw the frame ourselves, clipped to the card's own corner radius.

        setPixmap draws a SQUARE bitmap over a rounded widget: the stylesheet rounds the
        background, the picture covers it, and the four corners come back as hard edges
        sitting a pixel proud of every other card on the screen. Clipping to the same
        radius is the whole difference, and it costs one path per repaint.

        THE PICTURE IS NEVER CROPPED TO FILL. Letterboxing looks worse than an
        edge-to-edge picture and is not negotiable here: cropping would hide the pills
        nearest the edge of the tray, and a count is exactly what it would hide them from.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), T.R_LG, T.R_LG)
        painter.setClipPath(path)
        painter.fillRect(self.rect(), QColor(VIEW_BG))
        if self._pix is not None:
            x0, y0, _ = self._geom
            painter.drawPixmap(int(x0), int(y0), self._pix)
            return
        painter.setPen(QColor(T.INK_MUTED))
        painter.setFont(self.font())
        painter.drawText(self.rect(), Qt.AlignCenter, "กำลังเปิดกล้อง")

    def _to_frame(self, pos):
        """Widget point -> frame point, or None while nothing has been drawn yet.

        The picture is letterboxed inside the widget, so the point comes back through the
        same offset and scale it was drawn with -- otherwise the region lands somewhere
        near, but not where, the operator tapped it.

        AND THROUGH THE MIRROR, when the picture is being shown reversed. This is one of
        exactly two places the flip exists: the window mirrors the finished picture on its
        way to the screen, and this undoes it on the way back in. Between them the frame,
        the model, the counting region and the saved record never learn that the setting
        exists -- which is the point. A flip applied to the frame itself would feed the
        detector a different image and can move the count; measured on this bench it moved
        it by three on one tray. A preference about which way round a picture looks must
        not be able to change a number that goes in a record.
        """
        if self._geom is None:
            return None
        x0, y0, scale = self._geom
        if scale <= 0:
            return None
        x = (pos.x() - x0) / scale
        if self.flip and self._fw:
            x = self._fw - 1 - x
        return x, (pos.y() - y0) / scale

    def mousePressEvent(self, ev):
        if ev.button() == Qt.RightButton:
            self._press_at = None
            self.cancelled.emit()
            return
        self._press_at = ev.position() if self.arming else None

    def mouseReleaseEvent(self, ev):
        """The corner lands HERE, and only if the mouse did not wander on the way.

        Placing it on the press would be a fraction more responsive and would give the
        operator no way out of a click they had already started. Every button on every
        screen lets you slide off before letting go; a corner of the region that decides
        which pills are counted deserves at least as much.
        """
        start, self._press_at = self._press_at, None
        if ev.button() != Qt.LeftButton or start is None or not self.arming:
            return
        here = ev.position()
        if abs(here.x() - start.x()) > TAP_SLOP or abs(here.y() - start.y()) > TAP_SLOP:
            return
        point = self._to_frame(here)
        if point is not None:
            self.tapped.emit(*point)


class Tick(QWidget):
    """A drawn check mark, because the bundled font has no U+2713.

    The toast asked for "\u2713" and Qt substituted from somewhere else in the system: what
    appeared was a square-root sign. A tick is three points and two lines -- cheaper to draw
    than to depend on, and it then matches at every size and in any font the app is ever
    given.
    """

    def __init__(self, size=66):
        super().__init__()
        self.setFixedSize(size, size)

    def paintEvent(self, ev):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        side = self.width()
        ring = QPen(QColor(255, 255, 255, 90))
        ring.setWidth(max(2, side // 22))
        painter.setPen(ring)
        painter.drawEllipse(self.rect().adjusted(2, 2, -2, -2))
        stroke = QPen(QColor("#ffffff"))
        stroke.setWidth(max(4, side // 11))
        stroke.setCapStyle(Qt.RoundCap)
        stroke.setJoinStyle(Qt.RoundJoin)
        painter.setPen(stroke)
        path = QPainterPath()
        path.moveTo(side * 0.28, side * 0.52)
        path.lineTo(side * 0.44, side * 0.68)
        path.lineTo(side * 0.73, side * 0.33)
        painter.drawPath(path)


class Toast(QWidget):
    """The "saved" confirmation: big, centred, and gone again on its own.

    WHY NOT A MESSAGE BOX. A record is saved every few minutes all shift. A modal dialog
    would stop the window each time and demand a click before the next tray could be
    counted -- a hundred clicks a day to be told something went right. This lands in the
    middle of the screen where it cannot be missed, says what was written, and leaves after
    a second and a half. A click dismisses it early; nothing waits for one.

    WHY NOT JUST THE FOOTER. The footer is where this app reports things and it stays -- but
    it is one line of small print at the bottom of a screen whose whole point is a number in
    the middle. Somebody looking at the tray, not the screen, never saw it.

    It is a CHILD of the window, not a window of its own: it moves and stacks with the
    screen it belongs to, and never appears on the taskbar or behind the counter.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("toast")
        # WITHOUT THIS THE PANEL IS INVISIBLE. A plain QWidget does not paint the background
        # a stylesheet gives it unless it is told to; the first build of this toast put white
        # text straight onto the camera picture, which is unreadable and looks like a bug.
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setCursor(Qt.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(48, 30, 48, 30)
        lay.setSpacing(10)
        mark = QHBoxLayout()
        mark.addStretch(1)
        mark.addWidget(Tick())
        mark.addStretch(1)
        lay.addLayout(mark)
        self.title = styled(QLabel(""), 32, QFont.Bold, "#ffffff")
        self.title.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.title)
        self.detail = styled(QLabel(""), 19, QFont.Normal, "#d9f2e2")
        self.detail.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.detail)

        self._fade_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._fade_effect)
        self._fade = QPropertyAnimation(self._fade_effect, b"opacity", self)
        self._fade.setDuration(280)
        self._fade.finished.connect(self._faded)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.dismiss)
        self.hide()

    def flash(self, title, detail="", seconds=1.8):
        self.title.setText(title)
        self.detail.setText(detail)
        self.detail.setVisible(bool(detail))
        self.adjustSize()
        self.recentre()
        self._fade.stop()
        self._fade_effect.setOpacity(1.0)
        self.show()
        self.raise_()                       # above both cards, whatever built them first
        self._timer.start(int(seconds * 1000))

    def recentre(self):
        parent = self.parentWidget()
        if parent is None:
            return
        self.move((parent.width() - self.width()) // 2,
                  (parent.height() - self.height()) // 2)

    def dismiss(self):
        if not self.isVisible():
            return
        self._timer.stop()
        self._fade.stop()
        self._fade.setStartValue(self._fade_effect.opacity())
        self._fade.setEndValue(0.0)
        self._fade.start()

    def _faded(self):
        if self._fade_effect.opacity() <= 0.01:
            self.hide()

    def mousePressEvent(self, ev):
        self.dismiss()


# ----------------------------------------------------------------------------- window
class Window(QWidget):
    def __init__(self, capture, infer, target=0, camera=0):
        super().__init__()
        self.capture, self.infer = capture, infer
        self.camera = camera
        self.target = int(target)
        self.arming = False
        #: COUNTS ALREADY TAKEN AND ALREADY TIPPED AWAY, in the order they were taken.
        #:
        #: The tray holds about sixty tablets before they start lying on top of one another,
        #: and a detector cannot count what it cannot see -- so a prescription for a hundred
        #: is physically two pours, and was unanswerable here until this list existed. Every
        #: figure in it came off a still tray, was read for half a second before it was
        #: taken, and was confirmed by a person. What is on screen is this list plus
        #: whatever is on the tray NOW; an empty list makes that the plain live count, which
        #: is exactly what the screen did before and what it still does for one pour.
        self.rounds = []
        #: (frame, boxes, confidences) for each of those, so a disputed total can be
        #: re-counted by eye pour by pour instead of being taken on trust.
        self.round_shots = []
        #: True between taking a round and seeing the tray empty. See CLEAR_S.
        self.clearing = False
        self._steady_n = None               # the count the stillness timer is timing
        self._steady_at = 0.0               # when it last changed
        self._can_round = False             # set every repaint, alongside the button
        self._round_block = ""              # and why not, for the footer to say
        self._save_kind = "primary"         # or "warn"; restyled only when it changes
        self._reset_armed_at = 0.0          # the two-press guard on starting over
        #: Show the picture left-to-right reversed. A webcam is built to be pointed at a
        #: face and hands over a mirror image because that is what a face expects; a tray
        #: does not, and the operator reaches left for a tablet the screen shows on the
        #: right. On until somebody turns it off, because that is what the hardware does.
        #:
        #: IT TOUCHES THE DRAWING AND NOTHING ELSE. See CameraView._to_frame.
        self.flip = load_flip(camera)
        self.pending = []                   # corners tapped so far, in frame pixels
        self.note = ""                      # footer message
        self._fit = None                    # the PANEL_FITS entry now applied
        self._bar_colour = None             # so the bar is restyled only on a change
        self._block = ""                    # why saving is refused; "" when it is allowed
        self._note_colour = None            # so the footer is restyled only on a change
        self._fitting = False

        self.setObjectName("root")
        self.setWindowTitle(T.TITLE)
        self.resize(T.WIN_W, T.WIN_H)
        # THE MINIMUM IS MEASURED, not chosen: 740 is the shortest window in which the
        # tightest PANEL_FITS entry still fits, and below it Qt squeezes every widget at
        # once -- clipping the count and cutting the tone marks off the Thai on the
        # buttons. Declaring it here lets Windows refuse the size instead of drawing it
        # wrong.
        self.setMinimumSize(1100, 740)
        self.setStyleSheet(self._sheet())

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._header())

        body = QWidget()
        body.setObjectName("body")
        row = QHBoxLayout(body)
        row.setContentsMargins(20, 20, 20, 20)
        row.setSpacing(20)
        # The number left, the tray right: the figure is what is being read, the picture
        # is only there to check it against.
        row.addWidget(self._panel(), 0)
        row.addWidget(self._camera(), 1)
        root.addWidget(body, 1)
        root.addWidget(self._footer())

        self.view.set_flip(self.flip)
        frame, _ = self.capture.latest()
        if frame is not None:
            pts, why = load_roi(camera, (frame.shape[1], frame.shape[0]))
            if pts:
                self.infer.roi = pts
            self.note = why
        self._disarm()

        self._meta()
        self.toast = Toast(self)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(DRAW_MS)

    # -------------------------------------------------------------------------- style
    def _sheet(self) -> str:
        return T.sheet() + f"""
            #root, #body {{ background: {T.BG}; }}
            #header {{ background: {T.SURFACE}; border-bottom: 1px solid {T.LINE}; }}
            #footer {{ background: {T.SURFACE}; border-top: 1px solid {T.LINE}; }}
            #card {{ background: {T.SURFACE}; border: 1px solid {T.LINE};
                     border-radius: {T.R_XL}px; }}
            #view {{ background: #0f1714; border-radius: {T.R_LG}px; }}
            #rule {{ background: {T.LINE}; max-height: 1px; border: 0; }}
            #toast {{ background: {T.GREEN_700}; border-radius: {T.R_XL}px;
                      border: 1px solid {T.GREEN_900}; }}

            QPushButton#primary {{
                font-family: {T.FONT_STACK}; background: {T.GREEN_700}; color: #fff;
                border: 0; border-radius: {T.R_MD}px; padding: 20px;
                font-size: 22px; font-weight: 700; }}
            QPushButton#primary:hover {{ background: {T.GREEN_500}; }}
            QPushButton#primary:pressed {{ background: {T.GREEN_900}; }}
            QPushButton#primary:disabled {{ background: {T.LINE};
                                            color: {T.INK_MUTED}; }}
            /* THE SAME BUTTON IN A DIFFERENT COAT, for a save that files a SHORT count.
               Not a second button somewhere else on the panel: there is one save here and
               adding a rival to it would make the operator choose between two things that
               both file a record, which is exactly the decision they should not be asked
               to get right at speed. It is the same control, wearing what it is about to
               do. Orange rather than red because a short dispense is a legitimate act --
               the stock ran out -- and red is reserved here for the machine having
               stopped. */
            QPushButton#warn {{
                font-family: {T.FONT_STACK}; background: {T.WARN}; color: #fff;
                border: 0; border-radius: {T.R_MD}px; padding: 20px;
                font-size: 22px; font-weight: 700; }}
            QPushButton#warn:hover {{ background: #f0954f; }}
            QPushButton#warn:pressed {{ background: #cf6f28; }}
            QPushButton#warn:disabled {{ background: {T.LINE};
                                         color: {T.INK_MUTED}; }}
            QPushButton#ghost {{
                font-family: {T.FONT_STACK}; background: {T.SURFACE}; color: {T.INK};
                border: 1px solid {T.LINE_STRONG}; border-radius: {T.R_MD}px;
                padding: 16px; font-size: 19px; font-weight: 600; }}
            QPushButton#ghost:hover {{ background: {T.GREEN_TINT};
                                       border-color: {T.GREEN_500}; }}
            QPushButton#ghost:disabled {{ color: {T.INK_MUTED};
                                          border-color: {T.LINE}; }}
            QPushButton#topbtn {{
                font-family: {T.FONT_STACK}; background: {T.SURFACE}; color: {T.INK};
                border: 1px solid {T.LINE_STRONG}; border-radius: {T.R_MD}px;
                padding: 9px 18px; font-size: 19px; font-weight: 600;
                min-height: 26px; }}
            QPushButton#topbtn:hover {{ background: {T.GREEN_TINT};
                                        border-color: {T.GREEN_500}; }}
            QPushButton#topbtn:pressed {{ background: {T.LINE}; }}
            QPushButton#topbtn:checked {{ background: {T.GREEN_700}; color: #fff;
                                          border-color: {T.GREEN_700}; }}
            /* 30px, not the 999px that was here. A radius Qt considers absurd is a
               radius Qt declines to draw: it fell back to a barely-rounded box, which is
               why the phone's controls were capsules and the bench's were rectangles
               side by side in the same screenshot. These end up 59x60 -- nearly square,
               because the bench panel is narrower than the phone's -- so the radius has
               to clear HALF THE WIDTH as well, not just half the height, or Qt declines
               it again and draws the box it drew before. */
            QPushButton#chip {{
                font-family: {T.FONT_STACK}; background: {T.BG};
                border: 1px solid {T.LINE}; border-radius: 24px;
                padding: 12px 6px; font-size: 19px; font-weight: 600;
                color: {T.INK_SOFT}; }}
            QPushButton#chip:hover {{ background: {T.GREEN_TINT};
                                      border-color: {T.GREEN_500};
                                      color: {T.GREEN_700}; }}
            QPushButton#step {{
                font-family: {T.FONT_STACK}; background: {T.BG};
                border: 1px solid {T.LINE_STRONG}; border-radius: {T.R_MD}px;
                color: {T.GREEN_700}; font-size: 26px; font-weight: 700;
                min-width: 56px; min-height: 56px; }}
            QPushButton#step:hover {{ background: {T.GREEN_TINT}; }}
            /* PLAIN WORDS. These two say what the camera is doing -- which part of the
               picture is counted, and how long a pass takes. They are not pressable and
               never were, and every shape a chip could wear (an outline, then a tint)
               said otherwise: on a screen whose header is two real buttons, anything
               with a box round it is read as a third. Type on the card, and the weight
               carries it. */
            QLabel#chip {{
                font-family: {T.FONT_STACK}; background: transparent; color: {T.INK_SOFT};
                border: 0; padding: 0; font-size: 18px; font-weight: 600; }}
            QProgressBar {{ background: {T.LINE}; border: 0; border-radius: 6px;
                            min-height: 12px; max-height: 12px; }}
            QProgressBar::chunk {{ border-radius: 6px; background: {T.GREEN_500}; }}
            QLineEdit {{ font-family: {T.FONT_STACK}; border: 1px solid {T.LINE_STRONG};
                         border-radius: {T.R_SM}px; padding: 12px;
                         font-size: 28px; font-weight: 700; }}
        """

    # ------------------------------------------------------------------------- header
    def _header(self):
        """Logo left on white, the time right. The green moved to where it means something.

        A coloured bar across the top is decoration; on this screen green means "the count
        matches what was asked for", and spending it on a title bar blunts the one place it
        has to be read instantly.
        """
        bar = QWidget()
        bar.setObjectName("header")
        bar.setFixedHeight(72)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(24, 0, 24, 0)
        if os.path.isfile(LOGO):
            logo = QLabel()
            logo.setPixmap(QPixmap(LOGO).scaledToHeight(40, Qt.SmoothTransformation))
            lay.addWidget(logo)
        lay.addStretch(1)

        # THE RECORD BUTTONS LIVE UP HERE, not in the panel beside the count. Two reasons.
        # They are not counting controls -- nobody presses them while a tray is on the
        # bench -- and the panel is full: adding a row to it pushed the save button off the
        # bottom of the card on a 1366x768 screen, which is the size of the bench this runs
        # on. The header is 72px of empty white and these are the only other things the
        # window does.
        self.list_btn = sized(QPushButton("ดูรายการที่บันทึก"), 19)
        self.list_btn.setObjectName("topbtn")
        self.list_btn.clicked.connect(self._show_records)
        lay.addWidget(self.list_btn)
        self.export_btn = sized(QPushButton("ส่งออก CSV"), 19)
        self.export_btn.setObjectName("topbtn")
        self.export_btn.clicked.connect(self._export)
        lay.addWidget(self.export_btn)
        lay.addSpacing(16)

        self.clock = styled(QLabel(""), 18, QFont.Normal, T.INK_MUTED)
        lay.addWidget(self.clock)
        return bar

    # -------------------------------------------------------------------------- panel
    def _panel(self):
        card = _card()
        card.setFixedWidth(430)
        self.panel = card
        lay = self.panel_lay = QVBoxLayout(card)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(14)

        # WHERE THE BIG NUMBER CAME FROM, and only when that is a question. On one pour it
        # says nothing and takes no room, because on one pour the figure below is simply
        # what the camera can see and a caption explaining that would be noise. From the
        # second pour on, the figure is PARTLY MEMORY -- 35 of it is in a bottle and cannot
        # be checked against the picture -- and a number on a screen that the picture does
        # not corroborate has to say so out loud, or the operator has no way to tell a
        # working total from a stuck one.
        self.rounds_lbl = styled(QLabel(""), 19, QFont.DemiBold, T.INK_SOFT)
        self.rounds_lbl.setAlignment(Qt.AlignCenter)
        self.rounds_lbl.setVisible(False)
        lay.addWidget(self.rounds_lbl)

        self.count_lbl = styled(QLabel("0"), PANEL_FITS[0][0], QFont.ExtraBold,
                                T.GREEN_700)
        self.count_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.count_lbl)

        # THE VERDICT IS A BADGE NOW: tinted pill, dark text, sized to its words. Plain
        # coloured text had to carry the whole judgement on hue and weight alone; a filled
        # shape is visible from further back, and the dark-text-on-tint pairing keeps it
        # legible under the fluorescent strip this bench sits beneath. The words stay --
        # the colour is still not the message.
        vrow = QHBoxLayout()
        vrow.addStretch(1)
        self.verdict = styled(QLabel("พร้อมนับ"), 26, QFont.Bold)
        vrow.addWidget(self.verdict)
        vrow.addStretch(1)
        lay.addLayout(vrow)
        self._set_badge(T.BADGE_IDLE)

        # HOW FAR ALONG, without having to do the subtraction. The number says 47 and the
        # target says 60; the bar says "nearly there" before either has been read, which is
        # what somebody glancing up from the tray actually wants. It hides itself when no
        # target is set, because a progress bar towards nothing is decoration.
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 100)
        lay.addWidget(self.progress)

        lay.addWidget(_rule())

        lay.addWidget(styled(QLabel("จำนวนที่ต้องการ"), 19, QFont.DemiBold, T.INK_SOFT))

        trow = QHBoxLayout()
        trow.setSpacing(10)
        minus = sized(QPushButton("−"), 26, QFont.Bold)
        minus.setObjectName("step")
        minus.clicked.connect(lambda: self._set_target(self.target - 1))
        trow.addWidget(minus)
        self.target_edit = sized(QLineEdit(str(self.target)), 28, QFont.Bold)
        self.target_edit.setAlignment(Qt.AlignCenter)
        self.target_edit.editingFinished.connect(self._target_typed)
        trow.addWidget(self.target_edit, 1)
        plus = sized(QPushButton("+"), 26, QFont.Bold)
        plus.setObjectName("step")
        plus.clicked.connect(lambda: self._set_target(self.target + 1))
        trow.addWidget(plus)
        lay.addLayout(trow)

        # In a widget of its own, not a bare layout: the fitter hides this row on a short
        # screen, and a layout cannot be hidden.
        self.chips = QWidget()
        chips = QHBoxLayout(self.chips)
        chips.setContentsMargins(0, 0, 0, 0)
        chips.setSpacing(6)
        for n in PRESETS:
            b = sized(QPushButton(str(n)), 19)
            b.setObjectName("chip")
            b.setMinimumWidth(52)
            b.clicked.connect(lambda _=False, v=n: self._set_target(v))
            chips.addWidget(b)
        lay.addWidget(self.chips)

        lay.addStretch(1)

        # THE ROUND CONTROLS SIT ABOVE THE REGION CONTROLS, and the order is the order of
        # the job: the region is set once when the camera is aimed and then never touched,
        # while these two are pressed once per pour, with a tray in the other hand. The
        # thing used every minute belongs nearer the thumb than the thing used every week.
        rrow = QHBoxLayout()
        rrow.setSpacing(10)

        self.round_btn = sized(QPushButton("เก็บรอบที่ 1"), 19)
        self.round_btn.setObjectName("ghost")
        self.round_btn.clicked.connect(self._take_round)
        rrow.addWidget(self.round_btn, 1)

        # THE WAY OUT, and it throws the whole total away rather than one pour.
        #
        # It used to step back a single pour, which sounds gentler and is false precision.
        # When something has gone wrong mid-prescription -- a tray tipped twice, a pour
        # banked off the wrong tray, a number nobody is sure of -- the operator does not
        # know WHICH pour is wrong, and a button that removes the last one invites them to
        # guess. There is no evidence left to check against either: those tablets are in a
        # bottle. Tipping the bottle back out and counting the prescription again is what
        # actually restores certainty, so that is what the button does and what it says.
        self.reset_btn = sized(QPushButton("นับใหม่"), 19)
        self.reset_btn.setObjectName("ghost")
        self.reset_btn.clicked.connect(self._reset_clicked)
        rrow.addWidget(self.reset_btn)
        lay.addLayout(rrow)

        # ONE ROW, as on the phone. The two are a pair -- set the frame, clear the frame
        # -- and a full-width button each read as two unrelated commands while spending
        # two rows of panel to say it. Side by side they are visibly one control with two
        # ends, and the row they give back goes to the space above the save button, which
        # is what keeps save from being just another button in a stack.
        frow = QHBoxLayout()
        frow.setSpacing(10)

        self.roi_btn = sized(QPushButton("กำหนดกรอบนับ"), 19)
        self.roi_btn.setObjectName("ghost")
        self.roi_btn.clicked.connect(self._roi_clicked)
        frow.addWidget(self.roi_btn)

        self.roi_clear = sized(QPushButton("ล้างกรอบ"), 19)
        self.roi_clear.setObjectName("ghost")
        self.roi_clear.clicked.connect(self._roi_cleared)
        frow.addWidget(self.roi_clear)
        lay.addLayout(frow)

        self.save_btn = sized(QPushButton("บันทึกผล"), 22, QFont.Bold)
        self.save_btn.setObjectName("primary")
        self.save_btn.clicked.connect(self._save)
        lay.addWidget(self.save_btn)
        return card

    # ------------------------------------------------------------------------- camera
    def _camera(self):
        card = _card()
        lay = QVBoxLayout(card)
        lay.setContentsMargins(20, 20, 20, 20)
        lay.setSpacing(12)
        # A header row rather than a bare caption: the two things worth knowing ABOUT the
        # picture -- which part of it is being counted, and how long the model is taking --
        # belong on the picture, not buried in the footer with the settings.
        head = QHBoxLayout()
        head.setSpacing(24)                 # the padding the chips used to carry
        head.addWidget(styled(QLabel("ภาพจากกล้อง"), 19, QFont.DemiBold, T.INK_SOFT))
        head.addStretch(1)
        # THE FLIP LIVES ON THE PICTURE'S CARD, not in the counting panel. It is a property
        # of the picture rather than of the count, it is set once when the camera is aimed
        # and then never again, and the panel has no row to spare -- the header of this card
        # is the only place on the screen with room that is also the right place.
        self.flip_btn = sized(QPushButton("พลิกภาพ"), 19)
        self.flip_btn.setObjectName("topbtn")
        # CHECKABLE, so the button says which way the picture is rather than only offering
        # to change it. Somebody who walks up to this bench cannot tell a mirrored tray
        # from an unmirrored one by looking at the tray -- both are a tray from above --
        # and a plain button would leave them pressing it twice to find out.
        self.flip_btn.setCheckable(True)
        self.flip_btn.setChecked(self.flip)
        self.flip_btn.clicked.connect(self._flip_clicked)
        head.addWidget(self.flip_btn)
        head.addSpacing(8)
        self.roi_chip = styled(QLabel(""), 18, QFont.DemiBold)
        head.addWidget(self.roi_chip)
        self.ms_chip = styled(QLabel(""), 18, QFont.DemiBold)
        head.addWidget(self.ms_chip)
        lay.addLayout(head)

        self.view = CameraView()
        self.view.tapped.connect(self._corner)
        self.view.cancelled.connect(self._corner_undo)
        lay.addWidget(self.view, 1)
        return card

    # ------------------------------------------------------------------------- footer
    def _footer(self):
        """Every piece of small print lives here, with room to be read.

        It used to sit under the buttons, in type small enough to break, competing with
        the controls beside it. One strip along the bottom keeps the panel to the job it
        does and gives the status somewhere to be legible.
        """
        bar = QWidget()
        bar.setObjectName("footer")
        bar.setFixedHeight(56)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(24, 0, 24, 0)
        lay.setSpacing(24)
        self.note_lbl = styled(QLabel(""), 18, QFont.Bold, T.GREEN_700)
        lay.addWidget(self.note_lbl)
        lay.addStretch(1)
        self.meta_lbl = styled(QLabel(""), 18, QFont.Normal, T.INK_MUTED)
        lay.addWidget(self.meta_lbl)
        return bar

    # ------------------------------------------------------------------------ actions
    def _set_badge(self, pair):
        self.verdict.setStyleSheet(self.verdict.base + badge_css(pair))

    @staticmethod
    def _chip(label, text):
        label.setText(text)
        label.setObjectName("chip")
        label.setStyleSheet("")             # let the sheet's #chip rule take over
        label.style().unpolish(label)
        label.style().polish(label)
        label.setVisible(bool(text))

    def _meta(self):
        """The footer's right-hand side: the settings, and how much has been saved today.

        Both are things somebody asks about once an hour and never while counting, which is
        exactly what a footer is for. The record count is read from disk, so it is worked
        out when a record is written and when the window opens -- never on the repaint.
        """
        try:
            saved = len(glob.glob(os.path.join(RECORDS, "count_*.json")))
        except OSError:
            saved = 0
        self.meta_lbl.setText(
            f"conf {self.infer.conf}   iou {self.infer.iou}   "
            f"imgsz {self.infer.imgsz}   บันทึกไว้ {saved} รายการ")

    # ------------------------------------------------------------------------- rounds
    def banked(self) -> int:
        """Tablets already counted and already tipped out of the tray."""
        return sum(self.rounds)

    def _take_round(self):
        """Freeze what is on the tray into the total, and refuse the tray until it is empty.

        The guard is not the disabled button repeated for neatness. A disabled button is a
        drawing; the rule that a pour is only counted when the figure has settled, the
        picture is alive and the tray has been seen empty since the last pour is what stops
        the same tablets being counted twice, and it has to live where the addition happens.
        """
        if not self._can_round:
            self.note = self._round_block or self.note
            return
        frame, (boxes, confs, count, _ms) = self.infer.snapshot()
        if count <= 0:
            return
        self.rounds.append(int(count))
        self.round_shots.append((frame, boxes, confs))
        self.clearing = True
        self._steady_n = None               # the stillness timer restarts on the new state
        sound.round_taken()
        self.note = CLEAR_NOTE
        self.toast.flash(f"เก็บรอบที่ {len(self.rounds)}  {count} เม็ด",
                         f"สะสมแล้ว {self.banked()} เม็ด   {CLEAR_NOTE}")

    def _reset_clicked(self):
        """Start the prescription again. TWICE, because the total cannot be got back.

        The same guard the mid-pour save carries, for the same reason and in the lighter
        form: what is being discarded is a count of tablets that are already in a bottle,
        so it cannot be recovered by looking at anything. Arming the button and saying on
        it what the next press costs turns an accident into two accidents in a row.
        """
        if not (self.rounds or self.clearing):
            return
        if time.time() - self._reset_armed_at < CONFIRM_S:
            banked = self.banked()
            self._clear_rounds()
            self.note = f"เริ่มนับใหม่  ทิ้งยอดสะสม {banked} เม็ดแล้ว"
            return
        self._reset_armed_at = time.time()
        self.note = (f"จะทิ้งยอดสะสม {self.banked()} เม็ด  "
                     "กดอีกครั้งเพื่อเริ่มนับใหม่")

    def _clear_rounds(self):
        """Back to a single-pour screen. After a save, or when the operator starts over."""
        self.rounds = []
        self.round_shots = []
        self.clearing = False
        self._steady_n = None
        self._reset_armed_at = 0.0

    def _set_target(self, value):
        self.target = max(0, int(value))
        self.target_edit.setText(str(self.target))

    def _target_typed(self):
        text = self.target_edit.text().strip()
        self._set_target(int(text) if text.isdigit() else 0)

    def _flip_clicked(self):
        """Turn the picture round. NOTHING ELSE MOVES, and that is the whole design.

        An earlier version of this flipped the camera frame itself and then had to mirror
        the counting region to match, because the region is stored in frame pixels and
        would otherwise have jumped to the other side of the bench. It worked, and it was
        wrong twice over: the detector was handed a different image -- measured on this
        bench, mirroring moved the count by as much as three on one tray -- and the record
        it saved was of a picture that only existed because of a display preference.

        Now the frame is never touched. The window mirrors the finished picture on its way
        to the screen and CameraView undoes it on the way back in, so the region, the
        model, the marks and the saved JPEG all go on living in the camera's own
        coordinates and none of them can tell this setting apart from the other one.
        """
        self.flip = not self.flip
        save_flip(self.camera, self.flip)
        self.view.set_flip(self.flip)
        self.flip_btn.setChecked(self.flip)
        self._disarm()
        self.note = "พลิกภาพซ้าย-ขวาแล้ว" if self.flip else "เลิกพลิกภาพแล้ว"

    def _roi_clicked(self):
        """Start placing corners, or stop if they are already being placed."""
        if self.arming:
            self._disarm()
            return
        self.arming = True
        self.pending = []
        self.view.set_arming(True)
        self.roi_btn.setText("ยกเลิก")
        self._roi_prompt()

    def _roi_cleared(self):
        """The second button in the row: take a corner back, or clear the whole region.

        ONE BUTTON, TWO JOBS, and they are the same job at two moments. While corners are
        being placed the only thing anybody wants from it is the last one back; once the
        region is set the only thing anybody wants is it gone. A separate undo would sit
        dead and unexplained for all the hours nobody is drawing a region.
        """
        if self.arming:
            self._corner_undo()
            return
        self.infer.roi = None
        self._persist()
        self._disarm()

    def _corner(self, x, y):
        """One corner placed. The fourth closes the shape and sets the region."""
        if not self.arming:
            return
        self.pending.append((x, y))
        if len(self.pending) < ROI_POINTS:
            self._roi_prompt()
            return
        frame, _ = self.capture.latest()
        size = (frame.shape[1], frame.shape[0]) if frame is not None else (0, 0)
        pts = quad(self.pending, size)
        if pts is None:
            # The four corners enclose nothing worth counting -- tapped on one spot, or
            # strung out in a line. Only the last one is dropped: three good corners and a
            # slip is the likely case, and throwing all four away would make the operator
            # pay for the slip four times.
            self.pending.pop()
            self.note = f"มุมนี้แคบเกินไป  แตะให้ห่างจากมุมอื่นกว่า {MIN_ROI} จุดภาพ"
            return
        self.infer.roi = pts
        self._persist()
        self._disarm()
        self.note = "กำหนดกรอบแล้ว"

    def _corner_undo(self):
        """Right click, or the second button: the last corner back, then the whole mode."""
        if not self.arming:
            return
        if self.pending:
            self.pending.pop()
            self._roi_prompt()
            return
        self._disarm()

    def _roi_prompt(self):
        """What the footer says while corners are going down, and what the row offers."""
        left = ROI_POINTS - len(self.pending)
        self.note = (f"แตะมุมถาดทีละมุม  อีก {left} จุด"
                     "     คลิกขวาเพื่อถอยจุดล่าสุด")
        self.roi_clear.setText("ถอยจุด")
        self.roi_clear.setEnabled(bool(self.pending))

    def _disarm(self):
        self.arming = False
        self.pending = []
        self.view.set_arming(False)
        self.roi_btn.setText("กำหนดกรอบใหม่" if self.infer.roi
                             else "กำหนดกรอบนับ")
        self.roi_clear.setText("ล้างกรอบ")
        self.roi_clear.setEnabled(bool(self.infer.roi))
        self.note = ""

    def _persist(self):
        frame, _ = self.capture.latest()
        size = (frame.shape[1], frame.shape[0]) if frame is not None else (0, 0)
        save_roi(self.camera, size, self.infer.roi)
        self.note = ""

    def _save(self):
        """The number, and enough beside it to argue with months later.

        A bare count in a log is unfalsifiable, so the frame, the boxes and the settings go
        with it: a disputed record can be re-counted by eye.

        The guard repeats what the disabled button already prevents. A disabled button is a
        presentation detail -- a keyboard, a touch driver that sends a release without a
        press, or a later shortcut could all reach this method anyway -- and the rule that
        an over-count is not filed belongs with the writing, not with the widget.
        """
        if self._block:
            self.note = self._block
            return
        frame, (boxes, confs, count, ms) = self.infer.snapshot()
        # THE SAME ARITHMETIC THE SCREEN DID, not the live count on its own. A save that
        # filed the tray while the screen showed the total would put 25 in the record for a
        # prescription of 60 -- a record that is not merely wrong but wrong in the direction
        # that reads as a short dispense, and unfalsifiable afterwards because the other 35
        # are in a bottle. `clearing` is honoured here for the same reason it is honoured
        # on screen: mid-sweep those tablets are already in `rounds`.
        live = 0 if self.clearing else int(count)
        rounds = list(self.rounds) + ([live] if live or not self.rounds else [])
        total = self.banked() + live
        # EVERY NUMBER IS ALREADY FIXED BEFORE THE QUESTION IS ASKED, and that is why the
        # question is asked here rather than before the snapshot. A modal runs a nested
        # event loop, so the repaint timer keeps firing behind it: the tray can empty, the
        # count can move, `clearing` can end. Re-reading any of it afterwards would file a
        # number the operator was never shown, which is the one outcome a confirmation is
        # supposed to make impossible.
        if not self._confirm_partial(total, rounds):
            self.note = "ยกเลิกการบันทึก"
            return
        os.makedirs(RECORDS, exist_ok=True)
        stamp = self._free_stamp()
        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            # `count` STAYS THE TOTAL. Every reader of these files -- the list, the detail
            # pane, the CSV, and whatever a pharmacy opens them with next year -- asks this
            # field how many tablets were dispensed, and the answer must not depend on
            # knowing that a newer field exists. `rounds` adds detail; it never corrects.
            "count": int(total),
            "target": int(self.target),
            "difference": int(total) - int(self.target) if self.target else None,
            "rounds": [int(n) for n in rounds],
            # Deliberately short, and recorded as such. `difference` already carries the
            # arithmetic, but a reader with a spreadsheet has to know to do the comparison
            # before a short dispense becomes visible to them; a flag is visible without
            # being looked for. It is false when no target was set, because with nothing to
            # fall short OF the question does not arise.
            "short": bool(self.target and total < self.target),
            "conf": self.infer.conf, "iou": self.infer.iou, "imgsz": self.infer.imgsz,
            "roi": [[int(x), int(y)] for x, y in self.infer.roi] if self.infer.roi else None,
            "model_ms": round(float(ms), 1),
            "boxes": [[round(float(v), 1) for v in b] for b in boxes],
            "confidences": [round(float(c), 3) for c in confs],
        }
        # ONE FRAME PER POUR. `count_<stamp>.jpg` stays the last tray, because that is the
        # one the `boxes` above belong to and a record whose picture and boxes disagree is
        # worse than a record with no picture. The earlier pours go beside it numbered, so a
        # total of 60 that nobody can re-count on a tray can still be re-counted on two.
        shots = []
        for i, (shot_frame, _b, _c) in enumerate(self.round_shots, start=1):
            if shot_frame is None:
                continue
            name = f"count_{stamp}_r{i}.jpg"
            if cv2.imwrite(os.path.join(RECORDS, name), shot_frame):
                shots.append(name)
        if shots:
            record["round_images"] = shots
        with open(os.path.join(RECORDS, f"count_{stamp}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
        if frame is not None:
            cv2.imwrite(os.path.join(RECORDS, f"count_{stamp}.jpg"), frame)
        detail = f"  ({' + '.join(str(n) for n in rounds)})" if len(rounds) > 1 else ""
        self.note = f"บันทึกแล้ว {total} เม็ด{detail}"
        self._meta()
        target = f"  จากที่ต้องการ {self.target}" if self.target else ""
        sound.saved()
        self.toast.flash(f"บันทึกแล้ว  {total} เม็ด{detail}",
                         f"{time.strftime('%H:%M:%S')}{target}   ไฟล์ count_{stamp}")
        # BACK TO A BLANK SCREEN, and this is the last line for a reason: the rounds are
        # only safe to forget once they are on disk. A crash or a full disk above this point
        # leaves the total on screen, where the operator can press save again.
        self._clear_rounds()

    def _confirm_partial(self, total, rounds) -> bool:
        """Ask before filing a total that has pours banked in it and has not reached target.

        ONLY THAT CASE. A short count on ONE tray is the stock running out and is already
        told apart by the button's colour and its words -- adding a dialog to it would put
        a modal in front of an errand the operator runs all day, and a modal that appears
        every day is a modal nobody reads by the end of the week.

        What is different mid-pour is what a stray press DESTROYS. The tablets in `rounds`
        are in the bottle: they cannot be re-counted, they are not on the tray, and the only
        record of them is the list this method is standing in front of. Saving clears it. So
        the operator who meant to press it after the next pour and pressed it now does not
        merely file a wrong number -- they lose the count and have to tip the bottle out and
        start the prescription again. That is worth one question.

        The safe answer is the default, so the dialog can be dismissed with Escape or Enter
        by somebody who did not mean to open it and nothing will have happened.
        """
        if not (self.rounds and self.target and total < self.target):
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("ยังเทค้างอยู่")
        box.setText("ยังเทค้างอยู่")
        box.setInformativeText(
            f"เก็บแล้ว {len(self.rounds)} รอบ = {' + '.join(str(n) for n in rounds)} "
            f"= {total} เม็ด\n"
            f"จากที่ต้องการ {self.target} เม็ด  ขาดอีก {self.target - total} เม็ด\n\n"
            f"ถ้าบันทึกตอนนี้ ยอดสะสมจะถูกล้าง และนับต่อจากเดิมไม่ได้")
        # 19px, the same floor the rest of this screen holds to, and set on the box rather
        # than inherited: a QMessageBox picks up the window's sheet, whose QWidget rule is
        # 16px, and Thai tone marks do not survive 16px at this DPI.
        box.setStyleSheet(f"QLabel {{ font-family: {T.FONT_STACK}; font-size: 19px; }}"
                          f"QPushButton {{ font-family: {T.FONT_STACK}; font-size: 19px; "
                          f"font-weight: 600; padding: 10px 18px; }}")
        back = box.addButton("กลับไปเทต่อ", QMessageBox.RejectRole)
        go = box.addButton(f"บันทึก {total} เม็ด", QMessageBox.AcceptRole)
        box.setDefaultButton(back)
        box.setEscapeButton(back)
        box.exec()
        return box.clickedButton() is go

    def _free_stamp(self) -> str:
        """A file name no record already has.

        The stamp counts in whole seconds, and two saves inside one second gave two records
        the same name: the second quietly replaced the first, and the count that was filed
        first was simply gone. It takes a double-click on the save button. The suffix keeps
        the first fifteen characters exactly as they were, which is what the list sorts on
        and what the range filter parses.
        """
        stamp = time.strftime("%Y%m%d-%H%M%S")
        candidate, n = stamp, 1
        while os.path.exists(os.path.join(RECORDS, f"count_{candidate}.json")):
            n += 1
            candidate = f"{stamp}-{n}"
        return candidate

    def _show_records(self):
        """The saved records, in a window of their own.

        IMPORTED HERE, NOT AT THE TOP. The counting screen must open as fast as the splash
        can hand it over; a table nobody has asked for yet can wait until they ask. The
        camera keeps running behind the dialog -- Qt still services the repaint timer --
        so nothing is lost by standing in the list for a minute.
        """
        from .records_view import RecordsDialog

        dialog = RecordsDialog(self)
        dialog.exec()
        if dialog.note:
            self.note = dialog.note
        # The footer counts what is on disk, and that window can now delete from it.
        self._meta()

    def _export(self):
        """Straight to a CSV of everything saved, without going through the list first.

        The common errand is "give me today's counts", and making that a two-step trip
        through a table the operator does not need to read would be ceremony.
        """
        from .records_view import export_with_dialog

        note = export_with_dialog(self)
        if note:
            self.note = note

    def _draw_roi(self, shown):
        """The region, drawn two different ways on purpose.

        WHILE THE CORNERS ARE GOING DOWN each one is a bright ring with its number beside
        it, joined by the edges so far, and the shape closes itself the moment the fourth
        lands. The numbers are there because they answer the question the operator has --
        how many more -- on the picture they are looking at rather than in the footer they
        are not.

        RINGS, NOT FILLED DOTS. A corner of a full tray has a pill under it, and a solid
        mark would hide the very thing the corner is being placed around.

        ONCE IT IS SET the treatment goes quiet: a thin outline, a wash of green so faint it
        would not survive being described as a colour, and corner ticks. The region is then
        background information, and a bright box over a tray that is being worked in would
        compete with the markers, which are the thing that has to be visible.
        """
        if self.arming:
            pts = [(int(x), int(y)) for x, y in self.pending]
            if len(pts) > 1:
                cv2.polylines(shown, [np.array(pts, np.int32).reshape(-1, 1, 2)],
                              False, ROI_BAND, 2, cv2.LINE_AA)
            for i, (x, y) in enumerate(pts, start=1):
                cv2.circle(shown, (x, y), 9, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.circle(shown, (x, y), 9, ROI_BAND, 2, cv2.LINE_AA)
                cv2.putText(shown, str(i), (x + 14, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(shown, str(i), (x + 14, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, ROI_BAND, 2, cv2.LINE_AA)
            return

        pts = self.infer.roi or []
        if len(pts) < 3:
            return
        poly = np.array(pts, np.int32).reshape(-1, 1, 2)
        wash = shown.copy()
        cv2.fillPoly(wash, [poly], ROI_LINE)
        cv2.addWeighted(wash, 0.10, shown, 0.90, 0, shown)
        cv2.polylines(shown, [poly], True, ROI_LINE, 2, cv2.LINE_AA)
        self._corner_ticks(shown, pts, ROI_LINE, 22, 3)

    @staticmethod
    def _corner_ticks(shown, pts, colour, length, width):
        """Short elbows at each corner instead of dots.

        A filled dot sits ON the tray and hides whatever is under it -- at the corners of a
        full tray that is a pill. An elbow marks the same corner from outside the content.
        """
        h, w = shown.shape[:2]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        for x, y in pts:
            sx = length if x < cx else -length
            sy = length if y < cy else -length
            x, y = int(min(max(x, 0), w - 1)), int(min(max(y, 0), h - 1))
            cv2.line(shown, (x, y), (int(x + sx), y), colour, width, cv2.LINE_AA)
            cv2.line(shown, (x, y), (x, int(y + sy)), colour, width, cv2.LINE_AA)

    # ------------------------------------------------------------------------ repaint
    def _tick(self):
        frame, _ = self.capture.latest()
        if frame is None:
            return
        (boxes, confs, count, ms), _ = self.infer.result()
        # Asked once, used twice: the picture wears the warning and the save button obeys
        # the same number, so they can never disagree about whether the camera is alive.
        stale = self.capture.stale()
        shown = frame.copy()

        for b in boxes:
            if not self.infer.inside(b):
                continue
            cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
            cv2.circle(shown, (cx, cy), 7, MARK_RIM, -1, cv2.LINE_AA)
            cv2.circle(shown, (cx, cy), 5, MARK, -1, cv2.LINE_AA)

        self._draw_roi(shown)
        if stale > STALE_S:
            # A red edge on the picture, not words on it: cv2 cannot draw Thai, and the
            # footer and the chip are already saying it in a language people read.
            cv2.rectangle(shown, (0, 0), (shown.shape[1] - 1, shown.shape[0] - 1),
                          STALE_EDGE, 10)

        # THE MIRROR, IN ONE LINE AND AT THE VERY END. Everything above drew in the
        # camera's coordinates -- the marks, the region, the corners going down, the red
        # border on a dead picture -- so one flip of the finished picture carries all of
        # them together and none of them had to know. The other half is
        # CameraView._to_frame, which undoes it for every tap.
        if self.flip:
            shown = cv2.flip(shown, 1)
        self.view.show_frame(shown)

        # WHAT THE BIG NUMBER IS, when the prescription took more than one pour.
        #
        #     total = what is already in the bottle + what is on the tray right now
        #
        # and the second term is dropped while `clearing`, which is the whole of the
        # safety. Between taking a round and the tray being seen empty, those tablets are
        # counted in BOTH terms -- the list remembers them and the camera can still see
        # them -- so adding the two would report a tray of 35 as 70. Dropping the live term
        # until the tray has been observed empty means the machine has watched the tablets
        # leave before it agrees to count anything else, and a pour cannot be counted twice
        # without physically pouring it twice.
        now = time.perf_counter()
        if count != self._steady_n:
            self._steady_n = count
            self._steady_at = now
        steady = now - self._steady_at
        if self.clearing and count == 0 and steady >= CLEAR_S:
            self.clearing = False
            if self.note == CLEAR_NOTE:
                self.note = "ถาดว่างแล้ว  เทรอบต่อไปได้"
        live = 0 if self.clearing else count
        total = self.banked() + live
        self.count_lbl.setText(str(total))

        if self.clearing:
            strip = (f"เก็บแล้ว {len(self.rounds)} รอบ  รวม {self.banked()} เม็ด"
                     f"   ·   รอกวาดถาด")
        elif self.rounds:
            strip = (f"เก็บแล้ว {len(self.rounds)} รอบ  รวม {self.banked()}"
                     f"   +   ในถาด {live}")
        else:
            strip = ""
        if strip != self.rounds_lbl.text():
            self.rounds_lbl.setText(strip)
            self.rounds_lbl.setVisible(bool(strip))
        round_text = f"เก็บรอบที่ {len(self.rounds) + 1}"
        if round_text != self.round_btn.text():
            self.round_btn.setText(round_text)

        # The verdict is words as well as colour. A dispensary is not the place to make
        # somebody read a hue: colour-blindness aside, a glance across a room resolves a
        # word faster than a shade of orange.
        if self.target:
            diff = total - self.target
            if diff == 0:
                text, colour, pair = "ครบตามจำนวน", T.GREEN_700, T.BADGE_OK
            elif diff > 0:
                text, colour, pair = f"เกิน {diff} เม็ด", T.WARN, T.BADGE_WARN
            else:
                text, colour, pair = f"ขาด {-diff} เม็ด", T.DANGER, T.BADGE_BAD
        else:
            text, colour, pair = "ยังไม่กำหนดจำนวน", T.INK_MUTED, T.BADGE_IDLE
        if text != self.verdict.text():
            # Only on a change: restyling a widget makes Qt reparse the rule, and this runs
            # sixty times a second.
            self.verdict.setText(text)
            self._set_badge(pair)

        # WHAT WOULD MAKE THIS SAVE A LIE, in the order that matters.
        #
        # A dead picture first: if frames stopped arriving, the number beside it is the
        # count of whatever was under the camera when they stopped, and saving files it
        # under now. That is the one failure here that produces a WRONG RECORD rather than
        # no record, so it outranks everything else on screen.
        #
        # Then the model: a detector that is throwing is not counting, and the last figure
        # it managed is just as stale.
        #
        # Then over the target, which is not a fault at all -- it is a tray with too much
        # in it, and the answer is to take some out rather than to write it down. Under the
        # target stays saveable: a part-filled tray is a normal thing to record.
        if stale > STALE_S:
            block = f"ภาพจากกล้องหยุด {stale:.0f} วินาที  ตรวจสายกล้องหรือโปรแกรมที่ใช้กล้องอยู่"
        elif getattr(self.infer, "error", ""):
            block = f"โมเดลผิดพลาด  {self.infer.error}"
        elif self.target and total > self.target:
            block = OVER_NOTE
        else:
            block = ""

        # WHY A ROUND MAY NOT BE TAKEN, in the same order and for the same reason: the
        # button is drawn dead AND _take_round refuses, because a disabled button is a
        # picture of a rule and this one is load-bearing. The two transient reasons are the
        # ones worth naming -- an empty tray and a figure still moving both look like the
        # app ignoring a press, and an operator who thinks a control is broken presses it
        # harder rather than waiting the half second it is asking for.
        if block:
            round_block = block
        elif self.clearing:
            round_block = CLEAR_NOTE
        elif count <= 0:
            round_block = "ถาดว่าง  ยังไม่มีอะไรให้เก็บ"
        elif steady < SETTLE_S:
            round_block = "ตัวเลขยังไม่นิ่ง  รอสักครู่"
        else:
            round_block = ""
        self._round_block = round_block
        self._can_round = not round_block
        self.round_btn.setEnabled(self._can_round)
        armed_reset = time.time() - self._reset_armed_at < CONFIRM_S
        reset_text = "กดอีกครั้ง" if armed_reset else "นับใหม่"
        if reset_text != self.reset_btn.text():
            self.reset_btn.setText(reset_text)
        # Live while there is a total to discard, INCLUDING mid-sweep: the tray that has
        # just been banked is exactly the moment somebody notices it was the wrong tray.
        self.reset_btn.setEnabled(bool(self.rounds or self.clearing))

        # THE SAVE BUTTON SAYS WHAT IT IS ABOUT TO FILE, and a short count is filed under
        # protest. Saving under the target stays possible on purpose -- the stock runs out,
        # and a screen that refuses to record 47 of 60 does not create the missing thirteen,
        # it just sends the number onto a scrap of paper where nothing can audit it. What
        # was wrong was that it looked identical to filing a complete one: same green, same
        # word, and the operator learns the gesture rather than the state. So the button
        # keeps the job and loses the disguise.
        if self.target and total < self.target:
            save_text, save_kind = f"บันทึกว่าไม่ครบ (ขาด {self.target - total})", "warn"
        else:
            save_text, save_kind = "บันทึกผล", "primary"
        if save_text != self.save_btn.text():
            self.save_btn.setText(save_text)
        if save_kind != self._save_kind:
            self._save_kind = save_kind
            self.save_btn.setObjectName(save_kind)
            # setObjectName alone changes nothing that is already drawn: Qt matched the
            # #primary rule when the widget was polished and does not go looking again.
            self.save_btn.style().unpolish(self.save_btn)
            self.save_btn.style().polish(self.save_btn)

        if block != self._block:
            was_broken = self._block not in ("", OVER_NOTE)
            self._block = block
            self.save_btn.setEnabled(not block)
            if block:
                self.note = block
                if block != OVER_NOTE:
                    sound.problem()         # once, on the way in, not every frame
            elif self.note in (OVER_NOTE, "") or was_broken:
                self.note = ""
        recolour(self.count_lbl, colour if self.target else T.GREEN_700)

        # The bar is coloured with the verdict, not with the fill: at 61 of 60 a full green
        # bar would say "done" while the words beside it say there is one too many.
        if self.target:
            self.progress.setVisible(True)
            self.progress.setValue(min(100, int(total * 100 / self.target)))
        else:
            self.progress.setVisible(False)
        if colour != self._bar_colour:
            self._bar_colour = colour
            self.progress.setStyleSheet(
                f"QProgressBar::chunk {{ border-radius: 6px; background: {colour}; }}")

        # Chips are restyled only when their words change. setStyleSheet reparses the rule
        # every time it is called, and this runs sixty times a second.
        roi_text = ("ภาพค้าง" if stale > STALE_S
                    else "เฉพาะในกรอบ" if self.infer.roi else "นับทั้งภาพ")
        if roi_text != self.roi_chip.text():
            self._chip(self.roi_chip, roi_text)
        ms_text = f"{ms:.0f} ms" if ms else ""
        if ms_text != self.ms_chip.text():
            self._chip(self.ms_chip, ms_text)

        self.clock.setText(time.strftime("%H:%M"))
        self.note_lbl.setText(self.note)
        # The footer is green because most of what it says is "saved". A refusal in the
        # colour of success is read as the opposite of itself, so the three cases are told
        # apart: green for what went right, orange for a tray with too much in it (which
        # the operator fixes by hand), red for the machine having stopped counting.
        if not self._block or self.note != self._block:
            colour = T.GREEN_700
        elif self._block == OVER_NOTE:
            colour = T.WARN
        else:
            colour = T.DANGER
        if colour != self._note_colour:
            self._note_colour = colour
            recolour(self.note_lbl, colour)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._fit_panel()
        if getattr(self, "toast", None) is not None:
            self.toast.recentre()

    def _fit_panel(self):
        """Pick the roomiest PANEL_FITS entry the card can actually hold.

        Measured against the layout's own sizeHint rather than guessed from the window
        height: the hint already knows what the fonts, the padding and the borders cost,
        and it is the number Qt would otherwise ignore on its way to squeezing everything.
        """
        # Worked out from the window, not read off the card: a resizeEvent arrives BEFORE
        # the children are laid out, so the card still reports its previous height here and
        # the fitter would choose against a stale number.
        available = self.height() - CHROME_H
        if available <= 0 or self._fitting:
            return                          # setFont below re-enters resizeEvent
        self._fitting = True
        try:
            for fit in PANEL_FITS:
                self._apply_fit(fit)
                if (self.panel_lay.sizeHint().height() <= available
                        or fit is PANEL_FITS[-1]):
                    return
        finally:
            self._fitting = False

    def _apply_fit(self, fit):
        if fit == self._fit:
            return
        self._fit = fit
        count_px, spacing, margin, chips = fit
        styled(self.count_lbl, count_px, QFont.ExtraBold, T.GREEN_700)
        # _tick recolours it from the rule styled() just rebuilt, within 16ms.
        self.panel_lay.setSpacing(spacing)
        self.panel_lay.setContentsMargins(margin, margin, margin, margin)
        self.chips.setVisible(chips)
        # invalidate() before activate(), or sizeHint() answers from the cache it filled
        # for the PREVIOUS fit: a font change only posts a layout request, it does not drop
        # the cached hint, so the fitter would compare the new layout against the old
        # number and walk all the way down to the tightest entry on a 1080p screen.
        self.panel_lay.invalidate()
        self.panel_lay.activate()

    def closeEvent(self, ev):
        self.timer.stop()
        self.infer.stop()
        self.capture.stop()
        super().closeEvent(ev)
