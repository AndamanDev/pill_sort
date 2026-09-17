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
                               QLabel, QLineEdit, QProgressBar, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

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
MIN_ROI = 40                # frame pixels; under this, the drag was a click that slipped

#: Said in the footer while the save button is dead, so the grey button is never a mystery.
OVER_NOTE = "เกินจำนวนที่ต้องการ  นำออกก่อนจึงบันทึกได้"

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
PANEL_FITS = ((104, 14, 24, True), (92, 14, 24, True), (80, 12, 20, True),
              (68, 10, 16, True), (68, 8, 14, False), (56, 8, 12, False))

#: Header + footer + the body's top and bottom margins: everything between the window and
#: the height the two cards get to share.
CHROME_H = 72 + 56 + 40


# --------------------------------------------------------------------------- settings
def roi_path(camera):
    return os.path.join(SETTINGS, f"roi-cam{camera}.json")


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
    """The live picture, and the rubber band the tray region is dragged out with.

    DRAGGED, NOT CLICKED CORNER BY CORNER. It used to take four clicks: four chances to put
    a corner in the wrong place, and a mode the operator had to be told they were in. A drag
    is the gesture everyone already owns from every photo tool they have used -- press at
    one corner of the tray, pull to the other, let go. What it makes is a rectangle, and a
    tray IS a rectangle from above; the odd quadrilateral the four clicks allowed was never
    worth the cost of getting there.

    Every coordinate leaving this class is in FRAME pixels, the only space the model and the
    saved region agree on. The widget's own pixels stop meaning anything the moment somebody
    resizes the window.
    """

    started = Signal(float, float)          # press, in frame coordinates
    dragged = Signal(float, float)          # moved while held
    finished = Signal()                     # released
    cancelled = Signal()                    # right click

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignCenter)
        self.setObjectName("view")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._geom = None                   # (x0, y0, scale) of the last drawn pixmap
        self._pix = None                    # the frame as painted, letterboxed
        self.arming = False                 # set by the window; drives the cursor
        self._down = False

    def set_arming(self, on):
        self.arming = on
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

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
        near, but not where, the operator dragged it.
        """
        if self._geom is None:
            return None
        x0, y0, scale = self._geom
        if scale <= 0:
            return None
        return (pos.x() - x0) / scale, (pos.y() - y0) / scale

    def mousePressEvent(self, ev):
        if ev.button() == Qt.RightButton:
            self._down = False
            self.cancelled.emit()
            return
        point = self._to_frame(ev.position())
        if not self.arming or point is None:
            return
        self._down = True
        self.started.emit(*point)

    def mouseMoveEvent(self, ev):
        if not self._down:
            return
        point = self._to_frame(ev.position())
        if point is not None:
            self.dragged.emit(*point)

    def mouseReleaseEvent(self, ev):
        if ev.button() != Qt.LeftButton or not self._down:
            return
        point = self._to_frame(ev.position())
        if point is not None:
            self.dragged.emit(*point)
        self._down = False
        self.finished.emit()


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
        self.drag = None                    # (x0, y0, x1, y1) in frame pixels, mid-drag
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
        self.roi_chip = styled(QLabel(""), 18, QFont.DemiBold)
        head.addWidget(self.roi_chip)
        self.ms_chip = styled(QLabel(""), 18, QFont.DemiBold)
        head.addWidget(self.ms_chip)
        lay.addLayout(head)

        self.view = CameraView()
        self.view.started.connect(self._drag_start)
        self.view.dragged.connect(self._drag_to)
        self.view.finished.connect(self._drag_done)
        self.view.cancelled.connect(self._drag_cancel)
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

    def _set_target(self, value):
        self.target = max(0, int(value))
        self.target_edit.setText(str(self.target))

    def _target_typed(self):
        text = self.target_edit.text().strip()
        self._set_target(int(text) if text.isdigit() else 0)

    def _roi_clicked(self):
        """Arm the drag, or cancel it if it is armed already."""
        if self.arming:
            self._disarm()
            return
        self.arming = True
        self.drag = None
        self.view.set_arming(True)
        self.roi_btn.setText("ยกเลิก")
        self.roi_clear.setEnabled(bool(self.infer.roi))
        self.note = "ลากเมาส์คลุมพื้นที่ถาด     คลิกขวาเพื่อยกเลิก"

    def _roi_cleared(self):
        self.infer.roi = None
        self._persist()
        self._disarm()

    def _drag_start(self, x, y):
        self.drag = (x, y, x, y)

    def _drag_to(self, x, y):
        if self.drag:
            self.drag = (self.drag[0], self.drag[1], x, y)

    def _drag_done(self):
        """Keep the region if the drag drew one, and say so plainly if it did not.

        A press that barely moves is a CLICK, not a region -- an operator steadying the
        mouse, or one who has not noticed the screen is waiting for a drag. Taken as a 20px
        box it would drop the count to zero and look like the model had failed, so the rule
        is explicit and the window asks again instead.
        """
        rect, self.drag = self.drag, None
        if not rect:
            return
        pts = self._rect_points(rect)
        if pts is None:
            self.note = f"กรอบเล็กเกินไป  ลากให้กว้างกว่า {MIN_ROI} จุดภาพ"
            return
        self.infer.roi = pts
        self._persist()
        self._disarm()
        self.note = "กำหนดกรอบแล้ว"

    def _drag_cancel(self):
        if self.arming:
            self.drag = None
            self._disarm()

    def _rect_points(self, rect):
        """The drag as four clockwise corners, clamped to the frame, or None if too small.

        Clamped because dragging from the middle of the tray out past the edge of the
        picture is the ordinary way to say "all of it", and a polygon with coordinates
        outside the frame would quietly drop the pills nearest that edge.
        """
        frame, _ = self.capture.latest()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        x0, x1 = sorted((rect[0], rect[2]))
        y0, y1 = sorted((rect[1], rect[3]))
        x0, x1 = max(0, int(x0)), min(w - 1, int(x1))
        y0, y1 = max(0, int(y0)), min(h - 1, int(y1))
        if x1 - x0 < MIN_ROI or y1 - y0 < MIN_ROI:
            return None
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def _disarm(self):
        self.arming = False
        self.drag = None
        self.view.set_arming(False)
        self.roi_btn.setText("กำหนดกรอบใหม่" if self.infer.roi
                             else "กำหนดกรอบนับ")
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
        os.makedirs(RECORDS, exist_ok=True)
        stamp = self._free_stamp()
        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "count": int(count),
            "target": int(self.target),
            "difference": int(count) - int(self.target) if self.target else None,
            "conf": self.infer.conf, "iou": self.infer.iou, "imgsz": self.infer.imgsz,
            "roi": [[int(x), int(y)] for x, y in self.infer.roi] if self.infer.roi else None,
            "model_ms": round(float(ms), 1),
            "boxes": [[round(float(v), 1) for v in b] for b in boxes],
            "confidences": [round(float(c), 3) for c in confs],
        }
        with open(os.path.join(RECORDS, f"count_{stamp}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
        if frame is not None:
            cv2.imwrite(os.path.join(RECORDS, f"count_{stamp}.jpg"), frame)
        self.note = f"บันทึกแล้ว {count} เม็ด"
        self._meta()
        target = f"  จากที่ต้องการ {self.target}" if self.target else ""
        sound.saved()
        self.toast.flash(f"บันทึกแล้ว  {count} เม็ด",
                         f"{time.strftime('%H:%M:%S')}{target}   ไฟล์ count_{stamp}")

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

        WHILE DRAGGING the picture OUTSIDE the band is darkened and the band itself is left
        at full brightness. Dimming the rest is what makes a selection read as a selection
        -- the eye is pulled to the bright part, and the operator can see at a glance which
        pills will be counted, which is the whole question they are answering.

        ONCE IT IS SET the treatment goes quiet: a thin outline, a wash of green so faint it
        would not survive being described as a colour, and corner ticks. The region is then
        background information, and a bright box over a tray that is being worked in would
        compete with the markers, which are the thing that has to be visible.
        """
        if self.drag:
            pts = self._band_points(self.drag)
            if pts is None:
                return
            (x0, y0), (x1, y1) = pts[0], pts[2]
            dim = shown.copy()
            cv2.rectangle(dim, (0, 0), (shown.shape[1], shown.shape[0]), (0, 0, 0), -1)
            cv2.rectangle(dim, (x0, y0), (x1, y1), (0, 0, 0), -1)   # hole, filled back in
            inside = shown[y0:y1, x0:x1].copy()
            cv2.addWeighted(dim, 0.45, shown, 0.55, 0, shown)
            if inside.size:
                shown[y0:y1, x0:x1] = inside
            cv2.rectangle(shown, (x0, y0), (x1, y1), ROI_BAND, 2, cv2.LINE_AA)
            self._corner_ticks(shown, pts, ROI_BAND, 18, 3)
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

    def _band_points(self, rect):
        """The live drag as four corners -- the same maths as _rect_points, minus the veto.

        No minimum here: the band has to follow the mouse from the first pixel, or the
        gesture feels broken for the first half-inch. The size rule belongs at the end of
        the drag, where it decides whether to KEEP the region.
        """
        frame, _ = self.capture.latest()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        x0, x1 = sorted((rect[0], rect[2]))
        y0, y1 = sorted((rect[1], rect[3]))
        x0, x1 = max(0, int(x0)), min(w - 1, int(x1))
        y0, y1 = max(0, int(y0)), min(h - 1, int(y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

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

        self.view.show_frame(shown)
        self.count_lbl.setText(str(count))

        # The verdict is words as well as colour. A dispensary is not the place to make
        # somebody read a hue: colour-blindness aside, a glance across a room resolves a
        # word faster than a shade of orange.
        if self.target:
            diff = count - self.target
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
        elif self.target and count > self.target:
            block = OVER_NOTE
        else:
            block = ""
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
            self.progress.setValue(min(100, int(count * 100 / self.target)))
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
