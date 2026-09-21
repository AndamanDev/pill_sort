"""The saved records: a window that lists them, and the CSV an operator sends onwards.

WHY A SEPARATE FILE. window.py is the counting screen and is already the longest thing in
app/; reading a folder of JSON, sorting it and rendering a table has nothing to do with
counting pills. Keeping it apart means the screen that runs all day imports this only when
somebody presses the button.

WHAT A RECORD IS. Whatever _save() in window.py wrote: count_<stamp>.json, and beside it
count_<stamp>.jpg -- the frame the count came from. The pair is the point. A row in a list
is a claim; the picture is how the claim gets checked, so every row carries a thumbnail of
its own frame and the selected one is shown large, rather than making anybody go digging
through a folder of timestamps.

THE THUMBNAILS ARE LOADED A FEW AT A TIME, ON A TIMER. Decoding one 640x480 JPEG costs a
handful of milliseconds; decoding four hundred of them the moment the window opens is two
seconds of frozen UI, and the folder only grows. The loader hands back four per tick and
caches what it has read, so the window is up immediately and fills in as fast as the disk
allows -- and a scroll to the bottom of a long list is never waiting on the top of it.

THE CSV IS WRITTEN utf-8-sig, AND THAT MATTERS HERE. Excel on a Thai Windows opens a plain
UTF-8 file in the system codepage and turns every Thai heading into mojibake; the BOM is
what tells it otherwise. The boxes and confidences are deliberately left out of it -- they
are hundreds of numbers per row, they are what the JSON is for, and a spreadsheet full of
them is not a report anybody reads.
"""
from __future__ import annotations

import csv
import ctypes
import datetime as dt
import glob
import json
import os
import time
from ctypes import wintypes

from PySide6.QtCore import (QDate, QEvent, QLocale, QPoint, QPointF, QRectF,
                            QSize, Qt, QTimer)
from PySide6.QtGui import (QColor, QFont, QMouseEvent, QPainter, QPainterPath,
                           QPen, QPixmap)
from PySide6.QtWidgets import (QAbstractItemView, QCalendarWidget, QDateEdit,
                               QDialog, QFileDialog, QFrame, QHBoxLayout,
                               QHeaderView, QLabel, QMessageBox, QPushButton,
                               QStyle, QStyledItemDelegate, QStyleOptionComboBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from . import RECORDS
from . import theme as T

#: The list. The thumbnail leads, because it is what the eye lands on, and the verdict
#: closes, because it is what the row is being read FOR.
COLUMNS = ("", "เวลา", "นับได้", "ต้องการ", "ผล")

#: The CSV. Wider than the list on purpose: this one leaves the building, so the settings
#: that produced the count travel with it, and the file names let any row be traced back to
#: the JSON and the frame it came from.
#: `rounds` sits next to `count` rather than at the end because that is where it is read:
#: a 60 that was poured 35 and 25 and a 60 that came off one tray are the same dispense and
#: a different piece of evidence, and the column that says which must be beside the number
#: it qualifies, not out past the model settings where nobody scrolls.
CSV_FIELDS = ("time", "count", "rounds", "target", "difference", "short", "detections",
              "conf", "iou", "imgsz", "model_ms", "roi", "json", "image")

THUMB = QSize(104, 62)      # the row's picture; the row is sized from it
ROW_H = 78
THUMBS_PER_TICK = 4         # see the module docstring: enough to fill fast, few enough
                            # that the window never stops answering the mouse

#: The verdict, as a value the delegate can paint and the sort can order.
OK, OVER, SHORT, NONE = "ok", "over", "short", "none"
BADGES = {OK: T.BADGE_OK, OVER: T.BADGE_WARN, SHORT: T.BADGE_BAD, NONE: T.BADGE_IDLE}


# ------------------------------------------------------------------------------ reading
def load_records(folder=RECORDS):
    """Every saved record, newest first, with the ones that cannot be read left out.

    A half-written JSON -- the machine was switched off mid-save -- must not stop the
    other two hundred from being listed, so a broken file is skipped rather than raised.
    """
    rows = []
    for path in glob.glob(os.path.join(folder, "count_*.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
        except Exception:                                           # noqa: BLE001
            continue
        image = os.path.splitext(path)[0] + ".jpg"
        rec["json"] = path
        rec["image"] = image if os.path.isfile(image) else ""
        # The per-pour frames, resolved here so that everything which acts on a record --
        # and deleting is the one that matters -- sees them as part of it. Left unresolved
        # they would survive the record they belong to and sit in the folder for ever,
        # unreachable from the list because nothing but a count_*.json puts a row in it.
        rec["round_paths"] = [
            q for q in (os.path.join(folder, n) for n in (rec.get("round_images") or []))
            if os.path.isfile(q)]
        rec["stamp"] = os.path.basename(path)[6:-5]     # count_<stamp>.json
        rows.append(rec)
    rows.sort(key=lambda r: r.get("stamp", ""), reverse=True)
    return rows


def when(rec):
    """The moment a record was taken, from its file name, or None if it cannot be read.

    THE FILE NAME, NOT THE "time" FIELD. The stamp is what the file is called and what the
    list is sorted by; the field beside it is a human string that a future change of format
    would break silently. Parsing the name keeps one source of truth for the ordering, the
    range filter and the day filter alike.
    """
    # THE FIRST FIFTEEN CHARACTERS ONLY. A second save inside the same second is filed as
    # count_20260917-103640-2, and parsing the whole name would throw on the suffix -- which
    # would quietly drop that record out of every range filter while leaving it in the list.
    stamp = rec.get("stamp", "")[:15]
    try:
        return dt.datetime.strptime(stamp, "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def verdict(rec):
    """(kind, words) for a record, in the same language the counting screen uses.

    A bare signed number in a column is read wrong at a glance as often as not; the screen
    beside it says ครบ / เกิน / ขาด, and a record that disagrees with the screen about what
    it means is worse than no record.
    """
    target = int(rec.get("target") or 0)
    if not target:
        return NONE, "ไม่ได้ตั้ง"
    diff = int(rec.get("count") or 0) - target
    if diff == 0:
        return OK, "ครบ"
    return (OVER, f"เกิน {diff}") if diff > 0 else (SHORT, f"ขาด {-diff}")


def difference_text(rec) -> str:
    return verdict(rec)[1]


def rounds_text(rec) -> str:
    """"35 + 25" for a dispense poured twice, and "" for one poured once.

    Records written before multi-round counting existed have no `rounds` key at all, and
    records written after it exists but used once carry a single-element list. Both mean
    the same thing -- one tray, nothing to explain -- so both come back empty here, and no
    reader of these files has to know which era a record is from.
    """
    rounds = rec.get("rounds") or []
    return " + ".join(str(int(n)) for n in rounds) if len(rounds) > 1 else ""


#: Thai month abbreviations, indexed by month number.
MONTHS = ("", "ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
          "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค.")


def short_when(rec) -> str:
    """"17 ก.ย. 09:30" -- what fits, and what a person says out loud.

    The column was showing 2026-09-17 09:30:20 and the list is not wide enough for it, so
    Qt elided it to "2026-09-17 ..." and threw away the only part anybody was reading. The
    full timestamp is still one click away in the pane, and the CSV carries it untouched.
    """
    stamp = rec.get("stamp", "")
    if len(stamp) < 15:
        return rec.get("time", stamp)
    year, month, day = int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8])
    when = f"{day} {MONTHS[month]} {stamp[9:11]}:{stamp[11:13]}"
    return when if year == int(time.strftime("%Y")) else f"{when} {year}"


# ------------------------------------------------------------------------------ writing
def export_csv(rows, path) -> int:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in rows:
            roi = rec.get("roi")
            writer.writerow({
                "time": rec.get("time", ""),
                "count": rec.get("count", ""),
                # Blank for a single pour, not "60": a cell that repeats the count adds
                # nothing and makes a column of them look like something worth reading.
                "rounds": rounds_text(rec),
                "target": rec.get("target", ""),
                # An empty cell, not the word "None": a spreadsheet reads a blank as "no
                # target was set" and reads None as text that breaks the column's sums.
                "difference": "" if rec.get("difference") is None else rec["difference"],
                # A column a spreadsheet can FILTER on. "show me every short dispense this
                # month" is the question these files exist to answer, and answering it off
                # `difference` means sorting a signed column and reading where it turns
                # negative -- which is a thing to get right rather than a thing to click.
                "short": "TRUE" if rec.get("short") else "",
                "detections": len(rec.get("boxes") or []),
                "conf": rec.get("conf", ""), "iou": rec.get("iou", ""),
                "imgsz": rec.get("imgsz", ""), "model_ms": rec.get("model_ms", ""),
                "roi": " ".join(f"{int(x)},{int(y)}" for x, y in roi) if roi else "",
                "json": os.path.basename(rec.get("json", "")),
                "image": os.path.basename(rec.get("image", "")),
            })
    return len(rows)


def export_with_dialog(parent, rows=None) -> str:
    """Ask where, write it, and hand back the line the footer should show.

    This says nothing itself; every outcome comes back as text so the caller puts it in the
    one place this app reports things -- and a cancelled save stays silent.
    """
    rows = load_records() if rows is None else rows
    if not rows:
        return "ยังไม่มีรายการให้ส่งออก"
    default = os.path.join(os.path.expanduser("~"), "Desktop",
                           f"pillsort_{time.strftime('%Y%m%d-%H%M%S')}.csv")
    if not os.path.isdir(os.path.dirname(default)):
        default = os.path.join(RECORDS, os.path.basename(default))
    path, _ = QFileDialog.getSaveFileName(parent, "ส่งออกรายการเป็น CSV", default,
                                          "ไฟล์ CSV (*.csv)")
    if not path:
        return ""
    if not path.lower().endswith(".csv"):
        path += ".csv"
    try:
        n = export_csv(rows, path)
    except OSError as exc:
        # The usual one: the file is already open in Excel, so Windows refuses to replace
        # it. The reason is the only useful thing left to give, so it is shown, not eaten.
        QMessageBox.warning(parent, "ส่งออกไม่สำเร็จ", f"{exc}")
        return "ส่งออกไม่สำเร็จ"
    return f"ส่งออกแล้ว {n} รายการ  →  {os.path.basename(path)}"


class _SHFILEOPSTRUCTW(ctypes.Structure):
    _fields_ = [("hwnd", wintypes.HWND),
                ("wFunc", wintypes.UINT),
                ("pFrom", wintypes.LPCWSTR),
                ("pTo", wintypes.LPCWSTR),
                ("fFlags", ctypes.c_uint16),
                ("fAnyOperationsAborted", wintypes.BOOL),
                ("hNameMappings", ctypes.c_void_p),
                ("lpszProgressTitle", wintypes.LPCWSTR)]


#: SHFileOperation, deleting, undoably, without asking or reporting anything of its own --
#: this window has already asked, in Thai, and says the result in its own footer.
_FO_DELETE = 3
_UNDO_QUIETLY = 0x0040 | 0x0010 | 0x0004 | 0x0400


def recycle(paths) -> bool:
    """Move files to the Recycle Bin. True if Windows took them all.

    WHY NOT os.remove. It does not use the bin: the file is gone the instant the button is
    pressed, with no way back. This window's wipe destroys a dispensary's record of what it
    counted, and on this bench it already did, permanently, while the feature was being
    tried out -- an undo that Windows offers for free should never have been left on the
    floor. The bin also means the confirmation can tell the truth about what happens next,
    which makes the whole dialog less frightening and more honest at the same time.

    Falls back to a plain delete only if the shell call cannot be made at all, which on
    Windows means something is very wrong; the caller then reports whatever is left.
    """
    paths = [p for p in paths if p and os.path.isfile(p)]
    if not paths:
        return True
    try:
        # Double-null-terminated, which is what the shell expects for a file list.
        op = _SHFILEOPSTRUCTW(None, _FO_DELETE, "\0".join(paths) + "\0\0", None,
                              _UNDO_QUIETLY, False, None, None)
        result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
        return result == 0 and not op.fAnyOperationsAborted
    except Exception:                                               # noqa: BLE001
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                return False
        return True


def delete_records(rows):
    """Recycle each record's frame and its JSON. Returns (deleted, [failures]).

    ONE CALL PER RECORD, AND THE PAIR GOES TOGETHER. The JSON is what puts a record in this
    list; the JPEG is what the row shows. If they were sent separately and the picture
    refused to move -- open in a viewer, read-only, a share that blinked -- the JSON could
    still go, leaving an orphan that nothing can see any more, so the button meant to clear
    the folder could never reach it. Taking them in one operation keeps that from splitting,
    and anything left behind afterwards is reported and stays in the list, where pressing
    the button again will pick it up.

    A missing half is not a failure: a record whose picture was tidied away by hand still
    has to be removable, or the button would leave exactly the rows nobody can clear.
    """
    deleted, failed = 0, []
    for rec in rows:
        paths = [p for p in (rec.get("image"), rec.get("json")) if p and os.path.isfile(p)]
        paths.extend(rec.get("round_paths") or [])
        recycle(paths)
        left = [p for p in paths if os.path.isfile(p)]
        if left:
            failed.extend(os.path.basename(p) for p in left)
            continue
        deleted += 1
    return deleted, failed


def open_folder(folder=RECORDS) -> str:
    os.makedirs(folder, exist_ok=True)
    try:
        os.startfile(folder)                                        # noqa: S606
    except Exception as exc:                                        # noqa: BLE001
        return f"เปิดโฟลเดอร์ไม่ได้ {exc}"
    return ""


# ------------------------------------------------------------------------------- pieces
def _sized(button, size, weight=QFont.DemiBold):
    """A button's font on the button -- see window.sized() for what goes wrong without it."""
    font = QFont()
    font.setFamilies(T.FONT_FAMILIES)
    font.setPixelSize(size)
    font.setWeight(weight)
    button.setFont(font)
    return button


def _styled(label, size, weight=QFont.Normal, colour=None):
    """Set the font both ways, for the reason window.styled() gives at length.

    Duplicated rather than imported: this module must not pull the counting screen -- and
    the camera, the model and torch behind it -- along with it. Six lines is cheaper than
    that import.

    THE WEIGHT FLOOR IS A FALLBACK, AND IT IS MEASURED. This window is the only one in the
    app with ordinary body text -- a list and a caption block, not headings -- so it was the
    one that found what theme.py now documents: a Qt STYLESHEET setting Leelawadee UI at
    weight 400 renders Thai about two thirds the height of the Latin beside it, tone marks
    smeared. With the bundled font registered that is gone and 400 reads properly; without
    it, every label here is pushed to 600, the weight that survives.
    """
    if not T.BUNDLED:
        weight = max(weight, QFont.DemiBold, key=int)
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


def _recolour(label, colour):
    """Change the colour without throwing the size away with it."""
    label.setStyleSheet(getattr(label, "base", "") + f" color: {colour};")


class Cell(QTableWidgetItem):
    """A cell that shows what it was given and sorts by a number underneath.

    TWO TRAPS IN ONE CLASS, BOTH FOUND ON THIS BENCH.

    Handing Qt an int for the display role -- the usual trick for making a column sort
    numerically -- lets QLocale format it, and the locale here is th_TH: a count of 13
    appeared in the list as ๑๓ and 60 as ๖๐, while the JSON, the CSV and the number on the
    counting screen all said 13 and 60. Thai numerals are not wrong in themselves; a record
    that does not read the same as the thing it is a record OF is.

    So the text is formatted in Python and the number kept beside it for sorting, which is
    also what stops the count column ordering 10 before 9.
    """

    def __init__(self, text="", sort_key=None, centre=False):
        super().__init__(str(text))
        self._key = text if sort_key is None else sort_key
        if centre:
            self.setTextAlignment(Qt.AlignCenter)

    def __lt__(self, other):
        mine, theirs = self._key, getattr(other, "_key", None)
        try:
            return mine < theirs
        except TypeError:                   # a number against a string: fall back to text
            return str(mine) < str(theirs)


class BadgeDelegate(QStyledItemDelegate):
    """Paints the verdict column as a filled pill instead of bare words.

    A DELEGATE, NOT A WIDGET PER ROW. setCellWidget on a QLabel would work for five records
    and crawl at five hundred -- every row would be a live widget with its own stylesheet.
    A delegate paints; nothing is allocated per row.
    """

    def paint(self, painter, option, index):
        kind = index.data(Qt.UserRole + 1) or NONE
        text = index.data(Qt.DisplayRole) or ""
        bg, fg = BADGES.get(kind, T.BADGE_IDLE)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        metrics = option.fontMetrics
        width = min(option.rect.width() - 12, metrics.horizontalAdvance(text) + 34)
        height = min(option.rect.height() - 18, metrics.height() + 14)
        rect = QRectF(0, 0, width, height)
        rect.moveCenter(QRectF(option.rect).center())
        # The outline is not decoration. The idle tint (#eef2ef) and the row selection
        # tint (#e7f4ec) are within a hair of each other, so on the selected row the pill
        # vanished and that row alone looked unstyled.
        outline = QColor(fg)
        outline.setAlpha(60)
        painter.setPen(QPen(outline, 1))
        painter.setBrush(QColor(bg))
        painter.drawRoundedRect(rect, height / 2, height / 2)
        painter.setPen(QPen(QColor(fg)))
        painter.setFont(option.font)
        painter.drawText(rect, Qt.AlignCenter, text)
        painter.restore()


class Shot(QLabel):
    """The selected frame, painted into rounded corners like the live view next door.

    A pixmap set with setPixmap is a SQUARE bitmap laid over a rounded card, and the four
    hard corners are visible against every other card in the app. Same fix as CameraView:
    clip to the radius and draw the picture ourselves.

    The full-size frame is kept and scaled per paint. Re-reading the JPEG on every resize
    would hit the disk for each pixel the window is dragged by.
    """

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 220)
        self.setAlignment(Qt.AlignCenter)
        self._pix = QPixmap()
        self._empty = "เลือกรายการเพื่อดูภาพ"

    def show_image(self, path):
        self._pix = QPixmap(path) if path else QPixmap()
        self._empty = "ไม่พบไฟล์ภาพ" if path else "เลือกรายการเพื่อดูภาพ"
        self.update()

    def paintEvent(self, ev):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), T.R_LG, T.R_LG)
        painter.setClipPath(path)
        painter.fillRect(self.rect(), QColor("#0f1714"))
        if not self._pix.isNull():
            scaled = self._pix.scaled(self.size(), Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)
            painter.drawPixmap((self.width() - scaled.width()) // 2,
                               (self.height() - scaled.height()) // 2, scaled)
            return
        painter.setPen(QColor(T.INK_MUTED))
        painter.setFont(self.font())
        painter.drawText(self.rect(), Qt.AlignCenter, self._empty)


def _tile(caption, value, colour):
    """One summary figure with its caption. Three of these are the top of the window."""
    card = QWidget()
    card.setObjectName("tile")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(20, 14, 20, 14)
    lay.setSpacing(2)
    number = _styled(QLabel(value), 34, QFont.Bold, colour)
    lay.addWidget(number)
    lay.addWidget(_styled(QLabel(caption), 18, QFont.Normal, T.INK_MUTED))
    card.value_label = number
    return card


# ------------------------------------------------------------------------------- window
class WhenEdit(QDateEdit):
    """A date box that is CHOSEN, never typed: any click opens the calendar.

    TWO THINGS QT DOES NOT DO ON ITS OWN, both found by clicking this window rather than
    reading it.

    FIRST, the calendar opens only when the click lands on the little arrow at the
    right-hand edge -- twenty pixels of a two-hundred-pixel box. Everywhere else the click
    parks a text cursor in a section, which from arm's length looks exactly like a control
    that does not work. Every left click is retargeted onto the arrow instead.

    SECOND, the keyboard still edits the sections, so a stray keypress could set a filter to
    a date nobody chose, and a caret blinking in a box that is meant to be pressed invites
    exactly that. The keys are dropped, and so is the mouse wheel: leaning on a trackpad
    over this box must not silently move the range while somebody reads the list under it.

    IT IS A DATE, NOT A DATE AND A TIME. A calendar can only offer days, and with typing
    gone there is no way left to say 09:25 -- so the box says what it can deliver, and the
    range is read as whole days (see _apply). Half a control that pretends to take a time
    and cannot is worse than one that is honest about being a day.
    """

    def mousePressEvent(self, ev):
        if (ev.button() != Qt.LeftButton or not self.calendarPopup()
                or self.isReadOnly() or not self.isEnabled()):
            super().mousePressEvent(ev)
            return
        centre = self._arrow_centre()
        if centre is None:
            super().mousePressEvent(ev)
            return
        super().mousePressEvent(QMouseEvent(
            ev.type(), QPointF(centre), ev.globalPosition(), ev.button(), ev.buttons(),
            ev.modifiers()))

    def keyPressEvent(self, ev):
        """Nothing types into this box. Tab still leaves it, Escape still closes things."""
        if ev.key() in (Qt.Key_Tab, Qt.Key_Backtab, Qt.Key_Escape):
            super().keyPressEvent(ev)
            return
        ev.ignore()

    def wheelEvent(self, ev):
        ev.ignore()

    def _arrow_centre(self):
        """Where Qt thinks the arrow is, asked of the style rather than guessed.

        A calendar-popup QDateTimeEdit is drawn as a combo box, so the arrow is that
        style's SC_ComboBoxArrow. If a style ever answers with nothing, the fall-back is
        the inside of the right-hand edge, which is where every style has put it so far.
        """
        option = QStyleOptionComboBox()
        option.initFrom(self)
        option.editable = True
        option.subControls = QStyle.SubControl.SC_All
        rect = self.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, option,
                                           QStyle.SubControl.SC_ComboBoxArrow, self)
        if rect.isValid() and not rect.isEmpty():
            return rect.center()
        if self.width() < 24:
            return None
        return self.rect().adjusted(0, 0, -12, 0).topRight() + QPoint(0, self.height() // 2)


class RecordsDialog(QDialog):
    """Summary, filters and the list on the left; the frame and its numbers on the right.

    WHY A SUMMARY AT ALL. A list answers "what happened at 09:25". The three figures above
    it answer "how did today go", which is the question somebody opening this window at the
    end of a shift actually has, and which a list of two hundred timestamps answers only
    after arithmetic nobody does.

    NOTHING IS SMALLER THAN 18px, the same rule the counting screen holds to: below it this
    font stops placing Thai tone marks correctly at this DPI, and a window full of dates and
    captions is exactly what tempts small type.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("รายการที่บันทึก")
        self.resize(1240, 820)
        self.setMinimumSize(1000, 660)
        # OPENS FILLING THE SCREEN. The counting screen is maximised all day; a window that
        # comes up over it at two thirds the size reads as a box that landed on top rather
        # than as the app's other page -- and the list is a list, so every row this buys is
        # a row nobody has to scroll for. Maximised, not fullscreen: the title bar stays,
        # and with it the way back that people reach for before finding the ปิด button.
        self.setWindowState(self.windowState() | Qt.WindowMaximized)
        self.setStyleSheet(self._sheet())

        self.rows = load_records()          # everything on disk
        self.shown = list(self.rows)        # what the filter left
        self.filter = 0
        self.note = ""                      # what the caller's footer should say after
        self._thumbs = {}                   # path -> QPixmap, loaded once
        self._queue = []                    # paths still to read

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 22, 24, 22)
        lay.setSpacing(16)
        lay.addLayout(self._head())
        lay.addLayout(self._tiles())
        lay.addLayout(self._filters())
        lay.addLayout(self._range())

        split = QHBoxLayout()
        split.setSpacing(16)
        split.addWidget(self._list(), 3)
        split.addWidget(self._preview(), 2)
        lay.addLayout(split, 1)
        lay.addLayout(self._buttons())

        # The thumbnails fill in behind the window rather than in front of it.
        self._loader = QTimer(self)
        self._loader.timeout.connect(self._load_thumbs)
        self._loader.start(16)

        self._apply()

    # --------------------------------------------------------------------------- style
    def _sheet(self) -> str:
        return T.sheet() + f"""
            QDialog {{ background: {T.BG}; }}
            #card, #tile {{ background: {T.SURFACE}; border: 1px solid {T.LINE};
                            border-radius: {T.R_XL}px; }}

            QTableWidget {{ background: {T.SURFACE}; border: 1px solid {T.LINE};
                            border-radius: {T.R_XL}px; font-size: 19px;
                            outline: none; gridline-color: transparent;
                            selection-background-color: {T.GREEN_TINT};
                            selection-color: {T.INK}; }}
            QTableWidget::item {{ padding: 8px 10px;
                                  border-bottom: 1px solid {T.LINE}; }}
            QTableWidget::item:selected {{ background: {T.GREEN_TINT}; color: {T.INK}; }}
            QHeaderView::section {{ background: {T.SURFACE}; color: {T.INK_MUTED};
                                    border: 0; border-bottom: 1px solid {T.LINE_STRONG};
                                    padding: 14px 10px; font-size: 18px;
                                    font-weight: 600; }}
            /* The sort arrow, parked inside the section it belongs to. Left to itself it
               floats above the header row and reads as a stray mark on the table. */
            QHeaderView::up-arrow, QHeaderView::down-arrow {{
                subcontrol-origin: padding; subcontrol-position: center right;
                width: 10px; height: 10px; margin-right: 6px; }}
            QScrollBar:vertical {{ background: transparent; width: 12px; margin: 8px 2px; }}
            QScrollBar::handle:vertical {{ background: {T.LINE_STRONG};
                                           border-radius: 5px; min-height: 40px; }}
            QScrollBar::handle:vertical:hover {{ background: {T.INK_MUTED}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: transparent; }}

            QPushButton#tab {{
                font-family: {T.FONT_STACK}; background: transparent; color: {T.INK_SOFT};
                border: 1px solid {T.LINE}; border-radius: 999px;
                padding: 10px 22px; font-size: 19px; font-weight: 600; }}
            QPushButton#tab:hover {{ border-color: {T.GREEN_500}; color: {T.GREEN_700}; }}
            QPushButton#tab:checked {{ background: {T.GREEN_700}; color: #fff;
                                       border-color: {T.GREEN_700}; }}

            QPushButton#primary {{
                font-family: {T.FONT_STACK}; background: {T.GREEN_700}; color: #fff;
                border: 0; border-radius: {T.R_MD}px; padding: 14px 30px;
                font-size: 20px; font-weight: 700; }}
            QPushButton#primary:hover {{ background: {T.GREEN_500}; }}
            QPushButton#primary:pressed {{ background: {T.GREEN_900}; }}
            QPushButton#ghost {{
                font-family: {T.FONT_STACK}; background: {T.SURFACE}; color: {T.INK};
                border: 1px solid {T.LINE_STRONG}; border-radius: {T.R_MD}px;
                padding: 14px 24px; font-size: 19px; font-weight: 600; }}
            QPushButton#ghost:hover {{ background: {T.GREEN_TINT};
                                       border-color: {T.GREEN_500}; }}
            QPushButton#ghost:disabled {{ color: {T.INK_MUTED};
                                          border-color: {T.LINE}; }}
            QPushButton#danger {{
                font-family: {T.FONT_STACK}; background: {T.SURFACE}; color: {T.DANGER};
                border: 1px solid {T.DANGER}; border-radius: {T.R_MD}px;
                padding: 14px 24px; font-size: 19px; font-weight: 600; }}
            QPushButton#danger:hover {{ background: {T.DANGER}; color: #fff; }}
            QPushButton#danger:disabled {{ color: {T.INK_MUTED};
                                           border-color: {T.LINE}; }}

            QDateTimeEdit {{
                font-family: {T.FONT_STACK}; background: {T.SURFACE}; color: {T.INK};
                border: 1px solid {T.LINE_STRONG}; border-radius: {T.R_MD}px;
                padding: 8px 12px; font-size: 19px; min-width: 210px; }}
            QDateTimeEdit:focus {{ border-color: {T.GREEN_500}; }}
            QDateTimeEdit:disabled {{ color: {T.INK_MUTED}; background: {T.BG};
                                      border-color: {T.LINE}; }}
            QDateTimeEdit::drop-down {{ width: 26px; border: 0; }}
            /* A Qt calendar that is styled by halves comes out BLACK: the parts nobody
               names keep a palette the rest of the sheet has already moved away from. So
               every piece of it is named here -- the bar, its buttons, the grid, and the
               days belonging to the months either side. */
            QCalendarWidget {{ background: {T.SURFACE}; }}
            QCalendarWidget QWidget {{ font-size: 18px;
                                       alternate-background-color: {T.SURFACE}; }}
            QCalendarWidget QWidget#qt_calendar_navigationbar {{
                background: {T.GREEN_700}; border-top-left-radius: {T.R_MD}px;
                border-top-right-radius: {T.R_MD}px; }}
            QCalendarWidget QToolButton {{
                color: #fff; background: transparent; border: 0;
                padding: 8px 14px; font-size: 19px; font-weight: 600; }}
            QCalendarWidget QToolButton:hover {{ background: {T.GREEN_500};
                                                 border-radius: {T.R_SM}px; }}
            QCalendarWidget QToolButton::menu-indicator {{ image: none; }}
            QCalendarWidget QSpinBox {{ font-size: 18px; }}
            QCalendarWidget QAbstractItemView:enabled {{
                background: {T.SURFACE}; color: {T.INK}; font-size: 18px;
                outline: none; gridline-color: transparent;
                selection-background-color: {T.GREEN_700}; selection-color: #fff; }}
            QCalendarWidget QAbstractItemView:disabled {{ color: {T.INK_MUTED}; }}
            #rule {{ background: {T.LINE}; max-height: 1px; border: 0; }}
        """

    # ---------------------------------------------------------------------- the pieces
    def _head(self):
        """The title, and nothing else.

        A search box lived here and has been taken out. It matched typed text against the
        timestamp, which is the same question the range below answers -- and answers better,
        with a calendar instead of a guess at the format. Two controls for one job is how a
        window starts feeling cluttered, and the one that needed typing was the weaker.
        """
        row = QHBoxLayout()
        row.addWidget(_styled(QLabel("รายการที่บันทึก"), 28, QFont.Bold, T.INK))
        row.addStretch(1)
        return row

    def _tiles(self):
        row = QHBoxLayout()
        row.setSpacing(14)
        self.tile_all = _tile("บันทึกทั้งหมด", "0", T.INK)
        self.tile_ok = _tile("ครบตามจำนวน", "0", T.GREEN_700)
        self.tile_off = _tile("ไม่ตรงจำนวน", "0", T.DANGER)
        for tile in (self.tile_all, self.tile_ok, self.tile_off):
            row.addWidget(tile, 1)
        return row

    def _filters(self):
        """Three buttons, not a combo box: on a touch bench a menu is two gestures."""
        row = QHBoxLayout()
        row.setSpacing(10)
        self.tabs = []
        for i, name in enumerate(("ทั้งหมด", "วันนี้", "ไม่ตรงจำนวน", "ช่วงวันเวลา")):
            b = _sized(QPushButton(name), 19)
            b.setObjectName("tab")
            b.setCheckable(True)
            b.setChecked(i == 0)
            b.clicked.connect(lambda _=False, n=i: self._filtered(n))
            self.tabs.append(b)
            row.addWidget(b)
        row.addStretch(1)
        self.count_lbl = _styled(QLabel(""), 18, QFont.Normal, T.INK_MUTED)
        row.addWidget(self.count_lbl)
        return row

    def _range(self):
        """From when, to when. ALWAYS LIVE -- never greyed, never hidden.

        It was greyed at first, until the "ช่วงวันเวลา" tab was chosen, and that was simply
        wrong: the first thing anybody did was click the date, and a disabled widget does
        not even get the click, so the window looked broken. Touching a date now IS choosing
        the range filter, which is what the gesture meant anyway.
        """
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(_styled(QLabel("ตั้งแต่"), 18, QFont.DemiBold, T.INK_SOFT))
        self.from_edit = self._when_edit()
        row.addWidget(self.from_edit)
        row.addWidget(_styled(QLabel("ถึง"), 18, QFont.DemiBold, T.INK_SOFT))
        self.to_edit = self._when_edit()
        row.addWidget(self.to_edit)
        row.addWidget(_styled(QLabel("(นับทั้งวัน)"), 18, QFont.Normal, T.INK_MUTED))
        row.addStretch(1)
        self._set_range_bounds()
        return row

    def _when_edit(self):
        """A date box that shows 17/09/2026, not ๑๗/๐๙/๒๐๒๖.

        SAME TRAP THE COUNT COLUMN FELL INTO, arriving through a different door. Qt formats
        a QDateTimeEdit with the application's locale, which on this bench is th_TH, and
        Qt's Thai locale numbers itself in Thai digits: the pickers opened reading
        "๑๖ ก.ย. ๒๐๒๖ ๐๐:๐๐" while every timestamp they filter -- the list, the pane, the
        CSV, the file names -- was written 2026-09-16. A control has to read like the thing
        it controls, so this one is given a locale with Western digits and an explicit
        numeric format; day/month/year, which is the order it is said in here.
        """
        edit = WhenEdit()
        edit.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        edit.setCalendarPopup(True)
        edit.setDisplayFormat("dd/MM/yyyy")
        edit.setCursor(Qt.PointingHandCursor)   # it is a button, so it says so
        # The popup does not inherit the box's locale: it is a separate widget tree, and
        # left alone it opened as a grid of Thai numerals under a Thai month name while the
        # box above it read 16/09/2026.
        calendar = edit.calendarWidget()
        if calendar is not None:
            calendar.setLocale(edit.locale())
            calendar.setGridVisible(False)
            calendar.setVerticalHeaderFormat(
                QCalendarWidget.VerticalHeaderFormat.NoVerticalHeader)
        _sized(edit, 19, QFont.Normal)
        edit.dateTimeChanged.connect(self._range_changed)
        return edit

    def _set_range_bounds(self):
        """Open on the span the records actually cover, so the first use needs no typing.

        An empty folder gets today, which is the only honest default when there is nothing
        to take the dates from.
        """
        moments = [m for m in (when(r) for r in self.rows) if m]
        first = min(moments).date() if moments else dt.date.today()
        last = dt.date.today()
        for edit, value in ((self.from_edit, first), (self.to_edit, last)):
            edit.blockSignals(True)         # or setting them re-filters twice on open
            edit.setDate(QDate(value.year, value.month, value.day))
            edit.blockSignals(False)

    def _range_changed(self):
        """Moving a date IS choosing the range filter -- nobody sets dates for fun."""
        if self.filter != 3:
            self._filtered(3)
            return
        self._apply()

    def _list(self):
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.setShowGrid(False)
        self.table.setIconSize(THUMB)
        self.table.verticalHeader().setDefaultSectionSize(ROW_H)
        _sized(self.table, 19, QFont.Normal)
        self.table.setItemDelegateForColumn(4, BadgeDelegate(self.table))

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.setColumnWidth(0, THUMB.width() + 24)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        for col in range(2, len(COLUMNS)):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)

        # Point the indicator at the column the list is actually ordered by. Qt parks it
        # on column 0 by default, which here is the thumbnails -- an arrow over a column of
        # pictures says the list is sorted by something that has no order.
        header.setSortIndicator(1, Qt.DescendingOrder)

        self.table.itemSelectionChanged.connect(self._show_selected)
        self.table.doubleClicked.connect(self._open_image)
        return self.table

    def _preview(self):
        card = QWidget()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(14)

        top = QHBoxLayout()
        top.addWidget(_styled(QLabel("ภาพขณะบันทึก"), 19, QFont.DemiBold, T.INK_SOFT))
        top.addStretch(1)
        self.when_lbl = _styled(QLabel(""), 19, QFont.DemiBold, T.INK)
        top.addWidget(self.when_lbl)
        lay.addLayout(top)

        self.shot = Shot()
        lay.addWidget(self.shot, 1)

        # The numbers as pairs, not a paragraph. A wall of "conf 0.45 iou 0.4 imgsz 640"
        # is read by nobody; a label above its value is read by everybody.
        self.facts = {}
        grid = QHBoxLayout()
        grid.setSpacing(10)
        for key, caption in (("count", "นับได้"), ("target", "ต้องการ"),
                             ("result", "ผล"), ("ms", "เวลาโมเดล (ms)")):
            box = QVBoxLayout()
            box.setSpacing(0)
            value = _styled(QLabel("—"), 22, QFont.Bold, T.INK)
            box.addWidget(value)
            box.addWidget(_styled(QLabel(caption), 18, QFont.Normal, T.INK_MUTED))
            grid.addLayout(box, 1)
            self.facts[key] = value
        lay.addLayout(grid)

        line = QFrame()
        line.setObjectName("rule")
        line.setFrameShape(QFrame.HLine)
        lay.addWidget(line)

        self.settings_lbl = _styled(QLabel(""), 18, QFont.Normal, T.INK_MUTED)
        self.settings_lbl.setWordWrap(True)
        lay.addWidget(self.settings_lbl)

        # NO BUTTONS UNDER THE PICTURE. There were two -- open the image, open the folder
        # -- and neither is something this window is for: one repeats what a double-click
        # on the row already does, and the other hands the operator a folder of JPEGs and
        # a file manager, which is the answer to a question nobody asked here. The panel
        # is for looking at the record, and the phone's has no room for them either.
        #
        # _open_image and _open_folder are still there and still reachable: the table's
        # double-click opens the picture, and the export tells you where the folder is.
        lay.addStretch(1)
        return card

    def _buttons(self):
        row = QHBoxLayout()
        row.setSpacing(12)
        refresh = _sized(QPushButton("รีเฟรช"), 19)
        refresh.setObjectName("ghost")
        refresh.clicked.connect(self._reload)
        row.addWidget(refresh)
        # Far from the primary button, and wearing the only red on the window: this is the
        # one control here that destroys something.
        self.clear_btn = _sized(QPushButton("ล้างข้อมูลทั้งหมด"), 19)
        self.clear_btn.setObjectName("danger")
        self.clear_btn.clicked.connect(self._clear_all)
        row.addWidget(self.clear_btn)
        row.addStretch(1)
        self.export_btn = _sized(QPushButton("ส่งออก CSV"), 20, QFont.Bold)
        self.export_btn.setObjectName("primary")
        self.export_btn.clicked.connect(self._export)
        row.addWidget(self.export_btn)
        close = _sized(QPushButton("ปิด"), 19)
        close.setObjectName("ghost")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        return row

    # ----------------------------------------------------------------- filter and fill
    def _filtered(self, index):
        self.filter = index
        for i, tab in enumerate(self.tabs):
            tab.setChecked(i == index)
        self._apply()

    def _apply(self):
        """Work out what the filter leaves, then rebuild the list."""
        today = time.strftime("%Y%m%d")
        # WHOLE DAYS, both ends inclusive. The boxes offer days, so a range that stopped at
        # midnight on the closing date would silently drop everything counted that day --
        # which is the day somebody filtering "up to today" most wants to see.
        start = dt.datetime.combine(self.from_edit.date().toPython(), dt.time.min)
        end = dt.datetime.combine(self.to_edit.date().toPython(), dt.time.max)
        rows = []
        for rec in self.rows:
            kind = verdict(rec)[0]
            if self.filter == 1 and not rec.get("stamp", "").startswith(today):
                continue
            if self.filter == 2 and kind in (OK, NONE):
                continue
            if self.filter == 3:
                moment = when(rec)
                # A record whose name cannot be parsed has no place on a time line, so it
                # is left out of a range rather than guessed into one.
                if moment is None or not start <= moment <= end:
                    continue
            rows.append(rec)
        self.shown = rows
        self._fill()
        self._summary()

    def _summary(self):
        kinds = [verdict(r)[0] for r in self.rows]
        self.tile_all.value_label.setText(str(len(self.rows)))
        self.tile_ok.value_label.setText(str(kinds.count(OK)))
        self.tile_off.value_label.setText(str(kinds.count(OVER) + kinds.count(SHORT)))
        if not self.rows:
            self.count_lbl.setText("ยังไม่มีรายการที่บันทึก")
        elif len(self.shown) == len(self.rows) and self.filter == 0:
            self.count_lbl.setText(f"ทั้งหมด {len(self.rows)} รายการ")
        else:
            self.count_lbl.setText(f"แสดง {len(self.shown)} จาก {len(self.rows)} รายการ")

    def _fill(self):
        # Sorting OFF while the rows go in. Left on, Qt re-sorts after every insert and the
        # row being filled moves out from under the column loop -- cells land in the wrong
        # rows, which reads as a data bug and is not one.
        self.table.setSortingEnabled(False)
        self.table.clearSpans()             # or the empty-state span outlives the message
        self.table.setRowCount(0)
        self._queue = []
        for index, rec in enumerate(self.shown):
            r = self.table.rowCount()
            self.table.insertRow(r)
            count = int(rec.get("count") or 0)
            target = int(rec.get("target") or 0)
            kind, words = verdict(rec)

            shot = Cell()
            shot.setData(Qt.UserRole, index)        # the index into self.shown
            if rec.get("image"):
                cached = self._thumbs.get(rec["image"])
                if cached is None:
                    self._queue.append((r, rec["image"]))
                else:
                    shot.setIcon(cached)
            self.table.setItem(r, 0, shot)

            # Shown short, SORTED BY THE STAMP: the label drops the year and
            # the seconds, and sorting on what is drawn would order 9 ก.ย. after
            # 10 ม.ค. and look like the list had shuffled itself.
            self.table.setItem(r, 1, Cell(short_when(rec), rec.get("stamp", "")))
            self.table.setItem(r, 2, Cell(count, centre=True))
            self.table.setItem(r, 3, Cell(target if target else "—", target, centre=True))
            # Sorted by the signed difference, so short trays gather at one end of the
            # column and over-counts at the other.
            badge = Cell(words, count - target if target else 0, centre=True)
            badge.setData(Qt.UserRole + 1, kind)
            self.table.setItem(r, 4, badge)
        self.table.setSortingEnabled(True)
        if self.shown:
            self.table.selectRow(0)
            return
        # An empty list with nothing in it looks like a window that failed to load. One
        # row saying why, spanning the whole width and not selectable, says which of the
        # two it is -- no records yet, or a filter that matched none.
        self.table.insertRow(0)
        message = Cell("ยังไม่มีรายการที่บันทึก" if not self.rows
                       else "ไม่มีรายการที่ตรงกับตัวกรอง", centre=True)
        message.setFlags(Qt.ItemIsEnabled)
        message.setForeground(QColor(T.INK_MUTED))
        self.table.setItem(0, 0, message)
        self.table.setSpan(0, 0, 1, len(COLUMNS))
        self.shot.show_image("")
        self._clear_facts()

    def _load_thumbs(self):
        """A few JPEGs per tick, cached, until the queue is empty -- then stop the timer.

        Stopping matters: a timer that keeps firing on an idle window is a wakeup every
        16ms for nothing, on a machine whose other job is running a detector.
        """
        if not self._queue:
            self._loader.stop()
            return
        for _ in range(THUMBS_PER_TICK):
            if not self._queue:
                break
            row, path = self._queue.pop(0)
            pix = self._thumbs.get(path)
            if pix is None:
                pix = QPixmap(path).scaled(THUMB, Qt.KeepAspectRatio,
                                           Qt.SmoothTransformation)
                self._thumbs[path] = pix
            item = self.table.item(row, 0)
            if item is not None:
                item.setIcon(pix)

    # ------------------------------------------------------------------------ the pane
    def _selected(self):
        """The record under the cursor, found by the index carried in column 0.

        Not by row number: the list can be sorted by any column, and after a sort row 3 is
        no longer self.shown[3].
        """
        items = self.table.selectedItems()
        if not items:
            return None
        item = self.table.item(items[0].row(), 0)
        index = item.data(Qt.UserRole) if item else None
        if index is None or index >= len(self.shown):
            return None
        return self.shown[index]

    def _clear_facts(self):
        self.when_lbl.setText("")
        self.settings_lbl.setText("")
        for label in self.facts.values():
            label.setText("—")

    def _show_selected(self):
        rec = self._selected()
        if rec is None:
            return
        self.shot.show_image(rec.get("image", ""))
        self.when_lbl.setText(rec.get("time", ""))
        target = int(rec.get("target") or 0)
        kind, words = verdict(rec)
        self.facts["count"].setText(f"{int(rec.get('count') or 0)}")
        self.facts["target"].setText(str(target) if target else "—")
        self.facts["result"].setText(words)
        _recolour(self.facts["result"], BADGES[kind][1])
        self.facts["ms"].setText(f"{rec.get('model_ms', '—')}")
        roi = "เฉพาะในกรอบ" if rec.get("roi") else "นับทั้งภาพ"
        # THE BREAKDOWN GOES FIRST AND ON ITS OWN LINE. For a two-pour record the count in
        # the box above is the only number here the picture cannot be used to check -- the
        # frame shown is the LAST tray, so a viewer who reads 60 and counts 25 tablets has
        # found the app lying unless this line is there to say it did not.
        rounds = rounds_text(rec)
        breakdown = ((f"เทเป็น {len(rec.get('rounds') or [])} รอบ: {rounds} = "
                     f"{int(rec.get('count') or 0)} เม็ด   "
                     f"(ภาพที่เห็นคือรอบสุดท้าย)\n") if rounds else "")
        self.settings_lbl.setText(
            f"{breakdown}"
            f"conf {rec.get('conf', '-')}   iou {rec.get('iou', '-')}   "
            f"imgsz {rec.get('imgsz', '-')}   {roi}   "
            f"เจอในภาพนี้ {len(rec.get('boxes') or [])} เม็ด")

    # ------------------------------------------------------------------------- actions
    def _reload(self):
        self.rows = load_records()
        self._apply()
        if self._queue and not self._loader.isActive():
            self._loader.start(16)

    def _export(self):
        """Exports WHAT IS ON SCREEN, filters and all.

        The errand is usually "give me the ones that did not match" or "give me today", and
        a button that ignores the filter the operator just set would hand them the whole
        folder and look like it had misheard.
        """
        note = export_with_dialog(self, self.shown)
        if note:
            self.note = note
            self.count_lbl.setText(note)

    def _clear_all(self):
        """Wipe the folder, but make the operator mean it.

        THREE CHOICES, AND THE SAFE ONE IS THE DEFAULT. This clears a dispensary's record
        of what it counted -- so the dialog says how many and what goes with them, Enter
        falls on ยกเลิก, and there is a way OUT of the dialog that keeps the data: exporting
        first, which is what somebody clearing down a full folder at the end of a month
        actually wants. Exporting does not then delete; they come back and press it again,
        having seen the file land. The files themselves go to the Recycle Bin, so a wipe
        nobody meant is recoverable -- see recycle().

        It clears EVERYTHING, not the filtered rows, because the button says so. A delete
        that quietly obeys a filter set five minutes ago is how the wrong month disappears.
        """
        if not self.rows:
            self.count_lbl.setText("ยังไม่มีรายการให้ลบ")
            return
        images = sum(1 for r in self.rows if r.get("image"))
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("ล้างข้อมูลทั้งหมด")
        box.setText(f"ลบรายการที่บันทึกทั้งหมด {len(self.rows)} รายการ")
        box.setInformativeText(
            f"รวมไฟล์ภาพ {images} ไฟล์ ในโฟลเดอร์\n{RECORDS}\n\n"
            "ไฟล์จะถูกย้ายไปถังรีไซเคิลของ Windows  กู้คืนจากที่นั่นได้")
        _sized(box, 19, QFont.Normal)
        keep = box.addButton("ส่งออก CSV ก่อน", QMessageBox.ActionRole)
        wipe = box.addButton("ลบทั้งหมด", QMessageBox.DestructiveRole)
        cancel = box.addButton("ยกเลิก", QMessageBox.RejectRole)
        # The one button that destroys something wears the only red in the dialog. It is
        # NOT made the prominent one: the default stays on cancel, and Enter with it.
        wipe.setStyleSheet(f"color: {T.DANGER}; font-weight: 700;")
        box.setDefaultButton(cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is keep:
            note = export_with_dialog(self, self.rows)
            self.count_lbl.setText(
                f"{note}   กดล้างข้อมูลอีกครั้งเพื่อลบ" if note else "")
            return
        if clicked is not wipe:
            return
        deleted, failed = delete_records(self.rows)
        self._thumbs.clear()
        self.rows = load_records()
        self._set_range_bounds()
        self._apply()
        self.note = f"ย้ายไปถังรีไซเคิลแล้ว {deleted} รายการ"
        if failed:
            QMessageBox.warning(self, "ลบไม่ครบ", "\n".join(failed[:8]))
            self.note = (f"ย้ายไปถังรีไซเคิลแล้ว {deleted} รายการ  "
                         f"เหลือ {len(failed)} ไฟล์ที่ลบไม่ได้")
        self.count_lbl.setText(self.note)

    def _open_folder(self):
        note = open_folder()
        if note:
            self.count_lbl.setText(note)

    def _open_image(self):
        """Opens the frame full size in whatever views JPEGs on this machine.

        The pane is for checking at a glance; a disputed count gets looked at properly, and
        rebuilding an image viewer inside this window would be silly.
        """
        rec = self._selected()
        if not rec or not rec.get("image"):
            return
        try:
            os.startfile(rec["image"])                              # noqa: S606
        except Exception as exc:                                    # noqa: BLE001
            self.count_lbl.setText(f"เปิดภาพไม่ได้ {exc}")

    def closeEvent(self, ev):
        self._loader.stop()
        super().closeEvent(ev)
