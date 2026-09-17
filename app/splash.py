"""The window that holds the screen while the heavy things load.

WHY IT EXISTS. Opening this program means importing torch and ultralytics, reading a
10 MB model, opening a camera and running one frame through to warm the graph up. On this
bench that is about ten seconds, and for all of it the old build showed nothing at all --
no window, no taskbar entry, nothing to say the double-click had worked. An operator
double-clicks again, and now two processes are fighting over one camera.

So the splash appears FIRST, before any of the loading starts, and the loading happens on
a worker thread that reports where it has got to. The percentage is not decorative: each
step publishes when it actually finishes, so a stall at 60% says the model is the slow
part rather than leaving someone to guess.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget)

from . import LOGO
from . import theme as T
import os


class Loader(QThread):
    """Every slow thing at start-up, off the UI thread so the splash can paint.

    Emits (percent, what) as each step completes. The objects it built come back through
    `result` rather than a signal payload, because Qt would have to marshal a camera
    handle and a loaded network across threads and there is nothing to gain by it.
    """

    progress = Signal(int, str)
    failed = Signal(str)
    done = Signal()

    def __init__(self, source, model_path, conf, iou, imgsz, exposure):
        super().__init__()
        self.source, self.model_path = source, model_path
        self.conf, self.iou, self.imgsz, self.exposure = conf, iou, imgsz, exposure
        self.result = None

    def run(self):
        try:
            self.progress.emit(5, "กำลังเริ่มระบบ")
            import os as _os

            import torch
            from ultralytics import YOLO
            self.progress.emit(35, "โหลดไลบรารีประมวลผลภาพ")

            torch.set_num_threads(max(1, (_os.cpu_count() or 2) // 2))
            if not _os.path.isfile(self.model_path):
                self.failed.emit(f"ไม่พบไฟล์โมเดล\n{self.model_path}")
                return
            model = YOLO(self.model_path)
            self.progress.emit(60, "โหลดโมเดลตรวจจับเม็ดยา")

            from .worker import Capture, Infer
            capture = Capture(self.source, exposure=self.exposure)
            if not capture.open():
                self.failed.emit(f"เปิดกล้องไม่ได้ (กล้อง {self.source})\n"
                                 f"ตรวจว่ามีโปรแกรมอื่นใช้กล้องอยู่หรือไม่")
                return
            self.progress.emit(80, "เชื่อมต่อกล้อง")

            infer = Infer(capture, model, conf=self.conf, iou=self.iou,
                          imgsz=self.imgsz)
            capture.start()
            # One pass before the window opens, so the first number the operator sees is
            # a real one rather than a zero that corrects itself a second later.
            frame = None
            for _ in range(60):
                frame, _seq = capture.latest()
                if frame is not None:
                    break
                self.msleep(50)
            if frame is not None:
                model.predict(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                              max_det=1000, verbose=False)
            self.progress.emit(95, "เตรียมการนับ")
            infer.start()

            self.result = (capture, infer)
            self.progress.emit(100, "พร้อมใช้งาน")
            self.done.emit()
        except Exception as exc:                                # noqa: BLE001
            self.failed.emit(str(exc))


class Splash(QWidget):
    """Full screen, logo centred, green on white.

    Full screen rather than a small dialog because this is the first thing the operator
    sees and a 560px box on a 1920px bench monitor reads as something that went wrong.
    It also means the transition to the main window is a change of content, not a window
    appearing somewhere else on the desktop.
    """

    def __init__(self):
        super().__init__()
        # T.TITLE, not a name typed here. The taskbar entry that appears while the app
        # loads is the first thing the operator sees, and it said PillSort -- the
        # project's old name -- while the window it turns into says DrugCount.
        self.setWindowTitle(T.TITLE)
        self.setObjectName("splash")
        self.setStyleSheet(f"""
            #splash {{ background: {T.SURFACE}; }}
            #tint {{ background: {T.GREEN_TINT}; border-radius: 0px; }}
            QProgressBar {{ background: {T.LINE}; border: 0;
                            border-radius: 6px; height: 12px; }}
            QProgressBar::chunk {{ background: {T.GREEN_700};
                                   border-radius: 6px; }}
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(64, 48, 64, 40)
        outer.setSpacing(0)

        outer.addStretch(3)

        if os.path.isfile(LOGO):
            logo = QLabel()
            # Tall: the mark is the identity of the screen while there is nothing else
            # on it, and it is the only thing here that is not text.
            logo.setPixmap(QPixmap(LOGO).scaledToHeight(96, Qt.SmoothTransformation))
            logo.setAlignment(Qt.AlignCenter)
            outer.addWidget(logo)
            outer.addSpacing(28)

        self.title = self._label(30, QFont.Bold, T.INK)
        self.title.setText("ระบบนับเม็ดยา")
        self.title.setAlignment(Qt.AlignCenter)
        outer.addWidget(self.title)
        outer.addSpacing(44)

        # The bar is held to a readable width and centred, rather than stretched across
        # a 1920px screen where the eye cannot see both ends of it at once.
        barrow = QHBoxLayout()
        barrow.addStretch(1)
        holder = QVBoxLayout()
        holder.setSpacing(14)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedWidth(520)
        holder.addWidget(self.bar)
        self.pct = self._label(22, QFont.Bold, T.GREEN_700)
        self.pct.setText("0%")
        self.pct.setAlignment(Qt.AlignCenter)
        holder.addWidget(self.pct)

        # WHICH STEP IS RUNNING, under the thing it is the caption for. It used to sit in
        # the bottom-right corner of the screen, half a metre from the bar on a bench
        # monitor and nowhere near the eye that is watching the percentage -- and the
        # phone, which was built from this file, quite reasonably put it under the bar.
        # One of the two had to move and it was not going to be the one that was right.
        holder.addSpacing(18)
        self.status = self._label(20, QFont.Normal, T.INK_SOFT)
        self.status.setText("กำลังเริ่มระบบ")
        self.status.setAlignment(Qt.AlignCenter)
        holder.addWidget(self.status)

        barrow.addLayout(holder)
        barrow.addStretch(1)
        outer.addLayout(barrow)

        outer.addStretch(4)

    @staticmethod
    def _label(size, weight, colour):
        """Font on the widget AND in the rule -- see app/window.py: styled()."""
        lab = QLabel("")
        css_weight = {QFont.Normal: 400, QFont.DemiBold: 600,
                      QFont.Bold: 700, QFont.ExtraBold: 800}.get(weight, 400)
        font = QFont()
        font.setFamilies(T.FONT_FAMILIES)
        font.setPixelSize(size)
        font.setWeight(weight)
        lab.setFont(font)
        lab.base = (f"font-family: {T.FONT_STACK}; "
                    f"font-size: {size}px; font-weight: {css_weight};")
        lab.setStyleSheet(lab.base + f" color: {colour};")
        return lab

    def step(self, percent, what):
        self.bar.setValue(percent)
        self.pct.setText(f"{percent}%")
        self.status.setText(what)

    def fail(self, message):
        self.title.setText("เปิดโปรแกรมไม่สำเร็จ")
        self.title.setStyleSheet(self.title.base + f" color: {T.DANGER};")
        self.pct.setText("")
        self.bar.hide()
        self.status.setStyleSheet(self.status.base + f" color: {T.DANGER};")
        self.status.setText(message.replace("\n", "   "))
