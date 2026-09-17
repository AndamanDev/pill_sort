"""Wire the splash, the camera, the detector and the window together.

THE ORDER MATTERS. The QApplication and the splash come first, before a single heavy
import, so something is on screen within a moment of the double-click. Everything slow --
torch, ultralytics, the weights, the camera, the warm-up pass -- runs on a worker thread
that reports its progress, and the main window is only built once that thread says it has
something to build it from.
"""
from __future__ import annotations

import argparse
import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from . import MODEL
from .worker import exposure_arg


def _identify(app) -> None:
    """The name and the icon, set once for the whole process.

    THE APP USER MODEL ID IS WHAT MAKES THE TASKBAR OBEY. Windows groups taskbar buttons by
    that id, and a plain `python -m app` inherits Python's -- so the button shows the Python
    logo and sits in Python's group no matter what icon the window itself carries. Setting
    an id of our own before the first window exists is the whole fix; it is cosmetic, it is
    Windows-only, and it must never stop the counter opening, hence the bare except.
    """
    from PySide6.QtGui import QIcon

    from . import ICON
    from . import theme as T

    app.setApplicationName(T.TITLE)
    if os.path.isfile(ICON):
        app.setWindowIcon(QIcon(ICON))
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "PharmaFlow.DrugCount.Counter")
        except Exception:                                           # noqa: BLE001
            pass


def run(source=0, model_path=MODEL, conf=0.45, iou=0.4, imgsz=640, target=0,
        exposure="auto") -> int:
    # HIGH-DPI, AND THIS IS WHAT MADE THE THAI UNREADABLE. This bench runs Windows at
    # 125%. With the default rounding policy Qt rounds a fractional factor to 1x, renders
    # at 1x, and lets Windows scale the finished BITMAP up by a quarter -- so every glyph
    # is stretched and the tone marks smear into the letters below. Large type survived
    # it; small Thai did not. PassThrough renders AT 1.25 instead, and must be set before
    # the QApplication exists.
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication(sys.argv)

    # The bundled Thai font, before the first label exists. Registering it later would
    # leave the splash -- and only the splash -- in the system face.
    from . import theme as T

    T.load_fonts()
    _identify(app)

    from .splash import Loader, Splash

    splash = Splash()
    splash.showFullScreen()
    app.processEvents()                     # paint it before the thread starts working

    loader = Loader(source, model_path, conf, iou, imgsz, exposure)
    state = {"window": None, "code": 0}

    def ready():
        from .window import Window

        capture, infer = loader.result
        win = Window(capture, infer, target=target, camera=source)
        win.showMaximized()
        state["window"] = win           # held, or Python would collect it
        splash.close()

    def failed(message):
        splash.fail(message)
        state["code"] = 1
        # Left on screen rather than closed: a window that vanishes tells the operator
        # nothing, and the reason is the only useful thing left to give them.

    loader.progress.connect(splash.step)
    loader.done.connect(ready)
    loader.failed.connect(failed)
    loader.start()

    code = app.exec()
    if loader.result:
        capture, infer = loader.result
        infer.stop()
        capture.stop()
    return code or state["code"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--target", type=int, default=0)
    ap.add_argument("--no-sound", action="store_true",
                    help="เงียบ: สำหรับบริเวณที่เสียงรบกวน")
    ap.add_argument("--exposure", type=exposure_arg, default="auto",
                    help="auto (default) or a number. A number is shutter time, so the "
                         "bright end of the scale costs frame rate -- see "
                         "set-camera-exposure.py, which measures both.")
    a = ap.parse_args(argv)
    if a.no_sound:
        from . import sound
        sound.enabled = False
    return run(a.camera, model_path=a.model, conf=a.conf, iou=a.iou, imgsz=a.imgsz,
               target=a.target, exposure=a.exposure)


if __name__ == "__main__":
    raise SystemExit(main())
