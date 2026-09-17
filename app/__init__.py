"""A bench window for the model3 detector: camera right, the number left.

    D:\\pill-counter-lite\\pillcount-v12\\.venv-train\\Scripts\\python.exe -m model3.ui

What it is for, and how it differs from the two windows already in this project:

    model1/ui_qt/   the pharmacy screen. Segmentation model, outlines, odd-one-out,
                    history, a beep. It is what a counter ships as.
    model2/count.py a bench instrument. One number, no smoothing, prints the spread on
                    exit -- built to MEASURE a model, not to use one. It stays exactly
                    as it is: every default in it is a measurement, and a window with
                    buttons is not the place to keep those honest.
    app/            this. The v3 detector behind a screen an operator can work: draw the
                    tray once, set how many the prescription wants, watch the number,
                    save the result.

SEPARATE FROM model2/count.py ON PURPOSE. That file opens a camera and prints numbers,
and it is how every setting this window uses was decided. Folding it into a GUI would
make the measurements harder to repeat and the GUI harder to trust; they answer different
questions and share only the model.

THE THEME AND THE LOGO CAME FROM model1 and are copied into app/assets and app/theme.py,
not imported. Same look, no dependency: model1 and model2 are history now and nothing
running every day should break when history is tidied away.

NOTHING HERE SMOOTHS THE COUNT. model1 holds a figure until three of five passes agree,
which is right for a pharmacy. This shows what the model says, now, because the point of
model3 is still to find out what v3 actually does.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MODEL = os.path.join(ROOT, "model3", "weights", "pillcount-det-v3.pt")
RECORDS = os.path.join(ROOT, "app", "records")

#: Where the drawn region is kept between runs. PER BENCH, NEVER IN THE REPO -- the same
#: reason model1 gitignores its settings/ folder: a polygon is pixel coordinates from one
#: camera bolted at one height over one tray, and handing it to another bench points that
#: bench's counter at a rectangle calibrated for this room.
SETTINGS = os.path.join(ROOT, "app", "settings")

# Theme and logo live HERE. They started as model1's and were copied rather than
# imported: a window in daily use must not stop opening because a folder kept only as
# history was moved. Nothing in app/ reaches outside app/ except the weights file.
ASSETS = os.path.join(HERE, "assets")
LOGO = os.path.join(ASSETS, "logo.png")

#: The window and taskbar icon: the leaf mark from the logo, cropped square and
#: saved at every size Windows asks for. The wordmark itself is useless as an icon
#: -- at 32px a 223x72 strip of text is a grey smudge.
ICON = os.path.join(ASSETS, "icon.ico")

__all__ = ["ROOT", "MODEL", "RECORDS", "SETTINGS", "ASSETS", "LOGO", "ICON"]
