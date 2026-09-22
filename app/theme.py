"""Palette, scale and stylesheet.

COPIED from model1/ui_qt/theme.py rather than imported. app/ used to import it, which
made a window in daily use depend on a folder that is now kept only as history -- move
or archive model1 and the counter stops opening. The duplication is the price of that
independence, and the two are allowed to drift: this file answers to app/ alone.


Two sources, deliberately: the COLOUR identity is DrugCount's own green, sampled from
the logo the header shows; the STRUCTURE -- spacing and radius scales, badge and
progress treatments, the KPI/muted type pair -- is lifted from the PharmaFlow design
system next door (pharmaflow_drugcount/app/globals.css) so this window belongs to the
same family as the rest of the product.
"""

# --- palette -------------------------------------------------------------------------
BG = "#f4f7f4"
SURFACE = "#ffffff"
LINE = "#dfe7e1"
LINE_STRONG = "#cfdad3"
INK = "#17211c"
INK_SOFT = "#5b6b62"
INK_MUTED = "#94a3a0"

GREEN_900 = "#015e13"
GREEN_700 = "#108644"
GREEN_500 = "#17a050"
GREEN_TINT = "#e7f4ec"

WARN = "#e8833a"
DANGER = "#c0392b"

# Badge pairs: a tint behind dark text reads at a glance without shouting, and stays
# legible on a bench under fluorescent light. Same trio as the sibling app.
BADGE_OK = ("#dcfce7", "#166534")
BADGE_WARN = ("#fef3c7", "#92400e")
BADGE_BAD = ("#fee2e2", "#991b1b")
BADGE_IDLE = ("#eef2ef", "#5b6b62")

# Overlay strokes are BGR: cv2 draws them onto the frame before Qt sees it.
OUTLINE_OK = (106, 196, 43)     # #2bc46a
OUTLINE_ODD = (61, 157, 255)    # #ff9d3d

# --- scale ---------------------------------------------------------------------------
SP_2, SP_3, SP_4, SP_6, SP_8 = 8, 12, 16, 24, 32
R_SM, R_MD, R_LG, R_XL = 6, 8, 12, 16

#: THE NAME OF THE APP, and the only place it is written on the desktop side.
#:
#: It is "DrugCount" and nothing longer, because the same name has to fit under an icon on
#: a phone, where Android truncates at about eleven characters -- "PharmaFlow DrugCount"
#: appears there as "PharmaFlow…", which is a different product as far as anybody reading
#: the home screen is concerned. The full wordmark is still in the logo at the top of the
#: window, where there is room for it, and "โปรแกรมนับยา" was a description rather than a
#: name and belongs in the documentation.
#:
#: The phone's copy of this is model3/android/app/src/main/res/values/strings.xml, and the
#: two have to be changed together -- they were not, once, and the desktop said one thing
#: while the launcher said another.
TITLE = "DrugCount"
WIN_W, WIN_H = 1280, 820
HEADER_H = 60

#: THE FONT SHIPS WITH THE APP NOW -- app/assets/fonts, IBM Plex Sans Thai, OFL.
#:
#: What it replaced and why. Leelawadee UI is the system's own Thai UI face and was the
#: honest choice while nothing was bundled, but when a Qt STYLESHEET sets it at weight 400
#: its Thai renders about two thirds the height of the Latin beside it, tone marks smeared:
#: "ทั้งหมด 13 รายการ" came out with the digits full size and the Thai shrunken. Measured on
#: this bench across 16-22px: broken at 400 at every size, clean at 600 and above, clean at
#: any weight through a plain QFont. The screens dodged it by accident -- every label on
#: them was already DemiBold or Bold -- and the records window, which has ordinary body
#: text, walked straight into it.
#:
#: IBM Plex Sans Thai renders both scripts at one size at every weight, so normal-weight
#: text is usable again, and a bundled file means the window looks the same on the next
#: bench instead of depending on what Windows happens to have. The old stack stays behind
#: it: if the files are ever missing, the app falls back to what it used before rather
#: than to whatever Qt picks.
FONT_FAMILIES = ["IBM Plex Sans Thai", "Leelawadee UI", "TH Sarabun New", "Tahoma"]
FONT_STACK = ('"IBM Plex Sans Thai", "Leelawadee UI", "TH Sarabun New", '
              "Tahoma, sans-serif")

#: True once load_fonts() has found the bundled family. Read by the label helpers, which
#: hold Thai to DemiBold when it is False -- the weight that survives Leelawadee UI.
BUNDLED = False


def load_fonts() -> bool:
    """Register app/assets/fonts with Qt. Call ONCE, after the QApplication exists.

    Before it, not after: a font registered once labels are already built does not restyle
    them, so this belongs beside the QApplication in app.py, ahead of the splash.
    """
    global BUNDLED
    import glob
    import os

    from PySide6.QtGui import QFontDatabase

    from . import ASSETS

    for path in sorted(glob.glob(os.path.join(ASSETS, "fonts", "*.ttf"))):
        if QFontDatabase.addApplicationFont(path) < 0:
            continue
    BUNDLED = FONT_FAMILIES[0] in QFontDatabase.families()
    return BUNDLED


DRAW_MS = 16          # ~60 fps repaint, independent of inference

#: Mean absolute difference (0-255) between the current view and the frame the outlines
#: came from, above which the view counts as moving and the outlines are held back --
#: otherwise they float over bare bench while the camera is repositioned.
STILL_DRIFT = 6.0

DEFAULT_TARGET = 0
STEP = 1
PRESETS = (30, 60, 90, 100, 200)


def sheet() -> str:
    return f"""
    QWidget {{ font-family: {FONT_STACK}; color: {INK}; font-size: 16px; }}
    #root {{ background: {BG}; }}
    #header {{ background: {GREEN_700}; }}
    #title {{ color: #ffffff; font-size: 19px; font-weight: 700; }}

    #card {{ background: {SURFACE}; border: 1px solid {LINE};
             border-radius: {R_XL}px; }}
    #paneHead {{ color: {INK_SOFT}; font-size: 15px; font-weight: 700; }}

    #label {{ color: {INK_SOFT}; font-size: 16px; font-weight: 600; }}
    #muted {{ color: {INK_SOFT}; font-size: 16px; }}
    #kpi {{ font-size: 132px; font-weight: 800; color: {GREEN_700}; }}
    #progressNum {{ font-size: 22px; font-weight: 700; }}
    #odd {{ color: {WARN}; font-size: 15px; font-weight: 600; }}
    #fps {{ color: {INK_MUTED}; font-size: 13px; }}
    #rule {{ background: {LINE}; }}

    QLineEdit {{ background: {SURFACE}; border: 1px solid {LINE_STRONG};
                 border-radius: {R_MD}px; font-size: 20px; font-weight: 700;
                 padding: 8px; selection-background-color: {GREEN_500}; }}
    QLineEdit:focus {{ border: 1px solid {GREEN_500}; }}

    QPushButton#step {{ background: {BG}; border: 1px solid {LINE_STRONG};
                        border-radius: {R_MD}px; font-size: 20px; font-weight: 700;
                        color: {GREEN_700}; min-width: 44px; min-height: 44px; }}
    QPushButton#step:hover {{ border-color: {GREEN_500}; background: {GREEN_TINT}; }}
    QPushButton#step:pressed {{ background: {LINE}; }}

    QPushButton#chip {{ background: {BG}; border: 1px solid {LINE};
                        border-radius: 999px; padding: 6px 14px;
                        font-size: 15px; font-weight: 600; color: {INK_SOFT};
                        min-height: 34px; }}
    QPushButton#chip:hover {{ border-color: {GREEN_500}; color: {GREEN_700}; }}
    QPushButton#chip:checked {{ background: {GREEN_TINT}; border-color: {GREEN_500};
                                color: {GREEN_700}; }}

    QProgressBar {{ background: {LINE}; border: none; border-radius: 5px;
                    max-height: 10px; min-height: 10px; text-align: center; }}
    QProgressBar::chunk {{ border-radius: 5px; background: {GREEN_500}; }}

    /* A MESSAGE BOX HAS TO PAINT ITS OWN SURFACE.
       Everything else on this screen sits on a card this sheet paints, so the sheet only
       ever had to name the colour of the INK -- and dark ink is all it named. A message
       box has no card: its background comes from the Windows palette, and on a machine
       set to the dark theme that palette is near-black. Dark ink on a near-black panel is
       how the change-the-number question shipped: a black rectangle with a blue question
       mark, two ghost buttons and not one readable word between them.
       So the surface, the ink and the buttons are all stated here rather than inherited,
       and stated ONCE: the static helpers (QMessageBox.warning) build their own box that
       no call site can reach with setStyleSheet, and those were just as unreadable. */
    QMessageBox {{ background: {SURFACE}; }}
    QMessageBox QLabel {{ background: transparent; color: {INK};
                          font-size: 19px; }}
    QMessageBox QPushButton {{ background: {BG}; color: {INK};
                               border: 1px solid {LINE_STRONG};
                               border-radius: {R_MD}px; padding: 10px 20px;
                               font-size: 19px; font-weight: 600; min-width: 150px; }}
    QMessageBox QPushButton:hover {{ border-color: {GREEN_500};
                                     background: {GREEN_TINT}; color: {GREEN_700}; }}
    QMessageBox QPushButton:pressed {{ background: {LINE}; }}
    /* The default button is the SAFE answer everywhere this app asks a question -- keep
       the number, go back and pour more, cancel the delete -- so the one that is filled
       in and obvious is also the one that costs nothing. */
    QMessageBox QPushButton:default {{ background: {GREEN_700}; color: #ffffff;
                                       border-color: {GREEN_900}; }}
    QMessageBox QPushButton:default:hover {{ background: {GREEN_500};
                                             color: #ffffff; }}
    """


def badge_css(pair) -> str:
    bg, fg = pair
    return (f"background: {bg}; color: {fg}; border-radius: 999px; "
            f"padding: 7px 16px; font-size: 16px; font-weight: 700;")
