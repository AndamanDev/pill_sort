"""Draw every string the phone can show, once, on the PC. Run when strings.py changes.

    python -m model3.tools.render_text

WHAT IT MAKES. model3/phone/assets/text/ : one PNG per string in strings.ALL, one per
character in strings.DIGITS, and index.json listing what each file holds and how wide it
is. The phone loads that folder and blits; it never measures or shapes anything.

WHY WHITE ON TRANSPARENT. A glyph sheet in one colour can be tinted to any other by
multiplying, so the same file serves white text on the green header, dark text on a card,
and the red of a warning. Rendering each string once per colour would be three times the
assets and a fourth one the day the palette changes.

THE HEIGHT IS FIXED AND LARGE. Everything is drawn at RENDER_PX and scaled DOWN on the
phone, because scaling type up is what makes a screen look cheap. 96px covers the biggest
thing the phone screen asks for except the count itself, which is digits -- and those are
rendered at COUNT_PX so the one number everybody reads across a room stays sharp.
"""
from __future__ import annotations

import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL3 = os.path.dirname(HERE)
ROOT = os.path.dirname(MODEL3)
sys.path.insert(0, ROOT)

from model3.phone import strings                                    # noqa: E402

FONT_DIR = os.path.join(ROOT, "app", "assets", "fonts")
OUT_DIR = os.path.join(MODEL3, "phone", "assets", "text")

RENDER_PX = 96              # words
COUNT_PX = 260              # digits, which the count is drawn from
PAD = 8                     # room for tone marks above and descenders below


def face(weight="Regular", px=RENDER_PX):
    path = os.path.join(FONT_DIR, f"IBMPlexSansThai-{weight}.ttf")
    if not os.path.isfile(path):
        raise SystemExit(f"ไม่พบฟอนต์ {path}  (ต้องมี app/assets/fonts)")
    return ImageFont.truetype(path, px)


def render(text, font):
    """One string -> (RGBA image, advance). White, on a canvas with a FIXED baseline.

    EVERY IMAGE THE SAME HEIGHT, and the baseline in the same place in all of them. The
    first version cropped each word tight to its ink, which is smaller and quite wrong: the
    phone scales a word to the height it wants, so a word with no ascender came out bigger
    than the word beside it, and one with a tone mark came out smaller. Type size would have
    depended on which letters a word happened to contain.

    The advance is the font's own, not the ink width -- that is what the space after a word
    is measured from, and it is why "ครบ" and "ครบ " do not collide with what follows.
    """
    ascent, descent = font.getmetrics()
    advance = int(round(font.getlength(text)))
    w = advance + PAD * 2
    h = ascent + descent + PAD * 2
    img = Image.new("RGBA", (max(1, w), h), (255, 255, 255, 0))
    ImageDraw.Draw(img).text((PAD, PAD), text, font=font, fill=(255, 255, 255, 255),
                             anchor="la")
    return img, advance


def safe_name(text, n):
    """ASCII, and only ASCII. The Thai belongs in index.json, not in a path.

    The first version put the word in the file name, which reads beautifully in the folder
    and does not open: cv2.imread goes through the ANSI codepage on Windows and cannot
    find D:\...	-จำนวนทตองการ.png at all, whatever the file system thinks. Inside an
    APK it would be a zip entry read by a different library again. A number is a number
    everywhere."""
    return f"w{n:03d}.png"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for old in os.listdir(OUT_DIR):
        if old.endswith(".png") or old == "index.json":
            os.remove(os.path.join(OUT_DIR, old))

    words = face("SemiBold", RENDER_PX)
    digits = face("Bold", COUNT_PX)
    w_asc, w_desc = words.getmetrics()
    d_asc, d_desc = digits.getmetrics()
    index = {
        "pad": PAD,
        # What the phone divides by to turn a wanted pixel size into a scale factor, and
        # where the baseline sits inside every image so two runs can be lined up.
        "word": {"px": RENDER_PX, "height": w_asc + w_desc + PAD * 2,
                 "baseline": PAD + w_asc, "ascent": w_asc},
        "digit": {"px": COUNT_PX, "height": d_asc + d_desc + PAD * 2,
                  "baseline": PAD + d_asc, "ascent": d_asc},
        "words": {}, "digits": {},
    }

    for n, text in enumerate(strings.ALL):
        img, advance = render(text, words)
        name = safe_name(text, n)
        img.save(os.path.join(OUT_DIR, name))
        index["words"][text] = {"file": name, "w": img.width, "adv": advance}

    for n, ch in enumerate(strings.DIGITS):
        # A PLAIN space, not U+2007. The figure space was meant to give the blank a
        # width the font agrees with, and in IBM Plex Sans Thai Bold it has a VISIBLE
        # glyph: every gap between words came out as a hollow box. An ordinary space
        # draws nothing and still carries its own advance.
        img, advance = render(ch, digits)
        name = f"d{n:02d}.png"
        img.save(os.path.join(OUT_DIR, name))
        index["digits"][ch] = {"file": name, "w": img.width, "adv": advance}

    with open(os.path.join(OUT_DIR, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=1)

    total = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR))
    print(f"เขียน {len(index['words'])} คำ และ {len(index['digits'])} ตัวอักขระ "
          f"ลง {OUT_DIR}")
    print(f"รวม {total / 1024:.0f} KB")


if __name__ == "__main__":
    main()
