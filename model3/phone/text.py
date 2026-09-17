"""Blitting the pre-rendered words. The phone's entire typography, in one small class.

HOW A LINE IS BUILT. A string is split into runs: anything made of DIGITS is composed
character by character from the digit sheet, and everything else is looked up whole in the
word sheet -- because Thai marks stack on the consonant they belong to and a per-character
blit would scatter them. So "บันทึกแล้ว 13 เม็ด" is three pieces: a word, two digits, a
word, with the space between them coming from the font's own advances.

RUNS ARE LINED UP BY THEIR BASELINES, not by their tops. Words are rendered at 96px and
digits at 260px, so their images are different heights; laying them out from the top would
leave the number sitting a centimetre above the word beside it. index.json carries where
the baseline is inside each sheet and this class does the arithmetic.

EVERY SCALED PIECE IS CACHED. The screen is redrawn for every frame the camera delivers,
and cv2.resize on forty glyphs a frame is measurable on a phone; the same word at the same
size is resized once and then copied. The cache is keyed on the size actually asked for,
rounded, so a layout that animates a size does not grow it without bound.
"""
from __future__ import annotations

import json
import os

import numpy as np

try:
    import cv2
except ImportError:                                             # pragma: no cover
    cv2 = None

from .strings import DIGITS


class Text:
    def __init__(self, folder):
        self.folder = folder
        with open(os.path.join(folder, "index.json"), encoding="utf-8") as fh:
            self.index = json.load(fh)
        self._raw = {}                      # file -> BGRA as loaded
        self._scaled = {}                   # (file, px) -> BGRA at that size
        self._tokens = {}                   # string -> how it was split last time
        self._longest = max((len(w) for w in self.index["words"]), default=1)

    # ------------------------------------------------------------------ the pieces --
    def _entry(self, piece, digit):
        table = self.index["digits"] if digit else self.index["words"]
        return table.get(piece)

    def _image(self, name):
        img = self._raw.get(name)
        if img is None:
            img = cv2.imread(os.path.join(self.folder, name), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(os.path.join(self.folder, name))
            if img.shape[2] == 3:           # a sheet saved without alpha: treat as opaque
                img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
            self._raw[name] = img
        return img

    def _piece(self, entry, kind, px):
        """The piece at the size asked for, from cache when it has been asked before."""
        key = (entry["file"], px)
        got = self._scaled.get(key)
        if got is None:
            src = self._image(entry["file"])
            scale = px / self.index[kind]["px"]
            w = max(1, int(round(src.shape[1] * scale)))
            h = max(1, int(round(src.shape[0] * scale)))
            got = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
            self._scaled[key] = got
        return got

    def _runs(self, s):
        """The string as [(piece, is_digit)], LONGEST KNOWN WORD FIRST.

        The obvious split -- cut wherever a digit-class character appears -- gets two
        things wrong, and both showed up the first time a real line was drawn. "17 ก.ย.
        10:24" came apart at the full stops into ก and ย, neither of which is a word this
        app ever renders, so the date drew as two empty boxes; and "ส่งออก CSV", which IS
        one entry in the sheet, was split at its space into two fragments that are not.

        Matching the longest entry at each position handles both: the month is found whole
        because "ก.ย." is in the index, and a label with a space in it is found before
        anything inside it is considered. Only what is left over falls through to the
        per-character digits.

        Tokenising is memoised because the screen redraws every frame and the strings on
        it barely change between one frame and the next.
        """
        cached = self._tokens.get(s)
        if cached is not None:
            return cached
        words = self.index["words"]
        runs, i, n = [], 0, len(s)
        while i < n:
            hit = None
            for length in range(min(self._longest, n - i), 0, -1):
                piece = s[i:i + length]
                if piece in words:
                    hit = piece
                    break
            if hit is not None:
                runs.append((hit, False))
                i += len(hit)
                continue
            runs.append((s[i], s[i] in DIGITS))
            i += 1
        self._tokens[s] = runs
        return runs

    # -------------------------------------------------------------------- measuring --
    def measure(self, s, px):
        """How wide the line will be. Missing words measure as the box that will be drawn."""
        width = 0.0
        for piece, digit in self._runs(s):
            entry = self._entry(piece, digit)
            kind = "digit" if digit else "word"
            if entry is None:
                width += px * 0.6 * len(piece)
                continue
            width += entry["adv"] * px / self.index[kind]["px"]
        return int(round(width))

    def height(self, px):
        return int(round(self.index["word"]["height"] * px / self.index["word"]["px"]))

    # --------------------------------------------------------------------- drawing --
    def draw(self, dst, s, x, y, px, colour=(255, 255, 255), align="left", alpha=1.0):
        """Draw `s` with its LINE TOP at y. Returns the width drawn.

        A missing word is drawn as a hollow box rather than skipped: a screen with a gap
        where a label should be is a bug you see, and a screen that silently drops the
        label is a bug you ship. Add the string to strings.py and re-run the render tool.
        """
        px = int(round(px))
        if px <= 0 or not s:
            return 0
        runs = self._runs(s)
        if align != "left":
            width = self.measure(s, px)
            x = x - width if align == "right" else x - width // 2

        # Where the baselines meet: the tallest ascender in the line decides.
        tops = []
        for piece, digit in runs:
            kind = "digit" if digit else "word"
            entry = self._entry(piece, digit)
            if entry is None:
                tops.append(px)
                continue
            tops.append(self.index[kind]["baseline"] * px / self.index[kind]["px"])
        baseline = y + (max(tops) if tops else px)

        cursor = float(x)
        for (piece, digit), top in zip(runs, tops):
            kind = "digit" if digit else "word"
            entry = self._entry(piece, digit)
            if entry is None:
                w = int(px * 0.6 * len(piece))
                cv2.rectangle(dst, (int(cursor), int(baseline - px)),
                              (int(cursor + w), int(baseline)), (0, 0, 255), 1)
                cursor += w
                continue
            img = self._piece(entry, kind, px)
            self._blit(dst, img, int(round(cursor)), int(round(baseline - top)),
                       colour, alpha)
            cursor += entry["adv"] * px / self.index[kind]["px"]
        return int(round(cursor - x))

    @staticmethod
    def _blit(dst, src, x, y, colour, alpha):
        """Alpha-composite one piece, tinted, clipped to whatever of it is on screen."""
        h, w = src.shape[:2]
        H, W = dst.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x0 >= x1 or y0 >= y1:
            return
        piece = src[y0 - y:y1 - y, x0 - x:x1 - x]
        a = (piece[:, :, 3:4].astype(np.float32) / 255.0) * alpha
        patch = dst[y0:y1, x0:x1].astype(np.float32)
        tint = np.array(colour, np.float32).reshape(1, 1, 3)
        dst[y0:y1, x0:x1] = (patch * (1 - a) + tint * a).astype(np.uint8)
