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
        self._lines = {}                    # (string, px) -> the whole line's coverage
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
        if align != "left":
            width = self.measure(s, px)
            x = x - width if align == "right" else x - width // 2

        # ONE BLEND FOR THE WHOLE LINE, from a mask built the first time it was asked for.
        # A line is a dozen pieces -- the footer is forty, a glyph at a time -- and a
        # blend is a dozen numpy calls whoever small it is, so on a board thirty times
        # slower than the desk the words cost more than the picture did. The same few
        # lines are drawn every frame, so each one is assembled once.
        line = self._line(s, px)
        if line is not None:
            mask, dx, dy, advance = line
            self._blend(dst, mask, x + dx, y + dy, colour, alpha)
            return advance
        runs = self._runs(s)

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

    #: How many lines are kept. Far more than one screen holds; the cap is for strings with
    #: a number in them, which would otherwise collect one entry per value ever shown.
    LINES_KEPT = 400

    def _line(self, s, px):
        """(coverage, x offset, y offset, advance) for the line drawn at (0, 0), or None.

        The coverage is what blitting the pieces one after another would have left:
        1 - the product of what each let through, which is exact for one colour. None for
        a line with a missing word, which keeps the hollow box the slow way.
        """
        key = (s, px)
        if key in self._lines:
            return self._lines[key]
        runs = self._runs(s)
        placed, tops = [], []
        for piece, digit in runs:
            kind = "digit" if digit else "word"
            entry = self._entry(piece, digit)
            if entry is None:
                self._lines[key] = None
                return None
            tops.append(self.index[kind]["baseline"] * px / self.index[kind]["px"])
        baseline = max(tops) if tops else px
        cursor = 0.0
        for (piece, digit), top in zip(runs, tops):
            kind = "digit" if digit else "word"
            entry = self._entry(piece, digit)
            img = self._piece(entry, kind, px)
            placed.append((img, int(round(cursor)), int(round(baseline - top))))
            cursor += entry["adv"] * px / self.index[kind]["px"]
        if not placed:
            self._lines[key] = None
            return None
        x0 = min(px_ for _i, px_, _y in placed)
        y0 = min(py_ for _i, _x, py_ in placed)
        x1 = max(px_ + i.shape[1] for i, px_, _y in placed)
        y1 = max(py_ + i.shape[0] for i, _x, py_ in placed)
        through = np.ones((y1 - y0, x1 - x0, 1), np.float32)
        for img, px_, py_ in placed:
            h, w = img.shape[:2]
            through[py_ - y0:py_ - y0 + h, px_ - x0:px_ - x0 + w] *= (
                1.0 - img[:, :, 3:4].astype(np.float32) / 255.0)
        line = (1.0 - through, x0, y0, int(round(cursor)))
        if len(self._lines) >= self.LINES_KEPT:
            self._lines.clear()
        self._lines[key] = line
        return line

    @staticmethod
    def _blend(dst, cover, x, y, colour, alpha):
        """Lay a coverage mask on dst in one colour, clipped to what is on screen."""
        h, w = cover.shape[:2]
        H, W = dst.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x0 >= x1 or y0 >= y1:
            return
        a = cover[y0 - y:y1 - y, x0 - x:x1 - x]
        if alpha != 1.0:
            a = a * alpha
        patch = dst[y0:y1, x0:x1].astype(np.float32)
        tint = np.array(colour, np.float32).reshape(1, 1, 3)
        dst[y0:y1, x0:x1] = (patch + (tint - patch) * a).astype(np.uint8)

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
