"""The two noises this screen makes: one when a count is filed, one when it stops counting.

WHY SOUND AT ALL. The operator's eyes are on the tray and their hands are in it. Every
confirmation this app has -- the number, the badge, the toast, the footer -- is something
they have to look up to see, and the moment they most need to know that the save landed is
the moment they are least likely to be looking. A note they can hear needs nobody's eyes.

TWO SOUNDS, AND THEY MUST NOT BE CONFUSABLE. Rising and short for a record filed; low,
doubled and slower for something that has gone wrong. That difference has to survive a
noisy dispensary, being heard from across the room, and a person who has never been told
what the sounds mean -- which is why it is pitch AND rhythm, not two different pitches.

EVERY BEEP RUNS ON ITS OWN THREAD. winsound.Beep BLOCKS for as long as the note lasts, and
the caller here is the repaint loop: a 140ms beep on the GUI thread is a 140ms freeze of
the picture, eight frames dropped, every single time a tray is saved.

IT IS ALSO ALLOWED TO FAIL. A bench with no sound device, a machine where the beeper is
disabled in the BIOS, a Windows that refuses -- none of that is a reason for the counter to
stop working, so every failure here is swallowed. Sound is a courtesy, not a mechanism.
"""
from __future__ import annotations

import sys
import threading

#: Turned off by --no-sound, for a bench where any noise is one too many.
enabled = True

#: (frequency Hz, milliseconds) pairs, played in order.
SAVED = ((1046, 70), (1568, 90))            # two rising notes: "filed"
PROBLEM = ((392, 160), (0, 60), (392, 220))  # low, twice, unhurried: "stopped"
#: One note, flat, between the two above: "that pour is on the total".
#:
#: IT HAD TO BE ITS OWN SOUND. Taking a round was sounding SAVED, which tells the operator
#: the record is written when it is not -- and the moment they hear it is the moment they
#: tip the tray into a bottle, so a mistaken "filed" is acted on irreversibly before anybody
#: looks at the screen. One note against two is the same rhythm difference that separates
#: SAVED from PROBLEM, and it reads as the smaller event that it is.
ROUND = ((1318, 80),)


def _play(notes):
    try:
        import winsound

        for freq, ms in notes:
            if freq <= 0:
                threading.Event().wait(ms / 1000)
                continue
            winsound.Beep(int(freq), int(ms))
    except Exception:                                           # noqa: BLE001
        pass


def play(notes) -> None:
    """Sound a sequence, on a thread of its own, or do nothing at all."""
    if not enabled or sys.platform != "win32":
        return
    threading.Thread(target=_play, args=(notes,), daemon=True).start()


def saved() -> None:
    play(SAVED)


def round_taken() -> None:
    play(ROUND)


def problem() -> None:
    play(PROBLEM)
