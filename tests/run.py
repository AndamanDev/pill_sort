# -*- coding: utf-8 -*-
"""Run every suite in this folder and say which ones are unhappy.

    python tests/run.py                 # all of them
    python tests/run.py zero ask        # only the ones whose name contains these
    python tests/run.py -v zero         # and print everything that suite printed

WHY A RUNNER AND NOT pytest. Every file here is a script that drives the real thing --
a real Window with a fake camera, a real Screen composing real frames -- prints a line
per check and exits non-zero if any failed. That shape came first because the failures
worth catching are visual and sequential ("the button was live at this moment"), and a
runner that just reports the last line of each is enough to know where to look. Adding a
framework now would mean rewriting sixteen working files to gain a summary they already
print themselves.

EACH SUITE GETS ITS OWN PROCESS, and that is not tidiness either. They pass an offscreen
QApplication around, monkey-patch module globals (RECORDS, SETTINGS) and lay the phone
canvas out for one screen size; sharing an interpreter would let one suite's leftovers
decide another's result. A process each is the only isolation that costs nothing to
maintain.

AND EACH GETS A TIMEOUT. A modal dialog with nobody to answer it waits for ever: the
question added to the target field hung two suites for as long as anyone let them run,
and a suite that hangs looks exactly like a suite that is being thorough. TIMEOUT turns
that into a failure with a name.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# PRINTING A FAILURE MUST NOT ITSELF FAIL, and this runner's first real failure was its
# own: the console here is cp874, the suites print Thai, and a byte that did not survive
# whatever encoding it went through arrives as U+FFFD, which cp874 cannot encode either.
# The traceback replaced the one thing anybody runs this for -- the name of the suite that
# broke -- with a stack trace about codecs.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):    # not a real console, or already wrapped
        pass

#: Long enough for the slowest suite (test_align builds a real ONNX session and pushes
#: frames through it), short enough that a wedged modal is noticed within a coffee.
TIMEOUT = 420

#: Not a test: it drives the phone bridge end to end and exits non-zero if anything in it
#: throws. Run last, as a smoke check that the pieces still fit together.
SIMULATOR = ("-m", "model3.tools.simulate_phone")


def suites(patterns):
    found = sorted(glob.glob(os.path.join(HERE, "test_*.py")))
    if not patterns:
        return found
    return [p for p in found
            if any(w.lower() in os.path.basename(p).lower() for w in patterns)]


def run(cmd, label, verbose):
    t0 = time.time()
    try:
        done = subprocess.run(cmd, cwd=ROOT, capture_output=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"  HUNG  {label:22} no answer in {TIMEOUT}s -- a modal with nobody to "
              f"close it?")
        return False
    out = (done.stdout or b"").decode("utf-8", "replace")
    err = (done.stderr or b"").decode("utf-8", "replace")
    ok = done.returncode == 0
    last = ([ln for ln in out.strip().splitlines() if ln.strip()] or ["(no output)"])[-1]
    print(f"  {'ok  ' if ok else 'FAIL'}  {label:22} {time.time() - t0:5.1f}s  {last}")
    # THE OUTPUT OF A FAILURE IS PRINTED HERE, not left for a second run to find. Half of
    # these suites take half a minute to reach the state that failed.
    if verbose or not ok:
        for line in (out + err).strip().splitlines():
            print(f"        {line}")
    return ok


def main():
    args = [a for a in sys.argv[1:] if a not in ("-v", "--verbose")]
    verbose = len(args) != len(sys.argv[1:])
    chosen = suites(args)
    if not chosen:
        print(f"nothing matches {args}")
        return 1

    print(f"{len(chosen)} suites, {sys.executable}")
    bad = [os.path.basename(p)[:-3]
           for p in chosen
           if not run([sys.executable, p], os.path.basename(p)[:-3], verbose)]

    if not args:                        # the simulator only belongs in a full run
        if not run([sys.executable, *SIMULATOR], "simulate_phone", verbose):
            bad.append("simulate_phone")

    print()
    print("FAILURES:", ", ".join(bad) if bad else "none")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
