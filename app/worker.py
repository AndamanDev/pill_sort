"""Two background threads, so the picture never waits on the model.

`model2/count.py` reads, infers and draws in one loop. That pins the VIDEO to the model's
rate -- measured at 42 ms a frame for pillcount-det-v3, so about 24 fps at best and 8 fps
in practice once drawing and the camera's own latency are added. Eight fps of live video
reads as a stutter even when the counting is perfect, and the operator blames the count.

Separated here, the way model1/ui_qt does it:

    Capture   reads the camera as fast as it gives frames and keeps only the newest.
              Nothing waits on it.
    Infer     takes whichever frame is newest, runs the model, publishes boxes. Frames
              that arrived while it was busy are DROPPED, never queued -- a queue would
              only add latency, and a tray from 300 ms ago is not worth counting.

The window repaints at 60 fps from the newest frame plus the newest boxes. The markers
trail the video by about one inference, which is invisible on a tray nobody is touching
and is the right trade while one is being positioned: the picture stays live.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

#: Failed reads before the camera is opened again. Each one costs the 10ms sleep beside
#: it, so this is about two seconds of silence -- long enough not to fire on a dropped
#: frame, short enough that somebody watching the screen has not yet walked away.
REOPEN_TRIES = 200

#: Size the staleness thumbnails are compared at -- small enough to be free, large
#: enough that a tray moving across the bench changes it.
THUMB = (64, 48)


def thumb(frame):
    return cv2.cvtColor(cv2.resize(frame, THUMB, interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY)


def exposure_arg(text):
    """"auto", or a number. An argparse type, so a typo fails at the command line."""
    if str(text).lower() == "auto":
        return "auto"
    return float(text)


def set_exposure(cap, value) -> bool:
    """Set exposure. `value` is a number, or "auto". Returns whether the camera took it.

    AUTOMATIC IS THE DEFAULT, and it used to be a fixed -3. That number came from a real
    measurement -- -3 gave brightness 97.6 and the steadiest count of any setting tried --
    but what it actually recorded was how much light -3 needed, in the room it was measured
    in. The same camera, the same spot, one evening later:

        exposure -3    8.0 fps   brightness   4.5    <- the same setting, unusable
        AUTOMATIC     20.0 fps   brightness  53.9

    And because this scale is SHUTTER TIME, the brighter manual steps cost frame rate: -1
    is about half a second a frame. An evening was spent blaming the counting app for lag
    that was this setting. Automatic can raise GAIN instead, which costs no time, so it is
    brighter AND faster here, and it keeps tracking the room when somebody turns a lamp on.

    Pass a number when a FIXED exposure is the point, and check the frame rate when you do.
    set-camera-exposure.py in the repository root measures both.
    """
    if value is None:
        return False
    if isinstance(value, str) and value.lower() == "auto":
        # 1 is automatic on this camera through DSHOW; 0.75 hands control to the EXPOSURE
        # property. Measured on this backend, not taken from the 0.25/0.75 convention.
        return bool(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1))
    if value == 0:
        return False
    # Manual first or DSHOW ignores the exposure write.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    return bool(cap.set(cv2.CAP_PROP_EXPOSURE, float(value)))


class Capture(threading.Thread):
    """Owns the camera. `latest()` gives the newest frame and its sequence number."""

    daemon = True

    def __init__(self, source=0, exposure="auto"):
        super().__init__(name="capture")
        self.source = source
        self.exposure = exposure
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = False
        self.fps = 0.0
        #: When the last frame actually arrived. THE WINDOW ASKS THIS BEFORE IT TRUSTS THE
        #: PICTURE. Everything above keeps the newest frame and nothing above it expires:
        #: pull the USB lead and read() simply returns False for ever, while the last frame
        #: sits on screen looking live and the number beside it looks current. In a room
        #: where somebody then presses save, that is a count of the wrong tray filed under
        #: the right time.
        self.last_frame_at = 0.0
        self.error = ""                     # what went wrong, for the window to show

    def open(self) -> bool:
        cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            return False
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:                                       # noqa: BLE001
            pass
        set_exposure(cap, self.exposure)
        self._cap = cap
        for _ in range(8):                                      # let the camera settle
            cap.read()
        self.last_frame_at = time.perf_counter()
        self.error = ""
        return True

    def latest(self):
        with self._lock:
            return self._frame, self._seq

    def stale(self) -> float:
        """Seconds since the last frame arrived. 0 before the camera has ever opened."""
        if not self.last_frame_at:
            return 0.0
        return time.perf_counter() - self.last_frame_at

    def run(self):
        n, t0 = 0, time.perf_counter()
        fails = 0
        while not self._stop:
            ok, frame = self._cap.read()
            if not ok:
                # A FAILED READ IS NOT FATAL AND NOT IGNORED. USB cameras drop a frame now
                # and then, and Windows re-enumerates a device that was jogged in its
                # socket; both recover. What does not recover on its own is a handle onto a
                # camera that has gone, so after a couple of seconds of nothing the device
                # is opened again from scratch -- which is what un-plugging and re-plugging
                # does by hand, and it is the fix for nine failures out of ten.
                fails += 1
                self.error = "กล้องไม่ส่งภาพ"
                time.sleep(0.01)
                if fails % REOPEN_TRIES == 0:
                    self._reopen()
                continue
            fails = 0
            self.error = ""
            with self._lock:
                self._frame = frame
                self._seq += 1
                self.last_frame_at = time.perf_counter()
            n += 1
            if n >= 30:
                now = time.perf_counter()
                self.fps = n / max(now - t0, 1e-6)
                n, t0 = 0, now
        self._cap.release()

    def _reopen(self):
        """Throw the handle away and ask Windows for the camera again."""
        try:
            self._cap.release()
        except Exception:                                       # noqa: BLE001
            pass
        try:
            if not self.open():
                self.error = "เปิดกล้องไม่ได้"
        except Exception as exc:                                # noqa: BLE001
            self.error = f"เปิดกล้องไม่ได้ {exc}"

    def stop(self):
        self._stop = True


class Infer(threading.Thread):
    """Runs the detector on the newest frame. `result()` gives boxes, count and timing."""

    daemon = True

    def __init__(self, capture, model, conf=0.45, iou=0.4, imgsz=640):
        super().__init__(name="infer")
        self.capture = capture
        self.model = model
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.roi = None                     # list of (x, y), set from the window
        self._out = (np.zeros((0, 4)), np.zeros((0,)), 0, 0.0)
        self._seq = 0
        self._shot = None                   # frame the boxes came from, for saving
        self._lock = threading.Lock()
        self._stop = False
        #: The last thing predict() raised, or "". A model that throws once -- a frame of
        #: the wrong shape, memory that ran out -- used to take this thread down with it,
        #: and the window went on showing the last number it had been given, for ever,
        #: with nothing anywhere to say the counting had stopped.
        self.error = ""

    def result(self):
        """(boxes, confidences, count_in_roi, ms), and the sequence they belong to."""
        with self._lock:
            return self._out, self._seq

    def snapshot(self):
        """The exact frame and boxes behind the number on screen, for the JSON record."""
        with self._lock:
            return self._shot, self._out

    def inside(self, box) -> bool:
        if not self.roi:
            return True
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return cv2.pointPolygonTest(np.array(self.roi, np.int32),
                                    (float(cx), float(cy)), False) >= 0

    def run(self):
        seen = -1
        while not self._stop:
            frame, seq = self.capture.latest()
            if frame is None or seq == seen:
                time.sleep(0.005)
                continue
            seen = seq
            t0 = time.perf_counter()
            try:
                r = self.model.predict(frame, conf=self.conf, iou=self.iou,
                                       imgsz=self.imgsz, max_det=1000, verbose=False)[0]
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
            except Exception as exc:                            # noqa: BLE001
                # Keep the thread alive and hand the reason upwards. Sleeping a moment
                # stops a permanent fault -- a model that is gone, a GPU that is out --
                # from becoming a hot loop that eats the machine as well.
                self.error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.2)
                continue
            self.error = ""
            ms = (time.perf_counter() - t0) * 1000
            count = sum(1 for b in boxes if self.inside(b))
            with self._lock:
                self._out = (boxes, confs, count, ms)
                self._shot = frame
                self._seq += 1

    def stop(self):
        self._stop = True
