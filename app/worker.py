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

#: The slider's range. 0 is the whole picture, ZOOM_MAX is ZOOM_X times closer.
ZOOM_MIN, ZOOM_MAX = 0, 100

#: THE ZOOM IS A CROP, DONE HERE, and the camera's own zoom is held at its widest. The
#: bench's UGREEN takes CAP_PROP_ZOOM only BEFORE it starts streaming: set while frames
#: are flowing, it answers True, reads back the new value, and the picture does not move
#: (measured: mean pixel difference 2.8, which is sensor noise). Driving it live would mean
#: re-opening the camera on every step of the slider -- seconds of black each time.
#:
#: SO THE CAMERA IS ASKED FOR MORE PIXELS THAN THE MODEL USES, and the crop spends them.
#: The model reads 640 across whatever it is handed; a 1920-wide frame cropped to a third
#: is still 640 real pixels, so up to ZOOM_X the model sees detail the sensor actually
#: recorded, not an enlargement. Measured on this camera through DSHOW: 1920x1080 MJPEG
#: holds the same 20 fps as 640x480 did, and 4K drops to 15 for detail the model would
#: throw away. MJPEG must be asked for AFTER the size, or DSHOW hands back raw YUY2 at
#: 5 fps.
CAPTURE_W, CAPTURE_H = 1920, 1080
#: What leaves this thread, at every zoom: EXACTLY WHAT 640x480 USED TO DELIVER AT 1x.
#:
#: The first version of this sent the whole 16:9 frame out at 1x, and the count went
#: wrong at the widest setting. Measured: the camera's own 640x480 is the middle 1440x1080
#: of its 1080p picture, scaled down -- so the model had always seen a pill at 640/1440 of
#: its 1080p size, and the wide frame showed it at 640/1920, a quarter smaller than every
#: tray it was tuned on. So 1x is now that same 4:3 middle at that same 640x480, and the
#: model, the saved regions and the records get the picture they got before zoom existed.
OUT_W, OUT_H = 640, 480
#: 1440 real pixels across the 4:3 middle, 640 out: up to 2.25x every output pixel is a
#: pixel the sensor recorded. Past that it would be an enlargement, so the slider stops.
ZOOM_X = 2.25


def zoom_factor(value) -> float:
    """Slider value -> how many times closer. Straight line from 1x to ZOOM_X."""
    v = max(ZOOM_MIN, min(ZOOM_MAX, value))
    return 1.0 + (ZOOM_X - 1.0) * (v - ZOOM_MIN) / (ZOOM_MAX - ZOOM_MIN)


def zoomed(frame, value):
    """The middle 4:3 of the frame, then the middle 1/f of that, at OUT_W x OUT_H."""
    h, w = frame.shape[:2]
    # The 4:3 middle first: on a 16:9 frame that trims the sides, on a 4:3 one nothing.
    bw = min(w, int(round(h * OUT_W / OUT_H)))
    bh = min(h, int(round(bw * OUT_H / OUT_W)))
    f = zoom_factor(value or 0)
    cw, ch = int(round(bw / f)), int(round(bh / f))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    crop = frame[y0:y0 + ch, x0:x0 + cw]
    if (cw, ch) == (OUT_W, OUT_H):
        return crop
    interp = cv2.INTER_AREA if cw > OUT_W else cv2.INTER_LINEAR
    return cv2.resize(crop, (OUT_W, OUT_H), interpolation=interp)

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
        #: When the camera was FIRST opened, and never moved afterwards. It gives `stale`
        #: something to count from before the first frame, so a device that opens and then
        #: says nothing ages like any other dead camera instead of reading as brand new.
        self.opened_at = 0.0
        self.error = ""                     # what went wrong, for the window to show
        #: ZOOM_MIN..ZOOM_MAX; see `zoomed`. APPLIED IN run(), BEFORE THE FRAME IS
        #: PUBLISHED: what latest() hands out is already the zoomed picture, so the model,
        #: the marks, the region and the saved JPEG all see the same thing and nothing
        #: downstream has to know zoom exists.
        self.zoom = ZOOM_MIN
        #: Always true now that the crop is done here; kept so the window can still ask.
        self.zoom_ok = True

    def set_zoom(self, value):
        """Ask for a zoom. One int written, read by run() at its next frame."""
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, int(value)))

    def open(self) -> bool:
        cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # RELEASED, NOT DROPPED. A VideoCapture that failed to open still holds a
            # DSHOW graph, and this runs every couple of seconds for as long as the camera
            # is missing -- an afternoon of that is thousands of them, and the reopen that
            # would have worked then fails for want of a handle. Measured against a fake
            # device: six leaked in the first 1.2 seconds.
            cap.release()
            cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            return False
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:                                       # noqa: BLE001
            pass
        # Size, THEN MJPEG -- see CAPTURE_W. A camera that cannot do this size gives the
        # nearest it has, and the crop still works on whatever arrives.
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:                                       # noqa: BLE001
            pass
        set_exposure(cap, self.exposure)
        # The camera's OWN zoom to its widest, before the first read -- the only moment it
        # listens. It is remembered by the device across power cycles, and a camera left
        # zoomed in by some other program would otherwise have the crop stacked on top.
        try:
            cap.set(cv2.CAP_PROP_ZOOM, 0.0)
        except Exception:                                       # noqa: BLE001
            pass
        self._cap = cap
        for _ in range(8):                                      # let the camera settle
            cap.read()
        # NOT `last_frame_at`, AND THAT USED TO BE A LIE THE WHOLE SCREEN BELIEVED. This
        # line stamped the frame clock on the strength of the device OPENING, which is not
        # the same thing as a picture arriving and on a sick USB camera is routinely not
        # even close: the handle comes back, the reads return nothing, and `stale` read
        # 0.1 seconds. So the window called a dead camera live, took the black pane down,
        # put the last frozen frame back up and RE-ENABLED THE SAVE BUTTON over it. The
        # clock now only ever moves when a frame is actually read, in run().
        if not self.opened_at:
            self.opened_at = time.perf_counter()
        self.error = ""
        return True

    def latest(self):
        with self._lock:
            return self._frame, self._seq

    def stale(self) -> float:
        """Seconds since a frame last ARRIVED. 0 before the camera has ever opened.

        MEASURED FROM A FRAME, NEVER FROM AN OPEN -- see the note in open(). Before the
        first frame it counts from the first open instead, so a camera that never delivers
        one is called dead after the same STALE_S as one that stops later, rather than
        sitting at zero for ever looking healthy.
        """
        ref = self.last_frame_at or self.opened_at
        if not ref:
            return 0.0
        return time.perf_counter() - ref

    def run(self):
        n, t0 = 0, time.perf_counter()
        fails = 0
        while not self._stop:
            # `self._cap` is None between a release and the open that failed to replace
            # it. Reading None would take this thread down with an AttributeError, and a
            # dead capture thread is a camera that never comes back at all.
            cap = self._cap
            ok, frame = cap.read() if cap is not None else (False, None)
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
                if fails % REOPEN_TRIES == 0 and self._reopen():
                    fails = 0               # the next attempt is REOPEN_TRIES from THIS one
                continue
            fails = 0
            self.error = ""
            frame = zoomed(frame, self.zoom)
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

    def _reopen(self) -> bool:
        """Throw the handle away and ask Windows for the camera again. Did it come back?

        `self._cap` IS CLEARED BEFORE THE ATTEMPT, because open() only assigns it on
        success -- so a failed reopen used to leave the released handle in place and run()
        went on reading from it. Harmless by luck rather than by design, and exactly the
        kind of thing that stops being harmless when a driver decides a read on a released
        graph is worth an exception.
        """
        try:
            self._cap.release()
        except Exception:                                       # noqa: BLE001
            pass
        self._cap = None
        try:
            if self.open():
                return True
            self.error = "เปิดกล้องไม่ได้"
        except Exception as exc:                                # noqa: BLE001
            self.error = f"เปิดกล้องไม่ได้ {exc}"
        return False

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
        #: How the size is written down -- "320x416" for an export, the phone's spelling,
        #: so a record from either machine says the same thing about the same model.
        self.imgsz_text = (f"{imgsz[0]}x{imgsz[1]}" if isinstance(imgsz, (tuple, list))
                           else str(imgsz))
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
