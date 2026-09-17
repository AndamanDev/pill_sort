"""The counter as a web page: the PC does the seeing, anything with a browser is a screen.

    D:\\pill-counter-lite\\pillcount-v12\\.venv-train\\Scripts\\python.exe -m web.server

Then open the address it prints -- on this PC, on a phone, on a tablet, on all three at
once. No install on the device, no APK, no store.

WHY THIS SHAPE. The model costs about 115 ms of CPU a frame on this machine and that is
the whole budget; a phone doing its own inference would be slower still, and a phone
running a UI toolkit that re-encodes every frame would steal from the very CPU the count
depends on. Here the PC keeps the camera and the model, and the device gets finished
JPEGs. The phone does nothing but display, which is the one job it can do without
competing.

WHAT IT COSTS, honestly: the phone is useless on its own. If the tray has to be counted
somewhere the PC is not, this is the wrong shape and model1/android -- which really does
run the model on the handset -- is the right one.

MJPEG rather than WebRTC or a websocket of frames, because <img src="/stream"> is one
line of HTML that every browser has supported for twenty years, and the alternative is a
dependency and a negotiation to deliver the same pixels.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from app import MODEL, RECORDS                                  # noqa: E402
from app.worker import Capture, Infer                           # noqa: E402

MARK = (0, 0, 0)
MARK_RIM = (255, 255, 255)
ROI_LINE = (90, 220, 110)

STATE = {"target": 0, "note": ""}
LOCK = threading.Lock()


def overlay(frame, infer, roi):
    shown = frame.copy()
    (boxes, _confs, count, _ms), _ = infer.result()
    for b in boxes:
        if not infer.inside(b):
            continue
        cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
        cv2.circle(shown, (cx, cy), 7, MARK_RIM, -1, cv2.LINE_AA)
        cv2.circle(shown, (cx, cy), 5, MARK, -1, cv2.LINE_AA)
    if roi and len(roi) >= 3:
        poly = np.array(roi, np.int32).reshape(-1, 1, 2)
        wash = shown.copy()
        cv2.fillPoly(wash, [poly], ROI_LINE)
        cv2.addWeighted(wash, 0.16, shown, 0.84, 0, shown)
        cv2.polylines(shown, [poly], True, ROI_LINE, 2, cv2.LINE_AA)
    return shown, count


class Handler(BaseHTTPRequestHandler):
    capture: Capture
    infer: Infer

    def log_message(self, *a):                                  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # ------------------------------------------------------------------------- GET
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            with open(os.path.join(HERE, "index.html"), "rb") as fh:
                self._send(200, fh.read(), "text/html; charset=utf-8")
        elif path == "/stream":
            self._stream()
        elif path == "/state":
            frame, _ = self.capture.latest()
            (_b, _c, count, ms), _ = self.infer.result()
            with LOCK:
                target, note = STATE["target"], STATE["note"]
            self._send(200, json.dumps({
                "count": int(count), "target": int(target),
                "fps": round(self.capture.fps, 1),
                "seconds": round(ms / 1000, 1),
                "roi": bool(self.infer.roi), "note": note,
                "size": [frame.shape[1], frame.shape[0]] if frame is not None else [0, 0],
            }))
        else:
            self._send(404, "{}")

    def _stream(self):
        """One long response, a JPEG at a time. The browser keeps the socket open."""
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                frame, _ = self.capture.latest()
                if frame is None:
                    time.sleep(0.05)
                    continue
                shown, _ = overlay(frame, self.infer, self.infer.roi)
                ok, buf = cv2.imencode(".jpg", shown,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if not ok:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: " + str(len(buf)).encode() +
                                 b"\r\n\r\n" + buf.tobytes() + b"\r\n")
                # 15 fps is plenty for a tray and leaves the CPU to the model, which is
                # the thing the number depends on.
                time.sleep(1 / 15)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ------------------------------------------------------------------------ POST
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or "{}")
        path = self.path.split("?")[0]
        if path == "/target":
            with LOCK:
                STATE["target"] = max(0, int(body.get("value", 0)))
            self._send(200, "{}")
        elif path == "/roi":
            pts = body.get("points")
            self.infer.roi = [(int(x), int(y)) for x, y in pts] if pts else None
            with LOCK:
                STATE["note"] = ""
            self._send(200, "{}")
        elif path == "/save":
            self._send(200, json.dumps(self._save()))
        else:
            self._send(404, "{}")

    def _save(self):
        frame, (boxes, confs, count, ms) = self.infer.snapshot()
        os.makedirs(RECORDS, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        with LOCK:
            target = STATE["target"]
        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "count": int(count), "target": int(target),
            "difference": int(count) - int(target) if target else None,
            "conf": self.infer.conf, "iou": self.infer.iou, "imgsz": self.infer.imgsz,
            "roi": [[int(x), int(y)] for x, y in self.infer.roi] if self.infer.roi else None,
            "model_ms": round(float(ms), 1),
            "boxes": [[round(float(v), 1) for v in b] for b in boxes],
            "confidences": [round(float(c), 3) for c in confs],
        }
        with open(os.path.join(RECORDS, f"count_{stamp}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
        if frame is not None:
            cv2.imwrite(os.path.join(RECORDS, f"count_{stamp}.jpg"), frame)
        with LOCK:
            STATE["note"] = f"บันทึกแล้ว {count} เม็ด"
        return {"count": int(count), "stamp": stamp}


def lan_address():
    """The address a phone on the same wifi can reach, not 127.0.0.1."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))                              # no packet is sent
        return s.getsockname()[0]
    except Exception:                                           # noqa: BLE001
        return "127.0.0.1"
    finally:
        s.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--exposure", type=float, default=-3.0)
    a = ap.parse_args(argv)

    import torch
    from ultralytics import YOLO

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    if not os.path.isfile(a.model):
        raise SystemExit(f"ไม่พบโมเดล: {a.model}")

    capture = Capture(a.camera, exposure=a.exposure)
    if not capture.open():
        raise SystemExit(f"เปิดกล้องไม่ได้: {a.camera}")
    infer = Infer(capture, YOLO(a.model), conf=a.conf, iou=a.iou, imgsz=a.imgsz)
    capture.start()
    infer.start()

    Handler.capture, Handler.infer = capture, infer
    server = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    ip = lan_address()
    print(f"model    : {os.path.basename(a.model)}")
    print(f"เปิดที่   : http://localhost:{a.port}")
    print(f"มือถือ   : http://{ip}:{a.port}      (ต้องอยู่ wifi เดียวกัน)")
    print("Ctrl+C เพื่อหยุด")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        infer.stop()
        capture.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
