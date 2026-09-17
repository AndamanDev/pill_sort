"""Saved counts on the phone: the same file format the bench writes, in the same shapes.

ONE FORMAT ACROSS BOTH. A record written here is count_<stamp>.json with the frame beside
it as count_<stamp>.jpg, field for field what app/window.py writes on the PC -- so a CSV
exported from a phone and one exported from the bench open in the same spreadsheet with
the same columns, and a record pulled off a handset can be read by the desktop window
without a converter in between.

WHERE IT GOES is decided by Android, not here: the activity passes the directory it is
allowed to write to (getExternalFilesDir, which needs no permission and is removed when
the app is uninstalled). This module only ever joins paths onto what it was handed.

NOTHING HERE IMPORTS Qt, torch, or anything the phone does not have. It is plain json,
csv, os -- which also means the desktop tests can run every line of it.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import time

CSV_FIELDS = ("time", "count", "target", "difference", "detections",
              "conf", "iou", "imgsz", "model_ms", "roi", "json", "image")


def free_stamp(folder) -> str:
    """A name no record already has -- see app/window.py for the double-save this stops."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate, n = stamp, 1
    while os.path.exists(os.path.join(folder, f"count_{candidate}.json")):
        n += 1
        candidate = f"{stamp}-{n}"
    return candidate


def roi_path(folder):
    return os.path.join(folder, "roi.json")


def load_roi(folder, size):
    """The counting frame from last time, if it can still mean what it meant.

    THE SAME RULE AS THE BENCH, word for word from app/window.py: a frame is raw pixel
    coordinates and is valid only at the resolution it was drawn at. Restoring a 640x480
    rectangle onto a 1280x720 picture would cover a quarter of the tray, and the count
    would be wrong while looking perfectly reasonable, which is the worst kind of wrong.
    So the picture's size is stored beside it and a mismatch drops the frame rather than
    scaling it -- a phone that was handed a different camera resolution by CameraX has to
    be told to draw the frame again, not quietly counted a corner of the tray.
    """
    path = roi_path(folder)
    if not os.path.isfile(path):
        return None, ""
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        pts = [(int(x), int(y)) for x, y in d["roi"]]
        if len(pts) < 3:
            return None, ""
        if tuple(d.get("frame", ())) != tuple(size):
            return None, "ขนาดภาพเปลี่ยน ต้องกำหนดกรอบใหม่"
        return pts, ""
    except Exception:                                               # noqa: BLE001
        return None, ""


def save_roi(folder, size, pts):
    """Write it, or delete it when the frame has been cleared."""
    path = roi_path(folder)
    if not pts:
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass
        return
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"frame": list(size),
                   "saved": time.strftime("%Y-%m-%d %H:%M"),
                   "roi": [[int(x), int(y)] for x, y in pts]}, fh,
                  ensure_ascii=False, indent=2)


def save(folder, frame, boxes, confs, count, target, ms, roi, conf, iou, imgsz,
         write_jpeg=None):
    """Write one record. Returns its stamp.

    `write_jpeg` is how the frame is encoded, injected because the phone has cv2 and the
    tests would rather not: the caller passes cv2.imwrite and a test passes a stub.
    """
    os.makedirs(folder, exist_ok=True)
    stamp = free_stamp(folder)
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "count": int(count),
        "target": int(target),
        "difference": int(count) - int(target) if target else None,
        "conf": conf, "iou": iou, "imgsz": imgsz,
        "roi": [[int(x), int(y)] for x, y in roi] if roi else None,
        "model_ms": round(float(ms), 1),
        "boxes": [[round(float(v), 1) for v in b] for b in boxes],
        "confidences": [round(float(c), 3) for c in confs],
        "source": "phone",
    }
    with open(os.path.join(folder, f"count_{stamp}.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    if frame is not None and write_jpeg is not None:
        write_jpeg(os.path.join(folder, f"count_{stamp}.jpg"), frame)
    return stamp


def load(folder):
    """Every readable record, newest first. A half-written one is skipped, not raised."""
    rows = []
    for path in glob.glob(os.path.join(folder, "count_*.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
        except Exception:                                           # noqa: BLE001
            continue
        image = os.path.splitext(path)[0] + ".jpg"
        rec["json"] = path
        rec["image"] = image if os.path.isfile(image) else ""
        rec["stamp"] = os.path.basename(path)[6:-5]
        rows.append(rec)
    rows.sort(key=lambda r: r.get("stamp", ""), reverse=True)
    return rows


def verdict(rec):
    """(kind, words) -- the same three words the bench and the records window use."""
    target = int(rec.get("target") or 0)
    if not target:
        return "none", "ไม่ได้ตั้ง"
    diff = int(rec.get("count") or 0) - target
    if diff == 0:
        return "ok", "ครบ"
    return ("over", f"เกิน {diff}") if diff > 0 else ("short", f"ขาด {-diff}")


def when(rec):
    """(year, month, day, hh, mm) from the file name, or None if it cannot be read."""
    stamp = rec.get("stamp", "")[:15]
    if len(stamp) < 15 or stamp[8] != "-":
        return None
    try:
        return (int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]),
                int(stamp[9:11]), int(stamp[11:13]))
    except ValueError:
        return None


def export_csv(rows, path) -> int:
    """utf-8-sig, because Excel on a Thai Windows reads plain UTF-8 as mojibake."""
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in rows:
            roi = rec.get("roi")
            writer.writerow({
                "time": rec.get("time", ""),
                "count": rec.get("count", ""),
                "target": rec.get("target", ""),
                "difference": "" if rec.get("difference") is None else rec["difference"],
                "detections": len(rec.get("boxes") or []),
                "conf": rec.get("conf", ""), "iou": rec.get("iou", ""),
                "imgsz": rec.get("imgsz", ""), "model_ms": rec.get("model_ms", ""),
                "roi": " ".join(f"{int(x)},{int(y)}" for x, y in roi) if roi else "",
                "json": os.path.basename(rec.get("json", "")),
                "image": os.path.basename(rec.get("image", "")),
            })
    return len(rows)


def delete(rows):
    """Remove records, picture first. Returns (gone, [names that would not go]).

    THE PICTURE BEFORE THE JSON, for the reason app/records_view.py gives at length: the
    JSON is what makes a record visible, so losing it first would strand a picture that
    nothing can see or reach again. There is no Recycle Bin on Android to fall back on,
    which makes the ordering matter more here rather than less.
    """
    gone, failed = 0, []
    for rec in rows:
        paths = [p for p in (rec.get("image"), rec.get("json")) if p and os.path.isfile(p)]
        stuck = False
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                failed.append(os.path.basename(path))
                stuck = True
                break
        if not stuck:
            gone += 1
    return gone, failed
