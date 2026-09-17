"""The v3 detector without PyTorch: letterbox in, boxes out, for a phone.

WHY THIS FILE EXISTS. app/ counts with ultralytics and torch, which is 2GB of wheels that
do not run on Android at all. The phone runs the same weights exported to ONNX, through
ONNX Runtime, and everything around the forward pass -- the letterbox, the thresholds, the
scaling back, the counting inside the tray -- has to be done here instead. If any of it
drifts from what app/ does, the phone and the bench disagree about how many pills are in
the same tray, which is worse than having no phone at all.

SO THE RAW HEAD, NOT AN EXPORT WITH NMS BAKED IN. The phone runs

    pillcount-det-v3-480x640.onnx    input [1,3,480,640], output [1,5,8400], no NMS

and de-duplicates here with cv2.dnn.NMSBoxes. An export with the NMS inside the graph is
less work and the wrong choice: the conf and iou it de-duplicates with are frozen at export
time, while the bench runs 0.45 and 0.40 -- numbers measured on this tray under this lamp
and changed from the window. Doing it out here means the phone takes the same two numbers.

MEASURED AGAINST THE BENCH, on four frames including a tray with sixty-one pills and one
with a single pill at the very edge: 13/13/61/1 both ways, and every box identical to
within 0.0 px. That is the bar this file has to keep clearing.

NO TORCH, NO ULTRALYTICS, NOTHING ABOVE OPENCV 4.5.1. Chaquopy's package index stops
there, so this file uses numpy and the parts of cv2 that have not moved in a decade.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:                                             # pragma: no cover
    cv2 = None

#: The shape the bench's export takes, as (height, width). NOT 640x640, AND THAT IS THE
#: WHOLE POINT. ultralytics does rectangular inference: a 640x480 camera frame is padded to
#: the next multiple of 32, which for 480 is 480, so the model on the PC sees a 480x640
#: tensor and no grey bars at all. Squaring it instead -- 80px of padding top and bottom --
#: is a different picture, and the detector says so: on tray_sparse.jpg the square input
#: found two pills at conf 0.47 and 0.51 that the bench does not see, in an image where the
#: right answer is one. Same weights, same thresholds, two extra pills. So the phone gets an
#: export of the same shape the bench runs, and the two agree exactly.
INPUT_HW = (480, 640)

#: The grey the letterbox pads with -- the value ultralytics uses, so a tray that reaches
#: the edge of the frame sees the same border here as it does on the bench.
PAD = 114


def letterbox(bgr, hw=INPUT_HW):
    """Fit the frame into the graph's input without distorting it. (boxed, gain, pads).

    Scaled by one factor and padded, never squashed: stretching would turn every pill from
    a circle into an ellipse, which is the shape cue the detector was trained on.

    A phone camera that gives 4:3 frames lands here with no padding at all, the same as the
    bench's 640x480 -- which is exactly why the export is 480x640.
    """
    want_h, want_w = hw
    h, w = bgr.shape[:2]
    gain = min(want_h / h, want_w / w)
    new_w, new_h = int(round(w * gain)), int(round(h * gain))
    pad_x, pad_y = (want_w - new_w) / 2, (want_h - new_h) / 2
    if (w, h) != (new_w, new_h):
        interp = cv2.INTER_LINEAR if gain > 1 else cv2.INTER_AREA
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=interp)
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    bgr = cv2.copyMakeBorder(bgr, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=(PAD, PAD, PAD))
    return bgr, gain, (left, top)


def blob(letterboxed):
    """BGR uint8 -> the NCHW float32 tensor the graph wants, in [0,1], RGB order.

    cv2.dnn.blobFromImage rather than numpy, and it is not a style choice. The numpy
    version does the swap, the transpose, the copy and the divide as four passes over
    900,000 values in the interpreter's own memory; blobFromImage is one call into
    OpenCV's C++ that does all four at once. It is also the one part of cv2.dnn that
    still works in the 4.5.1 the phone has -- it is image arithmetic, not a model
    importer -- which is the same reason model1 uses it there.
    """
    return cv2.dnn.blobFromImage(letterboxed, 1 / 255.0, swapRB=True, crop=False)


def nms(boxes, scores, iou):
    """Greedy non-maximum suppression, in numpy. Returns the indices that survive.

    WRITTEN OUT RATHER THAN CALLED, and the phone is why. cv2.dnn.NMSBoxes exists in every
    OpenCV, but the one Chaquopy ships is 4.5.1, and there its binding takes vector<Rect> --
    INTEGER boxes. Handing it the float boxes the model produces raises, the bridge caught
    it, and the handset showed "โมเดลผิดพลาด" with no count at all while the same code ran
    perfectly on a desktop with OpenCV 4.10. Twenty lines of numpy has no such version to
    disagree with, costs about a millisecond for the few hundred boxes that get this far,
    and means the phone and the bench de-duplicate detections by exactly the same rule.

    `boxes` is xywh, which is the shape the caller already has.
    """
    if not len(boxes):
        return np.zeros((0,), np.int32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores)[::-1]

    keep = []
    while order.size:
        best = order[0]
        keep.append(best)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[best], x1[rest])
        yy1 = np.maximum(y1[best], y1[rest])
        xx2 = np.minimum(x2[best], x2[rest])
        yy2 = np.minimum(y2[best], y2[rest])
        overlap = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[best] + areas[rest] - overlap
        # A zero-area box would divide by zero rather than simply lose: keep the guard
        # cheap and let it fall out as "no overlap".
        ratio = np.where(union > 0, overlap / np.maximum(union, 1e-9), 0.0)
        order = rest[ratio <= iou]
    return np.array(keep, np.int32)


def decode(raw, gain, pads, conf=0.45, iou=0.4, frame_shape=None):
    """[1,5,8400] -> (boxes in FRAME pixels, confidences), thresholded and de-duplicated.

    The 8400 columns are every anchor the head predicts, most of them empty; rows 0-3 are
    the box as centre-x, centre-y, width, height in letterboxed pixels, and row 4 is the
    confidence. Filtering before the NMS matters: passing eight thousand boxes to
    NMSBoxes costs more than the forward pass on a phone.
    """
    pred = np.squeeze(raw, 0)                   # (5, 8400)
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T                           # (8400, 5), whichever way it came
    scores = pred[:, 4]
    keep = scores >= conf
    pred, scores = pred[keep], scores[keep]
    if not len(pred):
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)    # xywh, for NMSBoxes

    keep_idx = nms(boxes, scores, iou)
    if not len(keep_idx):
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    boxes, scores = boxes[keep_idx], scores[keep_idx]

    # Back out of the letterbox: undo the padding first, then the scale.
    pad_x, pad_y = pads
    x1 = (boxes[:, 0] - pad_x) / gain
    y1 = (boxes[:, 1] - pad_y) / gain
    x2 = (boxes[:, 0] + boxes[:, 2] - pad_x) / gain
    y2 = (boxes[:, 1] + boxes[:, 3] - pad_y) / gain
    out = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    if frame_shape is not None:
        h_f, w_f = frame_shape[:2]
        np.clip(out[:, 0::2], 0, w_f - 1, out[:, 0::2])
        np.clip(out[:, 1::2], 0, h_f - 1, out[:, 1::2])
    return out, scores.astype(np.float32)


def inside(box, roi) -> bool:
    """Is the box's centre inside the drawn region? The same test app/ counts with."""
    if not roi:
        return True
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return cv2.pointPolygonTest(np.array(roi, np.int32), (float(cx), float(cy)), False) >= 0


def count_inside(boxes, roi) -> int:
    return sum(1 for b in boxes if inside(b, roi))


class Engine:
    """Holds whatever runs the graph. Two of them, and the phone only has the second.

    ON A PC it makes its own onnxruntime session, so every line above can be tested on the
    bench against the counts the real app produces -- which is the only way to know the
    phone will agree with it.

    ON THE PHONE the session lives in Kotlin (OrtEngine.kt: bytes in, bytes out, knows
    nothing about YOLO) and is handed in. Same arithmetic either way.
    """

    def __init__(self, session=None, kotlin=None, conf=0.45, iou=0.4, hw=None):
        if session is None and kotlin is None:
            raise ValueError("Engine needs an onnxruntime session or the Kotlin one")
        self.session = session
        self.kotlin = kotlin
        self.conf, self.iou = conf, iou
        # Asked of the graph rather than assumed, so re-exporting at another size needs no
        # change here -- and a mismatch cannot hide.
        if hw is None and session is not None:
            shape = session.get_inputs()[0].shape
            hw = (int(shape[2]), int(shape[3]))
        self.hw = hw or INPUT_HW

    @classmethod
    def from_file(cls, path, conf=0.45, iou=0.4):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return cls(session=ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]),
                   conf=conf, iou=iou)

    def forward(self, tensor):
        if self.session is not None:
            name = self.session.get_inputs()[0].name
            return self.session.run(None, {name: tensor})[0]
        # Kotlin hands back one flat float array plus the shapes it came with.
        flat = np.frombuffer(self.kotlin.run(tensor.tobytes()), dtype=np.float32)
        return flat.reshape(1, 5, -1)

    def detect(self, bgr):
        """A frame -> (boxes, confidences) in that frame's own pixels."""
        boxed, gain, pads = letterbox(bgr, self.hw)
        raw = self.forward(blob(boxed))
        return decode(raw, gain, pads, self.conf, self.iou, bgr.shape)
