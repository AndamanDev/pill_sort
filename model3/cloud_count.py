"""How many tablets Roboflow's `pill-count/1` sees in a still. One image, one number.

    python model3/cloud_count.py model3/samples/tray_full.jpg
    python model3/cloud_count.py model3/samples/tray_full.jpg --save out.jpg --json out.json
    python model3/cloud_count.py https://example.com/tray.jpg --conf 0.25
    python model3/cloud_count.py model2/frames/tray_full.mp4 --frame 20

Run it with the interpreter that has inference-sdk:

    D:\\pillsort\\model3\\.venv-rf\\Scripts\\python.exe model3/cloud_count.py <image>

THIS IS A STILL, NOT A CAMERA, and that is deliberate -- see the note at the top of
cloud.py. Each call is a network round trip, so pointing a 30 fps loop at it would be
thirty paid round trips a second to watch a tray being positioned.

WHAT TO DO WITH THE NUMBER. Compare it. Run model2/app.py on the same tray, press `s` to
drop a still into model2/shots/, then run this file on that still: two models, one tray,
two counts. Where they agree the tray is easy. Where they disagree, --save writes the
boxes out and the disagreement is visible rather than theoretical -- and a model that has
never seen this bench disagreeing is the outside check the local numbers cannot give
themselves.

WHY IT TAKES A VIDEO TOO. model2/frames holds mp4s, not jpgs, and the bench's stills come
out of them. `--frame N` pulls one out and sends that, so nothing has to be exported by
hand first.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cloud                                                    # noqa: E402

#: The same two colours model2's window uses, so a saved frame from either looks like it
#: came from the same bench.
INK = (245, 245, 245)
EDGE = (90, 220, 110)

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def _frame_from_video(path: str, index: int):
    """One frame of a clip, as a numpy image. Negative or past the end means the middle."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if index < 0 or (total and index >= total):
        index = max(total // 2, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"{path} has no frame {index} ({total} frames)")
    print(f"frame       : {index} of {total} from {os.path.basename(path)}")
    return frame


def _load_for_drawing(source):
    """The image as pixels, for the overlay. Only called when there is something to save.

    A URL has to be fetched a second time here -- Roboflow got its own copy over the
    wire -- which is a wasted download, but only on the --save path, and only for a URL.
    """
    import cv2
    import numpy as np

    if not isinstance(source, str):
        return source
    if source.startswith(("http://", "https://")):
        from urllib.request import urlopen

        with urlopen(source, timeout=30) as response:
            data = np.frombuffer(response.read(), np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"fetched {source} but it did not decode as an image")
        return image
    image = cv2.imread(source)
    if image is None:
        raise SystemExit(f"cannot read {source}")
    return image


def annotate(image, shot: cloud.Shot, labels: bool = False):
    """The boxes and the count, drawn on a copy. Returns the copy."""
    import cv2

    shown = image.copy()
    for pill in shot.pills:
        x1, y1, x2, y2 = pill.box
        cv2.rectangle(shown, (x1, y1), (x2, y2), EDGE, 2, cv2.LINE_AA)
        if labels:
            tag = f"{pill.confidence:.2f}"
            cv2.putText(shown, tag, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(shown, tag, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, INK, 1, cv2.LINE_AA)

    # The number, big, top left -- the same place model2 puts it, for the same reason.
    n = str(shot.count)
    cv2.putText(shown, n, (16, 74), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 0), 9,
                cv2.LINE_AA)
    cv2.putText(shown, n, (16, 74), cv2.FONT_HERSHEY_SIMPLEX, 2.4, INK, 3, cv2.LINE_AA)
    line = f"{cloud.MODEL_ID}   roboflow cloud"
    cv2.putText(shown, line, (16, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3,
                cv2.LINE_AA)
    cv2.putText(shown, line, (16, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1,
                cv2.LINE_AA)
    return shown


def run(source: str, conf: float = cloud.DEFAULT_CONF, iou: float = cloud.DEFAULT_IOU,
        model_id: str = cloud.MODEL_ID, frame: int = -1, save=None, json_path=None,
        labels: bool = False, show: bool = False) -> int:
    is_url = source.startswith(("http://", "https://"))
    if not is_url and not os.path.isfile(source):
        raise SystemExit(f"no such file: {source}")

    payload = source
    if not is_url and source.lower().endswith(VIDEO_SUFFIXES):
        payload = _frame_from_video(source, frame)

    print(f"model       : {model_id}   via {cloud.API_URL}")
    print(f"image       : {source}")
    print(f"conf        : {conf}   iou {iou}")

    started = time.perf_counter()
    shot = cloud.detect(payload, conf=conf, iou=iou, model_id=model_id)
    trip = (time.perf_counter() - started) * 1000

    # Both numbers, because they answer different questions: `model` is what to compare
    # against model1's ~100 ms forward pass, `round trip` is what the operator waits.
    print(f"size        : {shot.size[0]}x{shot.size[1]}")
    print(f"round trip  : {trip:.0f} ms   model {shot.ms:.0f} ms")
    print(f"count       : {shot.count} tablets")
    if shot.pills:
        scores = sorted(p.confidence for p in shot.pills)
        print(f"confidence  : min {scores[0]:.2f}  median {scores[len(scores) // 2]:.2f}  "
              f"max {scores[-1]:.2f}")
        kinds = {}
        for pill in shot.pills:
            kinds[pill.label] = kinds.get(pill.label, 0) + 1
        if len(kinds) > 1 or "pill" not in kinds:
            print("classes     : " + "  ".join(f"{n}x {k}" for k, n in
                                                sorted(kinds.items(), key=lambda kv: -kv[1])))

    if json_path:
        # The RAW response, not our dataclasses. Whatever reads this next should see what
        # Roboflow actually said, including any field this file does not model yet.
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(shot.raw, fh, indent=2)
        print(f"json        : {json_path}")

    if save or show:
        import cv2

        shown = annotate(_load_for_drawing(payload), shot, labels=labels)
        if save:
            os.makedirs(os.path.dirname(os.path.abspath(save)), exist_ok=True)
            cv2.imwrite(save, shown)
            print(f"saved       : {save}")
        if show:
            title = f"model3 -- {model_id} -- any key closes"
            cv2.imshow(title, shown)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image", help="an image path, an image URL, or an mp4 (with --frame)")
    ap.add_argument("--conf", type=float, default=cloud.DEFAULT_CONF)
    ap.add_argument("--iou", type=float, default=cloud.DEFAULT_IOU)
    ap.add_argument("--model", default=cloud.MODEL_ID, help="project/version on Roboflow")
    ap.add_argument("--frame", type=int, default=-1,
                    help="which frame, if the input is a video. default: the middle one")
    ap.add_argument("--save", default=None, help="write the annotated image here")
    ap.add_argument("--json", dest="json_path", default=None,
                    help="write Roboflow's raw response here")
    ap.add_argument("--labels", action="store_true", help="draw the confidence on each box")
    ap.add_argument("--show", action="store_true", help="open a window as well")
    a = ap.parse_args(argv)
    return run(a.image, conf=a.conf, iou=a.iou, model_id=a.model, frame=a.frame,
               save=a.save, json_path=a.json_path, labels=a.labels, show=a.show)


if __name__ == "__main__":
    raise SystemExit(main())
