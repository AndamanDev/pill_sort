"""Everything in model3 except the network, checked without spending a credit.

    D:\\pillsort\\model3\\.venv-rf\\Scripts\\python.exe model3/selftest.py

WHY THIS EXISTS. The live path costs a network round trip and a slice of a credit every
time, and it depends on a model that -- measured, see README.md -- does not work on this
bench. Neither of those should stand between somebody and knowing whether the CODE is
sound. This file puts a canned Roboflow response through the same path a real one takes
and checks the two things that are actually easy to get wrong.

    1. THE KEY IS IN THE HEADER, not the query string. This is the check worth keeping
       forever. api_key_transport="header" is one field in one constructor call, it is
       invisible when it is right, and if it regresses the key starts appearing in every
       proxy log between here and Roboflow. So: intercept the request, assert the key is
       in Authorization and assert it is NOT in the URL or the params.

    2. THE RESPONSE PARSES. Roboflow returns box centres; everything that draws wants
       corners. An off-by-half-a-width there looks plausible on screen and is wrong.

THE SEAM IS requests.Session.request, not requests.post. The SDK sends through a
thread-local Session, so patching the module-level requests.post intercepts nothing and
the test quietly goes to the real network -- which is how this file first "passed".
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import requests                                                 # noqa: E402

import cloud                                                    # noqa: E402
import cloud_count as count_cli                                       # noqa: E402

#: Shaped like a real serverless response, down to the keys we do not read. Two classes
#: on purpose: this model is not guaranteed to be single-class and the count must not
#: care.
CANNED = {
    "time": 0.0834,
    "image": {"width": 640, "height": 480},
    "predictions": [
        {"x": 100, "y": 120, "width": 40, "height": 40, "confidence": 0.91,
         "class": "pill", "class_id": 0, "detection_id": "a"},
        {"x": 300, "y": 220, "width": 38, "height": 42, "confidence": 0.62,
         "class": "pill", "class_id": 0, "detection_id": "b"},
        {"x": 420, "y": 300, "width": 44, "height": 40, "confidence": 0.47,
         "class": "capsule", "class_id": 1, "detection_id": "c"},
    ],
}

SENT = {}


class _Response:
    status_code = 200
    headers: dict = {}

    def raise_for_status(self):
        pass

    def json(self):
        return CANNED


def _intercept(self, method, url, *args, **kwargs):
    SENT["method"] = method
    SENT["url"] = url
    SENT["headers"] = dict(kwargs.get("headers") or {})
    SENT["params"] = dict(kwargs.get("params") or {})
    return _Response()


def main() -> int:
    key = cloud.load_api_key(required=False)
    if not key:
        raise SystemExit(
            f"this checks where the key is put, so it needs one. Set {cloud.KEY_VAR} or "
            f"write {cloud.ENV_FILE}. No credit is spent -- nothing leaves the machine.")

    requests.Session.request = _intercept
    sample = os.path.join(HERE, "samples", "tray_full.jpg")
    if not os.path.isfile(sample):
        raise SystemExit(f"missing sample: {sample}")
    shot = cloud.detect(sample)

    on_the_wire = json.dumps(SENT["url"]) + json.dumps(SENT["params"], default=str)
    assert key not in on_the_wire, "THE API KEY IS IN THE URL -- api_key_transport broke"
    auth = {k.lower(): v for k, v in SENT["headers"].items()}.get("authorization", "")
    assert auth == f"Bearer {key}", f"key is not in the Authorization header: {auth!r}"
    assert SENT["url"].endswith("/" + cloud.MODEL_ID), SENT["url"]
    print(f"auth        : Authorization: Bearer <key>, and no key in {SENT['url']}")

    assert shot.count == 3, shot.count
    assert shot.size == (640, 480), shot.size
    assert abs(shot.ms - 83.4) < 0.1, shot.ms
    # centre (100, 120) size 40x40 -> corners (80, 100) .. (120, 140)
    assert shot.pills[0].box == (80, 100, 120, 140), shot.pills[0].box
    assert shot.pills[0].centre == (100, 120), shot.pills[0].centre
    assert [p.label for p in shot.pills] == ["pill", "pill", "capsule"]
    print(f"parse       : {shot.count} pills, corners and centres agree")

    import cv2

    drawn = count_cli.annotate(cv2.imread(sample), shot, labels=True)
    assert drawn is not None and drawn.shape == cv2.imread(sample).shape
    print("annotate    : boxes drawn on a copy, original untouched")
    print()
    print("PASS        : the wiring is sound. Whether the MODEL is, see README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
