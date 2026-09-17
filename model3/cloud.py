"""The Roboflow-hosted detector, behind one function. No window, no camera.

    from model3 import cloud
    shot = cloud.detect("model3/samples/tray_full.jpg")
    print(shot.count)

WHY A THIRD MODEL. model1 sees with a segmentation net we trained and ships as ONNX;
model2 is the bench that measures how steady that net is on a real tray. Both answer to
us. `pill-count/1` is somebody else's answer to the same question -- an object-detection
model on Roboflow Universe, trained on a pill dataset that is not ours, served from
Roboflow's cloud. It is here as a SECOND OPINION, not a replacement: a number from a
model that has never seen this bench is the closest thing we have to an outside check on
the one that has.

WHAT IT COSTS, and why this cannot be the live path. Every call is an HTTPS round trip to
serverless.roboflow.com with a JPEG on it -- hundreds of milliseconds against ~100 ms for
the local ONNX, it needs the internet, and it is metered. So: stills and saved frames, on
demand. The camera loop in model1 and model2 stays local.

BOXES, NOT MASKS. This is a detection model, so a tablet is a rectangle here and an
outline in model1. For counting that difference does not matter -- a count is a number of
things, not their shape -- but do not expect the size and roundness features in
`pillsort.features` to work off these; they read a contour.

THE API KEY IS NOT IN THIS FILE and must never be. It is read from ROBOFLOW_API_KEY, or
from model3/.env, which .gitignore keeps out of the repo. It goes up in an Authorization
header rather than the query string, because a query string is the part of a URL that
gets logged by every proxy on the way.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

#: Roboflow's serverless endpoint, and the model on it. Both halves of "pill-count/1"
#: matter: the project slug and the TRAINED VERSION. Version 1 is what was measured here;
#: a later version is a different model and gets measured again before it is trusted.
API_URL = "https://serverless.roboflow.com"
MODEL_ID = "pill-count/1"

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(HERE, ".env")
KEY_VAR = "ROBOFLOW_API_KEY"

#: Roboflow's own default for the hosted preview, and left there ON PURPOSE after
#: measuring. Lowering it does not improve this model, it just buys more boxes: on a tray
#: of about sixty tablets it returns 3 at 0.40, 63 at 0.05 and 230 at 0.01, and the 63 is
#: a coincidence -- the boxes land on the gaps. See the table in README.md. Do not "tune"
#: this to make a count match; look at --save first.
#:
#: Note also that the confidence this model RETURNS is useless: every detection comes back
#: at 0.999996+, including all 230 of them. Nothing here can rank or re-threshold them.
DEFAULT_CONF = 0.4
DEFAULT_IOU = 0.5


@dataclass(frozen=True)
class Pill:
    """One detection, in pixels of the image that was sent."""

    x: float                      # centre, which is what Roboflow returns
    y: float
    width: float
    height: float
    confidence: float
    label: str = "pill"

    @property
    def box(self) -> tuple[int, int, int, int]:
        """(x1, y1, x2, y2) -- corners, which is what a drawing call wants."""
        return (int(self.x - self.width / 2), int(self.y - self.height / 2),
                int(self.x + self.width / 2), int(self.y + self.height / 2))

    @property
    def centre(self) -> tuple[int, int]:
        return int(self.x), int(self.y)


@dataclass(frozen=True)
class Shot:
    """What one call came back with."""

    pills: list
    ms: float                     # what Roboflow says it spent, not the round trip
    size: tuple                   # (width, height) of the image it inferred on
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.pills)


def load_api_key(required: bool = True):
    """The key, from the environment first and model3/.env second.

    Two places rather than one because the two ways this gets run want different things:
    a shell that already has ROBOFLOW_API_KEY exported (CI, a service) should win, and a
    bench where nobody wants to export anything before every run should still work. The
    .env file is a plain KEY=value list -- python-dotenv would do this, but it would also
    be a dependency for eleven lines.
    """
    key = os.environ.get(KEY_VAR, "").strip()
    if key:
        return key
    if os.path.isfile(ENV_FILE):
        with open(ENV_FILE, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                if name.strip() == KEY_VAR:
                    return value.strip().strip("'\"")
    if required:
        raise SystemExit(
            "no Roboflow API key.\n"
            "    set it for this shell :  $env:{var} = 'your-key'\n"
            "    or put it in a file   :  {path}\n"
            "                             {var}=your-key\n"
            "That file is gitignored. Do not paste the key into a source file."
            .format(var=KEY_VAR, path=ENV_FILE))
    return None


def client(conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU, api_key=None):
    """A configured InferenceHTTPClient. Header auth, thresholds applied server-side.

    api_key_transport="header" sends `Authorization: Bearer <key>` instead of putting the
    key in the query string; it needs inference-sdk 1.5.0 or newer, which is why
    requirements.txt pins a floor rather than leaving the version open.
    """
    try:
        from inference_sdk import InferenceConfiguration, InferenceHTTPClient
    except ImportError:
        raise SystemExit(
            "inference-sdk is not installed in this interpreter.\n"
            "    D:\\pillsort\\model3\\.venv-rf\\Scripts\\python.exe -m pip install "
            "-r model3/requirements.txt")
    return InferenceHTTPClient(
        api_url=API_URL,
        api_key=api_key or load_api_key(),
    ).configure(InferenceConfiguration(
        confidence_threshold=conf,
        iou_threshold=iou,
        api_key_transport="header",
    ))


def detect(image, conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU,
           model_id: str = MODEL_ID, api_key=None) -> Shot:
    """Count the tablets in one image. A path, a URL, a numpy frame or a PIL image.

    Raises SystemExit with something readable on the two failures that actually happen --
    a rejected key and a model id that is not on this account -- because a stack trace out
    of an HTTP library does not tell an operator which of those it was.
    """
    from inference_sdk.http.errors import HTTPCallErrorError

    try:
        raw = client(conf=conf, iou=iou, api_key=api_key).infer(image, model_id=model_id)
    except HTTPCallErrorError as exc:
        status = getattr(exc, "status_code", None)
        if status in (401, 403):
            raise SystemExit(
                f"Roboflow refused the key ({status}). Check {KEY_VAR} -- and check it is "
                f"a key for a workspace that can see {model_id}.") from exc
        if status == 402:
            # Seen on the first real run from this bench. The key was fine -- a 402 means
            # it got past auth and hit billing -- so do not send anyone looking at the key.
            raise SystemExit(
                "Roboflow accepted the key and refused the work (402): the workspace is "
                "out of serverless credits.\n"
                "    https://app.roboflow.com/settings/plan\n"
                "Raise the credit cap or wait for the cycle to reset. Nothing in model3 "
                "needs changing -- the call itself is correct.") from exc
        if status == 404:
            raise SystemExit(
                f"Roboflow has no {model_id}. The id is project/version, and the version "
                f"has to be one that finished training.") from exc
        raise SystemExit(f"Roboflow call failed ({status}): {exc}") from exc

    # A batch of one comes back as a list of one. Unwrap it so callers never have to.
    if isinstance(raw, list):
        raw = raw[0] if raw else {}
    meta = raw.get("image") or {}
    pills = [
        Pill(x=p.get("x", 0.0), y=p.get("y", 0.0),
             width=p.get("width", 0.0), height=p.get("height", 0.0),
             confidence=p.get("confidence", 0.0), label=p.get("class", "pill"))
        for p in raw.get("predictions", [])
    ]
    return Shot(pills=pills, ms=float(raw.get("time", 0.0) or 0.0) * 1000.0,
                size=(int(meta.get("width", 0)), int(meta.get("height", 0))), raw=raw)
