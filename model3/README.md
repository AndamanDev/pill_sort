# model3 — the outside opinion

`pill-count/1` from [Roboflow Universe](https://universe.roboflow.com/testpills/pill-count),
called over the network, on stills.

model1 counts with a segmentation net we trained. model2 is the bench that measures how
steady that net is on a real tray. Both of them answer to us, which is the problem: when
the number looks wrong there is nothing to check it against except another run of the
same model. `pill-count/1` was trained by somebody else on somebody else's pills and has
never seen this bench, so its count is an independent second opinion — which is the only
kind worth having.

It is **not** a replacement for the local model and cannot become one. Every call is an
HTTPS round trip with a JPEG on it, it needs the internet, and it is metered. The camera
loops in model1 and model2 stay local; this runs on stills, on demand.

## Setup

```powershell
python -m venv model3\.venv-rf
model3\.venv-rf\Scripts\python.exe -m pip install -r model3\requirements.txt
```

Then the key. It is read from `ROBOFLOW_API_KEY` first, and from `model3\.env` second —
so an exported key wins for CI, and the bench does not have to export anything:

```powershell
Copy-Item model3\.env.example model3\.env   # then edit it
# or, for one shell only:
$env:ROBOFLOW_API_KEY = "your-key"
```

`model3\.env` is in `.gitignore`. **The key never goes in a source file**, and it never
goes in a query string — `cloud.py` configures `api_key_transport="header"` so it travels
as `Authorization: Bearer …`, because a query string is the part of a URL that every
proxy on the way writes to a log. `selftest.py` asserts both of those and will fail loudly
if either regresses.

## Running it

```powershell
$py = "model3\.venv-rf\Scripts\python.exe"

# one still
& $py model3\cloud_count.py model3\samples\tray_full.jpg

# with the boxes drawn and the raw JSON kept
& $py model3\cloud_count.py model3\samples\tray_full.jpg --save model3\out\full.jpg --json model3\out\full.json

# a frame out of one of model2's clips, no exporting by hand
& $py model3\cloud_count.py model2\frames\tray_full.mp4 --frame 20 --save model3\out\f20.jpg

# a URL works too
& $py model3\cloud_count.py https://example.com/tray.jpg --conf 0.25
```

It prints the count, the spread of confidences, and two timings: `model` is what Roboflow
says it spent, to compare against model1's ~100 ms forward pass, and `round trip` is what
the operator actually waits for.

## The comparison this is for

1. `python model2/app.py`, point it at a tray, press `s`. That drops a still in
   `model2/shots/` — the frame, not the overlay.
2. `& $py model3\cloud_count.py model2\shots\shot_XXXX.jpg --save model3\out\XXXX.jpg`
3. Two models, one tray, two counts. Where they agree the tray is easy. Where they
   disagree, open the two annotated frames side by side and the disagreement is a
   specific tablet rather than a difference of opinion about a number.

Note that this is a **detection** model: a tablet is a rectangle here and an outline in
model1. That is fine for counting — a count is a number of things, not their shape — but
`pillsort.features` reads contours and will not work off these boxes.

## Files

| | |
|---|---|
| `cloud.py` | the client. `detect(image) -> Shot`. Key loading, header auth, readable errors. |
| `cloud_count.py` | the CLI above. One image in, one number out, boxes optional. |
| `selftest.py` | everything except the network, with a canned response. Spends nothing. |
| `samples/` | two frames pulled out of `model2/frames/`, to have something to run on. |
| `.env` | the key. Gitignored. |
| `dataset/` | pill-count v4, downloaded. 21,022 labelled tablets. Gitignored, CC BY 4.0. |
| `train_colab.ipynb` | trains on that data on a free Colab GPU, ~15 min. See below. |
| `weights/pillcount-det.pt` | round tablets only. Kept for comparison. |
| `weights/pillcount-det-v2.pt` | round tablets, and capsules only if solid-coloured. |
| `weights/pillcount-det-v3.pt` | **the one to use.** Two-tone capsules as well. |
| `train_colab_v2.ipynb` | how v2 was made: merge the capsule set in, fine-tune from v1. |

## Status, as of 16 Sep 2026

The integration works. **The model does not.**

Measured on `samples/tray_full.jpg` — a tray a human counts at about sixty — by sweeping
the only knob the API gives:

| conf | tray_full (≈60 real) | tray_sparse (4 real) |
|---|---|---|
| 0.40 | 3 | 5 |
| 0.20 | 12 | 6 |
| 0.10 | 29 | 7 |
| 0.05 | **63** | 11 |
| 0.01 | 230 | 25 |

That 63 against a true 60 looks like a result, and it is not one. Open
`out/full_c0.05.jpg` and the boxes are scattered: several sit on the gaps *between*
tablets, several are stacked two and three deep on one tablet, several are out on the
dark bench and on a pen beyond the tray, and a good third of the actual tablets have no
box at all. The count is right by coincidence at a threshold chosen after seeing the
answer, which is not a measurement. `tray_sparse` gives it away with no arithmetic: four
tablets on the tray, eleven reported, with doubled boxes on single tablets.

**Its confidences carry no information.** Every detection comes back at 0.999996–0.999999
— all 230 of them at `conf 0.01`. So there is nothing to rank by and nothing to threshold
sensibly; the `confidence` request parameter still changes the count, so the server is
filtering on some score it does not return to us. Either way, the usual fix for an
over-eager detector is not available here.

**The honest conclusion: `pill-count/1` is not an outside check on this bench.** Do not
wire it into anything that reports a number to an operator.

**But the fault is the trained version, not the data** — and the first draft of this file
got that wrong, so it is worth being precise about. `pill-count` version 1 was built from
**27 training images**, and reports mAP 98.81 / precision 99.99 on a test split of three.
That is the overfit showing up in the saturated confidences above. Versions 2, 3 and 4
exist as *datasets* and were never trained: `--model pill-count/4` returns 404.

Version 4's data, measured after downloading it (`dataset/`, CC BY 4.0):

| | pill-count v4 | this bench |
|---|---|---|
| images | 266 (112 unique) | 3 clips |
| boxes | 21,022 | — |
| pills per image | **79** (range 10–175) | ~60 |
| box size, % of frame | 4.6% × 6.2% | ~4.7% × 4.7% |
| scene | blue counting tray, white round tablets, top-down | blue tray, white round tablets, top-down |

It is the same problem, at the same scale, on the same kind of tray. 21,022 labelled
tablets is roughly two orders of magnitude more annotation than this project has of its
own, and nobody has ever trained a model on it.

What model3 is good for, then:

- the plumbing, which is correct and cheap to point at a better model — `--model
  other-project/3` is the whole change;
- a worked example of why an outside model has to be *looked at*, not just run. The
  number agreed. The picture did not;
- `dataset/` — the part actually worth having.

Reproduce any of the above:

```powershell
& $py model3\cloud_count.py model3\samples\tray_full.jpg --conf 0.05 --save model3\out\full_c0.05.jpg
& $py model3\cloud_count.py model3\samples\tray_sparse.jpg --conf 0.05 --save model3\out\sparse_c0.05.jpg
```

## Training on it

`train_colab.ipynb` — upload it to [colab.research.google.com](https://colab.research.google.com),
turn on the T4, Run all. It takes the zip in `dataset/`, trains `yolo11n` for 100 epochs
and hands back a `best.pt`.

Two things in it are there because of measurements made here, and should not be removed:

- **It rewrites `data.yaml`.** Roboflow exports `../train/images`, which resolves outside
  the dataset folder and fails everywhere. Only an absolute `path:` survives a change of
  working directory.
- **It ends by counting YOUR tray, not by reporting mAP.** Every one of the 21 validation
  images comes from the same source clip as a training image — checked, 21 of 21 — so
  frames in val have near-twins in train and the score is meaningless. This is the exact
  trap `model2/dataset.py` documents, and the reason the shipped model reports recall
  1.000 while miscounting a real tray by two. `samples/tray_full.jpg` is the only honest
  test available, because nothing has ever trained on it.

`yolo11n` is chosen to match the bench, which has no GPU — not to make training fast.

### What came back — 16 Sep 2026

100 epochs on a free T4, about fifteen minutes. `weights/pillcount-det.pt`.

Its own validation says mAP50 0.980, precision 0.974, recall 0.950, and **that is the
number to ignore**, for the reason above. Here is the number that counts, measured over
the forty frames of `model2/frames/tray_full.mp4` — a tray nobody is touching, which
nothing in training has ever seen:

| | median | spread | ms/frame | confidences |
|---|---|---|---|---|
| `pillsort-seg` (shipped, 448px) | 60 | **6** | ~100 | — |
| `pillcount-det` (new, 640px) | 61 | **1** | **42** | 0.72–0.93 |
| `pill-count/1` (Roboflow hosted) | 63 at conf 0.05 | — | ~350 + network | 0.999998 flat |

Six times steadier than the model this project ships, twice as fast, on the CPU the bench
actually has. The confidences spread properly, which is what a model that has learned
something looks like next to one that has memorised 27 images.

And — the lesson of this whole directory — **the boxes were looked at**, not just the
number: `out/bench_newmodel.jpg`. One box per tablet, none on the gaps, none stacked,
none out on the bench. That is the difference between 61 and pill-count/1's 63.

Use it from the bench window:

```powershell
python model2\count.py --model model3\weights\pillcount-det.pt --imgsz 640 --conf 0.25
```

640 rather than `cloud_count.py`'s default 448 because that is what it was trained at; at 448
the spread goes to 2, at 960 to 11. It is a detection model, so `cloud_count.py` draws dots
rather than outlines, and `pillsort.features` still cannot read it.

Cost: serverless inference is 1,000 calls per credit, so the whole sweep above ran for
about a fiftieth of one credit. The earlier `402` on the first workspace was not
inference — storage is billed monthly at 5,000 images per credit, and that is what ate a
15-credit cap.

## What it cannot do: capsules — 16 Sep 2026

Measured on the bench, with two dark capsules dropped onto a tray of round tablets
(`model2/shots/*_0916-124245.*`):

- one capsule got **one box covering half of it**;
- the other got **two boxes laid end to end along it**.

Two capsules, counted as three. The boxes it draws are all round-tablet shaped — every
box in that frame had an aspect ratio near 1.0, none above 1.4 — because every tablet in
`dataset/` is round and lying flat. The model has no concept of an elongated object, so
it tiles one.

**No threshold fixes this.** The two boxes on the second capsule overlap by 13%. Dropping
`--iou` to 0.13 to suppress one would start merging round tablets that merely touch,
which is a worse failure. It is a gap in the training data, and only data closes it. The
same weakness shows up as a tablet standing on its edge, and on the synthetic capsule in
`model1/samples/three_kinds.jpg` — three independent sightings of one cause.

What still works in that frame: `pillsort.grouping` read the boxes and returned
`21 white round / 2 dark round / 1 grey round`, flagging 3 as odd. So the tray can be
told "something here is not like the others" today, even though the count is wrong by
one. Counting and sorting fail separately, which is what `grouping.py` says it was built
for.

### Candidate data for fixing it

Searched Universe (`GET https://api.roboflow.com/universe/search?q=`) and vetted the same
way `pill-count` was — density and a look at the images, not the name:

| project | images | boxes | per image | licence | notes |
|---|---|---|---|---|---|
| `apisits-workspace-mffve/detection-medicine-capsule` | 346 | 20,354 | 58.8 | CC BY 4.0 | **dense trays of capsules, top-down, touching and overlapping.** The one to take. |
| `jeonghyeon-kim/pill-counting-im4ph` | 1,236 | 132,028 | 106.8 | CC BY 4.0 | much the largest; its preview image is a 2-capsule close-up, so the set needs downloading before it can be judged |
| `polyploy/pill-mkfke` | 613 | 24,957 | 40.7 | CC BY 4.0 | two pill classes |
| `hardee/capsules-mywnz` | 224 | 4,165 | 18.6 | CC BY 4.0 | capsules, lower density |
| `kkh72/kkh7-pill-counting` | 1,368 | 32,021 | 23.4 | **Private** | cannot be used |

The capsule set's tray is white where ours is blue. That is a reason to MERGE it with
`dataset/` rather than train on it alone: a model that sees both learns the shape and
stops keying on the background.

**Fine-tune, do not restart.** `weights/pillcount-det.pt` already counts round tablets at
a spread of 1; throwing that away to start from `yolo11n.pt` wastes it. Pass the existing
weights as the starting point and train on the MERGED set. Training on capsules alone
would be the classic mistake -- the model would learn capsules and forget the tablets it
can already count.

Keep it single-class for the first pass. The bug to fix is "one capsule, two boxes",
which is about shape, and one class of `pill` is enough to teach that. A `tablet` /
`capsule` split is a separate question and can come after the count is right.

## v2 — 16 Sep 2026

`train_colab_v2.ipynb`: merged `pill-count` v4 with
`apisits-workspace-mffve/detection-medicine-capsule` v4, collapsed both to one `pill`
class, and trained 60 epochs at lr0 0.005 on a free T4.

**It was meant to fine-tune from `pillcount-det.pt` and it did not.** The checkpoint says
`model: yolo11n.pt`, `data: /content/merged/data.yaml` -- so the merge worked and the
starting weights did not: the uploaded `.pt` never registered and the notebook fell back
to COCO weights instead of stopping. That fallback has been removed; the cell now refuses
to continue without the file. Everything measured below is real, and was produced by
training from scratch on the merged set, not by the fine-tune this section originally
claimed. A genuine fine-tune has not been tried yet and might do better still.

**Read the first row of this table and you would revert it. Read the second and you
would ship it. Both are the same forty frames.**

| tray_full.mp4, 40 frames | median | spread |
|---|---|---|
| v1, whole frame | 61 | 3 |
| v2, whole frame | 66 | 3 |
| v1, inside the tray | 61 | 3 |
| **v2, inside the tray** | **61** | **1** |

v2 adds six boxes and loses none — all 61 tablets v1 found are still found. Five of the
six sit at y≈9–15, the strip of desk above the tray; the sixth is on the lamp reflection
down the tray's left rim. Every one of them is elongated (aspect 1.28–2.81). That is the
capsule training showing through: the model now believes elongated things can be pills,
and a light streak is elongated.

Inside the region the operator draws — which `cloud_count.py` has always required, and which
exists precisely because the bench and the operator's hands are in shot — those six are
gone and what remains is **the same count, twice as steady**.

On capsules, the frame v1 got wrong: the capsule that v1 tiled with two boxes now gets
**one box, aspect 1.56**. v1 produced no box above 1.25 anywhere on that tray.

So: **use v2, and draw the region.** Without a region v2 is worse than v1, and that is
worth knowing rather than hiding — it is the same lesson as `pill-count/1`, one layer up.
A number measured over the whole frame answered the wrong question.

```powershell
python model2\count.py --model model3\weights\pillcount-det-v2.pt     --imgsz 640 --conf 0.45 --iou 0.4 --exposure -3
```

Still open: the tray in that capsule frame reads 24 against a true 23, in both versions.

### v2 on the bench, with capsules — 16 Sep 2026

500 frames, 64 seconds, region drawn, tray holding ~22 round tablets and 5–6 capsules of
assorted colours (`model2/shots/*_0916-131734.*`).

    count   min 29   max 32   median 31        (31 on 283 frames, 30 on 213)
    spread  3

**The over-count is exactly the number of capsules it cut in half.** Zooming in
(`out/caps_v2_zoom.jpg`) shows what v2 fixed and what it did not:

| | v1 | v2 |
|---|---|---|
| capsule, one solid colour | two boxes tiled along it | **one box** |
| capsule, two-tone (white/cyan) | two boxes | **still two — split at the colour join** |
| oblong tan tablet | two boxes | still two |

Three objects split, count 3 over a true ~27–28. The arithmetic closes, which is the
useful part: this is not general noise, it is one nameable failure happening three times.

It is also where the spread of 3 comes from. Round tablets alone give spread 1 — some
frames cut a two-tone capsule and some do not, so the number flickers between 30 and 31.
Steadiness and correctness are failing for the same reason, and one fix addresses both.

**The next dataset is not on Roboflow.** The capsule set that got v2 this far is red/black
on a white tray under bright light; this bench is white/cyan on dark blue. Nothing public
matches it better than the frames already sitting in `model2/shots/`. Labelling 30–50 of
those — different angles, positions, counts — is the shortest path from here, and
`train_colab_v2.ipynb` takes an extra source without modification.

## The split-capsule merge — no retraining required

The operator's question was the right one: *why does it not look at the shape?* A capsule
is obviously one object to a human eye, colour join and all.

It does not, because a detector has no prior saying an object may not contain a hard
internal edge — and white against cyan on dark blue is as hard an edge as the capsule's
own outline. But the tray does have that information: against dark plastic a capsule is
one connected bright region. `model2/count.py: merge_split()` uses it.

The naive form of the rule destroys the main case — one blob is one pill turns a packed
tray of 25 tablets into 1. Three tests together separate a split capsule from a pile:
**exactly two boxes** in the blob, **elongation ≥ 1.8** by `minAreaRect` (a diagonal
capsule's bounding box is nearly square, so the obvious test fails), and **blob area <
1.6×** the two boxes, so two tablets touching inside a bigger blob are left alone.

Measured on every frame saved off the bench, and the rule is wired to exactly these:

**The first version of this rule was wrong, and the bench caught it.** With only
elongation and a loose area bound it merged three pairs of *round tablets that were
merely touching* — undercounting by three. Two touching tablets are elongated (≈2.0) and
convex; neither elongation, solidity nor a waist measurement tells them from a capsule:

| | elongation | solidity | waist | **fill** |
|---|---|---|---|---|
| split capsules | 2.92, 2.38 | 0.93 | 0.75, 0.61 | **0.55, 0.55** |
| touching tablets | 1.62–2.02 | 0.89–0.95 | 0.21–0.61 | **0.75–0.86** |

Only **fill** — blob area over the two boxes' area — separates them, and by geometry
rather than luck: a round tablet fills π/4 = 0.785 of its own box, always, while a
capsule half wearing a box this detector sized for a round tablet fills about 0.55.

| frame | boxes | merged | true |
|---|---|---|---|
| capsules only | 6 | **4** | 4 |
| touching tablets | 19 | **19** | 19 |
| packed tablets | 69 | **69** | — |
| packed tablets | 69 | **69** | — |
| edge-on tablet | 34 | 34 | 32 |

Costs **3.3 ms a frame** against the model's 42.

The last row is what the fix gives up: the loose rule did repair an edge-on tablet, and
the strict one no longer does — a tablet on its edge splits into round-ish halves that
fill their boxes like tablets. That trade is deliberate. Merging two real tablets reads
30 as 29 and a pharmacy trusting it dispenses short; counting an edge-on tablet twice
reads 30 as 31, which is wrong in the direction an operator notices and shows on screen
as two markers on one tablet. **When the rule is uncertain it declines to merge.**

On by default for detection models; `--no-merge` turns it off, and it is skipped entirely
when the model returns masks, since that case was never broken. The capsule frame is
still 24 against a true 23: one capsule there splits into halves that do not share a
blob. Data is the fix for that one, not geometry.

## v3 — capsules, properly — 16 Sep 2026

Fine-tuned **from v2** (the checkpoint says `model: /content/pillcount-det-v2.pt`, so the
guard added after v2's silent fallback did its job), 60 epochs at lr0 0.005 on the merge
of four sources: `pill-count` v4, `detection-medicine-capsule` v4,
`kasetsart-university-rpmpb/pills-pills` and `tes-4ynfo/capsule-mifje` — the last two
added for mixed trays and for two-tone capsules specifically. All CC BY 4.0, all collapsed
to one `pill` class.

Its own validation reads mAP50 0.934 / precision 0.907 / recall 0.867 — **lower than v2's
0.952**, and irrelevant, for the reason this file has repeated throughout: those splits
leak. The three frames with counts a human established are what matter.

| | true | v2 | **v3** |
|---|---|---|---|
| capsules only | 4 | 6 | **4** |
| capsule in a pile of tablets | 23 | 24 | **23** |
| round tablets | 61 | 67 | **61** |
| clip, 40 frames, in the tray | ~61 | 61, spread 1 | 60, spread 1 |

Right on all three, on the raw boxes, with no post-processing. The six phantom boxes v2
put on the bench edge and the lamp reflection are gone too — so v3 no longer needs the
region drawn to be correct, though drawing it is still right for excluding the next tray
along.

**This retires the merge rule.** `--merge` is now opt-in rather than on by default: with
nothing left to repair it can only make mistakes, and on the full tray it makes one,
reading 61 as 60. It stays in the file, measured and documented, for anyone running v2 or
earlier.

Known remaining difference: v3 reads the clip at a median of 60 where v2 read 61. Which
is right has not been established — nobody has counted that tray by hand.
