## Project objective and implemented system

Investigate face identity verification and resistance to photo/screen presentation attacks for
the CDAC m-governance research setup. Treat identity recognition and liveness as separate tasks:
recognition asks whether the face matches the registered user; liveness asks whether the input
is a live presentation.

Implemented:

- Image/webcam registration and one-to-one identity verification using model-specific stored embeddings.
- Four InsightFace recognition packs: buffalo_l, buffalo_m, buffalo_s and buffalo_sc.
- A MiniFASNet liveness gate before identity comparison in the existing verification flow.
- A common experimental CLI and script for MiniFASNet, MobileNetV3 anti-spoof-mn3 and TinyLiveness.
- Image, webcam and video test inputs; shared single-frame, rolling-mean and rolling-vote reports.
- Ground-truth, lighting and attack annotations; CSV samples, JSON settings/metrics and readable Markdown.
- Model hashes and crop/runtime provenance, CPU execution, and Windows camera backend fallback.
- 36 automated tests passed in the latest verification. This checks software behavior, not model accuracy.

The selectable alternative models currently belong to the experiment runner. The existing identity
verification menu continues to use the MiniFASNet gate.

## Finding 1 — recognition results depend on model choice and framing

Seven local images were evaluated with every recognition pack. All four processed six images
and failed face detection on the same remaining image. That is a processing failure, not an
incorrect identity decision. Pose/tight framing are possible contributors, not established causes.

Four images confirmed to show one identity supplied six genuine pairs, at similarity threshold 0.5:

| Pack | Correct genuine matches | False rejections |
| --- | ---: | ---: |
| buffalo_l | 6/6 | 0 |
| buffalo_m | 6/6 | 0 |
| buffalo_s | 4/6 | 2 |
| buffalo_sc | 4/6 | 2 |

The larger packs performed better on these particular genuine pairs. One buffalo_l score was
approximately 0.501, close to the threshold. There are no confirmed different-person pair results
from which to estimate false accepts. These counts do not establish general recognition accuracy.

Evidence: [image evaluation](IMAGE_EVALUATION.md), [pair scores and image outcomes](../results/image_evaluation.json).

## Finding 2 — storage/performance tradeoffs need separate measurement

Measured extracted full-pack ONNX sizes: buffalo_l 341.27 MB, buffalo_m 327.64 MB,
buffalo_s 166.10 MB and buffalo_sc 16.14 MB. These include auxiliary files and are not RAM measurements.

Initial warmed CPU extraction measurements were 0.461, 0.305, 0.147 and 0.176 seconds/image,
respectively. They are single sanity-run timings, not stable hardware benchmarks. Repeated identical-image
similarity near 1 confirms pipeline consistency, not useful real-world accuracy.

Evidence: [sanity measurements](SANITY_RESULTS.md).

## Finding 3 — passive liveness can confidently accept a spoof

The initial MiniFASNet frontal-light phone-photo trial had 75 valid scores from 92 sampled frames:

| Rule | Incorrect LIVE decisions on a labelled spoof |
| --- | ---: |
| Single | 66/75 = 88.0% |
| Rolling mean | 53/53 completed windows = 100% |
| Rolling vote | 39/53 completed windows = 73.6% |

Voting reduced the observed false-live fraction, but still accepted most completed windows.
An average cannot remove a consistently wrong model signal. The results support a spoof vulnerability
on this trial; they do not prove reliance on eye-socket shadows. That explanation still requires controlled
lighting comparisons and checks of crop, color order, alignment and output interpretation.

Evidence: [labelled phone-photo trial](../results/liveness/20261015T151643942032Z.json).

## Finding 4 — temporal aggregation can help or hurt genuine users

The later 20-second trials annotated as live, frontal lighting and no attack show:

| Model / threshold | Valid frame scores | Single false-spoof rate | Mean false-spoof rate | Vote false-spoof rate |
| --- | ---: | ---: | ---: | ---: |
| MiniFASNet / 0.5 | 81/87 samples | 5/81 = 6.17% | 0/67 = 0% | 0/67 = 0% |
| MobileNetV3 / 0.5 | 78/86 samples | 44/78 = 56.41% | 37/51 = 72.55% | 51/51 = 100% |
| TinyLiveness / 0.99 | 84/87 samples | 84/84 = 100% | 69/69 = 100% | 69/69 = 100% |

Rates exclude errors and pending windows. The clips were separate webcam runs, not identical recorded
inputs; the table is a record of trial outcomes, not a controlled model ranking. Different model
adapters also use different crop/alignment and detector color conventions, recorded in each report.

MiniFASNet's completed windows avoided intermittent false rejections. However, its final result was
INCONCLUSIVE because the final sample had a face-crop error. MobileNetV3 also ended INCONCLUSIVE,
with only 13 valid samples rebuilt in its final window. TinyLiveness at 0.99 ended SPOOF.

A separate 10-second TinyLiveness run at threshold 0.5 ended LIVE, but classified 27/28 completed vote
windows as SPOOF. A successful final window therefore does not establish a successful full run.
Because this was a different capture, it also does not isolate the effect of changing the threshold.

Evidence:
[MiniFASNet genuine run](../results/liveness/minifasnet_20261016T045109482200Z.json),
[MobileNetV3 genuine run](../results/liveness/anti-spoof-mn3_20261016T045453253526Z.json),
[TinyLiveness 0.99 genuine run](../results/liveness/tinyliveness_20261016T045740795078Z.json),
[TinyLiveness 0.5 genuine run](../results/liveness/tinyliveness_20261016T045842687604Z.json).

## Finding 5 — coverage and final-result semantics matter

A sample is one evaluated camera frame. At a target of five samples/second, a 20-second run aims
for about 100 samples; actual counts depend on capture/processing timing. Webcam images are not saved.

The default window contains 15 consecutive valid samples, about three seconds. A vote requires
at least 12/15 scores to pass the selected model threshold. Detection errors and long capture gaps
clear the window. The terminal's final result uses the last sampled window; full-run counts include
the entire run. A final LIVE or SPOOF alone can therefore conceal many opposite decisions earlier.

Overlapping windows share frames and are correlated. Do not describe 67 completed windows as
67 independent presentations, or these descriptive rates as trial-level APCER/BPCER.

## Limits to state explicitly

- Recognition covers one confirmed identity and six genuine pairs; no established false-accept rate.
- Liveness trials are few, separately captured, and labelled by the operator.
- Several older spoof reports have attack type none and unspecified lighting; those annotations need clarification.
- Eye-shadow dependence is an observation/hypothesis, not a demonstrated causal mechanism.
- No production-readiness or best-model claim is supported yet.
- Changing a threshold on different runs does not show its isolated effect; model scores need calibration.

## Decisions to discuss with the mentor

1. Agree on the next controlled dataset: multiple consenting people, independent sessions, physical
   lighting conditions, printed photos, phone photos and video replays.
2. Decide whether to add opt-in video recording so every model can be evaluated on identical clips.
   Re-score the same recordings to compare window rules and thresholds without capture differences.
3. Verify each exact model/export's preprocessing and face crop/alignment before concluding that an
   architecture is unsuitable. Investigate why some genuine runs have high detection failure.
4. Split development and evaluation data by person/session. Tune thresholds on development data,
   freeze them, and evaluate independent live/spoof presentations.
5. Define a presentation-level decision rule and report false acceptance, false rejection, inconclusive
   rate/coverage, latency and time to decision. Current last-window output is an experimental convention.
6. Add more labelled same-person and different-person pairs to the identity evaluation.

## Suggested opening statement

We built a common face-verification and liveness research setup and tested several models on local
data. The main finding is that recognition success, face detection and spoof resistance must be
evaluated separately. Rolling windows helped intermittent genuine-face errors in one trial, but
could worsen rejection in another and did not solve consistently incorrect spoof scores. We now
need controlled recordings and independent evaluation before recommending a model or threshold.
