# Initial liveness trial analysis

Two 20-second webcam trials, both labelled frontal lighting, using the same MiniFASNet model,
threshold 0.5, 15-sample window, target 5 samples/second and 80% live-vote requirement.
Labels are supplied by the operator; images were not saved and cannot be independently inspected.

Sources:

- Live: [summary](../results/liveness/20261015T151246814002Z.json), [samples](../results/liveness/20261015T151246814002Z.csv).
- Phone photo: [summary](../results/liveness/20261015T151643942032Z.json), [samples](../results/liveness/20261015T151643942032Z.csv).

## Observed results

| Measure | Labelled live | Labelled phone-photo spoof |
| --- | ---: | ---: |
| Sampled frames | 92 | 92 |
| Valid scores | 17 (18.5%) | 75 (81.5%) |
| Face-crop errors | 75 (81.5%) | 17 (18.5%) |
| Single-frame LIVE | 5/17 | 66/75 |
| Single-frame SPOOF | 12/17 | 9/75 |
| Completed rolling windows | 1 | 53 |
| Rolling mean LIVE | 0/1 | 53/53 |
| Rolling vote LIVE | 0/1 | 39/53 |
| Median real-face score | 0.266 | 0.978 |
| Mean real-face score | 0.376 | 0.839 |

Rates in decision rows exclude errors and incomplete windows. Missing evidence is never acceptance.
The single completed live window cannot establish reliable temporal performance.

## Interpretation

1. Capture/detection is the first obstacle for the live trial. Every error says
   `Expected exactly one complete face for liveness; found 0`. In this implementation,
   that means zero faces survived detector output and the size/border filters, not necessarily
   zero raw detector detections. The required face size is at least 60 pixels, with a five-pixel
   image-border margin; the detector confidence threshold is 0.8. The logs do not identify which
   condition failed. There were no valid live scores before 16.109 seconds, and the longest
   consecutive valid sequence was just 15 samples, from 16.531 to 19.562 seconds.
2. Classification also failed on valid live frames: 12/17 were rejected. Detection improvement
   alone will therefore not resolve the observed problem.
3. The spoof failure is persistent rather than occasional noise. Its median real-face score is
   0.978, and 66/75 valid samples pass the current threshold. The long valid sequence spans
   5.703 to 19.781 seconds (67 samples). Temporal aggregation repeatedly sees confidently
   incorrect evidence.
4. Averaging worsened spoof rejection in this trial. On exactly the 53 frames with completed
   windows, single-frame decisions falsely accepted 47/53 (88.7%), averaging accepted 53/53
   (100%), and voting accepted 39/53 (73.6%). Voting helps relative to those baselines, but
   still fails to reject most evaluated spoof windows.
5. Raising the threshold is not a sufficient demonstrated fix: at 0.9, the recorded raw scores
   would accept 51/75 spoof frames and only 3/17 live frames. At 0.99, they would accept
   25/75 spoof frames and none of the live frames. These are retrospective single-frame
   calculations, not new trials or recalculated temporal results.

The reports demonstrate poor separation on these two labelled trials. They do not establish
that eye-socket shadows caused it: both runs have the same lighting label, no saved imagery,
and no controlled lighting comparison. Incorrect labels or a model/preprocessing/output-contract
mismatch also cannot be excluded from scores alone. Do not reverse class interpretation based
only on these two examples.

## Next experiment

First repeat the real-face trial with one centered, fully visible face held at a stable distance
from the start. If detection still fails, inspect raw detector confidence, box size and border
rejection reasons before changing filters. Test the phone photo in the same framing and lighting.
Keep labels and presentation type constant throughout each run.

Then repeat the pair under diffuse light (weak eye shadows), side lighting and dim light,
with several independent runs. Check the current model's crop, channel order, normalization
and class order against its exact export before ranking alternatives. Compare a different
model on the same recorded trials once a reproducible capture protocol is available.

These two runs do not justify replacing the existing verification gate with either temporal
method. Overlapping windows are correlated; the reported rates are descriptive and cannot
be treated as independent presentation-level APCER/BPCER or population accuracy.
