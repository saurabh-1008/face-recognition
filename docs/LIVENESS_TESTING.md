# Test and compare liveness models in one project

Run these commands from the project root in PowerShell. All three models use the existing Python environment,
OpenCV, NumPy and ONNX Runtime. There is no separate project or environment per model.
Experiments save scores, not camera images, and do not register or verify identities.

## Select a model

Start `.\.venv\Scripts\python.exe cli.py` and choose **5: Test a liveness model and generate reports**.
Select the model, webcam/image/video input, optional ground truth and lighting description, and settings.
Lighting and attack type now show numbered choices with plain-language descriptions. Enter a number
or the option name; choose **Other** to enter a custom description. Lighting choices include front,
side, dim, backlit, diffuse, overhead, mixed and daylight. Attack choices include none, phone-photo,
print, replay, monitor-photo, mask and unknown. Spoof-labelled tests default to unknown attack type
instead of none; select the actual attack for useful comparisons. These choices annotate the report
and do not change lighting, model settings or predictions.
Option **6** lists models. The InsightFace model selector in registration/identity verification is separate.
Existing verification options 3/4 continue to use the MiniFASNet gate.

| Script model name | Architecture | Default threshold | Input / output |
| --- | --- | ---: | --- |
| `minifasnet` | Existing MiniFASNet V2 SE baseline | 0.5 | 128px RGB /255; two logits, class 0 real |
| `anti-spoof-mn3` | MobileNetV3, CelebA-Spoof | 0.5 | 128px RGB with channel mean/std; two probabilities, class 0 real |
| `tinyliveness` | EfficientNet-B0 release | 0.99 | 224px aligned RGB, ImageNet normalization; one live probability |

The weights are installed locally. On a fresh checkout, download classifier weights with:

```powershell
.\.venv\Scripts\python.exe scripts/liveness_models.py --download all
.\.venv\Scripts\python.exe scripts/test_liveness.py --list-models
```

Downloads require network access and verify registered SHA-256 hashes before writing.
The shared YuNet detector remains `models/minifasnet/detector_quantized.onnx`; install it as described in
[RUNNING.md](RUNNING.md) if missing. ONNX weights are excluded from Git. Model-specific directories contain
weights, notices and documentation only; all inference code lives in `scripts/liveness_models.py`.

Run any supported model by changing only `--model`:

```powershell
.\.venv\Scripts\python.exe scripts/test_liveness.py --model anti-spoof-mn3 --webcam --label live --lighting front
.\.venv\Scripts\python.exe scripts/test_liveness.py --model tinyliveness --webcam --label spoof --attack phone-photo --lighting diffuse
.\.venv\Scripts\python.exe scripts/test_liveness.py --model minifasnet --image data/images/face.jpg --label unknown
```

Replace example input paths with existing files. `--label` defaults to `unknown` and `--lighting` defaults to
`unspecified`; labels are annotations for error metrics and never influence predictions.
Thresholds are experimental defaults, not directly comparable confidence calibrations. TinyLiveness's 0.99
default follows its pinned release policy. Use `--threshold` to override and record the change.

## First two runs

Present your real face under even frontal lighting:

```powershell
.\.venv\Scripts\python.exe scripts/test_liveness.py --webcam --label live --lighting front --seconds 20
```

Then repeat with a phone displaying a face photo, under the same lighting:

```powershell
.\.venv\Scripts\python.exe scripts/test_liveness.py --webcam --label spoof --attack phone-photo --lighting front --seconds 20
```

The preview starts immediately after models load. Keep one person/presentation in view for the whole run.
Use Q or Esc to finish early. Close the existing webcam preview first if another application is using the camera.
Use `--camera 1` if your camera is not device 0.

On Windows the default camera selection tries DirectShow, then Media Foundation, then OpenCV's
automatic selection. To force an interface, use `--camera-backend dshow` or `--camera-backend msmf`.
If camera opening fails, close other camera apps, confirm camera access for desktop apps is enabled
in Windows Settings, check the hardware privacy switch, and retry with camera 0.
After updating the code, exit and restart an already-running CLI so it loads these changes.
An FFmpeg/libavdevice camera warning concerns camera opening, not a failed liveness classification.

Each sampled video/webcam frame displays three separate experimental decisions (MiniFASNet defaults below):

| Decision | Default rule |
| --- | --- |
| Single | Current frame score >= 0.5 |
| Mean | Mean score of the last 15 consecutive valid samples >= 0.5 |
| Vote | At least 80% (12/15) of those samples have a score >= 0.5 |

The target is 5 evaluated frames/second; a full window spans about 3 seconds.
Actual sampling can be slower on a busy CPU. Processing/capture gaps longer than two sampling intervals reset the window.
Missing faces and inference errors reset the window. `PENDING` means insufficient consecutive evidence, never acceptance.
The model and detector load once per run.

## Lighting and attack matrix

Repeat both commands with `--lighting side`, `dim`, `backlit`, and `diffuse`.
Change the physical lighting yourself; the flag records your description only.
Include the condition where eye-socket shadows disappear. Keep distance, face size, and camera settings as consistent as possible.
Repeat with a printed photo (`--attack print`) and a screen video (`--attack replay`).
For each condition, collect at least five separate runs for each presentation type and several people/devices before drawing conclusions.
Do not change presentation mid-run: stop and start another labelled run.

Repeat with windows 5, 10, and 15 using `--window`. Keep threshold fixed for your initial comparison.
Record the single-frame baseline alongside every temporal method on the same frames.
Tune on development recordings, then evaluate on different held-out recordings; avoid selecting and reporting a threshold on the same examples.

To replay an existing constant-frame-rate recording without a preview:

```powershell
.\.venv\Scripts\python.exe scripts/test_liveness.py --video data/videos/phone_front.mp4 --label spoof --attack phone-photo --lighting front --seconds 20 --no-preview
```

Video sampling uses frame index and the file's reported FPS, not inference speed. Convert variable-frame-rate clips to constant frame rate first.
Each method receives identical evaluated frames. This tool does not record webcam video.

## Read the results

New model-prefixed, timestamped CSV, JSON and Markdown files are saved in `results/liveness/`.
Supply `--output results/liveness/my_run.csv`
to choose a name; existing results cannot be overwritten.
CSV contains every sampled score, decision, processing time, window mean/vote fraction, and error.
JSON records the configuration, stop reason, counts, and descriptive error rates:

- For a spoof run: false-live rate among decided samples/windows.
- For a real-face run: false-spoof rate among decided samples/windows.
- Error and pending counts are separate; always inspect coverage alongside the rates.

The terminal and Markdown report show `LIVE`, `SPOOF` or `INCONCLUSIVE`. The result uses the **last sampled
frame/window** at the end of the run, not a majority over the whole recording. Choose `--decision single`,
`mean`, or `vote`; video/webcam defaults to vote. Images use single-frame decisions only.
Errors or incomplete windows yield `INCONCLUSIVE`; a processing failure or keyboard interruption also
invalidates the final result. The JSON records the chosen rule, actual model threshold, model and detector
hashes, source, crop convention, adapter version, and runtime versions. Existing reports are not modified.

Overlapping windows are correlated: these are descriptive rates, not independent presentation-level APCER/BPCER.
Do not interpret a run with no completed windows as good performance. There is no trial-level authentication decision in this tool.
It also does not track identity across frames: use one stable presentation per run. It is an experiment, not a replacement verification gate.

Smoothing reduces intermittent score changes only. If the spoof scores consistently exceed the threshold under flat lighting,
both temporal rules can continue to accept it. A rolling average does not learn motion, depth, or physiological cues.
The eye-shadow explanation remains a hypothesis until controlled testing rules out framing, exposure, screen reflections, and other changes.

## Adapter provenance and extending the system

The supported alternatives are not proven improvements for this camera:

- [Lightweight Face Anti Spoofing](https://github.com/kprokofi/light-weight-face-anti-spoofing):
  MobileNetV3 trained on CelebA-Spoof, with a webcam/video demo, cross-dataset evaluation, and ONNX conversion.
  Its different architecture/training makes it a useful next baseline. Published cross-domain results do not establish lighting robustness here.
- [Open Model Zoo anti-spoof-mn3 contract](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/anti-spoof-mn3):
  the original ONNX export expects RGB bytes normalized with mean `[151.2405,119.595,107.8395]`
  and std `[63.0105,56.457,55.0035]`. Outputs are probabilities; do not softmax them again.
  Our adapter uses YuNet instead of the original demo's SSD detector, retaining its 1.1x/1.05x crop expansion.
- [TinyLiveness pinned release](https://github.com/yuvrajraina/TinyLiveness/tree/8d92198e7ee122f39b2585e691a3d9450ca0cb0f):
  its API expects an externally aligned face. Our adapter supplies YuNet landmarks and a least-squares
  five-point similarity transform to a 224px ArcFace template. This alignment is our integration choice;
  the upstream benchmark is not reproduced by this integration. Its checked-in release policy uses a
  single 0.99 threshold (reject and accept thresholds are equal).

The MiniFASNet adapter preserves the original crop, interpolation, normalization and RGB detector input
for baseline continuity. The alternative adapters use BGR detector input. Compare these as complete
pipelines; detector/color/crop differences can affect results independently of classifier architecture.

To add a further model, add a `ModelSpec` entry, implement its crop/preprocessing and output interpretation
in the adapter, and test those contracts with representative outputs. The shared runner/CLI discovers
registry entries automatically. Models requiring a different runtime or actual temporal tensors need a
dedicated adapter; arbitrary ONNX files cannot be safely interpreted from shape alone.

`--liveness-model` overrides the selected adapter's weights only. It must match `--model`'s exact contract;
the runtime validates shape and values, and reports whether the file matches the registered release hash.
