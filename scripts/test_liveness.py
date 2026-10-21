"""Compare single-frame and rolling-window liveness on a webcam or video."""
import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from anti_spoof import DETECTOR_MODEL, LivenessError
from liveness_models import MODEL_SPECS, LivenessEngine, list_models
from camera_io import open_webcam


class RollingLiveness:
    """Require a full consecutive valid window; reset on detection/inference error."""

    def __init__(self, size=15, threshold=0.5, live_fraction=0.8):
        if size < 2 or not 0 < threshold < 1 or not 0 < live_fraction <= 1:
            raise ValueError("Use window >= 2, threshold in (0,1), live fraction in (0,1]")
        self.scores = deque(maxlen=size)
        self.size = size
        self.threshold = threshold
        self.live_fraction = live_fraction

    def update(self, probability):
        if probability is None or not math.isfinite(probability) or not 0 <= probability <= 1:
            self.scores.clear()
        else:
            self.scores.append(probability)
        mean = float(np.mean(self.scores)) if self.scores else None
        fraction = (sum(p >= self.threshold for p in self.scores) / len(self.scores)
                    if self.scores else None)
        ready = len(self.scores) == self.size
        return {
            "window_count": len(self.scores),
            "window_mean": mean,
            "window_live_fraction": fraction,
            "mean_decision": ("LIVE" if mean >= self.threshold else "SPOOF") if ready else "PENDING",
            "vote_decision": ("LIVE" if fraction >= self.live_fraction else "SPOOF") if ready else "PENDING",
        }


def summarize(rows, label):
    """Descriptive frame/window rates, not independent trial accuracy."""
    result = {"samples": len(rows), "errors": sum(bool(r["error"]) for r in rows)}
    for method in ("single", "mean", "vote"):
        decisions = [r[f"{method}_decision"] for r in rows]
        live = decisions.count("LIVE")
        spoof = decisions.count("SPOOF")
        decided = live + spoof
        result[method] = {
            "live": live, "spoof": spoof, "pending_or_error": len(rows) - decided,
            "decided": decided,
            "false_live_rate_on_decided": live / decided if label == "spoof" and decided else None,
            "false_spoof_rate_on_decided": spoof / decided if label == "live" and decided else None,
        }
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--webcam", action="store_true")
    source.add_argument("--video", type=Path)
    source.add_argument("--image", type=Path)
    parser.add_argument("--model", choices=MODEL_SPECS, default="minifasnet")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--decision", choices=("single", "mean", "vote"),
                        help="Final decision rule; default single for images, vote for webcam/video")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-backend", choices=("auto", "dshow", "msmf", "default"), default="auto",
                        help="Camera interface; auto tries DirectShow, MSMF, then default on Windows")
    parser.add_argument("--label", choices=("live", "spoof", "unknown"), default="unknown",
                        help="Known ground truth; keep the same subject/presentation for the entire run")
    parser.add_argument("--lighting", default="unspecified", help="For example front, side, dim, backlit")
    parser.add_argument("--attack", default="none", help="For example none, print, phone-photo, replay")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--sample-fps", type=float, default=5)
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--threshold", type=float, help="Override the selected model's default threshold")
    parser.add_argument("--live-fraction", type=float, default=0.8)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--output", type=Path, help="New CSV path; existing files are never overwritten")
    parser.add_argument("--liveness-model", type=Path, help="Override weights; must match --model's contract")
    parser.add_argument("--liveness-detector", type=Path, default=DETECTOR_MODEL)
    return parser


def run(args):
    import cv2

    if sum(bool(value) for value in (args.webcam, args.video, args.image)) != 1:
        raise ValueError("Choose exactly one of --webcam, --video or --image")
    if args.webcam and args.camera < 0:
        raise ValueError("Camera number must be non-negative; start with 0")
    if args.image and args.decision in ("mean", "vote"):
        raise ValueError("Images support --decision single only; temporal rules need webcam or video")
    args.decision = args.decision or ("single" if args.image else "vote")
    args.threshold = MODEL_SPECS[args.model].threshold if args.threshold is None else args.threshold
    window = RollingLiveness(args.window, args.threshold, args.live_fraction)
    if (not math.isfinite(args.seconds) or args.seconds <= 0
            or not math.isfinite(args.sample_fps) or args.sample_fps <= 0):
        raise ValueError("seconds and sample-fps must be finite and positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path(__file__).resolve().parents[1] / "results" / "liveness" / f"{args.model}_{stamp}.csv"
    report = output.with_suffix(".json")
    markdown = output.with_suffix(".md")
    if output.suffix.lower() != ".csv" or any(path.exists() for path in (output, report, markdown)):
        raise ValueError("Choose a new .csv output path; its .json and .md companions must also be new")
    engine = LivenessEngine(args.model, args.threshold, args.liveness_model, args.liveness_detector)
    print(f"Model: {args.model}; threshold: {args.threshold}; final rule: {args.decision}")
    camera = None
    rows = []
    reason = "duration"
    try:
        if args.webcam:
            camera = open_webcam(args.camera, cv2_module=cv2, backend=args.camera_backend)
            engine.metadata["camera_backend"] = camera.getBackendName()
        elif args.video:
            camera = cv2.VideoCapture(str(args.video))
        if camera is not None and not camera.isOpened():
            raise ValueError(f"Cannot open {'video' if args.video else 'webcam'}")
        video_fps = camera.get(cv2.CAP_PROP_FPS) if args.video else None
        if args.video and (not math.isfinite(video_fps) or video_fps <= 0):
            raise ValueError("Video has no valid frame rate; use a constant-frame-rate video")
        start = time.monotonic()
        next_sample = 0.0
        frame_index = 0
        last_sample = None
        status = "Waiting for a full window"
        print("Test started. Keep one presentation in view. Q/Esc stops. Scores only; no images saved.")
        while True:
            if args.image:
                if rows:
                    reason = "image_complete"
                    break
                frame = cv2.imread(str(args.image.resolve()))
                ok = frame is not None
                if not ok:
                    raise ValueError(f"Cannot read image: {args.image}")
            else:
                ok, frame = camera.read()
            if not ok or frame is None:
                if args.video:
                    reason = "end_of_video"
                    break
                raise ValueError("Webcam frame read failed")
            elapsed = 0.0 if args.image else (frame_index / video_fps if args.video else time.monotonic() - start)
            frame_index += 1
            if elapsed >= args.seconds:
                break
            if elapsed + 1e-9 >= next_sample:
                next_sample = elapsed + 1 / args.sample_fps
                # Do not carry evidence across long processing/capture gaps.
                if last_sample is not None and elapsed - last_sample > 2 / args.sample_fps:
                    window.update(None)
                last_sample = elapsed
                row = {"sample": len(rows) + 1, "seconds": round(elapsed, 4),
                       "model": args.model,
                       "label": args.label, "lighting": args.lighting, "attack": args.attack,
                       "real_probability": None, "single_decision": "ERROR", "error": ""}
                before = time.perf_counter()
                try:
                    result = engine.predict(frame)
                    row["real_probability"] = result.real_probability
                    row["single_decision"] = "LIVE" if result.is_real else "SPOOF"
                except LivenessError as error:
                    row["error"] = str(error)
                row["inference_ms"] = round((time.perf_counter() - before) * 1000, 3)
                row.update(window.update(row["real_probability"]))
                rows.append(row)
                probability = row["real_probability"]
                score = "n/a" if probability is None else f"{probability:.3f}"
                status = (f"p(real)={score} single={row['single_decision']} "
                          f"mean={row['mean_decision']} vote={row['vote_decision']} "
                          f"[{row['window_count']}/{args.window}]")
                print(f"{elapsed:6.2f}s {status}")
            if not args.no_preview and not args.image:
                preview = frame.copy()
                cv2.putText(preview, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("Liveness experiment - Q/Esc to stop", preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    reason = "user_stopped"
                    break
    except KeyboardInterrupt:
        reason = "interrupted"
    except Exception:
        reason = "error"
        raise
    finally:
        if camera is not None:
            camera.release()
        if not args.no_preview and not args.image:
            cv2.destroyAllWindows()
        if rows:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            verdict = rows[-1][f"{args.decision}_decision"]
            if reason in ("error", "interrupted") or verdict not in ("LIVE", "SPOOF"):
                verdict = "INCONCLUSIVE"
            summary = {"report_version": 2,
                       "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                       "model": engine.metadata,
                       "result": {"decision": verdict, "method": args.decision,
                                  "basis": "Last sampled frame/window at end of run; labels never affect prediction",
                                  "last_real_probability": rows[-1]["real_probability"]},
                       "stopped": reason, "metrics": summarize(rows, args.label),
                       "note": "Overlapping windows are correlated. These rates are descriptive, not trial-level APCER/BPCER."}
            with report.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(summary, indent=2, allow_nan=False))
            write_markdown(markdown, summary)
            print(f"\nRESULT: {verdict} ({args.decision})")
            print(f"Saved {output}\nSummary {report}\nReadable report {markdown}")
    if not rows:
        raise ValueError("No frames were evaluated; no results were saved")
    print(json.dumps(summarize(rows, args.label), indent=2))


def write_markdown(path, summary):
    config, result, metrics = summary["config"], summary["result"], summary["metrics"]
    lines = [f"# Liveness report: {config['model']}", "",
             f"**Result: {result['decision']}** (rule: {result['method']}; threshold: {config['threshold']})", "",
             "This is the last sampled frame/window decision, not a whole-run authentication result.", "",
             f"Ground truth: {config['label']}; lighting: {config['lighting']}; attack: {config['attack']}.", "",
             f"Samples: {metrics['samples']}; errors: {metrics['errors']}; stopped: {summary['stopped']}.", "",
             "| Method | LIVE | SPOOF | Pending/errors |", "| --- | ---: | ---: | ---: |"]
    for method in ("single", "mean", "vote"):
        counts = metrics[method]
        lines.append(f"| {method} | {counts['live']} | {counts['spoof']} | {counts['pending_or_error']} |")
    lines += ["", "Errors and pending windows never count as LIVE. Overlapping windows are correlated.",
              "These are experimental predictions; ground-truth labels are used only for evaluation.", "",
              f"Model SHA-256: `{summary['model']['sha256']}`", "",
              f"Crop: {summary['model']['crop']}", "",
              "The JSON companion contains full settings, model provenance and descriptive error rates."]
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.list_models:
        list_models()
        return
    try:
        run(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Liveness experiment failed: {error}\n")


if __name__ == "__main__":
    main()
