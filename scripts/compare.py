"""Compare model packs on a CSV of labelled image pairs; no database writes."""
import argparse
import csv
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from core import ROOT, MODEL_ROOT, MODELS, cosine, extract, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="CSV: image_a,image_b,same_person (1 or 0)")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default=str(ROOT / "results" / "comparison.json"))
    args = parser.parse_args()
    if not -1 <= args.threshold <= 1:
        parser.error("Threshold must be between -1 and 1")
    pairs_path = Path(args.pairs).resolve()
    with pairs_path.open(newline="", encoding="utf-8") as stream:
        pairs = list(csv.DictReader(stream))
    if not pairs or any(p.get("same_person") not in ("0", "1") or
                        not p.get("image_a") or not p.get("image_b") for p in pairs):
        parser.error("Provide nonempty pairs with image_a,image_b,same_person=0/1")
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
        "threshold": args.threshold, "provider": "CPUExecutionProvider",
        "det_size": [640, 640], "platform": platform.platform(),
        "versions": {p: version(p) for p in ("insightface", "onnxruntime", "numpy", "opencv-python")},
        "models": {}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        result = {"pairs": []}
        try:
            start = perf_counter()
            app = load_model(name)
            result["load_seconds_including_download"] = perf_counter() - start
            result["onnx_size_mb"] = sum(p.stat().st_size for p in
                (MODEL_ROOT / "models" / name).rglob("*.onnx")) / 1_000_000
            # Warm up separately; timed extraction includes image decode and detection.
            extract(app, pairs_path.parent / pairs[0]["image_a"])
            durations = []
            tp = tn = fp = fn = 0
            for pair in pairs:
                row = dict(pair)
                try:
                    start = perf_counter()
                    a = extract(app, pairs_path.parent / pair["image_a"])
                    b = extract(app, pairs_path.parent / pair["image_b"])
                    elapsed = perf_counter() - start
                    durations.append(elapsed / 2)
                    score = cosine(a, b)
                    actual, predicted = pair["same_person"] == "1", score >= args.threshold
                    tp += int(actual and predicted)
                    tn += int(not actual and not predicted)
                    fp += int(not actual and predicted)
                    fn += int(actual and not predicted)
                    row.update(similarity=score, match=predicted, pair_seconds=elapsed)
                except (ValueError, OSError) as error:
                    row["error"] = str(error)
                result["pairs"].append(row)
            count = tp + tn + fp + fn
            result.update(valid_pairs=count, failed_pairs=len(pairs)-count,
                true_positive=tp, true_negative=tn, false_positive=fp, false_negative=fn,
                accuracy=(tp+tn)/count if count else None,
                false_accept_rate=fp/(fp+tn) if fp+tn else None,
                false_reject_rate=fn/(fn+tp) if fn+tp else None,
                mean_image_seconds=sum(durations)/len(durations) if durations else None)
            del app
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        report["models"][name] = result
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(name, json.dumps({k:v for k,v in result.items() if k != "pairs"}))
    print(f"Report: {output.resolve()}")
    if any("error" in r or r.get("failed_pairs", 0) for r in report["models"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
