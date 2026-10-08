"""Evaluate every local image with each model; identity labels are optional."""
import argparse
import csv
import hashlib
import itertools
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import cv2
from core import ROOT, MODELS, load_model, normalize, cosine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default=str(ROOT / "data/images"))
    parser.add_argument("--labels", help="CSV with filename,person_id; blank IDs remain unknown")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not -1 <= args.threshold <= 1:
        parser.error("Threshold must be between -1 and 1")
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not paths:
        parser.error("No images found")
    labels = {}
    if args.labels:
        with open(args.labels, encoding="utf-8", newline="") as stream:
            labels = {row["filename"]: row["person_id"].strip() for row in csv.DictReader(stream)}
    output = ROOT / "results/image_evaluation.json"
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "threshold": args.threshold,
        "provider": "CPUExecutionProvider", "det_size": [640, 640], "det_threshold": 0.5,
        "platform": platform.platform(), "versions": {p: version(p) for p in
        ("insightface", "onnxruntime", "numpy", "opencv-python")},
        "timing": "Single pass after synthetic warm-up; decode, detection and recognition included.",
        "images": [{"filename": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        "models": {}}
    all_rows = []
    for name in MODELS:
        result = {"images": [], "pairs": []}
        embeddings = {}
        try:
            app = load_model(name)
            import numpy as np
            app.get(np.zeros((640, 640, 3), dtype=np.uint8))
            for path in paths:
                row = {"filename": path.name, "model": name, "success": False}
                start = perf_counter()
                try:
                    image = cv2.imread(str(path.resolve()))
                    if image is None:
                        raise ValueError("Image could not be read")
                    row["width"], row["height"] = image.shape[1], image.shape[0]
                    faces = app.get(image)
                    row["face_count"] = len(faces)
                    row["detection_scores"] = [float(f.det_score) for f in faces]
                    if len(faces) != 1:
                        row["failure"] = "no_face_detected" if not faces else "multiple_faces_detected"
                    else:
                        embeddings[path.name] = normalize(faces[0].embedding)
                        row["success"] = True
                        row["embedding_dimension"] = 512
                except Exception as error:
                    row["failure"] = f"{type(error).__name__}: {error}"
                row["seconds"] = perf_counter() - start
                result["images"].append(row)
                all_rows.append(row)
            for a, b in itertools.combinations(paths, 2):
                pair = {"image_a": a.name, "image_b": b.name}
                if a.name not in embeddings or b.name not in embeddings:
                    pair["status"] = "not_scored_extraction_failed"
                else:
                    score = cosine(embeddings[a.name], embeddings[b.name])
                    pair.update(status="scored", similarity=score, match=score >= args.threshold)
                    if labels.get(a.name) and labels.get(b.name):
                        pair["same_person"] = labels[a.name] == labels[b.name]
                        pair["correct"] = pair["same_person"] == pair["match"]
                result["pairs"].append(pair)
            valid = [p for p in result["pairs"] if "correct" in p]
            result.update(successful_images=sum(r["success"] for r in result["images"]),
                failed_images=sum(not r["success"] for r in result["images"]),
                labelled_scored_pairs=len(valid),
                accuracy=sum(p["correct"] for p in valid)/len(valid) if valid else None)
            del app
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        report["models"][name] = result
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(name, result.get("successful_images"), "successful;", result.get("failed_images"), "failed", result.get("error", ""))
    with (output.parent / "image_evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["model", "filename", "success", "face_count", "failure", "seconds", "width", "height"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    lines = ["# My evaluation of the available face images", "",
        f"I tested {len(paths)} images with all four Buffalo packs using CPU execution, 640 x 640 detection input and detection threshold 0.5.",
        "Success means exactly one face was detected and a valid embedding was extracted. It does not mean the identity was verified correctly.", "",
        "| Image | buffalo_l | buffalo_m | buffalo_s | buffalo_sc |", "| --- | --- | --- | --- | --- |"]
    for path in paths:
        cells = []
        for name in MODELS:
            row = next((r for r in report["models"][name]["images"] if r["filename"] == path.name), None)
            cells.append("Success" if row and row["success"] else (row.get("failure", "Failed") if row else "Model load failed"))
        lines.append("| " + " | ".join([path.name] + cells) + " |")
    lines += ["", "## My labelled same-person results", "",
        f"I used the supplied person labels with similarity threshold {args.threshold}. Unlabelled pairs are excluded.", "",
        "| Model | Same-person pairs | Correct matches | False rejections | Score range |",
        "| --- | ---: | ---: | ---: | --- |"]
    for name, result in report["models"].items():
        genuine = [p for p in result["pairs"] if p.get("same_person") is True]
        scores = [p["similarity"] for p in genuine]
        bounds = f"{min(scores):.3f} to {max(scores):.3f}" if scores else "Unavailable"
        lines.append(f"| {name} | {len(genuine)} | {sum(p['match'] for p in genuine)} | {sum(not p['match'] for p in genuine)} | {bounds} |")
    lines += ["", "My confirmed labels currently cover one person. These genuine-pair results measure acceptance on this person only; false accept rate needs confirmed different-person pairs.", "",
        "![Same-person pair scores](../results/graphs/genuine_pairs.png)", ""]
    lines += ["", "## My interpretation", "",
        "I recorded extraction failures separately from identity decisions. Pairwise scores are saved in the JSON report; pairs involving failed images are marked as not scored.",
        "Accuracy is calculated only for successfully scored pairs with supplied person labels. Without those labels I cannot judge correct matches or false matches.",
        "This small local dataset and single-pass timings do not establish general model accuracy or stable latency.", "",
        "![Per-image extraction outcomes](../results/graphs/image_evaluation.png)", "",
        "Detailed evidence: [JSON report](../results/image_evaluation.json) and [CSV outcomes](../results/image_evaluation.csv)."]
    (ROOT / "reports/IMAGE_EVALUATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot(report, paths)


def plot(report, paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    values = []
    annotations = []
    for path in paths:
        line, text = [], []
        for name in MODELS:
            row = next((r for r in report["models"][name]["images"] if r["filename"] == path.name), None)
            ok = bool(row and row["success"])
            line.append(int(ok))
            text.append("Success" if ok else ("No face" if row and row.get("face_count") == 0 else "Failed"))
        values.append(line)
        annotations.append(text)
    fig, ax = plt.subplots(figsize=(12, max(5, len(paths)*0.65 + 2)))
    ax.imshow(values, cmap=ListedColormap(["#fee2e2", "#dcfce7"]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), MODELS)
    ax.set_yticks(range(len(paths)), [p.name for p in paths])
    for i, line in enumerate(annotations):
        for j, value in enumerate(line):
            ax.text(j, i, value, ha="center", va="center", color="#1e293b")
    ax.set_title("Face detection and embedding extraction by image", pad=20, fontsize=16)
    fig.text(0.5, 0.025, "Success = exactly one detected face + valid embedding; identity accuracy requires person labels.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    folder = ROOT / "results/graphs"
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(folder / f"image_evaluation.{extension}", dpi=180)
    plt.close(fig)
    genuine = {name: [p for p in result["pairs"] if p.get("same_person") is True]
               for name, result in report["models"].items()}
    if any(genuine.values()):
        fig, ax = plt.subplots(figsize=(11, 6))
        for name, pairs in genuine.items():
            ax.plot(range(1, len(pairs)+1), [p["similarity"] for p in pairs], marker="o", label=name)
        ax.axhline(report["threshold"], linestyle="--", color="gray", label="Threshold")
        ax.set(xlabel="Same-person pair number (see report table)", ylabel="Cosine similarity",
               title="Confirmed same-person pairs across four models", ylim=(-1, 1.05))
        ax.legend()
        ax.grid(alpha=0.2)
        fig.text(0.5, 0.02, "One confirmed identity; these results do not estimate false accept rate or general accuracy.", ha="center")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        for extension in ("png", "svg"):
            fig.savefig(folder / f"genuine_pairs.{extension}", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
