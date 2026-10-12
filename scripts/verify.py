"""Verify a claimed user identity with the same model used for registration."""
import argparse
from anti_spoof import DETECTOR_MODEL, LIVENESS_MODEL, check_liveness
from core import MODELS, capture_webcam_frame, cosine, extract, load_model, read_database


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Path to a photo containing exactly one face")
    source.add_argument("--webcam", action="store_true", help="Capture one frame from webcam 0")
    parser.add_argument("--model", choices=MODELS, default="buffalo_l")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--liveness-threshold", type=float, default=0.5,
                        help="Minimum MiniFASNet real-face probability (default: 0.5)")
    parser.add_argument("--liveness-model", default=str(LIVENESS_MODEL),
                        help="Path to the MiniFASNet ONNX classifier")
    parser.add_argument("--liveness-detector", default=str(DETECTOR_MODEL),
                        help="Path to the YuNet ONNX face detector")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not -1 <= args.threshold <= 1:
        parser.error("Threshold must be between -1 and 1")
    if not 0 < args.liveness_threshold < 1:
        parser.error("Liveness threshold must be greater than 0 and less than 1")
    user = read_database()["users"].get(args.user_id)
    if user is None or args.model not in user["embeddings"]:
        parser.error("No registration for this user/model. Register with the selected model first.")
    try:
        source = capture_webcam_frame() if args.webcam else args.image
        liveness = check_liveness(
            source,
            threshold=args.liveness_threshold,
            liveness_model=args.liveness_model,
            detector_model=args.liveness_detector,
        )
        print(
            f"Liveness: {'REAL' if liveness.is_real else 'SPOOF'}; "
            f"real probability: {liveness.real_probability:.6f}; "
            f"threshold: {args.liveness_threshold}"
        )
        if not liveness.is_real:
            parser.exit(3, "Verification rejected: possible spoof detected. "
                        "InsightFace was not run.\n")
        score = cosine(extract(load_model(args.model), source),
                       user["embeddings"][args.model]["vector"])
    except ValueError as error:
        parser.exit(2, f"Verification could not be completed: {error}\n"
                    "Use a clear image containing one complete face. No similarity or match decision was calculated.\n")
    print(f"Model: {args.model}; similarity: {score:.6f}; threshold: {args.threshold}")
    print("MATCH" if score >= args.threshold else "NO MATCH")


if __name__ == "__main__":
    main()
