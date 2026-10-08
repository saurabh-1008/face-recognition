"""Verify a claimed user identity with the same model used for registration."""
import argparse
from core import MODELS, cosine, extract, load_model, read_database


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", choices=MODELS, default="buffalo_l")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not -1 <= args.threshold <= 1:
        parser.error("Threshold must be between -1 and 1")
    user = read_database()["users"].get(args.user_id)
    if user is None or args.model not in user["embeddings"]:
        parser.error("No registration for this user/model. Register with the selected model first.")
    try:
        score = cosine(extract(load_model(args.model), args.image),
                       user["embeddings"][args.model]["vector"])
    except ValueError as error:
        parser.exit(2, f"Verification could not be completed: {error}\n"
                    "Use a clear image containing one complete face. No similarity or match decision was calculated.\n")
    print(f"Model: {args.model}; similarity: {score:.6f}; threshold: {args.threshold}")
    print("MATCH" if score >= args.threshold else "NO MATCH")


if __name__ == "__main__":
    main()
