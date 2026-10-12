"""Register one user from a photo or a webcam capture."""
import argparse
from datetime import datetime, timezone
from core import (
    MODELS,
    capture_webcam_frame,
    extract,
    load_model,
    read_database,
    save_database,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--name", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Path to a photo containing exactly one face")
    source.add_argument("--webcam", action="store_true", help="Capture one frame from webcam 0")
    parser.add_argument("--model", choices=MODELS, default="buffalo_l")
    parser.add_argument("--replace", action="store_true", help="Replace this user's existing model embedding")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    data = read_database()
    user = data["users"].get(args.user_id, {"name": args.name, "embeddings": {}})
    if args.model in user["embeddings"] and not args.replace:
        parser.error("User already registered for this model; use --replace to update")
    try:
        source = capture_webcam_frame() if args.webcam else args.image
        embedding = extract(load_model(args.model), source)
    except ValueError as error:
        parser.exit(2, f"Registration could not be completed: {error}\n")
    user["name"] = args.name
    user["embeddings"][args.model] = {"vector": embedding.tolist(),
        "created_at": datetime.now(timezone.utc).isoformat()}
    data["users"][args.user_id] = user
    save_database(data)
    print(f"Registered {args.user_id} using {args.model}")


if __name__ == "__main__":
    main()
