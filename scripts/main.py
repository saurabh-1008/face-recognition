"""Register one user from one image."""
import argparse
from datetime import datetime, timezone
from core import MODELS, extract, load_model, read_database, save_database


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", choices=MODELS, default="buffalo_l")
    parser.add_argument("--replace", action="store_true", help="Replace this user's existing model embedding")
    args = parser.parse_args()
    data = read_database()
    user = data["users"].get(args.user_id, {"name": args.name, "embeddings": {}})
    if args.model in user["embeddings"] and not args.replace:
        parser.error("User already registered for this model; use --replace to update")
    embedding = extract(load_model(args.model), args.image)
    user["name"] = args.name
    user["embeddings"][args.model] = {"vector": embedding.tolist(),
        "created_at": datetime.now(timezone.utc).isoformat()}
    data["users"][args.user_id] = user
    save_database(data)
    print(f"Registered {args.user_id} using {args.model}")


if __name__ == "__main__":
    main()
