"""Interactive menu for InsightFace registration and liveness-protected verification."""

import sys
from datetime import datetime, timezone
from pathlib import Path


# The existing project modules live in scripts/ and use direct imports.
ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from anti_spoof import check_liveness  # noqa: E402
from core import (  # noqa: E402
    MODELS,
    capture_webcam_frame,
    cosine,
    extract,
    load_model,
    read_database,
    save_database,
)


def ask_required(label):
    while True:
        value = input(f"{label}: ").strip().strip('"')
        if value:
            return value
        print(f"{label} cannot be empty.")


def ask_model():
    print("\nInsightFace model:")
    for index, model in enumerate(MODELS, start=1):
        default = " (default)" if index == 1 else ""
        print(f"  {index}. {model}{default}")
    value = input("Choose model [1]: ").strip()
    if not value:
        return MODELS[0]
    if value.isdigit() and 1 <= int(value) <= len(MODELS):
        return MODELS[int(value) - 1]
    if value in MODELS:
        return value
    raise ValueError("Invalid model selection")


def ask_threshold(label, default, minimum, maximum, inclusive=False):
    value = input(f"{label} [{default}]: ").strip()
    result = default if not value else float(value)
    valid = minimum <= result <= maximum if inclusive else minimum < result < maximum
    if not valid:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return result


def get_source(use_webcam):
    if use_webcam:
        print("Webcam opened. Press Space to capture, or Q/Esc to cancel.")
        return capture_webcam_frame()
    return ask_required("Image path")


def register_user(use_webcam=False):
    """Interactively register one user from an image or webcam frame."""
    print("\n--- Register user ---")
    user_id = ask_required("User ID")
    name = ask_required("Name")
    model = ask_model()

    data = read_database()
    user = data["users"].get(user_id, {"name": name, "embeddings": {}})
    if model in user["embeddings"]:
        replace = input("Registration already exists. Replace it? [y/N]: ").strip().lower()
        if replace not in {"y", "yes"}:
            print("Registration cancelled.")
            return

    source = get_source(use_webcam)
    embedding = extract(load_model(model), source)
    user["name"] = name
    user["embeddings"][model] = {
        "vector": embedding.tolist(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    data["users"][user_id] = user
    save_database(data)
    print(f"\nSUCCESS: Registered {user_id} using {model}.")


def verify_user(use_webcam=False):
    """Interactively run MiniFASNet and then verify a claimed identity."""
    print("\n--- Verify user ---")
    user_id = ask_required("User ID")
    model = ask_model()
    similarity_threshold = ask_threshold(
        "Similarity threshold", 0.5, -1.0, 1.0, inclusive=True
    )
    liveness_threshold = ask_threshold("Liveness threshold", 0.5, 0.0, 1.0)

    user = read_database()["users"].get(user_id)
    if user is None or model not in user["embeddings"]:
        raise ValueError("No registration exists for this user and model")

    source = get_source(use_webcam)
    liveness = check_liveness(source, threshold=liveness_threshold)
    print(
        f"Liveness: {'REAL' if liveness.is_real else 'SPOOF'} "
        f"(real probability {liveness.real_probability:.6f})"
    )
    if not liveness.is_real:
        print("\nREJECTED: Possible spoof detected. InsightFace was not run.")
        return

    score = cosine(
        extract(load_model(model), source),
        user["embeddings"][model]["vector"],
    )
    print(f"Similarity: {score:.6f} (threshold {similarity_threshold})")
    print("\nMATCH" if score >= similarity_threshold else "\nNO MATCH")


MENU = {
    "1": ("Register using an image", lambda: register_user(use_webcam=False)),
    "2": ("Register using the webcam", lambda: register_user(use_webcam=True)),
    "3": ("Verify using an image", lambda: verify_user(use_webcam=False)),
    "4": ("Verify using the webcam", lambda: verify_user(use_webcam=True)),
}


def show_menu():
    print("\n========================================")
    print(" Face Registration and Verification")
    print("========================================")
    for number, (label, _) in MENU.items():
        print(f"  {number}. {label}")
    print("  0. Exit")


def main():
    while True:
        show_menu()
        choice = input("\nEnter option number: ").strip()
        if choice == "0":
            print("Goodbye.")
            return
        route = MENU.get(choice)
        if route is None:
            print("Invalid option. Enter a number from 0 to 4.")
            continue
        try:
            route[1]()
        except (ValueError, OSError) as error:
            print(f"\nERROR: {error}")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")


if __name__ == "__main__":
    main()
