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
from liveness_models import MODEL_SPECS, list_models  # noqa: E402
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


LIGHTING_OPTIONS = (
    ("unspecified", "Not recorded / unsure"),
    ("front", "Light facing you"),
    ("side", "Light from one side"),
    ("dim", "Low light / dark room"),
    ("backlit", "Bright light or window behind you"),
    ("diffuse", "Soft, even light with weak shadows"),
    ("overhead", "Light from above"),
    ("mixed", "Light from several different sources"),
    ("daylight", "Natural daylight"),
)
ATTACK_OPTIONS = (
    ("none", "Real person; no attack"),
    ("phone-photo", "Still face photo displayed on a phone"),
    ("print", "Printed face photo / paper"),
    ("replay", "Face video playing on a phone or screen"),
    ("monitor-photo", "Still face photo displayed on a computer screen"),
    ("mask", "Face mask / 3D presentation"),
    ("unknown", "Attack type not known"),
)


def ask_annotation(label, options, default):
    """Choose a readable condition while retaining stable report values."""
    print(f"\n{label} (recorded in the report):")
    default_index = next(i for i, (value, _) in enumerate(options, 1) if value == default)
    for index, (value, description) in enumerate(options, 1):
        suffix = " (default)" if value == default else ""
        print(f"  {index}. {value}: {description}{suffix}")
    other_index = len(options) + 1
    print(f"  {other_index}. Other: enter your own description")
    while True:
        value = input(f"Choose {label.lower()} [{default_index}]: ").strip()
        if not value:
            return default
        if value.lower() in ("other", str(other_index)):
            return ask_required(f"Describe {label.lower()}")
        if value.isdigit() and 1 <= int(value) <= len(options):
            return options[int(value) - 1][0]
        if value.lower() in {key for key, _ in options}:
            return value.lower()
        print(f"Choose a number from 1 to {other_index}, an option name, or Other.")


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


def test_liveness():
    """Choose an anti-spoofing model and generate a shared experiment report."""
    from test_liveness import build_parser, run

    print("\n--- Liveness model test ---")
    names = list(MODEL_SPECS)
    for index, name in enumerate(names, 1):
        spec = MODEL_SPECS[name]
        status = "installed" if spec.path.is_file() else "needs download"
        print(f"  {index}. {name}: {spec.title} ({status})")
    value = input("Choose liveness model [1]: ").strip() or "1"
    if value.isdigit() and 1 <= int(value) <= len(names):
        name = names[int(value) - 1]
    elif value in MODEL_SPECS:
        name = value
    else:
        raise ValueError("Invalid liveness model selection")
    if not MODEL_SPECS[name].path.is_file():
        raise ValueError(f"Install this model first: python scripts/liveness_models.py --download {name}")
    source = input("Source: 1 webcam, 2 image, 3 video [1]: ").strip() or "1"
    arguments = ["--model", name]
    if source == "1":
        camera = int(input("Camera number [0]: ").strip() or "0")
        arguments += ["--webcam", "--camera", str(camera)]
    elif source in ("2", "3"):
        arguments += ["--image" if source == "2" else "--video", ask_required("File path")]
    else:
        raise ValueError("Invalid source selection")
    label = input("Known ground truth: live / spoof / unknown [unknown]: ").strip().lower() or "unknown"
    if label not in ("live", "spoof", "unknown"):
        raise ValueError("Ground truth must be live, spoof or unknown")
    lighting = ask_annotation("Lighting", LIGHTING_OPTIONS, "unspecified")
    attack = ask_annotation("Attack type", ATTACK_OPTIONS, "unknown" if label == "spoof" else "none")
    arguments += ["--label", label, "--lighting", lighting, "--attack", attack]
    threshold = ask_threshold("Liveness threshold", MODEL_SPECS[name].threshold, 0.0, 1.0)
    arguments += ["--threshold", str(threshold)]
    if source != "2":
        seconds = input("Duration in seconds [20]: ").strip() or "20"
        size = input("Rolling window in samples [15]: ").strip() or "15"
        rule = input("Final decision rule: single / mean / vote [vote]: ").strip() or "vote"
        if rule not in ("single", "mean", "vote"):
            raise ValueError("Decision rule must be single, mean or vote")
        # Validate numeric entries before handing them to the argument parser.
        arguments += ["--seconds", str(float(seconds)), "--window", str(int(size)), "--decision", rule]
    run(build_parser().parse_args(arguments))


MENU = {
    "1": ("Register using an image", lambda: register_user(use_webcam=False)),
    "2": ("Register using the webcam", lambda: register_user(use_webcam=True)),
    "3": ("Verify using an image", lambda: verify_user(use_webcam=False)),
    "4": ("Verify using the webcam", lambda: verify_user(use_webcam=True)),
    "5": ("Test a liveness model and generate reports", test_liveness),
    "6": ("List liveness models", list_models),
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
            print("Invalid option. Choose one of the displayed menu numbers.")
            continue
        try:
            route[1]()
        except (ValueError, OSError) as error:
            print(f"\nERROR: {error}")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")


if __name__ == "__main__":
    main()
