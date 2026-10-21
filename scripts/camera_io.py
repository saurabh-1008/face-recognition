"""Open a chosen webcam with Windows camera backend fallbacks."""
import sys


def open_webcam(camera_index=0, cv2_module=None, backend="auto"):
    if cv2_module is None:
        import cv2 as cv2_module
    if not isinstance(camera_index, int) or camera_index < 0:
        raise ValueError("Camera number must be a non-negative integer; start with 0")
    if backend not in ("auto", "dshow", "msmf", "default"):
        raise ValueError("Camera backend must be auto, dshow, msmf or default")
    choices = []
    if backend == "auto":
        if sys.platform == "win32":
            for name, constant in (("dshow", "CAP_DSHOW"), ("msmf", "CAP_MSMF")):
                value = getattr(cv2_module, constant, None)
                if value is not None:
                    choices.append((name, value))
        choices.append(("default", None))
    elif backend == "default":
        choices.append((backend, None))
    else:
        value = getattr(cv2_module, f"CAP_{backend.upper()}", None)
        if value is None:
            raise ValueError(f"Camera backend {backend} is unavailable in this OpenCV build")
        choices.append((backend, value))
    attempted = []
    for name, preference in choices:
        camera = None
        opened = False
        attempted.append(name)
        try:
            camera = (cv2_module.VideoCapture(camera_index) if preference is None
                      else cv2_module.VideoCapture(camera_index, preference))
            opened = camera.isOpened()
            if opened:
                return camera
        except Exception:
            # A backend can raise rather than return a closed capture. Try the next one.
            pass
        finally:
            if camera is not None and not opened:
                camera.release()
    raise ValueError(
        f"Cannot open webcam {camera_index} (tried {', '.join(attempted)}). "
        "Close other camera apps, check that the webcam is connected and its hardware privacy switch is open, "
        "and allow desktop apps to access the camera in Windows Settings. "
        "Use camera number 0 first; try 1 if a second camera is connected."
    )
