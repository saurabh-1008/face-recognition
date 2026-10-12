"""Tests for image/webcam input selection and interactive frame capture."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import core  # noqa: E402
import main as registration  # noqa: E402
import verify  # noqa: E402


class FakeCapture:
    def __init__(self, opened=True, frames=None):
        self.opened = opened
        self.frames = list(frames or [])
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


class FakeCv2:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 0

    def __init__(self, capture, keys):
        self.capture = capture
        self.keys = list(keys)
        self.destroyed = False
        self.camera_index = None

    def VideoCapture(self, camera_index):
        self.camera_index = camera_index
        return self.capture

    def putText(self, frame, *args, **kwargs):
        return frame

    def imshow(self, title, frame):
        pass

    def waitKey(self, delay):
        return self.keys.pop(0)

    def destroyAllWindows(self):
        self.destroyed = True


def capture_result(fake_cv2):
    """Turn a missing/broken capture API into an assertion-friendly result."""
    try:
        frame = core.capture_webcam_frame(cv2_module=fake_cv2)
        return "captured", frame
    except Exception as error:  # The test compares the externally visible outcome.
        return "error", f"{type(error).__name__}: {error}"


def parser_result(module, arguments):
    try:
        return "parsed", vars(module.build_parser().parse_args(arguments))
    except Exception as error:
        return "error", f"{type(error).__name__}: {error}"


class WebcamCaptureTests(unittest.TestCase):
    def test_space_captures_current_frame_and_releases_camera(self):
        frame = np.full((2, 2, 3), 7, dtype=np.uint8)
        capture = FakeCapture(frames=[frame])
        fake_cv2 = FakeCv2(capture, keys=[ord(" ")])

        status, result = capture_result(fake_cv2)

        self.assertEqual(status, "captured")
        np.testing.assert_array_equal(result, frame)
        self.assertEqual(fake_cv2.camera_index, 0)
        self.assertTrue(capture.released)
        self.assertTrue(fake_cv2.destroyed)

    def test_captured_frame_can_be_sent_directly_to_face_extraction(self):
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 2

        class FakeFaceApp:
            def get(self, image):
                return [SimpleNamespace(embedding=vector)]

        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        try:
            result = core.extract(FakeFaceApp(), frame)
            status = "extracted"
        except Exception as error:
            status = f"error: {type(error).__name__}: {error}"
            result = None

        self.assertEqual(status, "extracted")
        np.testing.assert_array_equal(result, np.r_[1.0, np.zeros(511)])

    def test_q_cancels_capture_and_releases_camera(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        capture = FakeCapture(frames=[frame])
        fake_cv2 = FakeCv2(capture, keys=[ord("q")])

        status, result = capture_result(fake_cv2)

        self.assertEqual(status, "error")
        self.assertEqual(result, "ValueError: Webcam capture cancelled")
        self.assertTrue(capture.released)
        self.assertTrue(fake_cv2.destroyed)

    def test_closed_camera_reports_clear_error(self):
        capture = FakeCapture(opened=False)
        fake_cv2 = FakeCv2(capture, keys=[])

        status, result = capture_result(fake_cv2)

        self.assertEqual(status, "error")
        self.assertEqual(result, "ValueError: Cannot open webcam 0")
        self.assertTrue(capture.released)
        self.assertTrue(fake_cv2.destroyed)


class CommandLineSourceTests(unittest.TestCase):
    def test_registration_accepts_webcam_instead_of_image(self):
        status, arguments = parser_result(
            registration,
            ["--user-id", "user_001", "--name", "Kumar", "--webcam"],
        )

        self.assertEqual(status, "parsed")
        self.assertTrue(arguments["webcam"])
        self.assertIsNone(arguments["image"])

    def test_verification_accepts_image_input(self):
        status, arguments = parser_result(
            verify,
            ["--user-id", "user_001", "--image", "face.jpg"],
        )

        self.assertEqual(status, "parsed")
        self.assertEqual(arguments["image"], "face.jpg")
        self.assertFalse(arguments["webcam"])


if __name__ == "__main__":
    unittest.main()
