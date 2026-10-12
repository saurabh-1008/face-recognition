"""Tests for the MiniFASNet gate and its placement before InsightFace."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import anti_spoof  # noqa: E402
import verify  # noqa: E402


class FakeInput:
    name = "input"


class FakeSession:
    def __init__(self, logits):
        self.logits = np.asarray([logits], dtype=np.float32)
        self.received = None

    def get_inputs(self):
        return [FakeInput()]

    def run(self, outputs, inputs):
        self.received = inputs["input"]
        return [self.logits]


class MiniFASNetDecisionTests(unittest.TestCase):
    def _decision(self, logits, threshold=0.5):
        session = FakeSession(logits)
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        tensor = np.zeros((1, 3, 128, 128), dtype=np.float32)
        with (
            patch.object(anti_spoof, "_read_rgb", return_value=(image, "test")),
            patch.object(anti_spoof, "_load_detector", return_value=object()),
            patch.object(anti_spoof, "_detect_one_face", return_value=(0, 0, 100, 100)),
            patch.object(anti_spoof, "_expanded_square_crop", return_value=image),
            patch.object(anti_spoof, "_preprocess", return_value=tensor),
            patch.object(anti_spoof, "_load_liveness_session", return_value=session),
        ):
            result = anti_spoof.check_liveness(
                image,
                threshold=threshold,
                liveness_model="unused.onnx",
                detector_model="unused.onnx",
                cv2_module=object(),
                ort_module=object(),
            )
        return result, session

    def test_class_zero_is_real_and_passes_gate(self):
        result, session = self._decision([2.0, -1.0])

        self.assertTrue(result.is_real)
        self.assertGreater(result.real_probability, 0.95)
        self.assertEqual(session.received.shape, (1, 3, 128, 128))

    def test_class_one_is_spoof_and_fails_gate(self):
        result, _ = self._decision([-1.0, 2.0])

        self.assertFalse(result.is_real)
        self.assertLess(result.real_probability, 0.05)

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaisesRegex(anti_spoof.LivenessError, "threshold"):
            anti_spoof.check_liveness(np.zeros((1, 1, 3)), threshold=1.0)

    def test_large_image_is_downscaled_and_bbox_is_mapped_back(self):
        import cv2

        face = np.zeros(15, dtype=np.float32)
        face[:4] = [64, 32, 128, 128]
        face[14] = 0.95

        class FakeDetector:
            def __init__(self):
                self.input_size = None

            def setInputSize(self, size):
                self.input_size = size

            def detect(self, image):
                return None, np.asarray([face])

        detector = FakeDetector()
        image = np.zeros((1000, 500, 3), dtype=np.uint8)

        bbox = anti_spoof._detect_one_face(image, detector, cv2)

        self.assertEqual(detector.input_size, (320, 640))
        self.assertEqual(bbox, (100, 50, 300, 250))


class VerificationGateTests(unittest.TestCase):
    def test_spoof_result_stops_before_insightface(self):
        database = {
            "users": {
                "user_001": {
                    "embeddings": {"buffalo_l": {"vector": [0.0] * 512}}
                }
            }
        }
        spoof = anti_spoof.LivenessResult(False, 0.02, -1.0, 2.0)
        insightface_loader = Mock()
        arguments = ["verify.py", "--user-id", "user_001", "--image", "face.jpg"]

        with (
            patch.object(sys, "argv", arguments),
            patch.object(verify, "read_database", return_value=database),
            patch.object(verify, "check_liveness", return_value=spoof),
            patch.object(verify, "load_model", insightface_loader),
            self.assertRaises(SystemExit) as exit_result,
        ):
            verify.main()

        self.assertEqual(exit_result.exception.code, 3)
        insightface_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
