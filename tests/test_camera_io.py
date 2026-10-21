"""Backend fallback keeps failed captures released and preserves the chosen device."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from camera_io import open_webcam


class CameraBackendTests(unittest.TestCase):
    def test_windows_prefers_directshow_and_releases_failed_backend(self):
        failed = Mock()
        failed.isOpened.return_value = False
        working = Mock()
        working.isOpened.return_value = True
        cv = SimpleNamespace(CAP_DSHOW=700, CAP_MSMF=1400,
                             VideoCapture=Mock(side_effect=[failed, working]))
        with patch('camera_io.sys.platform', 'win32'):
            self.assertIs(open_webcam(1, cv), working)
        self.assertEqual(cv.VideoCapture.call_args_list, [call(1, 700), call(1, 1400)])
        failed.release.assert_called_once()
        working.release.assert_not_called()

    def test_all_failed_backends_are_released(self):
        captures = [Mock() for _ in range(3)]
        for capture in captures:
            capture.isOpened.return_value = False
        cv = SimpleNamespace(CAP_DSHOW=700, CAP_MSMF=1400,
                             VideoCapture=Mock(side_effect=captures))
        with patch('camera_io.sys.platform', 'win32'), self.assertRaisesRegex(ValueError, 'dshow, msmf, default'):
            open_webcam(0, cv)
        for capture in captures:
            capture.release.assert_called_once()

    def test_explicit_backend_does_not_fall_back_silently(self):
        capture = Mock()
        capture.isOpened.return_value = False
        cv = SimpleNamespace(CAP_DSHOW=700, VideoCapture=Mock(return_value=capture))
        with self.assertRaisesRegex(ValueError, 'tried dshow'):
            open_webcam(0, cv, backend='dshow')
        cv.VideoCapture.assert_called_once_with(0, 700)

    def test_invalid_camera_number_never_opens_capture(self):
        cv = SimpleNamespace(VideoCapture=Mock())
        with self.assertRaisesRegex(ValueError, 'non-negative'):
            open_webcam(-1, cv)
        cv.VideoCapture.assert_not_called()

    def test_backend_exception_does_not_prevent_next_attempt(self):
        working = Mock()
        working.isOpened.return_value = True
        cv = SimpleNamespace(CAP_DSHOW=700, CAP_MSMF=1400,
                             VideoCapture=Mock(side_effect=[RuntimeError('backend failed'), working]))
        with patch('camera_io.sys.platform', 'win32'):
            self.assertIs(open_webcam(0, cv), working)


if __name__ == '__main__':
    unittest.main()
