"""Validate distinct model contracts and the common report/CLI workflow."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import cli
import liveness_models as models
import test_liveness as runner
from anti_spoof import LivenessError


class ModelContractTests(unittest.TestCase):
    def test_probability_outputs_are_not_softmaxed_again(self):
        self.assertAlmostEqual(models.probability_from_output([[0.9, 0.1]], models.MODEL_SPECS['anti-spoof-mn3']), 0.9)
        self.assertAlmostEqual(models.probability_from_output([[0.997]], models.MODEL_SPECS['tinyliveness']), 0.997)
        self.assertGreater(models.probability_from_output([[2, -1]], models.MODEL_SPECS['minifasnet']), 0.95)

    def test_malformed_or_nonprobability_outputs_are_rejected(self):
        cases = [('anti-spoof-mn3', [[0.9, 0.9]]), ('anti-spoof-mn3', [[-1, 2]]),
                 ('tinyliveness', [[float('nan')]]), ('tinyliveness', [[0.8, 0.2]]),
                 ('tinyliveness', [[1.2]])]
        for name, output in cases:
            with self.subTest(name=name, output=output), self.assertRaises(LivenessError):
                models.probability_from_output(output, models.MODEL_SPECS[name])

    def test_each_alternative_uses_its_rgb_normalization(self):
        pixels = np.full((128, 128, 3), [255, 128, 0], dtype=np.uint8)
        tensor = models.preprocess(pixels, models.MODEL_SPECS['anti-spoof-mn3'], cv2)
        expected = (np.array([255, 128, 0]) - [151.2405, 119.595, 107.8395]) / [63.0105, 56.457, 55.0035]
        np.testing.assert_allclose(tensor[0, :, 0, 0], expected, atol=1e-5)
        tensor = models.preprocess(pixels, models.MODEL_SPECS['tinyliveness'], cv2)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        expected = (np.array([255, 128, 0]) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        np.testing.assert_allclose(tensor[0, :, 0, 0], expected, atol=1e-5)
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)

    def test_degenerate_landmarks_are_rejected(self):
        with self.assertRaises(LivenessError):
            models.align_face(np.zeros((224, 224, 3), np.uint8), np.zeros((5, 2)), cv2)

    def test_missing_model_stops_before_opening_camera(self):
        with tempfile.TemporaryDirectory() as folder:
            args = runner.build_parser().parse_args(['--webcam', '--liveness-model', str(Path(folder) / 'missing.onnx')])
            with patch('cv2.VideoCapture') as camera, self.assertRaisesRegex(LivenessError, 'Model missing'):
                runner.run(args)
            camera.assert_not_called()


class SharedReportTests(unittest.TestCase):
    def image_report(self, folder, model, label='unknown', prediction=None):
        args = runner.build_parser().parse_args([
            '--model', model, '--image', 'face.jpg', '--label', label,
            '--output', str(Path(folder) / f'{model}_{label}.csv')])
        engine = Mock()
        engine.metadata = {'sha256': 'test-hash', 'crop': 'test crop', 'name': model}
        if isinstance(prediction, Exception):
            engine.predict.side_effect = prediction
        else:
            engine.predict.return_value = prediction or SimpleNamespace(real_probability=0.999, is_real=True)
        with patch.object(runner, 'LivenessEngine', return_value=engine) as loader, \
             patch('cv2.imread', return_value=np.zeros((128, 128, 3), dtype=np.uint8)), \
             patch('cv2.VideoCapture') as camera, contextlib.redirect_stdout(io.StringIO()):
            runner.run(args)
        camera.assert_not_called()
        self.assertEqual(loader.call_args.args[0], model)
        self.assertEqual(loader.call_args.args[1], models.MODEL_SPECS[model].threshold)
        output = args.output
        self.assertTrue(output.is_file())
        self.assertTrue(output.with_suffix('.md').is_file())
        return json.loads(output.with_suffix('.json').read_text()), args

    def test_each_selected_model_generates_the_same_report_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            for model in models.MODEL_SPECS:
                report, args = self.image_report(folder, model)
                self.assertEqual(report['result']['decision'], 'LIVE')
                self.assertEqual(report['result']['method'], 'single')
                self.assertEqual(report['stopped'], 'image_complete')
                self.assertEqual(report['model']['name'], model)
                self.assertEqual(report['metrics']['samples'], 1)
                self.assertIsNone(report['metrics']['single']['false_live_rate_on_decided'])
                with self.assertRaisesRegex(ValueError, 'new .csv'):
                    runner.run(args)

    def test_ground_truth_does_not_determine_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            report, _ = self.image_report(folder, 'tinyliveness', label='spoof')
            self.assertEqual(report['result']['decision'], 'LIVE')
            self.assertEqual(report['metrics']['single']['false_live_rate_on_decided'], 1)

    def test_face_error_produces_inconclusive_report(self):
        with tempfile.TemporaryDirectory() as folder:
            report, _ = self.image_report(folder, 'anti-spoof-mn3', prediction=LivenessError('No complete face'))
            self.assertEqual(report['result']['decision'], 'INCONCLUSIVE')
            self.assertEqual(report['metrics']['errors'], 1)

    def test_single_image_cannot_be_used_for_a_temporal_decision(self):
        args = runner.build_parser().parse_args(['--image', 'face.jpg', '--decision', 'vote'])
        with self.assertRaisesRegex(ValueError, 'Images support'):
            runner.run(args)

    def test_interactive_selection_passes_selected_model_to_shared_runner(self):
        # Model, webcam, camera, ground truth, lighting, attack, threshold, duration, window, rule.
        answers = ['anti-spoof-mn3', '1', '', 'spoof', '6', '2', '', '', '', '']
        with patch('builtins.input', side_effect=answers), patch.object(runner, 'run') as run, \
             patch.object(Path, 'is_file', return_value=True), contextlib.redirect_stdout(io.StringIO()):
            cli.test_liveness()
        args = run.call_args.args[0]
        self.assertEqual(args.model, 'anti-spoof-mn3')
        self.assertEqual(args.label, 'spoof')
        self.assertEqual(args.lighting, 'diffuse')
        self.assertEqual(args.attack, 'phone-photo')
        self.assertEqual(args.decision, 'vote')


if __name__ == '__main__':
    unittest.main()
