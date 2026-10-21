"""Temporal behavior and descriptive metric accounting."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from test_liveness import RollingLiveness, summarize


class RollingWindowTests(unittest.TestCase):
    def test_full_window_required(self):
        window = RollingLiveness(size=3)
        self.assertEqual(window.update(0.9)["mean_decision"], "PENDING")
        self.assertEqual(window.update(0.9)["vote_decision"], "PENDING")
        self.assertEqual(window.update(0.9)["vote_decision"], "LIVE")

    def test_error_discards_previous_evidence(self):
        window = RollingLiveness(size=2)
        window.update(0.9)
        window.update(0.9)
        self.assertEqual(window.update(None)["window_count"], 0)
        self.assertEqual(window.update(0.9)["mean_decision"], "PENDING")

    def test_mean_and_vote_can_disagree(self):
        window = RollingLiveness(size=3, live_fraction=0.8)
        window.update(0.99)
        window.update(0.99)
        result = window.update(0.1)
        self.assertEqual(result["mean_decision"], "LIVE")
        self.assertEqual(result["vote_decision"], "SPOOF")

    def test_old_scores_expire(self):
        window = RollingLiveness(size=2)
        window.update(0.99)
        window.update(0.99)
        window.update(0.01)
        result = window.update(0.01)
        self.assertEqual(result["window_count"], 2)
        self.assertEqual(result["mean_decision"], "SPOOF")

    def test_consistently_wrong_scores_are_not_corrected_by_aggregation(self):
        window = RollingLiveness(size=3)
        for _ in range(10):
            result = window.update(0.99)
        self.assertEqual(result["mean_decision"], "LIVE")
        self.assertEqual(result["vote_decision"], "LIVE")

    def test_invalid_scores_reset(self):
        for probability in (float("nan"), float("inf"), -0.1, 1.1):
            window = RollingLiveness(size=2)
            window.update(0.9)
            self.assertEqual(window.update(probability)["window_count"], 0)

    def test_invalid_configuration(self):
        for kwargs in ({"size": 1}, {"threshold": 1}, {"live_fraction": 0}):
            with self.assertRaises(ValueError):
                RollingLiveness(**kwargs)

    def test_metrics_expose_errors_and_pending_without_counting_as_spoofs(self):
        rows = [
            {"single_decision": "LIVE", "mean_decision": "PENDING", "vote_decision": "PENDING", "error": ""},
            {"single_decision": "ERROR", "mean_decision": "PENDING", "vote_decision": "PENDING", "error": "no face"},
            {"single_decision": "SPOOF", "mean_decision": "LIVE", "vote_decision": "SPOOF", "error": ""},
        ]
        result = summarize(rows, "spoof")
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["single"]["false_live_rate_on_decided"], 0.5)
        self.assertEqual(result["mean"]["pending_or_error"], 2)
        self.assertEqual(result["mean"]["false_live_rate_on_decided"], 1)
        self.assertEqual(result["vote"]["false_live_rate_on_decided"], 0)
        self.assertIsNone(summarize(rows[:1], "live")["mean"]["false_spoof_rate_on_decided"])


if __name__ == "__main__":
    unittest.main()
