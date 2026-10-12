"""Tests for the interactive CLI menu routes."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cli  # noqa: E402


class InteractiveMenuTests(unittest.TestCase):
    def test_image_verification_route_is_selected(self):
        route = Mock()
        menu = {"3": ("Verify using an image", route)}
        with (
            patch.object(cli, "MENU", menu),
            patch("builtins.input", side_effect=["3", "0"]),
            patch.object(cli, "show_menu"),
        ):
            cli.main()

        route.assert_called_once_with()

    def test_invalid_option_returns_to_menu(self):
        with (
            patch("builtins.input", side_effect=["9", "0"]),
            patch.object(cli, "show_menu"),
            patch("builtins.print") as output,
        ):
            cli.main()

        self.assertTrue(
            any("Invalid option" in str(call) for call in output.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
