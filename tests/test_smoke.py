"""Milestone M0 smoke tests for package scaffold, config validation, and CLI."""

from __future__ import annotations

import copy
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from eeg_p3 import __version__
from eeg_p3.cli import main
from eeg_p3.config import load_config, validate_config_dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "subject01.yaml"


class TestM0ScaffoldSmoke(unittest.TestCase):
    """Smoke tests verifying M0 repository scaffold and leakage guardrails."""

    def test_package_version(self) -> None:
        """Package exposes expected semantic version string."""
        self.assertEqual(__version__, "0.1.0")

    def test_load_default_config(self) -> None:
        """Default subject01.yaml loads and enforces Subject 1 Runs 4/8 dev and Run 12 test."""
        cfg = load_config(DEFAULT_CONFIG)
        self.assertEqual(cfg.data.subject_id, 1)
        self.assertEqual(cfg.data.development_runs, (4, 8))
        self.assertEqual(cfg.data.held_out_test_run, 12)
        self.assertNotIn(cfg.data.held_out_test_run, cfg.data.development_runs)
        self.assertEqual(cfg.data.label_mapping, {"T1": 0, "T2": 1})
        self.assertIn("T0", cfg.data.excluded_annotations)
        self.assertEqual(cfg.data.expected_channels, 64)
        self.assertEqual(cfg.data.expected_sfreq, 160.0)
        self.assertEqual(cfg.preprocessing.expected_n_times, 320)

    def test_leakage_in_config_is_rejected(self) -> None:
        """Config validator rejects held-out Run 12 inside development_runs."""
        cfg = load_config(DEFAULT_CONFIG)
        bad_raw = copy.deepcopy(cfg.raw)
        bad_raw["data"]["development_runs"] = [4, 8, 12]
        with self.assertRaises(ValueError):
            validate_config_dict(bad_raw)

    def test_wrong_t1_t2_mapping_is_rejected(self) -> None:
        """Config validator rejects inverted or missing T1/T2 mappings."""
        cfg = load_config(DEFAULT_CONFIG)
        bad_raw = copy.deepcopy(cfg.raw)
        bad_raw["data"]["label_mapping"] = {"T1": 1, "T2": 0}
        with self.assertRaises(ValueError):
            validate_config_dict(bad_raw)

    def test_cli_info_command_outputs_valid_json(self) -> None:
        """CLI 'info' subcommand runs cleanly and prints JSON metadata without reading EDFs."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = main(["info", "--config", str(DEFAULT_CONFIG)])
        self.assertEqual(exit_code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["subject_id"], 1)
        self.assertEqual(payload["development_runs"], [4, 8])
        self.assertEqual(payload["held_out_test_run"], 12)
        self.assertEqual(payload["expected_shape_per_trial"], [64, 320])

    def test_all_cli_subcommands_are_registered_in_m6(self) -> None:
        """All M0-M6 CLI subcommands are registered in the argument parser."""
        from eeg_p3.cli import build_parser

        parser = build_parser()
        help_text = parser.format_help()
        for cmd in (
            "info",
            "inspect-data",
            "plot-eeg",
            "run-baselines",
            "run-hosvd",
            "run-rks",
            "run-markov",
            "evaluate-locked",
            "build-report",
            "audit-cohort",
            "validate-cohort",
        ):
            self.assertIn(cmd, help_text)


if __name__ == "__main__":
    unittest.main()
