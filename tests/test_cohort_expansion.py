"""Stage 1 (Milestone E1) tests for expanded cohort config, eligibility auditing, and split manifest."""

from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np

from eeg_p3.cli import run_build_report
from eeg_p3.config import load_config, validate_config_dict
from eeg_p3.data import (
    CANONICAL_EEGBCI_64_CHANNELS,
    audit_cohort_eligibility,
    audit_single_subject_eligibility,
    build_cohort_manifest_report,
    combine_recording_results,
    prepare_cohort_dataset,
)
from eeg_p3.evaluation import (
    build_run_swapped_splits,
    compute_participant_grouped_metrics,
    run_e2_cohort_development_validation,
    run_m5_locked_evaluation,
    select_best_dev_configuration,
)
from eeg_p3.preprocessing import (
    extract_epochs_from_filtered_run,
    filter_raw_run_independently,
    format_global_trial_id,
    format_recording_id,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_CONFIG = PROJECT_ROOT / "configs" / "subject01.yaml"
COHORT_CONFIG = PROJECT_ROOT / "configs" / "cohort_expanded.yaml"


def _make_canonical_synthetic_raw(
    subject_id: int,
    run_id: int,
    sfreq: float = 160.0,
    channel_names: tuple[str, ...] = CANONICAL_EEGBCI_64_CHANNELS,
    descriptions: list[str] | None = None,
    onsets: list[float] | None = None,
) -> mne.io.RawArray:
    """Build a synthetic MNE RawArray with canonical 64 EEGBCI channel names for unit testing ONLY."""
    rng = np.random.default_rng(1000 * subject_id + run_id)
    n_channels = len(channel_names)
    duration_sec = 25.0
    n_samples = int(round(duration_sec * sfreq))
    signals = rng.normal(0.0, 15e-6, size=(n_channels, n_samples))

    info = mne.create_info(ch_names=list(channel_names), sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(signals, info, verbose="ERROR")

    ann_onsets = onsets if onsets is not None else [1.0, 5.0, 9.0, 13.0, 17.0]
    ann_descs = descriptions if descriptions is not None else ["T0", "T1", "T2", "T1", "T2"]
    ann_durations = [3.0] * len(ann_onsets)
    raw.set_annotations(
        mne.Annotations(onset=ann_onsets, duration=ann_durations, description=ann_descs)
    )
    return raw


class TestE1CohortExpansionAndAudit(unittest.TestCase):
    """Unit tests for expanded cohort config, eligibility auditing, metadata, and split invariants."""

    def test_expanded_cohort_config_excludes_subject1_and_preserves_pilot(self) -> None:
        """cohort_expanded.yaml excludes Subject 1 by default and considers Subjects 2..109."""
        pilot_cfg = load_config(PILOT_CONFIG)
        self.assertFalse(pilot_cfg.data.is_multi_subject)
        self.assertEqual(pilot_cfg.data.subject_id, 1)
        self.assertEqual(pilot_cfg.data.candidate_subject_ids, (1,))

        cohort_cfg = load_config(COHORT_CONFIG)
        self.assertTrue(cohort_cfg.data.is_multi_subject)
        self.assertEqual(cohort_cfg.data.excluded_subject_ids, (1,))
        self.assertNotIn(1, cohort_cfg.data.candidate_subject_ids)
        self.assertEqual(cohort_cfg.data.candidate_subject_ids, tuple(range(2, 110)))
        self.assertEqual(len(cohort_cfg.data.candidate_subject_ids), 108)
        self.assertFalse(cohort_cfg.data.allow_download_fallback)
        self.assertEqual(cohort_cfg.data.development_runs, (4, 8))
        self.assertEqual(cohort_cfg.data.held_out_test_run, 12)

    def test_config_rejects_non_hand_imagery_runs_and_auto_download(self) -> None:
        """Validator rejects executed-movement or feet-imagery runs and automatic downloads."""
        cohort_cfg = load_config(COHORT_CONFIG)

        bad_runs = copy.deepcopy(cohort_cfg.raw)
        bad_runs["data"]["development_runs"] = [4, 6, 8]  # Run 6 is fists vs feet
        with self.assertRaises(ValueError):
            validate_config_dict(bad_runs)

        bad_dl = copy.deepcopy(cohort_cfg.raw)
        bad_dl["data"]["allow_download_fallback"] = True
        with self.assertRaises(ValueError):
            validate_config_dict(bad_dl)

    def test_unique_recording_and_trial_ids_and_chronological_order(self) -> None:
        """Recording IDs distinguish same run across subjects and enforce chronological cue order."""
        self.assertEqual(format_recording_id(2, 4), "S002R04")
        self.assertEqual(format_recording_id(3, 4), "S003R04")
        self.assertNotEqual(format_recording_id(2, 4), format_recording_id(3, 4))
        self.assertEqual(format_global_trial_id(2, 4, 0), "S002R04_T00")

        raw_s2 = _make_canonical_synthetic_raw(subject_id=2, run_id=4)
        res_s2 = extract_epochs_from_filtered_run(raw_s2, subject_id=2, run_id=4)
        self.assertEqual(res_s2.recording_id, "S002R04")
        self.assertEqual(res_s2.trials_metadata[0].recording_id, "S002R04")
        self.assertEqual(res_s2.trials_metadata[0].global_trial_id, "S002R04_T00")
        self.assertEqual(res_s2.trials_metadata[1].global_trial_id, "S002R04_T01")

        # Verify strictly increasing onset_sec across extracted trials
        onsets = [m.onset_sec for m in res_s2.trials_metadata]
        self.assertEqual(onsets, sorted(onsets))

    def test_eligibility_audit_logs_exact_reasons_for_exclusions(self) -> None:
        """Eligibility auditor accepts valid participants and logs specific reasons for invalid ones."""
        base_cfg = load_config(COHORT_CONFIG)
        raw_dict = copy.deepcopy(base_cfg.raw)
        raw_dict["data"]["candidate_subject_ids"] = [1, 2, 3, 4, 5, 6]
        raw_dict["data"]["excluded_subject_ids"] = [1]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            raw_dict["data"]["local_data_dir"] = str(tmp_root)
            cfg = replace(
                base_cfg,
                data=replace(
                    base_cfg.data,
                    local_data_dir=tmp_root,
                    candidate_subject_ids=(2, 3, 4, 5, 6),
                    excluded_subject_ids=(1,),
                ),
                raw=raw_dict,
            )

            # Create dummy EDF placeholders for subjects 2, 3, 4, 5 (subject 6 has missing R12 file)
            for sid in (2, 3, 4, 5):
                sdir = tmp_root / f"S{sid:03d}"
                sdir.mkdir(parents=True, exist_ok=True)
                for rid in (4, 8, 12):
                    (sdir / f"S{sid:03d}R{rid:02d}.edf").write_bytes(b"DUMMY")
            s6dir = tmp_root / "S006"
            s6dir.mkdir(parents=True, exist_ok=True)
            (s6dir / "S006R04.edf").write_bytes(b"DUMMY")
            (s6dir / "S006R08.edf").write_bytes(b"DUMMY")
            # S006R12.edf intentionally omitted -> missing_edf_file

            def _mock_read_raw_edf(edf_path_str: str, preload: bool = True, verbose: str = "ERROR"):
                name = Path(edf_path_str).stem  # e.g. S002R04
                sid = int(name[1:4])
                rid = int(name[5:7])
                if sid in (2, 3):
                    return _make_canonical_synthetic_raw(subject_id=sid, run_id=rid, sfreq=160.0)
                if sid == 4:
                    # Wrong sampling rate (128 Hz instead of 160 Hz)
                    return _make_canonical_synthetic_raw(subject_id=sid, run_id=rid, sfreq=128.0)
                if sid == 5:
                    # Missing class T2 (only T0 and T1 present)
                    return _make_canonical_synthetic_raw(
                        subject_id=sid,
                        run_id=rid,
                        sfreq=160.0,
                        descriptions=["T0", "T1", "T1", "T1", "T0"],
                    )
                raise RuntimeError(f"Unexpected mock EDF read for {edf_path_str}")

            with (
                patch("mne.io.read_raw_edf", side_effect=_mock_read_raw_edf),
                patch("mne.datasets.eegbci.standardize", side_effect=lambda r: r),
            ):
                records = audit_cohort_eligibility(cfg, workspace_data_dir=tmp_root)
                by_id = {r.subject_id: r for r in records}

                self.assertFalse(by_id[1].eligible)
                self.assertEqual(by_id[1].reason_code, "excluded_pilot_subject")

                self.assertTrue(by_id[2].eligible)
                self.assertEqual(by_id[2].reason_code, "ok")
                self.assertTrue(by_id[3].eligible)
                self.assertEqual(by_id[3].reason_code, "ok")

                self.assertFalse(by_id[4].eligible)
                self.assertEqual(by_id[4].reason_code, "sfreq_mismatch")

                self.assertFalse(by_id[5].eligible)
                self.assertEqual(by_id[5].reason_code, "missing_class_annotations")

                self.assertFalse(by_id[6].eligible)
                self.assertEqual(by_id[6].reason_code, "missing_edf_file")

                # Prepare multi-participant dataset from eligible subjects (2 and 3)
                dataset = prepare_cohort_dataset(cfg, workspace_data_dir=tmp_root)
                self.assertEqual(dataset.subject_ids, (2, 3))
                # 2 subjects * 2 dev runs * 4 trials/run = 16 dev trials
                self.assertEqual(dataset.dev.X.shape, (16, 64, 320))
                # 2 subjects * 1 test run * 4 trials/run = 8 test trials
                self.assertEqual(dataset.test.X.shape, (8, 64, 320))

                # Verify Option A split manifest and run-swapped folds
                manifest = build_cohort_manifest_report(cfg, records, workspace_data_dir=tmp_root)
                self.assertEqual(manifest["cohort_audit_summary"]["eligible_subject_ids"], [2, 3])
                self.assertEqual(manifest["cohort_audit_summary"]["n_eligible_subjects"], 2)
                self.assertEqual(manifest["cohort_audit_summary"]["n_excluded_subjects"], 4)

                splits = build_run_swapped_splits(
                    groups=dataset.dev.groups,
                    development_runs=(4, 8),
                    forbidden_test_run=12,
                    recording_ids=dataset.dev.recording_ids,
                    subject_ids=dataset.dev.subject_ids,
                )
                self.assertEqual(splits[0].train_recording_ids, ("S002R04", "S003R04"))
                self.assertEqual(splits[0].val_recording_ids, ("S002R08", "S003R08"))
                self.assertEqual(splits[1].train_recording_ids, ("S002R08", "S003R08"))
                self.assertEqual(splits[1].val_recording_ids, ("S002R04", "S003R04"))

    def test_build_run_swapped_splits_rejects_recording_leakage_or_subject_mismatch(self) -> None:
        """build_run_swapped_splits raises ValueError if a recording ID spans train and val."""
        groups = np.array([4, 4, 8, 8], dtype=np.int64)
        # Intentionally reuse 'S002R04' inside run 8 to test recording leakage detection
        bad_recording_ids = ("S002R04", "S003R04", "S002R04", "S003R08")
        with self.assertRaises(ValueError):
            build_run_swapped_splits(groups, recording_ids=bad_recording_ids)

        # Intentionally mismatch participants between Run 4 (subjects 2, 3) and Run 8 (subjects 2, 4)
        good_recording_ids = ("S002R04", "S003R04", "S002R08", "S004R08")
        bad_subject_ids = np.array([2, 3, 2, 4], dtype=np.int64)
        with self.assertRaises(ValueError):
            build_run_swapped_splits(
                groups, recording_ids=good_recording_ids, subject_ids=bad_subject_ids
            )

    def test_build_report_never_silently_triggers_final_test_evaluation(self) -> None:
        """run_build_report raises FileNotFoundError when M5 output is absent instead of running M5."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = Path(tmpdir)
            with self.assertRaises(FileNotFoundError):
                run_build_report(
                    config_path=PILOT_CONFIG,
                    metrics_dir=empty_dir,
                    figures_dir=empty_dir / "figures",
                )

    def test_participant_grouped_metrics_and_deterministic_tie_breaking(self) -> None:
        """Participant-grouped uncertainty computes valid 95% CIs and tie-breaking is deterministic."""
        import pandas as pd

        # 3 participants, 4 trials each (12 trials total)
        subject_ids = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=np.int64)
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        # Subj 2: 4/4 (100%), Subj 3: 3/4 (75%), Subj 4: 2/4 (50%)
        y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1], dtype=np.int64)

        summary, per_subj = compute_participant_grouped_metrics(y_true, y_pred, subject_ids)
        self.assertEqual(summary["n_participants"], 3)
        self.assertAlmostEqual(summary["participant_mean_accuracy"], 0.75)
        self.assertAlmostEqual(summary["participant_std_accuracy"], 0.25)
        self.assertLess(summary["participant_ci95_accuracy"][0], 0.75)
        self.assertGreater(summary["participant_ci95_accuracy"][1], 0.75)
        self.assertEqual(len(per_subj), 3)

        # Deterministic tie-breaking: identical mean_f1_macro/accuracy -> prefers smaller n_features, then smaller C
        candidates_df = pd.DataFrame([
            {
                "config_id": "cfg_large_feat",
                "params_json": '{"C": 1.0}',
                "C": 1.0,
                "n_features": 256,
                "mean_f1_macro": 0.70,
                "mean_balanced_accuracy": 0.70,
                "mean_accuracy": 0.70,
                "participant_mean_f1_macro": 0.70,
                "std_f1_macro": 0.02,
            },
            {
                "config_id": "cfg_compact_feat_high_c",
                "params_json": '{"C": 10.0}',
                "C": 10.0,
                "n_features": 128,
                "mean_f1_macro": 0.70,
                "mean_balanced_accuracy": 0.70,
                "mean_accuracy": 0.70,
                "participant_mean_f1_macro": 0.70,
                "std_f1_macro": 0.02,
            },
            {
                "config_id": "cfg_compact_feat_low_c",
                "params_json": '{"C": 0.1}',
                "C": 0.1,
                "n_features": 128,
                "mean_f1_macro": 0.70,
                "mean_balanced_accuracy": 0.70,
                "mean_accuracy": 0.70,
                "participant_mean_f1_macro": 0.70,
                "std_f1_macro": 0.02,
            },
        ])
        best = select_best_dev_configuration(candidates_df)
        self.assertEqual(best["config_id"], "cfg_compact_feat_low_c")

    def test_e3_multi_participant_locked_evaluation_on_synthetic_cohort(self) -> None:
        """run_m5_locked_evaluation supports multi-participant cohort with participant CIs and seed variability."""
        base_cfg = load_config(COHORT_CONFIG)
        raw_dict = copy.deepcopy(base_cfg.raw)
        raw_dict["data"]["candidate_subject_ids"] = [2, 3, 4]
        raw_dict["data"]["excluded_subject_ids"] = [1]
        # Compact hyperparameter grid for fast unit test execution
        raw_dict["baselines"]["linear_svc_c_grid"] = [0.1]
        raw_dict["baselines"]["rbf_svc_c_grid"] = [1.0]
        raw_dict["baselines"]["rbf_svc_gamma_grid"] = ["scale"]
        raw_dict["hosvd"]["channel_ranks"] = [4]
        raw_dict["hosvd"]["time_ranks"] = [10]
        raw_dict["hosvd"]["linear_svc_c_grid"] = [0.1]
        raw_dict["rks"]["n_components_grid"] = [128]
        raw_dict["rks"]["gamma_grid"] = [0.01]
        raw_dict["rks"]["linear_svc_c_grid"] = [1.0]
        raw_dict["rks"]["random_seeds"] = [42, 1337]
        raw_dict["markov"]["n_states_grid"] = [3]
        raw_dict["markov"]["linear_svc_c_grid"] = [0.1]
        raw_dict["markov"]["hosvd_fusion_r_c"] = 4
        raw_dict["markov"]["hosvd_fusion_r_t"] = 10
        raw_dict["markov"]["rks_fusion_dim"] = 128
        raw_dict["markov"]["rks_fusion_gamma"] = 0.01

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            raw_dict["data"]["local_data_dir"] = str(tmp_root)
            cfg = replace(
                base_cfg,
                data=replace(
                    base_cfg.data,
                    local_data_dir=tmp_root,
                    candidate_subject_ids=(2, 3, 4),
                    excluded_subject_ids=(1,),
                ),
                raw=raw_dict,
            )

            for sid in (2, 3, 4):
                sdir = tmp_root / f"S{sid:03d}"
                sdir.mkdir(parents=True, exist_ok=True)
                for rid in (4, 8, 12):
                    (sdir / f"S{sid:03d}R{rid:02d}.edf").write_bytes(b"DUMMY")

            def _mock_read_raw_edf(edf_path_str: str, preload: bool = True, verbose: str = "ERROR"):
                name = Path(edf_path_str).stem
                sid = int(name[1:4])
                rid = int(name[5:7])
                return _make_canonical_synthetic_raw(subject_id=sid, run_id=rid, sfreq=160.0)

            with (
                patch("mne.io.read_raw_edf", side_effect=_mock_read_raw_edf),
                patch("mne.datasets.eegbci.standardize", side_effect=lambda r: r),
            ):
                dataset = prepare_cohort_dataset(cfg, workspace_data_dir=tmp_root)
                _, _, _, _, e2_rep = run_e2_cohort_development_validation(dataset, cfg)
                locked_df, rank_df, m5_rep = run_m5_locked_evaluation(
                    dataset=dataset,
                    cfg=cfg,
                    precomputed_e2_report=e2_rep,
                )

                self.assertEqual(len(locked_df), 7)
                self.assertIn("test_balanced_accuracy", locked_df.columns)
                self.assertIn("test_participant_mean_accuracy", locked_df.columns)
                self.assertIn("test_participant_ci95_accuracy", locked_df.columns)
                self.assertEqual(m5_rep["n_eligible_participants"], 3)
                # 7 models * 3 subjects = 21 per-participant test rows
                self.assertEqual(len(m5_rep["per_participant_test_results"]), 21)
                self.assertEqual(
                    m5_rep["rks_predetermined_seed_variability"]["predetermined_seeds"],
                    [42, 1337],
                )
                self.assertEqual(
                    rank_df.iloc[0]["analysis_type"],
                    "exploratory_rank_curve_bases_fitted_on_dev_runs_4_8_only",
                )


if __name__ == "__main__":
    unittest.main()
