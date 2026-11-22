"""Synthetic mathematical, provenance, and workflow tests; not EEG evidence."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from eeg_svd.cli import main
from eeg_svd.config import DEFAULTS, check_config, digest, file_hash, new_directory, read_json, write_json
from eeg_svd.data import audit, epochs_from_raw, resolve_edf
from eeg_svd.evaluation import develop, evaluate, mean_ci, metrics, verify_lock
from eeg_svd.features import bandpower
from eeg_svd.models import make_pipeline, select_models, valid_ranks
from eeg_svd.reporting import build_report
from eeg_svd.svd import SVDProjector


@pytest.fixture
def cfg():
    return {**deepcopy(DEFAULTS), 'data_dir': '/synthetic-only', 'ranks': [2, 4],
            'c_values': [.1, 1.], 'gammas': ['scale', .01], 'timing_repeats': 2}


@pytest.mark.parametrize('field,value', [('prior_test_exposure', False), ('sfreq', 128),
    ('ranks', [0]), ('gammas', [-1]), ('c_values', [float('nan')])])
def test_invalid_configuration(cfg, field, value):
    with pytest.raises(ValueError):
        check_config({**cfg, field: value})


def test_no_overwrite(tmp_path):
    new_directory(tmp_path / 'experiment')
    with pytest.raises(FileExistsError):
        new_directory(tmp_path / 'experiment')


def test_bandpower_frequency_and_amplitude(cfg):
    t = np.arange(320) / 160
    X = np.broadcast_to(1e-5 * np.sin(2 * np.pi * 10 * t), (2, 64, 320)).copy()
    X[1] *= 2
    F = bandpower(X, cfg)
    assert F.shape == (2, 128)
    assert np.all(F[:, :64] > F[:, 64:])
    np.testing.assert_allclose(F[1, :64] - F[0, :64], np.log10(4), atol=1e-8)
    assert np.isfinite(bandpower(np.zeros_like(X), cfg)).all()


@pytest.mark.parametrize('rank', [1, 2, 4])
def test_svd_matches_optimal_reconstruction(rank):
    A = StandardScaler().fit_transform(np.random.default_rng(2).normal(size=(9, 7)))
    svd = SVDProjector(rank).fit(A)
    s = np.linalg.svd(A, compute_uv=False)
    assert svd.transform(A).shape == (9, rank)
    assert svd.reconstruction_error(A) == pytest.approx(np.sqrt(np.sum(s[rank:]**2) / np.sum(s**2)))
    np.testing.assert_allclose(svd.components_ @ svd.components_.T, np.eye(rank), atol=1e-12)


def test_rank_deficiency():
    with pytest.raises(ValueError):
        SVDProjector(2).fit(np.ones((8, 3)))
    with pytest.raises(ValueError):
        SVDProjector(1).fit(np.zeros((8, 3)))


def test_test_values_do_not_change_fitted_transforms():
    rng = np.random.default_rng(5)
    F = rng.normal(size=(12, 8))
    model = make_pipeline(dict(family='svd_rbf', rank=3, C=1., gamma='scale'))
    model.fit(F, np.tile([0, 1], 6))
    mean = model.named_steps['scale'].mean_.copy()
    basis = model.named_steps['svd'].components_.copy()
    model.predict(rng.normal(size=(4, 8)) * 1e6)
    np.testing.assert_array_equal(model.named_steps['scale'].mean_, mean)
    np.testing.assert_array_equal(model.named_steps['svd'].components_, basis)
    np.testing.assert_allclose(mean, F.mean(axis=0))


def test_search_rejects_run12_and_is_deterministic(cfg):
    rng = np.random.default_rng(6)
    F = {r: rng.normal(size=(4, 8)) for r in (4, 8)}
    y = {r: np.array([0, 1, 0, 1]) for r in F}
    assert valid_ranks(F, cfg) == [2]
    with pytest.raises(ValueError):
        select_models({**F, 12: F[4]}, y, cfg)
    best1, rows = select_models(F, y, cfg)
    best2, _ = select_models(F, y, cfg)
    assert best1 == best2
    assert all(r['invalid_ranks'] == [4] for r in rows)


def test_lock_tampering():
    manifest = dict(config={}, eligible_subjects=[2])
    lock = dict(manifest_sha256=digest(manifest), config={}, selections={'2': {}})
    lock['lock_sha256'] = digest(lock)
    verify_lock(lock, manifest)
    lock['selections']['2'] = {'tampered': True}
    with pytest.raises(ValueError):
        verify_lock(lock, manifest)


def test_metrics_and_uncertainty():
    m = metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert m['accuracy'] == .75
    assert m['confusion_matrix'] == [[1, 1], [0, 2]]
    assert mean_ci([.75])['ci95'] is None
    assert mean_ci([.5, .5, .5])['ci95'] == [.5, .5]


def raw_fixture(seed=42, sfreq=160.):
    names = mne.channels.make_standard_montage(
        'colin27_1005' if 'colin27_1005' in mne.channels.get_builtin_montages() else 'standard_1005').ch_names[:64]
    data = np.random.default_rng(seed).normal(0, 1e-5, (64, int(14 * sfreq)))
    raw = mne.io.RawArray(data, mne.create_info(names, sfreq, 'eeg'), verbose='ERROR')
    raw.set_annotations(mne.Annotations([0, 1, 4, 7, 10], [1] * 5,
                                       ['T0', 'T1', 'T2', 'T1', 'T2']))
    return raw


@pytest.fixture
def dataset(tmp_path, cfg):
    cfg.update(data_dir=str(tmp_path / 'data'), subject_start=2, subject_stop=3,
               ranks=[2], c_values=[1.], gammas=['scale'])
    raws = {}
    for s in (2, 3):
        folder = Path(cfg['data_dir']) / f'S{s:03d}'
        folder.mkdir(parents=True)
        for r in (4, 8, 12):
            path = folder / f'S{s:03d}R{r:02d}.edf'
            path.write_text('SYNTHETIC FIXTURE ONLY; NOT A REAL EDF')
            raws[str(path.resolve())] = raw_fixture(s * 100 + r)
    def reader(path, **kwargs):
        return raws[str(Path(path).resolve())].copy()
    return cfg, raws, reader


def test_epochs(cfg):
    X, y, rows = epochs_from_raw(raw_fixture(), cfg, 2, 4, filtered=False)
    assert X.shape == (4, 64, 320)
    assert y.tolist() == [0, 1, 0, 1]
    assert rows[0]['start_sample'] == 240
    assert len({r['trial_id'] for r in rows}) == 4
    with pytest.raises(ValueError, match='sfreq_mismatch'):
        epochs_from_raw(raw_fixture(sfreq=128), cfg, 2, 4)
    with pytest.raises(ValueError, match='Wrong task'):
        epochs_from_raw(raw_fixture(), cfg, 2, 3)


def test_invalid_annotations_and_bounds(cfg):
    raw = raw_fixture()
    raw.set_annotations(mne.Annotations([1, 13], [0, 0], ['T1', 'T2']))
    with pytest.raises(ValueError, match='bounds'):
        epochs_from_raw(raw, cfg, 2, 4, filtered=False)
    raw.set_annotations(mne.Annotations([1, 4, 7], [1] * 3, ['T1', 'T2', 'T3']))
    with pytest.raises(ValueError, match='annotations'):
        epochs_from_raw(raw, cfg, 2, 4, filtered=False)


def test_missing_data_never_downloads(tmp_path):
    with patch('mne.datasets.eegbci.load_data', side_effect=AssertionError('download forbidden')):
        with pytest.raises(FileNotFoundError):
            resolve_edf(tmp_path, 2, 4)


def test_audit_excludes_only_for_quality(dataset):
    cfg, raws, reader = dataset
    path = resolve_edf(Path(cfg['data_dir']), 3, 8)
    raws[str(path)] = raw_fixture(sfreq=128)
    with patch('mne.io.read_raw_edf', side_effect=reader):
        manifest = audit(cfg)
    assert manifest['eligible_subjects'] == [2]
    assert 'sfreq_mismatch' in manifest['exclusions'][0]['reason']


def test_full_synthetic_workflow(dataset, tmp_path):
    cfg, _, reader = dataset
    validation, output, report = (tmp_path / p for p in ('validation', 'output', 'report'))
    for path in (validation, output, report):
        path.mkdir()
    with patch('mne.io.read_raw_edf', side_effect=reader):
        manifest = audit(cfg)
        def dev_reader(path, **kwargs):
            assert 'R12' not in str(path)
            return reader(path, **kwargs)
        with patch('mne.io.read_raw_edf', side_effect=dev_reader):
            develop(manifest, validation)
        result = evaluate(validation, output)
        with pytest.raises(FileExistsError):
            evaluate(validation, tmp_path / 'second')
    assert result['n_participants'] == 2
    assert len(result['rows']) == 6
    assert result['status'] == 'complete'
    with patch('eeg_svd.evaluation.evaluate', side_effect=AssertionError('no retraining')):
        build_report(output, report)
    assert (report / 'accuracy.png').is_file()
    assert (report / 'results.tex').is_file()
    status = read_json(output / 'status.json')
    assert status['sha256']['results.json'] == file_hash(output / 'results.json')
    (output / 'results.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        build_report(output, report)


def test_report_requires_finished_evaluation(tmp_path):
    write_json(tmp_path / 'status.json', {'status': 'failed'})
    with pytest.raises(ValueError, match='not complete'):
        build_report(tmp_path, tmp_path)


def test_cli(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(['--help'])
    assert exc.value.code == 0
    assert main(['audit-data', '--config', str(tmp_path / 'absent.yaml'),
                 '--out', str(tmp_path / 'new')]) == 2


def test_optional_local_edf_integration():
    """Optional real pilot data smoke test, not a reported accuracy experiment."""
    import os
    root = os.environ.get('EEG_TEST_DATA_DIR')
    if not root:
        pytest.skip('Set EEG_TEST_DATA_DIR to existing EDF root; never downloads')
    from eeg_svd.data import load_run
    cfg = {**deepcopy(DEFAULTS), 'data_dir': root}
    loaded = {}
    for run in (4, 8, 12):
        X, y, rows, _ = load_run(resolve_edf(Path(root), 1, run), cfg, 1, run)
        assert X.shape == (15, 64, 320)
        assert len(rows) == 15 and set(y) == {0, 1}
        loaded[run] = bandpower(X, cfg), y
    for family in ('rbf_baseline', 'svd_rbf'):
        model = make_pipeline(dict(family=family, rank=2, C=1., gamma='scale'))
        model.fit(*loaded[4])
        assert model.predict(loaded[8][0]).shape == (15,)
    # Run 12 has only mechanical ingestion checks; no fresh accuracy claim.
