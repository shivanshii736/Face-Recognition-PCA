"""Tests for src/application.py (P4). Run from project root:  pytest tests/test_application.py"""
import numpy as np

from src.config import D, SEED
from src.application import (project_to_pca, recognize, accuracy_vs_k, reconstruct,
                             compression_ratio, denoise_demo, mse)


def _data(n_people=5, per=10, d=D):
    """Well-separated synthetic 'people' + a PCA basis from the SVD."""
    rng = np.random.default_rng(SEED)
    centers = 3 * rng.standard_normal((n_people, d))
    X = np.vstack([c + 0.3 * rng.standard_normal((per, d)) for c in centers])
    y = np.repeat(np.arange(n_people), per)
    test = np.arange(len(y)) % per >= 8
    Xtr, Xte, ytr, yte = X[~test], X[test], y[~test], y[test]
    mean = Xtr.mean(0)
    Xc_tr, Xc_te = Xtr - mean, Xte - mean
    _, _, Vt = np.linalg.svd(Xc_tr, full_matrices=False)
    return Xc_tr, Xc_te, ytr, yte, Vt.T, mean, Xte


def test_project_shapes():
    Xc_tr, Xc_te, *_ , W, _, _ = _data()
    assert project_to_pca(Xc_tr, W[:, :7]).shape == (Xc_tr.shape[0], 7)
    assert project_to_pca(Xc_te[0], W[:, :7]).shape == (7,)


def test_recognize_perfect_on_separated_data():
    Xc_tr, Xc_te, ytr, yte, W, _, _ = _data()
    pred = recognize(project_to_pca(Xc_tr, W[:, :10]), ytr, project_to_pca(Xc_te, W[:, :10]))
    assert pred.shape == yte.shape
    assert np.array_equal(pred, yte)


def test_recognize_matches_bruteforce():
    rng = np.random.default_rng(0)
    Zt, Zs, y = rng.normal(size=(20, 4)), rng.normal(size=(6, 4)), rng.integers(0, 5, 20)
    brute = [y[np.argmin(np.linalg.norm(Zt - z, axis=1))] for z in Zs]
    assert np.array_equal(recognize(Zt, y, Zs), brute)


def test_accuracy_vs_k_returns_list_in_range():
    Xc_tr, Xc_te, ytr, yte, W, _, _ = _data()
    accs = accuracy_vs_k(Xc_tr, Xc_te, ytr, yte, W, ks=[1, 2, 5])
    assert len(accs) == 3 and all(0.0 <= a <= 1.0 for a in accs)


def test_full_rank_reconstruction_is_exact_and_error_decreases():
    Xc_tr, _, _, _, W, mean, _ = _data()
    X = Xc_tr + mean
    errs = []
    for k in (2, 10, W.shape[1]):
        Wk = W[:, :k]
        errs.append(mse(X, reconstruct(project_to_pca(Xc_tr, Wk), Wk, mean)))
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-20


def test_compression_ratio():
    assert np.isclose(compression_ratio(320, 100), 320 * 4096 / (320 * 100 + 4096 * 100 + 4096))
    assert compression_ratio(320, 10) > compression_ratio(320, 100) > 1


def test_denoise_shapes_and_improves():
    Xc_tr, _, _, _, W, mean, Xte = _data()
    noisy, den = denoise_demo(Xte[0], W, mean, sigma=0.5, seed=123)
    assert noisy.shape == den.shape == (D,)
    assert mse(Xte[0], den) < mse(Xte[0], noisy)