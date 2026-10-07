"""Tests for src/eigen_pca.py (P3). Run from the project root:  pytest tests/test_eigen_pca.py"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `import src...` works

from src.config import D, IMG_SHAPE, SEED
from src.eigen_pca import (
    compute_pca,
    choose_k,
    verify_diagonalization,
    plot_eigenfaces,
    run_eigen_pca,
)

N = 320
_CACHE = {}


def _data():
    """Synthetic stand-in for P1's Xc_train: low-rank structure + noise, shape (320, 4096), centred."""
    if "Xc" not in _CACHE:
        rng = np.random.default_rng(SEED)
        X = rng.normal(size=(N, 30)) @ rng.normal(size=(30, D)) + 0.3 * rng.normal(size=(N, D))
        _CACHE["Xc"] = X - X.mean(axis=0)
    return _CACHE["Xc"]


def _pca():
    if "pca" not in _CACHE:
        _CACHE["pca"] = compute_pca(_data())
    return _CACHE["pca"]


def test_shapes():
    pca = _pca()
    assert pca["eigvals"].shape == (N,)
    assert pca["W_full"].shape == (D, N)
    assert pca["explained_ratio"].shape == (N,)
    assert pca["cumulative"].shape == (N,)
    assert pca["W_full"].dtype == np.float64


def test_eigvals_sorted_desc_and_nonnegative():
    ev = _pca()["eigvals"]
    assert np.all(np.diff(ev) <= 1e-12)
    assert np.all(ev >= -1e-12)


def test_W_full_orthonormal():
    W = _pca()["W_full"]
    assert np.allclose(W.T @ W, np.eye(W.shape[1]), atol=1e-10)


def test_eigvals_equal_S_squared_over_n_minus_1():
    pca = _pca()
    assert np.allclose(pca["eigvals"], pca["S"] ** 2 / (N - 1))


def test_eigvals_match_independent_eigh():
    # The non-zero eigenvalues of C = Xc^T Xc/(n-1) equal those of the small Gram matrix Xc Xc^T/(n-1).
    Xc = _data()
    ref = np.sort(np.linalg.eigvalsh(Xc @ Xc.T / (N - 1)))[::-1]
    ev = _pca()["eigvals"]
    assert np.allclose(ev, ref, atol=1e-8 * ref[0])


def test_columns_are_eigenvectors_of_C():
    Xc, pca = _data(), _pca()
    for i in range(5):
        v, lam = pca["W_full"][:, i], pca["eigvals"][i]
        Cv = Xc.T @ (Xc @ v) / (N - 1)
        assert np.allclose(Cv, lam * v, atol=1e-8 * lam)


def test_explained_ratio_sums_to_one():
    pca = _pca()
    assert np.isclose(pca["explained_ratio"].sum(), 1.0)
    assert np.isclose(pca["cumulative"][-1], 1.0)
    assert np.all(np.diff(pca["cumulative"]) >= 0)


def test_choose_k(tmp_path):
    pca = _pca()
    out = tmp_path / "p3_variance_curve.png"
    k = choose_k(pca["eigvals"], threshold=0.95, save_path=out)
    cum = pca["cumulative"]
    assert isinstance(k, int) and 1 <= k <= N
    assert cum[k - 1] >= 0.95
    assert k == 1 or cum[k - 2] < 0.95  # k is the SMALLEST such value
    assert out.exists() and out.stat().st_size > 0


def test_verify_diagonalization_passes():
    pca = _pca()
    res = verify_diagonalization(_data(), pca["W_full"], pca["eigvals"], k=50)
    assert res["all_passed"]
    assert res["orth_err_topk"] < 1e-8
    assert res["diag_recon_err_small"] < 1e-8


def test_plot_eigenfaces_saves_file(tmp_path):
    out = tmp_path / "p3_eigenfaces.png"
    plot_eigenfaces(_pca()["W_full"], n_faces=10, save_path=out)
    assert out.exists() and out.stat().st_size > 0
    assert IMG_SHAPE == (64, 64)


def test_uncentred_input_is_centred_automatically():
    Xc = _data()
    pca2 = compute_pca(Xc + 5.0)  # shifted data -> function should re-centre
    assert np.allclose(pca2["eigvals"], _pca()["eigvals"], rtol=1e-6, atol=1e-8)
