"""Tests for orthogonal.py. Run with:  pytest -v"""

import numpy as np
import pytest

from src.orthogonal import gram_schmidt, project, least_squares


@pytest.fixture
def B():
    rng = np.random.default_rng(0)
    return rng.standard_normal((64, 8))


@pytest.fixture
def x():
    return np.random.default_rng(1).standard_normal(64)


# ---------- Gram-Schmidt ----------
def test_q_is_orthonormal(B):
    Q, _ = gram_schmidt(B)
    assert np.allclose(Q.T @ Q, np.eye(B.shape[1]), atol=1e-10)


def test_qr_reconstructs_b(B):
    Q, R = gram_schmidt(B)
    assert np.allclose(Q @ R, B, atol=1e-10)


def test_r_is_upper_triangular_with_positive_diagonal(B):
    _, R = gram_schmidt(B)
    assert np.allclose(R, np.triu(R))
    assert np.all(np.diag(R) > 0)


def test_matches_numpy_qr(B):
    Q, R = gram_schmidt(B)
    q_np, r_np = np.linalg.qr(B)
    signs = np.where(np.diag(r_np) < 0, -1.0, 1.0)  # fix sign ambiguity
    assert np.allclose(Q, q_np * signs, atol=1e-8)
    assert np.allclose(R, signs[:, None] * r_np, atol=1e-8)


def test_dependent_column_raises(B):
    B_bad = B.copy()
    B_bad[:, -1] = B[:, 0] + B[:, 1]
    with pytest.raises(ValueError, match="dependent"):
        gram_schmidt(B_bad)


def test_wide_matrix_rejected():
    with pytest.raises(ValueError):
        gram_schmidt(np.ones((3, 5)))


def test_non_2d_rejected():
    with pytest.raises(ValueError):
        gram_schmidt(np.ones(5))


# ---------- Projection ----------
def test_residual_orthogonal_to_subspace(B, x):
    _, residual = project(B, x)
    assert np.allclose(B.T @ residual, 0, atol=1e-10)


def test_pythagoras(B, x):
    p, residual = project(B, x)
    assert np.isclose(x @ x, p @ p + residual @ residual)


def test_projection_is_idempotent(B, x):
    p, _ = project(B, x)
    p2, _ = project(B, p)
    assert np.allclose(p, p2, atol=1e-10)


def test_vector_in_subspace_is_unchanged(B):
    v = B @ np.arange(1, B.shape[1] + 1, dtype=float)  # a combo of B's columns
    p, residual = project(B, v)
    assert np.allclose(p, v, atol=1e-10)
    assert np.allclose(residual, 0, atol=1e-10)


def test_precomputed_q_gives_same_result(B, x):
    Q, _ = gram_schmidt(B)
    p1, _ = project(B, x)
    p2, _ = project(B, x, Q)
    assert np.allclose(p1, p2)


def test_project_rejects_wrong_length(B):
    with pytest.raises(ValueError):
        project(B, np.ones(10))


# ---------- Least squares ----------
def test_matches_numpy_lstsq(B, x):
    w = least_squares(B, x)
    w_np, *_ = np.linalg.lstsq(B, x, rcond=None)
    assert np.allclose(w, w_np, atol=1e-8)


def test_fit_equals_projection(B, x):
    w = least_squares(B, x)
    p, _ = project(B, x)
    assert np.allclose(B @ w, p, atol=1e-8)


def test_recovers_exact_weights_when_no_noise(B):
    w_true = np.arange(1, B.shape[1] + 1, dtype=float)
    assert np.allclose(least_squares(B, B @ w_true), w_true, atol=1e-8)


def test_least_squares_rejects_wrong_length(B):
    with pytest.raises(ValueError):
        least_squares(B, np.ones(10))
