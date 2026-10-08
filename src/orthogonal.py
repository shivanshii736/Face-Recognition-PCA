"""Modified Gram-Schmidt, projection, and least-squares helpers. NumPy only."""

import numpy as np


def gram_schmidt(B):
    """Return Q, R using modified Gram-Schmidt, with positive diagonal R."""
    B = np.asarray(B, dtype=float)
    if B.ndim != 2:
        raise ValueError("B must be a two-dimensional matrix")

    m, n = B.shape
    if m < n:
        raise ValueError("B must have at least as many rows as columns")

    # Residual columns are updated one at a time: this is modified Gram-Schmidt.
    V = B.copy()
    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)

    # Scale-aware threshold, computed ONCE (spectral norm = one SVD, so keep it
    # out of the loop). It is >= every column norm, so no max() is needed.
    eps = np.finfo(float).eps
    threshold = eps * max(m, n) * np.linalg.norm(B, ord=2)

    for j in range(n):
        norm_v = np.linalg.norm(V[:, j])
        # A column that is (numerically) a combo of earlier ones leaves ~0 behind.
        if norm_v <= threshold:
            raise ValueError(f"B has a linearly dependent column (column {j})")

        R[j, j] = norm_v
        Q[:, j] = V[:, j] / norm_v

        for k in range(j + 1, n):
            # Remove the current basis direction from each remaining column.
            R[j, k] = Q[:, j] @ V[:, k]
            V[:, k] -= R[j, k] * Q[:, j]

    return Q, R


def verify_gram_schmidt(B):
    """Check orthogonality, reconstruction, and agreement with NumPy QR."""
    B = np.asarray(B, dtype=float)
    Q, R = gram_schmidt(B)

    q_np, r_np = np.linalg.qr(B, mode="reduced")

    # QR signs are ambiguous; flip matching columns/rows to make diag(R) positive.
    signs = np.where(np.diag(r_np) < 0, -1.0, 1.0)
    q_np = q_np * signs
    r_np = signs[:, None] * r_np

    print(f"||Q^T Q - I||_F: {np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1])):.3e}")
    print(f"||B - QR||_F:    {np.linalg.norm(B - Q @ R):.3e}")
    print(f"Q matches NumPy: {np.allclose(Q, q_np, rtol=1e-8, atol=1e-10)}")
    print(f"R matches NumPy: {np.allclose(R, r_np, rtol=1e-8, atol=1e-10)}")

    return Q, R


def project(B, x, Q=None):
    """Project x onto col(B) and return (projection, residual).

    Pass a precomputed Q (from gram_schmidt) to avoid redoing the work when
    projecting many vectors onto the same subspace.
    """
    B = np.asarray(B, dtype=float)
    x = np.asarray(x, dtype=float)

    if B.ndim != 2:
        raise ValueError("B must be a two-dimensional matrix")
    if x.ndim != 1 or x.shape[0] != B.shape[0]:
        raise ValueError("x must be a vector whose length equals the row count of B")

    if Q is None:
        Q, _ = gram_schmidt(B)

    # Q Q^T is the orthogonal projector onto the column space of B.
    p = Q @ (Q.T @ x)
    residual = x - p
    return p, residual


def verify_projection(B, x):
    """Check residual orthogonality, Pythagoras, and idempotence."""
    B = np.asarray(B, dtype=float)
    x = np.asarray(x, dtype=float)
    Q, _ = gram_schmidt(B)
    p, residual = project(B, x, Q)

    lhs = np.linalg.norm(x) ** 2
    rhs = np.linalg.norm(p) ** 2 + np.linalg.norm(residual) ** 2
    # Projecting something already in the subspace must change nothing.
    p_again, _ = project(B, p, Q)

    print(f"||B^T residual||:       {np.linalg.norm(B.T @ residual):.3e}  (should be ~0)")
    print(
        f"Pythagoras:             "
        f"{np.isclose(lhs, rhs, rtol=1e-10, atol=1e-10)} "
        f"(|difference|={abs(lhs - rhs):.3e})"
    )
    print(f"Idempotent ||P(p) - p||: {np.linalg.norm(p_again - p):.3e}  (should be ~0)")

    return p, residual


def least_squares(B, y):
    """Solve the full-column-rank normal equations for the least-squares fit."""
    B = np.asarray(B, dtype=float)
    y = np.asarray(y, dtype=float)

    if B.ndim != 2:
        raise ValueError("B must be a two-dimensional matrix")
    if y.ndim != 1 or y.shape[0] != B.shape[0]:
        raise ValueError("y must be a vector whose length equals the row count of B")

    # Solve directly; forming an explicit inverse is unnecessary and less stable.
    return np.linalg.solve(B.T @ B, B.T @ y)


def verify_least_squares(B, y):
    """Compare the normal-equation solution and fitted vector with references."""
    B = np.asarray(B, dtype=float)
    y = np.asarray(y, dtype=float)

    w = least_squares(B, y)
    w_np, *_ = np.linalg.lstsq(B, y, rcond=None)
    p, _ = project(B, y)

    print(
        "Weights match np.linalg.lstsq: "
        f"{np.allclose(w, w_np, rtol=1e-8, atol=1e-10)}"
    )
    print(f"Max weight difference:         {np.max(np.abs(w - w_np)):.3e}")
    print(f"||B w - projection(y)||:       {np.linalg.norm(B @ w - p):.3e}")

    return w


def demo_condition_number(B):
    """Why lstsq is preferred: normal equations square the condition number."""
    print(f"cond(B):        {np.linalg.cond(B):.3e}")
    print(f"cond(B^T B):    {np.linalg.cond(B.T @ B):.3e}")
    print(f"cond(B)^2:      {np.linalg.cond(B) ** 2:.3e}  (matches cond(B^T B))")


def demo_dependent_column(B):
    """Show that a dependent column is caught with a clear error."""
    B_bad = B.copy()
    B_bad[:, -1] = B[:, 0] + B[:, 1]  # last column = sum of two others
    try:
        gram_schmidt(B_bad)
        print("No error raised (unexpected)")
    except ValueError as e:
        print(f"Caught as expected: {e}")


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    B_demo = rng.standard_normal((64, 8))
    # Replace this assignment with P1's supplied B_demo when integrating.
    B = B_demo

    x = rng.standard_normal(64)
    y = rng.standard_normal(64)

    print("=== TASK 1: MODIFIED GRAM-SCHMIDT ===")
    verify_gram_schmidt(B)

    print("\n=== TASK 2: PROJECTION ===")
    verify_projection(B, x)

    print("\n=== TASK 3: LEAST SQUARES ===")
    verify_least_squares(B, y)

    print("\n=== EXTRA: CONDITION NUMBERS (normal equations vs lstsq) ===")
    demo_condition_number(B)

    print("\n=== EXTRA: DEPENDENT COLUMN CHECK ===")
    demo_dependent_column(B)
