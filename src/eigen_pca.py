"""
eigen_pca.py  --  Person 3 (P3)
Stages 8-9: covariance, eigenfaces, diagonalization, explained variance, choice of k.

Input  : Xc_train, float64, shape (320, 4096). Rows = samples (faces), columns = pixels,
         already mean-centred by P1 (the mean face of the TRAIN set has been subtracted).
Output : a results dict (see compute_pca) + two figures in outputs/figures/:
         p3_variance_curve.png, p3_eigenfaces.png

Public functions (the ones P4 / main.py should call):
    compute_pca(Xc_train)                         -> dict with eigvals, W_full, ...
    choose_k(eigvals, threshold=0.95)             -> int  (also saves the variance curve)
    verify_diagonalization(Xc_train, W_full, eigvals, k=50) -> dict of checks
    plot_eigenfaces(W_full, n_faces=10)           -> saves the eigenface grid
    run_eigen_pca(Xc_train)                       -> runs everything in order (for main.py)

Usage from the project root:   python -m src.eigen_pca      (standalone demo on Olivetti)
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # save figures to file; works on machines with no display
import matplotlib.pyplot as plt

from src.config import IMG_SHAPE, D

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"


# ----------------------------------------------------------------------------------------
# WHY SVD AND NOT np.linalg.eigh ON THE 4096 x 4096 COVARIANCE MATRIX?
#
# The covariance matrix is C = Xc^T Xc / (n-1), of size 4096 x 4096 (d x d).
#   * Forming C costs O(n d^2) and 4096^2 float64 numbers = ~134 MB.
#   * A full eigendecomposition (eigh) of a d x d matrix costs O(d^3) ~ 7e10 flops,
#     and it is slow and memory hungry.
# But we only have n = 320 samples, and the centred data matrix has rank <= n-1 = 319.
# So C has at most 319 non-zero eigenvalues; the other ~3777 are exactly 0 and eigh would
# waste almost all of its time computing them.
#
# The thin SVD of Xc (320 x 4096) costs only O(n^2 d) and gives us everything:
#       Xc = U S V^T          (V is d x r with orthonormal columns, S = singular values)
#   =>  C = Xc^T Xc / (n-1) = V S^2 V^T / (n-1) = V Lambda V^T
# Hence:  eigenvectors of C  = right singular vectors of Xc (columns of V),
#         eigenvalues of C   = lambda_i = S_i^2 / (n-1).
# This is the diagonalization C = V Lambda V^T, obtained without ever building C.
# It is also more accurate: forming Xc^T Xc squares the condition number.
# ----------------------------------------------------------------------------------------


def _header(title):
    print(f"\n=== {title} ===")


# ========================================================================================
# STAGE 8: covariance matrix, eigenvalues, eigenvectors (eigenfaces) via SVD
# ========================================================================================
def compute_pca(Xc_train):
    """
    Eigen-decomposition of the covariance matrix C = Xc^T Xc / (n-1), computed through SVD.

    Parameters
    ----------
    Xc_train : (n, d) float64 array, mean-centred training faces (n=320, d=4096).

    Returns
    -------
    dict with
        'eigvals'          (r,)    eigenvalues of C, sorted in DESCENDING order, = S^2/(n-1)
        'W_full'           (d, r)  eigenvectors of C as columns (eigenfaces), orthonormal,
                                   column i belongs to eigvals[i].  Top-k basis = W_full[:, :k]
        'S'                (r,)    singular values of Xc_train
        'explained_ratio'  (r,)    eigvals / sum(eigvals)
        'cumulative'       (r,)    cumulative explained variance ratio
        'n_samples'        int
    Here r = min(n, d) = 320 (only ~319 are non-zero because the data is centred).
    """
    _header("STAGE 8: COVARIANCE, EIGENVALUES & EIGENFACES (via SVD)")

    Xc = np.asarray(Xc_train, dtype=np.float64)
    n, d = Xc.shape
    assert d == D, f"expected {D} pixel columns, got {d}"

    # Safety: PCA needs centred data. P1 should already have done this.
    max_mean = np.abs(Xc.mean(axis=0)).max()
    if max_mean > 1e-8:
        print(f"WARNING: input not centred (max |column mean| = {max_mean:.2e}); centring now.")
        Xc = Xc - Xc.mean(axis=0)

    print(f"Xc_train shape            : {Xc.shape}")
    print(f"Covariance matrix C       : would be {d} x {d} ({d * d * 8 / 1e6:.0f} MB) -> NOT formed")

    # Thin SVD: Xc = U diag(S) Vt, with Vt of shape (r, d), r = min(n, d)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    eigvals = S ** 2 / (n - 1)  # eigenvalue of C = S^2 / (n-1)
    W_full = Vt.T.copy()  # columns = eigenvectors of C = right singular vectors

    # Deterministic sign convention (SVD signs are arbitrary): the largest-magnitude entry of
    # each eigenvector is made positive, so figures are reproducible between runs/machines.
    idx = np.argmax(np.abs(W_full), axis=0)
    signs = np.sign(W_full[idx, np.arange(W_full.shape[1])])
    signs[signs == 0] = 1.0
    W_full *= signs

    explained_ratio = eigvals / eigvals.sum()
    cumulative = np.cumsum(explained_ratio)

    r_nonzero = int(np.sum(eigvals > 1e-10 * eigvals[0]))
    print("Method                    : thin SVD of Xc  (Xc = U S V^T)")
    print("Relation                  : eigvecs of C = right singular vectors V,  eigval = S^2/(n-1)")
    print(f"Eigenvectors W_full shape : {W_full.shape}  (columns = eigenfaces)")
    print(f"Non-zero eigenvalues      : {r_nonzero} of {d} (rank of Xc <= n-1 = {n - 1})")
    print(f"Total variance (trace C)  : {eigvals.sum():.4f}")
    print(f"Top 5 eigenvalues         : {np.array2string(eigvals[:5], precision=4)}")
    print(f"Variance in top 5 comps   : {100 * cumulative[4]:.2f}%")

    return {
        "eigvals": eigvals,
        "W_full": W_full,
        "S": S,
        "explained_ratio": explained_ratio,
        "cumulative": cumulative,
        "n_samples": n,
    }


# ========================================================================================
# STAGE 9: explained variance and choice of k
# ========================================================================================
def choose_k(eigvals, threshold=0.95, save_path=None):
    """
    Smallest k such that the first k components explain >= `threshold` of total variance.
    Also saves the variance curve (p3_variance_curve.png) with the k marker.

    Parameters
    ----------
    eigvals   : (r,) eigenvalues sorted descending (from compute_pca()['eigvals']).
    threshold : target cumulative explained variance (0.95 -> "k_95").
    save_path : where to save the figure (default outputs/figures/p3_variance_curve.png).

    Returns
    -------
    k : int
    """
    _header("STAGE 9: EXPLAINED VARIANCE & CHOICE OF k")

    eigvals = np.asarray(eigvals, dtype=np.float64)
    ratio = eigvals / eigvals.sum()
    cum = np.cumsum(ratio)
    r = len(eigvals)

    def k_for(t):
        # first index where cumulative >= t, converted to a count; clipped for round-off
        return int(min(np.searchsorted(cum, t - 1e-12) + 1, r))

    k = k_for(threshold)
    for t in (0.80, 0.90, 0.95, 0.99):
        print(f"k for {int(t * 100):>2d}% variance : {k_for(t)}")
    print(f"Chosen k ({threshold:.0%} threshold)  : {k}   (cumulative variance = {cum[k - 1]:.4f})")
    print(f"Compression               : {D} pixels -> {k} coefficients per face "
          f"(~{D / k:.0f}x fewer numbers)")

    # ---- figure ----
    save_path = Path(save_path) if save_path else FIG_DIR / "p3_variance_curve.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    comps = np.arange(1, r + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(comps, 100 * ratio, lw=1.5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Component index i")
    axes[0].set_ylabel("Explained variance (%)  [log scale]")
    axes[0].set_title("Scree plot (eigenvalue spectrum)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(comps, 100 * cum, lw=2, label="Cumulative explained variance")
    axes[1].axhline(100 * threshold, color="gray", ls="--", lw=1, label=f"{threshold:.0%} threshold")
    axes[1].axvline(k, color="red", ls="--", lw=1.2)
    axes[1].scatter([k], [100 * cum[k - 1]], color="red", zorder=5,
                    label=f"k_{int(threshold * 100)} = {k}")
    axes[1].annotate(f"k = {k}", (k, 100 * cum[k - 1]), textcoords="offset points",
                     xytext=(12, -18), color="red")
    axes[1].set_xlabel("Number of components k")
    axes[1].set_ylabel("Cumulative explained variance (%)")
    axes[1].set_title("Cumulative explained variance")
    axes[1].set_ylim(0, 101)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure              : {save_path}")

    return k


# ========================================================================================
# STAGE 8 (cont.): symmetric, orthogonal eigenvectors and C = V Lambda V^T
# ========================================================================================
def _downsample_to_64d(X):
    """Block-average each 64x64 face down to 8x8 = 64 features (a small 64-dim version)."""
    h, w = IMG_SHAPE
    n = X.shape[0]
    bh, bw = h // 8, w // 8
    return X.reshape(n, 8, bh, 8, bw).mean(axis=(2, 4)).reshape(n, 64)


def verify_diagonalization(Xc_train, W_full, eigvals, k=50, tol=1e-8):
    """
    Numerically verifies the spectral theorem on our data, WITHOUT a full 4096 eigendecomposition.

    Check A (top-k of the real 4096-dim problem; C is applied implicitly as Xc^T (Xc w)/(n-1)):
        A1. orthonormal columns        :  W_k^T W_k = I
        A2. eigen-equation             :  C W_k = W_k Lambda_k
        A3. diagonalization in subspace:  W_k^T C W_k = Lambda_k  (off-diagonals ~ 0)
    Check B (small 64-dim version: faces block-averaged to 8x8; full np.linalg.eigh is cheap):
        B1. C_small is symmetric        :  C = C^T
        B2. eigenvectors orthogonal     :  V^T V = I
        B3. diagonalization             :  C = V Lambda V^T
        B4. eigh agrees with SVD        :  eigenvalues = S^2/(n-1), eigenvectors = right singular
                                           vectors (up to sign)

    Returns a dict of the measured errors and booleans; result['all_passed'] is True/False.
    """
    _header("STAGE 8b: DIAGONALIZATION CHECK  (C = V Lambda V^T)")

    Xc = np.asarray(Xc_train, dtype=np.float64)
    n = Xc.shape[0]
    k = int(min(k, W_full.shape[1]))
    Wk = W_full[:, :k]
    lam_k = np.asarray(eigvals)[:k]
    res = {}

    # ---------------- Check A: top-k on the full 4096-dim space ----------------
    CWk = Xc.T @ (Xc @ Wk) / (n - 1)  # C W_k without forming C
    res["orth_err_topk"] = np.abs(Wk.T @ Wk - np.eye(k)).max()
    res["eig_eq_err_topk"] = np.linalg.norm(CWk - Wk * lam_k) / np.linalg.norm(Wk * lam_k)
    M = Wk.T @ CWk  # should equal diag(lam_k)
    off = M - np.diag(np.diag(M))
    res["offdiag_err_topk"] = np.abs(off).max() / lam_k[0]
    res["diag_err_topk"] = np.abs(np.diag(M) - lam_k).max() / lam_k[0]

    print(f"--- A. Top-{k} eigenpairs of the real {Xc.shape[1]}-dim covariance matrix ---")
    print(f"A1  max|W_k^T W_k - I|                    = {res['orth_err_topk']:.2e}   (orthonormal)")
    print(f"A2  ||C W_k - W_k Lambda_k|| / ||..||     = {res['eig_eq_err_topk']:.2e}   (C v = lambda v)")
    print(f"A3  max off-diag of W_k^T C W_k / lambda1 = {res['offdiag_err_topk']:.2e}   (diagonal)")
    print(f"    max|diag(W_k^T C W_k) - lambda| / l1  = {res['diag_err_topk']:.2e}")

    # ---------------- Check B: small 64-dim version, full eigh ----------------
    Xs = _downsample_to_64d(Xc)
    Xs = Xs - Xs.mean(axis=0)
    Cs = Xs.T @ Xs / (n - 1)  # 64 x 64

    res["symmetry_err_small"] = np.abs(Cs - Cs.T).max()

    lam, V = np.linalg.eigh(Cs)  # eigh: for symmetric matrices, ascending order
    order = np.argsort(lam)[::-1]  # -> descending
    lam, V = lam[order], V[:, order]

    res["orth_err_small"] = np.abs(V.T @ V - np.eye(64)).max()
    recon = V @ np.diag(lam) @ V.T
    res["diag_recon_err_small"] = np.linalg.norm(Cs - recon) / np.linalg.norm(Cs)

    _, Ss, Vts = np.linalg.svd(Xs, full_matrices=False)
    res["eigval_vs_svd_small"] = np.abs(lam - Ss ** 2 / (n - 1)).max() / lam[0]
    m = 10  # compare the 10 leading vectors (well separated eigenvalues)
    res["eigvec_vs_svd_small"] = np.abs(np.abs(np.sum(V[:, :m] * Vts[:m].T, axis=0)) - 1).max()

    print("--- B. Small 64-dim version (8x8 block-averaged faces), full eigendecomposition ---")
    print(f"B1  max|C - C^T|                          = {res['symmetry_err_small']:.2e}   (symmetric)")
    print(f"B2  max|V^T V - I|                        = {res['orth_err_small']:.2e}   (orthogonal)")
    print(f"B3  ||C - V Lambda V^T|| / ||C||          = {res['diag_recon_err_small']:.2e}   (diagonalization)")
    print(f"B4  max|lambda - S^2/(n-1)| / lambda1     = {res['eigval_vs_svd_small']:.2e}   (eigh == SVD)")
    print(f"    max| |v_i . v_svd_i| - 1 | (top {m})    = {res['eigvec_vs_svd_small']:.2e}")

    checks = {
        "orth_err_topk": res["orth_err_topk"],
        "eig_eq_err_topk": res["eig_eq_err_topk"],
        "offdiag_err_topk": res["offdiag_err_topk"],
        "diag_err_topk": res["diag_err_topk"],
        "symmetry_err_small": res["symmetry_err_small"],
        "orth_err_small": res["orth_err_small"],
        "diag_recon_err_small": res["diag_recon_err_small"],
        "eigval_vs_svd_small": res["eigval_vs_svd_small"],
    }
    # tol is loose enough for float64 round-off (errors are normally ~1e-13..1e-15)
    res["all_passed"] = bool(all(v < tol for v in checks.values())
                             and res["eigvec_vs_svd_small"] < 1e-6)
    print(f"RESULT: {'ALL CHECKS PASSED' if res['all_passed'] else 'SOME CHECKS FAILED'} "
          f"(tolerance {tol:.0e})")
    return res


# ========================================================================================
# Plot: first eigenfaces
# ========================================================================================
def plot_eigenfaces(W_full, n_faces=10, eigvals=None, save_path=None):
    """Save the first `n_faces` eigenfaces (columns of W_full) as 64x64 images (2 x 5 grid)."""
    _header("STAGE 8c: EIGENFACES PLOT")

    save_path = Path(save_path) if save_path else FIG_DIR / "p3_eigenfaces.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ncols = 5
    nrows = int(np.ceil(n_faces / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    ratio = None if eigvals is None else np.asarray(eigvals) / np.sum(eigvals)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n_faces:
            continue
        face = W_full[:, i].reshape(IMG_SHAPE)  # 4096 vector -> 64x64 image
        ax.imshow(face, cmap="gray")
        title = f"Eigenface {i + 1}"
        if ratio is not None:
            title += f"\n{100 * ratio[i]:.1f}% var"
        ax.set_title(title, fontsize=9)

    fig.suptitle(f"First {n_faces} eigenfaces (principal directions of face space)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure              : {save_path}")


# ========================================================================================
# Convenience wrapper for main.py
# ========================================================================================
def run_eigen_pca(Xc_train, threshold=0.95, k_check=50):
    """
    Runs Stages 8-9 in order and returns the PCA results dict, extended with:
        'k_95'  : chosen number of components
        'W_k'   : W_full[:, :k]  (d x k projection basis for P4)
        'checks': output of verify_diagonalization
    """
    pca = compute_pca(Xc_train)
    checks = verify_diagonalization(Xc_train, pca["W_full"], pca["eigvals"], k=k_check)
    k = choose_k(pca["eigvals"], threshold=threshold)
    plot_eigenfaces(pca["W_full"], n_faces=10, eigvals=pca["eigvals"])
    pca["k_95"] = k
    pca["W_k"] = pca["W_full"][:, :k]
    pca["checks"] = checks
    return pca


# ----------------------------------------------------------------------------------------
# Standalone demo (only for testing P3 before P1's data_prep is ready):
#     python -m src.eigen_pca
# Takes the first 8 images of each person as train (320) and centres with the train mean.
# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import fetch_olivetti_faces

    cache = Path(__file__).resolve().parents[1] / "data" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    faces = fetch_olivetti_faces(data_home=str(cache), shuffle=False)
    X = faces.data.astype(np.float64)  # (400, 4096), 10 images per person in order
    train_idx = np.concatenate([np.arange(p * 10, p * 10 + 8) for p in range(40)])
    Xtr = X[train_idx]
    Xc = Xtr - Xtr.mean(axis=0)
    run_eigen_pca(Xc)
