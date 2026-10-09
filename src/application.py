"""
Stages 10-13 (P4): projection to eigenface space, 1-NN recognition,
accuracy vs k, reconstruction / MSE, compression ratio, denoising.

Conventions: rows = samples, columns = pixels, float64.  One face = (4096,).
Inputs from other people:
    Xc_train, Xc_test, mean_face, y_train, y_test  <- P1 (data_prep)
    W_full, k_95                                   <- P3 (eigen_pca)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import D, IMG_SHAPE, SEED

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "outputs" / "figures"


def _header(title):
    print(f"\n=== {title} ===")


# ------------------------------------------------------------ core functions
def project_to_pca(Xc, W_k):
    """Z = Xc @ W_k.   (n, d) @ (d, k) -> (n, k).  Accepts a single (d,) face too."""
    return np.asarray(Xc, dtype=np.float64) @ W_k


def recognize(Z_train, y_train, Z_test):
    """1-nearest-neighbour in eigenface space (Euclidean). Returns y_pred (n_test,)."""
    # ||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2  (no (n_test, n_train, k) tensor needed)
    d2 = ((Z_test ** 2).sum(1)[:, None]
          - 2 * Z_test @ Z_train.T
          + (Z_train ** 2).sum(1)[None, :])
    return np.asarray(y_train)[np.argmin(d2, axis=1)]


def accuracy_vs_k(Xc_train, Xc_test, y_train, y_test, W_full,
                  ks=(5, 10, 20, 30, 50, 100, 150)):
    """Recognition accuracy for each k (uses the first k columns of W_full)."""
    accs = []
    for k in ks:
        W_k = W_full[:, :k]
        y_pred = recognize(project_to_pca(Xc_train, W_k), y_train,
                           project_to_pca(Xc_test, W_k))
        accs.append(float(np.mean(y_pred == y_test)))
    return accs


def reconstruct(Z, W_k, mean_face):
    """X_hat = Z @ W_k.T + mean_face  -> (n, d)."""
    return Z @ W_k.T + mean_face


def compression_ratio(n, k, d=D):
    """(n*d) / (n*k + d*k + d): original storage / (scores + basis + mean face)."""
    return (n * d) / (n * k + d * k + d)


def mse(X, X_hat):
    """Mean squared error per pixel."""
    return float(np.mean((X - X_hat) ** 2))


def denoise_demo(x_test_face, W_k, mean_face, sigma=0.1, seed=SEED):
    """Add Gaussian noise to a RAW face (4096,), then project onto W_k and back.
    Returns (noisy, denoised), both (4096,)."""
    rng = np.random.default_rng(seed)
    noisy = x_test_face + sigma * rng.standard_normal(x_test_face.shape)
    denoised = (noisy - mean_face) @ W_k @ W_k.T + mean_face
    return noisy, denoised


# ------------------------------------------------------------------ figures
def plot_accuracy_vs_k(ks, accs, k95=None, save_path=None):
    save_path = Path(save_path or FIG_DIR / "p4_accuracy_vs_k.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, accs, "o-")
    if k95 is not None:
        ax.axvline(k95, ls="--", c="gray", label=f"k_95 = {k95}")
        ax.legend()
    ax.set_xlabel("number of eigenfaces k")
    ax.set_ylabel("test accuracy (1-NN)")
    ax.set_title("Recognition accuracy vs k")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_reconstructions(x_raw, Xc_row, W_full, mean_face, ks, save_path=None):
    save_path = Path(save_path or FIG_DIR / "p4_reconstructions.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(ks) + 1, figsize=(2.2 * (len(ks) + 1), 2.6))
    axes[0].imshow(x_raw.reshape(IMG_SHAPE), cmap="gray")
    axes[0].set_title("original", fontsize=9)
    for ax, k in zip(axes[1:], ks):
        W_k = W_full[:, :k]
        xh = reconstruct(project_to_pca(Xc_row, W_k), W_k, mean_face)
        ax.imshow(xh.reshape(IMG_SHAPE), cmap="gray")
        ax.set_title(f"k = {k}", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_denoise(x_raw, noisy, denoised, save_path=None):
    save_path = Path(save_path or FIG_DIR / "p4_denoise.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))
    for ax, img, t in zip(axes, (x_raw, noisy, denoised), ("original", "noisy", "denoised")):
        ax.imshow(img.reshape(IMG_SHAPE), cmap="gray")
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ----------------------------------------------------------- stage runners
def run_application(data, pca, ks=(5, 10, 20, 30, 50, 100, 150), show_ks=(5, 20, 50)):
    """data = dict from P1's run_stages_1_to_4; pca = dict from P3's run_eigen_pca."""
    Xc_train, Xc_test = data["Xc_train"], data["Xc_test"]
    X_test, y_train, y_test = data["X_test"], data["y_train"], data["y_test"]
    mean_face = data["mean_face"]
    W_full, k95 = pca["W_full"], int(pca["k_95"])
    n = Xc_train.shape[0]
    results = {}

    _header("STAGE 10: PROJECTION TO EIGENFACE SPACE")
    W_k95 = W_full[:, :k95]
    Z_train, Z_test = project_to_pca(Xc_train, W_k95), project_to_pca(Xc_test, W_k95)
    print(f"W_k shape : {W_k95.shape}  (k_95 = {k95})")
    print(f"Z_train   : {Z_train.shape},  Z_test: {Z_test.shape}")

    _header("STAGE 11: RECOGNITION (1-NN) AND ACCURACY vs k")
    y_pred = recognize(Z_train, y_train, Z_test)
    acc95 = float(np.mean(y_pred == y_test))
    print(f"accuracy at k_95 = {k95}: {acc95:.4f}  ({int((y_pred == y_test).sum())}/{len(y_test)})")
    ks = [k for k in ks if k <= W_full.shape[1]]
    ks = sorted(set(ks) | {k95})
    accs = accuracy_vs_k(Xc_train, Xc_test, y_train, y_test, W_full, ks)
    for k, a in zip(ks, accs):
        print(f"  k = {k:>3} -> accuracy {a:.4f}")
    print(f"saved {plot_accuracy_vs_k(ks, accs, k95)}")
    results.update(k95=k95, acc95=acc95, ks=ks, accs=accs)

    _header("STAGE 12: RECONSTRUCTION, MSE AND COMPRESSION")
    shown = sorted(set(list(show_ks) + [k95]))
    print(f"{'k':>5} {'MSE (test)':>12} {'compression':>12}")
    mses = {}
    for k in shown:
        W_k = W_full[:, :k]
        Xh = reconstruct(project_to_pca(Xc_test, W_k), W_k, mean_face)
        mses[k] = mse(X_test, Xh)
        print(f"{k:>5} {mses[k]:>12.6f} {compression_ratio(n, k):>11.2f}x")
    print(f"saved {plot_reconstructions(X_test[0], Xc_test[0], W_full, mean_face, shown)}")
    results["mse"] = mses

    _header("STAGE 13: DENOISING DEMO")
    noisy, den = denoise_demo(X_test[0], W_k95, mean_face, sigma=0.1)
    print(f"MSE noisy vs clean    : {mse(X_test[0], noisy):.6f}")
    print(f"MSE denoised vs clean : {mse(X_test[0], den):.6f}  (k = {k95})")
    print(f"saved {plot_denoise(X_test[0], noisy, den)}")
    results["denoise_mse"] = (mse(X_test[0], noisy), mse(X_test[0], den))
    return results