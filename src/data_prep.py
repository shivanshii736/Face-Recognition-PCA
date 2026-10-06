"""
Stages 1-4 of the pipeline: data matrix, matrix simplification (RREF),
rank / nullity, linear independence and basis selection.

Conventions (shared by the whole team):
    rows = samples, columns = pixels, float64 arrays.
    One face is a 1-D array of shape (4096,).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

from src.config import D, IMG_SHAPE, N_PEOPLE, SEED

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
FIG_DIR = ROOT / "outputs" / "figures"


def _header(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------- Stage 1
def load_faces():
    """Return X (400, 4096) in [0, 1] and y (400,) with person ids 0-39."""
    data = fetch_olivetti_faces(data_home=str(CACHE_DIR), shuffle=False)
    X = data.data.astype(np.float64)
    y = data.target.astype(int)
    assert X.shape == (400, D) and y.shape == (400,)
    return X, y


def load_and_split(X=None, y=None):
    """Stratified split: 8 train / 2 test images per person.

    Returns X_train (320, 4096), X_test (80, 4096), y_train (320,), y_test (80,)
    """
    if X is None or y is None:
        X, y = load_faces()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    _header("STAGE 1: DATA MATRIX")
    print(f"X (all faces)  : {X.shape}  -> 400 images, each flattened to {D} pixels")
    print(f"X_train        : {X_train.shape},  X_test: {X_test.shape}")
    print(f"labels         : {len(np.unique(y))} people, "
          f"{np.bincount(y_train)[0]} train / {np.bincount(y_test)[0]} test each")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------- Stage 2
def center(X_train, X_test):
    """Subtract the TRAIN mean face from both sets (no test leakage).

    Returns Xc_train (320, 4096), Xc_test (80, 4096), mean_face (4096,)
    """
    mean_face = X_train.mean(axis=0)
    Xc_train = X_train - mean_face
    Xc_test = X_test - mean_face
    _header("STAGE 2: MEAN-CENTERING")
    print(f"mean_face shape      : {mean_face.shape}")
    print(f"max |column mean| of Xc_train: {np.abs(Xc_train.mean(axis=0)).max():.2e} (should be ~0)")
    return Xc_train, Xc_test, mean_face


# ------------------------------------------------------- Stages 3-4 (demo)
def make_demo_matrix(X, y, block=8):
    """Build A_demo (64, 10): 10 images as COLUMNS, each downsampled to 8x8.

    Columns 0-6 : real faces (first image of persons 0..6)
    Column 7    : col0 + col1        (deliberately dependent)
    Column 8    : col2 - col3        (deliberately dependent)
    Column 9    : 2 * col4           (deliberately dependent)

    Real faces alone are almost always full rank, so the 3 dependent columns
    are added on purpose to show redundancy being detected by RREF.
    Entries are integers 0-255 so sympy can do EXACT arithmetic.
    """
    cols = []
    for p in range(7):
        i = np.where(y == p)[0][0]
        img = X[i].reshape(IMG_SHAPE)
        small = img.reshape(8, block, 8, block).mean(axis=(1, 3))  # 64x64 -> 8x8
        cols.append(np.rint(small * 255).astype(int).flatten())
    cols.append(cols[0] + cols[1])
    cols.append(cols[2] - cols[3])
    cols.append(2 * cols[4])
    A_demo = np.column_stack(cols).astype(np.float64)
    assert A_demo.shape == (64, 10)
    return A_demo


def rref_basis_demo(A_demo):
    """Exact RREF with sympy. Returns rref_demo (64, 10), pivots (list), B_demo (64, r)."""
    M = sp.Matrix(np.rint(A_demo).astype(int).tolist())
    R, pivots = M.rref()
    rref_demo = np.array(R.tolist(), dtype=np.float64)
    pivots = list(pivots)
    B_demo = A_demo[:, pivots]

    n_cols = A_demo.shape[1]
    rank_demo = len(pivots)
    nullity_demo = n_cols - rank_demo

    _header("STAGE 3: RREF, RANK AND NULLITY")
    print(f"A_demo shape : {A_demo.shape}  (columns = 10 downsampled images)")
    print(f"pivot columns: {pivots}")
    print(f"rank = {rank_demo},  nullity = {n_cols} - {rank_demo} = {nullity_demo}")
    print(f"cross-check with np.linalg.matrix_rank: {np.linalg.matrix_rank(A_demo)}")

    _header("STAGE 4: LINEAR INDEPENDENCE AND BASIS SELECTION")
    print(f"B_demo (basis = pivot columns): {B_demo.shape}")
    for j in range(n_cols):
        if j not in pivots:
            coeffs = {pivots[k]: rref_demo[k, j] for k in range(rank_demo)
                      if abs(rref_demo[k, j]) > 1e-12}
            terms = " + ".join(f"({c:g})*col{p}" for p, c in coeffs.items())
            print(f"col{j} is redundant: col{j} = {terms}")
    print("Pivot columns are linearly independent and span the same column space as A_demo.")
    return rref_demo, pivots, B_demo


# ---------------------------------------------------------------- Figure
def save_sample_figure(X_train, mean_face):
    """Save a grid of 9 sample faces plus the mean face."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for ax, i in zip(axes.flat[:9], range(0, 9 * 8, 8)):
        ax.imshow(X_train[i].reshape(IMG_SHAPE), cmap="gray")
        ax.set_title(f"face {i}", fontsize=8)
    axes.flat[9].imshow(mean_face.reshape(IMG_SHAPE), cmap="gray")
    axes.flat[9].set_title("mean face", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sample_faces.png", dpi=150)
    plt.close(fig)
    print(f"saved {FIG_DIR / 'sample_faces.png'}")


# ----------------------------------------------------------- Entry point
def run_stages_1_to_4():
    """Run everything in this module. main.py calls this."""
    X, y = load_faces()
    X_train, X_test, y_train, y_test = load_and_split(X, y)
    Xc_train, Xc_test, mean_face = center(X_train, X_test)
    A_demo = make_demo_matrix(X, y)
    rref_demo, pivots, B_demo = rref_basis_demo(A_demo)
    save_sample_figure(X_train, mean_face)
    return dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                Xc_train=Xc_train, Xc_test=Xc_test, mean_face=mean_face,
                A_demo=A_demo, rref_demo=rref_demo, pivots=pivots, B_demo=B_demo)


if __name__ == "__main__":
    run_stages_1_to_4()
