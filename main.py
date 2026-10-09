"""Runs all stages in order:  python main.py"""
import numpy as np

from src.data_prep import run_stages_1_to_4
from src.orthogonal import verify_gram_schmidt, verify_projection, verify_least_squares
from src.eigen_pca import run_eigen_pca
from src.application import run_application


def main():
    # Stages 1-4 (P1)
    data = run_stages_1_to_4()

    # Stages 5-7 (P2), demonstrated on P1's basis B_demo (64 x 7)
    B = data["B_demo"]
    rng = np.random.default_rng(11)
    x, y = rng.standard_normal(B.shape[0]), rng.standard_normal(B.shape[0])
    print("\n=== STAGE 5: MODIFIED GRAM-SCHMIDT ===")
    verify_gram_schmidt(B)
    print("\n=== STAGE 6: PROJECTION ===")
    verify_projection(B, x)
    print("\n=== STAGE 7: LEAST SQUARES ===")
    verify_least_squares(B, y)

    # Stages 8-9 (P3)
    pca = run_eigen_pca(data["Xc_train"])

    # Stages 10-13 (P4)
    run_application(data, pca)
    print("\n=== DONE: figures are in outputs/figures/ ===")


if __name__ == "__main__":
    main()