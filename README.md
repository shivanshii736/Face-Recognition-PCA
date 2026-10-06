# Face Recognition using PCA (Eigenfaces)

Mini project for UE25MA242A: Mathematical Foundation for AI & Data Science.

Pipeline: data matrix -> RREF and basis -> Gram-Schmidt -> projection -> least squares -> eigenvalues/eigenvectors -> PCA -> face recognition, reconstruction and compression.

## Setup

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Run

    python main.py

## Structure

- src/data_prep.py: dataset, split, centering, RREF, rank, basis
- src/orthogonal.py: Gram-Schmidt, projection, least squares
- src/eigen_pca.py: covariance, eigenfaces, diagonalization, SVD
- src/application.py: recognition, reconstruction, compression, denoising
- docs/: viva notes, demo script, report
