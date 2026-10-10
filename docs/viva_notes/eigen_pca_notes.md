# P3 Viva Notes: Stages 8-9 (Eigen analysis and PCA)

Code: `src/eigen_pca.py` | Tests: `tests/test_eigen_pca.py` | Figures: `p3_variance_curve.png`, `p3_eigenfaces.png`

## Stage 8: Covariance, eigenvalues, eigenvectors (eigenfaces)

**Concept**
- Input `Xc_train` is (320, 4096): 320 mean-centred faces, one per row.
- Covariance matrix: `C = Xcᵀ Xc / (n-1)`, size 4096 x 4096. It is symmetric and positive semi-definite.
- Spectral theorem: a real symmetric matrix has real eigenvalues and orthogonal eigenvectors, so `C = V Λ Vᵀ` with `VᵀV = I`.
- Eigenvectors of C reshaped to 64x64 images are called **eigenfaces**.
- Because the data is centred and n = 320, rank(Xc) <= 319, so C has at most 319 non-zero eigenvalues; the other ~3777 are exactly 0.

**Why SVD instead of `np.linalg.eigh` on the 4096x4096 matrix**
- `eigh` on a d x d matrix costs O(d³) and needs 134 MB just to store C. Almost all of that work finds zero eigenvalues.
- Thin SVD of Xc (320 x 4096) costs O(n²d) and never forms C.
- If `Xc = U S Vᵀ`, then `C = V (S²/(n-1)) Vᵀ`. So **eigenvectors of C = right singular vectors of Xc**, and **eigenvalue λᵢ = Sᵢ² / (n-1)**.
- Forming `XᵀX` squares the condition number; SVD on X directly is more accurate.

**Diagonalization check (`verify_diagonalization`)**
- Top-k (k = 50) on the real 4096-dim problem, using `C w = Xᵀ(Xw)/(n-1)` so C is never built:
  `W_kᵀW_k = I`, `C W_k = W_k Λ_k`, `W_kᵀ C W_k = Λ_k`.
- 64-dim version (faces block-averaged to 8x8): full `eigh` is cheap, so we show `C = Cᵀ`, `VᵀV = I`, `C = V Λ Vᵀ`, and that `eigh` agrees with SVD.
- All errors are around 1e-15 (float64 round-off).

**Purpose**: find the directions in pixel space along which faces vary most, and in an orthogonal coordinate system.

**Outcome**: `W_full` (4096 x 320, orthonormal columns, sorted by decreasing eigenvalue) and `eigvals`.

## Stage 9: Explained variance and choice of k

**Concept**
- Total variance = trace(C) = sum of eigenvalues.
- Explained variance ratio of component i = λᵢ / Σλ. Cumulative ratio = running sum.
- `k_95` = smallest k whose cumulative explained variance is >= 95%.

**Purpose**: keep only the top-k eigenfaces, so each face goes from 4096 numbers to k coefficients with little information lost (compression and noise removal).

**Outcome**: `k_95`, `W_k = W_full[:, :k]` (the projection basis P4 uses for recognition, reconstruction, compression and denoising), and the variance-curve figure.

## Likely viva questions
- *Why is C symmetric?* `(XᵀX)ᵀ = XᵀX`.
- *Why are the eigenvectors orthogonal?* Spectral theorem for real symmetric matrices.
- *Why do eigenvalues equal S²/(n-1)?* Substitute `X = USVᵀ` into `XᵀX/(n-1)`; `UᵀU = I` collapses the middle.
- *Why are only 319 eigenvalues non-zero?* Rank <= min(n, d); centring removes one more dimension.
- *Why centre the data first?* Covariance is defined about the mean; without centring, the first component just points at the mean.
- *How is k chosen?* Smallest k reaching 95% cumulative variance (a trade-off between compression and fidelity).
- *Sign of an eigenface?* Arbitrary (v and -v are both eigenvectors); we fix the sign deterministically.
