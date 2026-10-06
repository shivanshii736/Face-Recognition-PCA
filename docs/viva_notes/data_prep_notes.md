\# Viva Notes: Data Matrix and Matrix Simplification (Stages 1-4)



Module: `src/data\_prep.py`. Answer in the order \*\*Concept → Purpose → Outcome\*\*.



\## What our data matrix represents

Each row of X is one face image (64×64) flattened into a vector of 4096 pixel intensities in \[0, 1].

X is 400 × 4096 (40 people × 10 images). Rows = samples, columns = pixels.



\---



\## Stage 1: Matrix representation of data

\- \*\*Concept:\*\* Real-world data as a matrix. Each image is a vector in R^4096; stacking images gives X.

\- \*\*Purpose:\*\* Linear algebra only works on vectors and matrices, so images must be turned into numbers first.

\- \*\*Outcome:\*\* X (400, 4096). Stratified split gives X\_train (320, 4096) and X\_test (80, 4096), with 8 train and 2 test images per person.



\## Stage 2: Mean-centering

\- \*\*Concept:\*\* Subtract the mean face (average of the training rows) from every image.

\- \*\*Purpose:\*\* PCA looks at how faces \*vary\*, so the average must be removed. Otherwise the first direction mostly captures "the average face". We use the \*training\* mean for both sets so no test information leaks into training.

\- \*\*Outcome:\*\* `mean\_face` (4096,), `Xc\_train` (320, 4096), `Xc\_test` (80, 4096). Column means of `Xc\_train` are about 1e-16, i.e. zero.



\## Stage 3: Matrix simplification (RREF), rank and nullity

\- \*\*Concept:\*\* Row-reduce a matrix to reduced row echelon form. Pivot columns show the independent directions. Rank = number of pivots. Nullity = columns − rank.

\- \*\*Purpose:\*\* To find out how much of the data is independent and how much is redundant.

\- \*\*Outcome:\*\* On the demo matrix A\_demo (64 × 10): pivots are columns 0-6, \*\*rank = 7, nullity = 3\*\*. Confirmed with `np.linalg.matrix\_rank`.



\## Stage 4: Linear independence and basis selection

\- \*\*Concept:\*\* The pivot columns of the \*original\* matrix are linearly independent and span the same column space, so they form a basis.

\- \*\*Purpose:\*\* Remove redundant vectors so later steps (Gram-Schmidt) get a clean independent set.

\- \*\*Outcome:\*\* `B\_demo` (64, 7). The RREF shows each redundant column as a combination of pivot columns: col7 = col0 + col1, col8 = col2 − col3, col9 = 2·col4.



\## How this connects to the next step

`B\_demo` (independent columns) goes to \*\*Gram-Schmidt\*\*, which orthogonalizes it. `Xc\_train` goes to \*\*PCA\*\*, which finds eigenvectors of its covariance matrix.



\---



\## Likely questions, with short answers



\*\*Why is the demo matrix only 64 × 10, not the full data?\*\*

Exact RREF on 4096 columns is too slow and numerically fragile. The demo shows the concept on a small matrix. The full-size work (PCA and recognition) runs on all 4096 pixels.



\*\*Why is the rank 7 and not 10?\*\*

Because we built it that way: 7 real faces plus 3 columns made from them. Real faces are almost always independent, so without this the demo would show rank 10 and nothing to explain. The point is to show RREF detecting redundancy.



\*\*Why use the pivot columns of A, not the non-zero columns of the RREF?\*\*

Row operations change the column space, so the RREF's columns are not in the original space. The pivot \*positions\* tell us which \*original\* columns to keep.



\*\*What does nullity 3 mean?\*\*

There are 3 independent ways to combine the columns to get the zero vector, one per redundant column (for example col0 + col1 − col7 = 0).



\*\*Why subtract the mean before PCA?\*\*

Covariance measures spread around the mean. Without centering, the leading eigenvector would point at the mean, not at the main direction of variation.



\*\*Why did you compute the mean from the training set only?\*\*

The test set must be unseen. Using it in the mean would leak information and inflate accuracy.



\*\*Why sympy for RREF?\*\*

The demo entries are integers, so sympy does exact arithmetic with no rounding errors. NumPy floats could misjudge a tiny non-zero number as a pivot.



\*\*Why 8 train / 2 test per person?\*\*

Stratified splitting keeps every person in both sets, so every test face has a matching person in training.

