\## Data Representation and Matrix Simplification



\### Dataset and matrix representation

We use the Olivetti faces dataset: 400 grayscale images (64 × 64) of 40 people, 10 images each. Each image is flattened into a vector of 4096 pixel intensities and stored as a row of the data matrix \*\*X\*\* (400 × 4096). The data is split in a stratified way into a training set (320 images, 8 per person) and a test set (80 images, 2 per person).



\### Mean-centering

The mean face is computed from the training rows only and subtracted from both the training and test sets. This produces the centered matrices Xc\_train (320 × 4096) and Xc\_test (80 × 4096). After centering, every column of Xc\_train has mean zero (largest absolute column mean about 1e-16, i.e. floating-point error). Centering is required so that PCA captures variation between faces and not the average face. Using the training mean for the test set avoids information leakage.



\### Matrix simplification, rank and nullity

To demonstrate Gaussian elimination, we form a small matrix A\_demo (64 × 10) whose columns are 8 × 8 downsampled images. Seven columns are real faces and three are deliberate linear combinations (col7 = col0 + col1, col8 = col2 − col3, col9 = 2·col4), so that redundancy exists to be detected. The matrix has integer entries, which lets sympy compute the RREF exactly.



The RREF has pivots in columns 0 to 6. Therefore \*\*rank = 7\*\* and \*\*nullity = 10 − 7 = 3\*\*, which agrees with `numpy.linalg.matrix\_rank`.



\### Linear independence and basis selection

The pivot columns of A\_demo are linearly independent and span its column space, so they form a basis. We store them as B\_demo (64 × 7). The non-pivot columns are read off the RREF as combinations of the basis columns, matching the three constructed relations exactly. B\_demo is passed to the Gram-Schmidt stage, and Xc\_train is passed to the PCA stage.



\### Figure

`outputs/figures/sample\_faces.png` shows sample faces and the mean face.

