k	        Accuracy (1-NN)	    MSE (test)	Compression*
5	        80.0%	            0.00886	    50.1×
20	        91.25%	            0.00505	    14.2×
50	        91.25%	            0.00334	    5.8×
111 (k_95)	92.5% (74/80)	    0.00239	    2.65×

*Compression is the storage ratio: (n·d)/(n·k + d·k + d), with n=320 and d=4096. It counts the scores, the basis W_k and the mean face. Per face, 4096 numbers shrink to 111, about 37×.

Denoising (σ=0.1, k=111): the noise MSE dropped from 0.00995 to 0.00474, about 52% lower. The projection throws away the noise components that lie outside the face subspace.

Accuracy: it climbs fast up to about k=20 and then flattens, so about 20 eigenfaces already capture most of the identity information. Don't overclaim the small differences. With only 80 test images, one image equals 1.25%, so 91.25% vs 92.5% is a single face.

MSE falls steadily as k grows, which is expected because more components mean a better approximation. The reconstruction at k=111 looks a bit noisier than at k=50 because it also reproduces fine pixel-level detail. The MSE is still lower.

Why a 95% threshold: it gives k=111 out of 4096 dimensions, with 92.5% accuracy.

Why SVD: the log confirms the covariance matrix would be 4096×4096 (134 MB), and it was never formed. The SVD of the 320×4096 matrix gives the same eigenvectors, with eigenvalue = S²/(n−1).