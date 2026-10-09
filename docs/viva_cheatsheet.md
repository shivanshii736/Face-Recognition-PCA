Concept	                        Purpose	                                                Outcome
Projection Z = Xc·W_k	        Express each face by k eigenface coefficients	        4096 → 111 numbers per face
1-NN in eigenface space	        Recognize a face by its closest training face	        92.5% accuracy at k_95
Accuracy vs k	                Choose k without wasting dimensions	                    Plateau after k≈20
Reconstruction Z·W_kᵀ + mean	See what k components keep	                            MSE falls from 0.0089 to 0.0024 as k goes from 5 to 111
Compression ratio	            Quantify storage saved	                                50× at k=5, 2.65× at k=111 (basis included)
Denoising	                    Orthogonal projection drops noise outside the subspace	Noise MSE cut by about 52%