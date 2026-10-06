import numpy as np
from src.data_prep import center, load_and_split, make_demo_matrix, rref_basis_demo


def _fake_data():
    rng = np.random.default_rng(0)
    X = rng.random((400, 4096))
    y = np.repeat(np.arange(40), 10)
    return X, y


def test_split_and_center_shapes():
    X, y = _fake_data()
    X_train, X_test, y_train, y_test = load_and_split(X, y)
    assert X_train.shape == (320, 4096) and X_test.shape == (80, 4096)
    assert y_train.shape == (320,) and y_test.shape == (80,)
    Xc_train, Xc_test, mean_face = center(X_train, X_test)
    assert mean_face.shape == (4096,)
    assert np.allclose(Xc_train.mean(axis=0), 0)


def test_rref_demo():
    X, y = _fake_data()
    A = make_demo_matrix(X, y)
    rref, pivots, B = rref_basis_demo(A)
    assert A.shape == (64, 10)
    assert pivots == [0, 1, 2, 3, 4, 5, 6]
    assert B.shape == (64, 7)
    assert np.linalg.matrix_rank(A) == 7
