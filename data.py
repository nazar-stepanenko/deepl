import numpy as np
from sklearn.datasets import load_iris

N_TRAIN_PER_CLASS = 35


def load_data():

    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target.astype(np.int64)
    rng = np.random.default_rng(0)
    train_idx, test_idx = [], []

    for c in range(3):

        idx = rng.permutation(np.flatnonzero(y == c))
        train_idx.append(idx[:N_TRAIN_PER_CLASS])
        test_idx.append(idx[N_TRAIN_PER_CLASS:])

    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":

    Xtr, ytr, Xte, yte = load_data()

    print(Xtr.shape, ytr.shape, Xte.shape, yte.shape, Xtr.dtype)
    print("train class counts:", np.bincount(ytr), "test:", np.bincount(yte))