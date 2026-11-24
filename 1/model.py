import numpy as np

N_IN, N_HIDDEN, N_OUT = 4, 8, 3
SEED = 0


def init_params(seed=SEED):

    rng = np.random.default_rng(seed)
    std1 = np.sqrt(2 / N_IN)                    # He
    std2 = np.sqrt(2 / (N_HIDDEN + N_OUT))      # Xavier
    W1 = rng.normal(0.0, std1, size=(N_IN, N_HIDDEN))
    W2 = rng.normal(0.0, std2, size=(N_HIDDEN, N_OUT))
    b1 = np.zeros(N_HIDDEN)
    b2 = np.zeros(N_OUT)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def relu(Z):

    return np.maximum(0.0, Z)


def softmax(Z):

    m = Z.max(axis=1, keepdims=True)
    E = np.exp(Z - m)

    return E / E.sum(axis=1, keepdims=True)


def log_softmax(Z):

    shifted = Z - Z.max(axis=1, keepdims=True)

    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def one_hot(y, n_classes=N_OUT):

    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0

    return Y


def cross_entropy(logits, y):

    logp = log_softmax(logits)

    return float(-np.mean(logp[np.arange(len(y)), y]))


def forward(params, X):

    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2}

    return Z2, cache


def loss_fn(params, X, y):

    Z2, _ = forward(params, X)

    return cross_entropy(Z2, y)


def backward(params, cache, y, divide_by_n=True):

    X, Z1, A1, Z2 = cache["X"], cache["Z1"], cache["A1"], cache["Z2"]
    N = len(y)
    P = softmax(Z2)
    Y = one_hot(y)
    Delta2 = P - Y

    if divide_by_n:
        Delta2 = Delta2 / N

    dW2 = A1.T @ Delta2
    db2 = Delta2.sum(axis=0)
    dA1 = Delta2 @ params["W2"].T
    Delta1 = dA1 * (Z1 > 0)
    dW1 = X.T @ Delta1
    db1 = Delta1.sum(axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def loss_and_grads(params, X, y, divide_by_n=True):

    Z2, cache = forward(params, X)
    loss = cross_entropy(Z2, y)
    grads = backward(params, cache, y, divide_by_n=divide_by_n)

    return loss, grads


if __name__ == "__main__":

    from data import load_data

    X, y, _, _ = load_data()
    params = init_params()
    logits, cache = forward(params, X)
    probs = softmax(logits)
    loss = cross_entropy(logits, y)
    grads = backward(params, cache, y)

    for name, g in grads.items():

        same = g.shape == params[name].shape
