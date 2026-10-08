import sys
import numpy as np
from data import load_data
from model import init_params, loss_and_grads
from torch_ref import torch_loss_and_grads

TOL = 1e-12


def compare_with_torch(params, X, y, divide_by_n=True):

    loss_np, g_np = loss_and_grads(params, X, y, divide_by_n=divide_by_n)
    loss_t, g_t = torch_loss_and_grads(params, X, y)

    rows = [("Втрата", abs(loss_np - loss_t),
             bool(np.isfinite(loss_np) and np.isfinite(loss_t) and abs(loss_np - loss_t) <= TOL))]

    for name in ("W1", "b1", "W2", "b2"):

        a, b = g_np[name], g_t[name]
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        diff = float(np.max(np.abs(a - b)))
        rows.append((f"Градієнт {name}", diff, finite and diff <= TOL))

    return loss_np, loss_t, rows


def print_report(loss_np, loss_t, rows):

    print(f"Втрата NumPy  : {loss_np!r}")
    print(f"Втрата PyTorch: {loss_t!r}\n")
    print("| Величина | Макс. абс. різниця NumPy / PyTorch | Перевірку пройдено |")
    print("|---|---|---|")

    for name, diff, ok in rows:

        print(f"| {name} | {diff:.3e} | {'так' if ok else 'ні'} |")


def main(divide_by_n=True):

    X, y, _, _ = load_data()
    params = init_params()
    loss_np, loss_t, rows = compare_with_torch(params, X, y, divide_by_n)
    print_report(loss_np, loss_t, rows)

    return all(ok for _, _, ok in rows)


if __name__ == "__main__":

    ok = main(divide_by_n="--buggy" not in sys.argv)
    sys.exit(0 if ok else 1)