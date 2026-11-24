import sys
import numpy as np
from data import load_data
from model import init_params, loss_and_grads, loss_fn

EPS = 1e-6
TOL = 1e-7
CHECKS = [("W1", (0, 0)), ("b1", (0,)), ("W2", (0, 0)), ("b2", (0,))]


def numeric_check(params, X, y, divide_by_n=True):

    _, grads = loss_and_grads(params, X, y, divide_by_n=divide_by_n)
    rows = []

    for name, idx in CHECKS:

        arr = params[name]
        orig = arr[idx]
        arr[idx] = orig + EPS
        l_plus = loss_fn(params, X, y)
        arr[idx] = orig - EPS
        l_minus = loss_fn(params, X, y)
        arr[idx] = orig
        g_num = (l_plus - l_minus) / (2 * EPS)
        g_man = float(grads[name][idx])
        diff = abs(g_num - g_man)
        ok = bool(np.isfinite(g_num) and np.isfinite(g_man) and diff <= TOL)
        label = f"{name}[{', '.join(map(str, idx))}]"
        rows.append((label, g_man, g_num, diff, ok))

    return rows


def print_report(rows):

    print("| Параметр | Градієнт backward() | Чисельна похідна | Абсолютна різниця | Перевірку пройдено |")
    print("|---|---|---|---|---|")

    for label, g_man, g_num, diff, ok in rows:

        print(f"| {label} | {g_man:.12e} | {g_num:.12e} | {diff:.3e} | {'так' if ok else 'ні'} |")


def main(divide_by_n=True):

    X, y, _, _ = load_data()
    params = init_params()
    rows = numeric_check(params, X, y, divide_by_n)
    print_report(rows)

    return all(r[-1] for r in rows)


if __name__ == "__main__":
    
    ok = main(divide_by_n="--buggy" not in sys.argv)
    sys.exit(0 if ok else 1)