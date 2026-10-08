"""Додаткові виводи для звіту: дані, параметри, розмірності, градієнти, відношення при помилці."""
import numpy as np

from data import load_data
from model import (N_HIDDEN, N_IN, N_OUT, backward, cross_entropy, forward,
                   init_params, one_hot, softmax)


def _table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")


def print_data_setup():
    Xtr, ytr, Xte, yte = load_data()
    print(f"Навчальна вибірка: X {Xtr.shape}, y {ytr.shape}, dtype {Xtr.dtype}, "
          f"по класах {np.bincount(ytr).tolist()}")
    print(f"Тестова вибірка  : X {Xte.shape}, y {yte.shape}, dtype {Xte.dtype}, "
          f"по класах {np.bincount(yte).tolist()}")
    print(f"Навчальна після стандартизації: max|середнє| = {np.abs(Xtr.mean(axis=0)).max():.1e}, "
          f"std (ddof=0) = {np.round(Xtr.std(axis=0, ddof=0), 6).tolist()}")
    print(f"Тестова (ті самі статистики)  : середнє = {np.round(Xte.mean(axis=0), 3).tolist()}, "
          f"std = {np.round(Xte.std(axis=0, ddof=0), 3).tolist()}")


def print_params():
    p = init_params()
    rows = [
        ("W1", p["W1"].shape, "He", f"{np.sqrt(2 / N_IN):.4f}", f"{p['W1'].std():.4f}"),
        ("b1", p["b1"].shape, "нулі", "-", f"усі нулі: {bool((p['b1'] == 0).all())}"),
        ("W2", p["W2"].shape, "Xavier", f"{np.sqrt(2 / (N_HIDDEN + N_OUT)):.4f}", f"{p['W2'].std():.4f}"),
        ("b2", p["b2"].shape, "нулі", "-", f"усі нулі: {bool((p['b2'] == 0).all())}"),
    ]
    _table(["Параметр", "Розмірність", "Ініціалізація", "Теор. σ", "Вибіркова σ"], rows)
    print(f"\nРазом параметрів: {sum(v.size for v in p.values())}")


def print_forward():
    X, y, _, _ = load_data()
    p = init_params()
    logits, cache = forward(p, X)
    P = softmax(logits)
    Y = one_hot(y)
    loss = cross_entropy(logits, y)

    rows = [(n, cache[n].shape) for n in ("X", "Z1", "A1", "Z2")]
    rows += [("P = softmax(Z2)", P.shape), ("Y (one-hot)", Y.shape)]
    _table(["Масив", "Розмірність"], rows)
    print(f"\nВимкнених нейронів ReLU (Z1 <= 0): {np.mean(cache['Z1'] <= 0):.1%}")
    print(f"Макс. відхилення суми ймовірностей по рядку від 1: {np.abs(P.sum(axis=1) - 1).max():.1e}")
    print(f"Втрата на початкових вагах: {loss!r} (для рівномірних ймовірностей ln 3 = {np.log(3):.6f})")
    print(f"Частка правильних відповідей на початкових вагах: {(logits.argmax(axis=1) == y).mean():.1%}")


def print_backward():
    X, y, _, _ = load_data()
    p = init_params()
    _, cache = forward(p, X)
    g = backward(p, cache, y)

    print("Збережені у cache значення та навіщо вони потрібні:\n")
    _table(["Значення", "Розмірність", "Навіщо"], [
        ("X", cache["X"].shape, "dW1 = X^T · Delta1"),
        ("Z1", cache["Z1"].shape, "маска похідної ReLU [Z1 > 0]"),
        ("A1", cache["A1"].shape, "dW2 = A1^T · Delta2"),
        ("Z2", cache["Z2"].shape, "softmax(Z2) для Delta2 = (P - Y) / N"),
    ])
    print("\nГрадієнти:\n")
    rows = [(f"d{k}", g[k].shape, p[k].shape, g[k].shape == p[k].shape,
             f"{np.linalg.norm(g[k]):.6f}") for k in ("W1", "b1", "W2", "b2")]
    _table(["Градієнт", "Розмірність", "Розмірність параметра", "Збігається", "Норма"], rows)


def print_bug_ratio():
    """Відношення градієнтів з помилкою (без /N) до правильних."""
    X, y, _, _ = load_data()
    p = init_params()
    N = len(y)
    _, cache = forward(p, X)
    g_ok = backward(p, cache, y, divide_by_n=True)
    g_bug = backward(p, cache, y, divide_by_n=False)

    rows = []
    for k in ("W1", "b1", "W2", "b2"):
        a, b = g_ok[k].ravel(), g_bug[k].ravel()
        nz = a != 0
        ratio = b[nz] / a[nz]
        rows.append((f"d{k}", a.size, int((~nz).sum()),
                     f"{ratio.min():.12f}", f"{ratio.max():.12f}",
                     f"{np.abs(ratio - N).max():.2e}"))
    print(f"N = {N}\n")
    _table(["Градієнт", "Елементів", "Нульових", "min(з помилкою / правильний)",
            "max(з помилкою / правильний)", "Макс. відхилення відношення від N"], rows)


if __name__ == "__main__":
    for title, fn in [("Дані та налаштування", print_data_setup),
                      ("Параметри", print_params),
                      ("Прямий прохід", print_forward),
                      ("Зворотний прохід", print_backward),
                      ("Відношення градієнтів при помилці", print_bug_ratio)]:
        print(f"### {title}\n")
        fn()
        print()