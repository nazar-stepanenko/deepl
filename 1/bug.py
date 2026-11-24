from data import load_data
from model import init_params, loss_and_grads
import check_torch
import check_numeric
import report


PREDICTION = """\
Прогнозується:
1. Функція втрат залишається середнім, помилка лише у backward.
2. Усі ненульові градієнти збільшаться рівно у N = 105 разів,
   бо dZ2 = P - Y замість (P - Y) / N, а решта ланцюжка лінійна за dZ2.
3. Порівняння втрати з PyTorch пройде, порівняння градієнтів з PyTorch і
   чисельна перевірка мають провалитись для всіх чотирьох параметрів
   (різниця ~ (N-1) * |g| для правильного g).
"""


def run(divide_by_n):

    X, y, _, _ = load_data()
    params = init_params()
    loss, grads = loss_and_grads(params, X, y, divide_by_n=divide_by_n)

    print(f"Втрата (NumPy): {loss!r}")
    print("Норми градієнтів: " + ", ".join(
        f"|{k}|={float((v**2).sum()**0.5):.6e}" for k, v in grads.items()))
    print("\n-- Звірка з PyTorch --")

    ln, lt, rows = check_torch.compare_with_torch(params, X, y, divide_by_n)
    check_torch.print_report(ln, lt, rows)

    print("\n-- Чисельна перевірка --")

    check_numeric.print_report(check_numeric.numeric_check(params, X, y, divide_by_n))


if __name__ == "__main__":

    print(PREDICTION)
    print("=" * 30, "НЕПРАВИЛЬНА реалізація (без /N)", "=" * 30)

    run(divide_by_n=False)

    print("\n" + "=" * 30, "ПРАВИЛЬНА реалізація (з /N)", "=" * 30)

    run(divide_by_n=True)

    print("\n" + "=" * 30, "ВІДНОШЕННЯ ГРАДІЄНТІВ (з помилкою / правильний)", "=" * 30)
    report.print_bug_ratio()