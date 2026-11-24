import sys
import check_numeric
import check_torch
import report

if __name__ == "__main__":

    print("### Дані та налаштування\n")

    report.print_data_setup()

    print("\n### Параметри\n")

    report.print_params()

    print("\n### Прямий прохід\n")

    report.print_forward()

    print("\n### Зворотний прохід\n")

    report.print_backward()

    print("\n### Звірка з PyTorch\n")

    ok1 = check_torch.main()

    print("\n### Чисельна перевірка градієнтів\n")

    ok2 = check_numeric.main()

    print("\nУсі перевірки пройдено" if ok1 and ok2 else "\nЄ провалені перевірки")

    sys.exit(0 if ok1 and ok2 else 1)