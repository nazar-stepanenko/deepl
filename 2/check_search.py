from __future__ import annotations
import numpy as np
import pandas as pd
from thresholds import (
    best_threshold,
    best_threshold_brute,
    candidate_table,
    candidate_table_brute,
)

SETS = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0]),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0]),
    "C": ([0.8] + [0.5] * 11 + [0.2], [1] + [1] + [0] * 10 + [0]),
}


def run_checks(verbose: bool = True) -> dict[str, dict]:

    results = {}

    for name, (s, y) in SETS.items():

        s_arr, y_arr = np.array(s), np.array(y)
        fast = candidate_table(y_arr, s_arr)
        slow = candidate_table_brute(y, s)
        slow = slow.sort_values("threshold", ascending=False).reset_index(drop=True)

        same_table = (
            np.array_equal(fast["threshold"].to_numpy(), slow["threshold"].to_numpy())
            and all(np.array_equal(fast[c], slow[c]) for c in ("TP", "FP", "FN", "TN", "C"))
        )

        t_fast = best_threshold(fast)
        t_slow, c_slow = best_threshold_brute(slow)
        same_choice = float(t_fast["threshold"]) == t_slow and int(t_fast["C"]) == c_slow

        results[name] = {
            "ok": bool(same_table and same_choice),
            "t_star": t_slow,
            "C_min": c_slow,
            "fast": fast,
            "brute": slow,
        }

        if verbose:
            print(f"\n=== Набір {name} ===")
            print("Кумулятивні суми:\n", fast.to_string(index=False))
            print("Прямий перебір:\n", slow.to_string(index=False))
            print(f"Таблиці збігаються: {same_table}; t*={t_slow}, C={c_slow}; вибір збігається: {same_choice}")

    return results


if __name__ == "__main__":

    res = run_checks()
    assert all(r["ok"] for r in res.values()), "Перевірка не пройдена"
    assert res["A"]["t_star"] == 0.4 and res["A"]["C_min"] == 2
    assert res["B"]["t_star"] == float("inf") and res["B"]["C_min"] == 0
    assert res["C"]["t_star"] == 0.8 and res["C"]["C_min"] == 10
    print("\nУсі три перевірки пройдено.")
