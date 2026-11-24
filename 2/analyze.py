from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from check_search import run_checks
from lab2_starter import load_predictions
from thresholds import best_threshold, candidate_table, confusion, cost, precision, recall


def jnum(x: float) -> float | str | None:

    if isinstance(x, float) and math.isnan(x):

        return None

    if isinstance(x, float) and math.isinf(x):

        return "+inf"

    return x


def rule_row(label: str, y, s, t: float) -> dict:

    tp, fp, fn, tn = confusion(y, s, t)
    return {"rule": label, "threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost(fn, fp)}


def validation_part(data: dict, out: Path) -> float:

    y, s = data["validation_y"], data["validation_scores"].astype(np.float64)
    prec, rec, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    ax[0].plot(rec, prec)
    ax[0].set(xlabel="Recall", ylabel="Precision", title="PR-крива (validation)")
    ax[1].plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[1].set(xlabel="FPR", ylabel="TPR", title="ROC-крива (validation)")
    ax[1].legend()

    table = candidate_table(y, s)
    table.to_csv(out / "validation_candidates.csv", index=False)
    best = best_threshold(table)
    t_star = float(best["threshold"])

    for label, t in (("t=0.5", 0.5), ("t*", t_star)):

        if math.isinf(t):

            continue

        tp, fp, fn, _ = confusion(y, s, t)

        if tp + fp > 0:

            ax[0].scatter([recall(tp, fn)], [precision(tp, fp)], zorder=3, label=label)
            ax[1].scatter([fp / max(1, int((y == 0).sum()))], [tp / max(1, int(y.sum()))], zorder=3, label=label)

    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "validation_curves.png", dpi=150)
    plt.close(fig)

    summary = pd.DataFrame(
        [
            rule_row("Стандартний поріг", y, s, 0.5),
            rule_row("Мінімум вартості", y, s, t_star),
            rule_row("Усі негативні", y, s, float("inf")),
        ]
    )

    direct = summary.iloc[1]

    assert (int(best["TP"]), int(best["FP"]), int(best["FN"]), int(best["TN"]), int(best["C"])) == (
        int(direct["TP"]), int(direct["FP"]), int(direct["FN"]), int(direct["TN"]), int(direct["C"])
    )

    summary.to_csv(out / "validation_summary.csv", index=False)

    print(f"ROC-AUC (validation) = {auc:.6f}")
    print(f"Кандидатів: {len(table)}; t* = {t_star!r}")
    print(summary.to_string(index=False))

    (out / "threshold.json").write_text(
        json.dumps(
            {"t_star": jnum(t_star), "C_validation": int(best["C"]), "roc_auc_validation": auc},
            ensure_ascii=False, indent=2, allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    return t_star


def test_part(data: dict, t_star: float, out: Path) -> None:

    y, s = data["test_y"], data["test_scores"].astype(np.float64)
    ids, amount = data["test_row_ids"], data["test_amount"]
    tp, fp, fn, tn = confusion(y, s, t_star)

    res = {
        "t_star": jnum(t_star), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "C": cost(fn, fp), "precision": jnum(precision(tp, fp)), "recall": jnum(recall(tp, fn)),
    }

    ref = pd.DataFrame(
        [rule_row("Стандартний поріг", y, s, 0.5), rule_row("t* (зафіксований)", y, s, t_star),
         rule_row("Усі негативні", y, s, float("inf"))]
    )

    ref.to_csv(out / "test_rules_reference.csv", index=False)
    (out / "test_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print("\nTEST:", json.dumps(res, ensure_ascii=False))
    print(ref.to_string(index=False))

    if tp + fp == 0:

        print("precision не визначена: TP+FP=0 (модель нікого не позначила шахраєм).")

    if tp + fn == 0:

        print("recall не визначений: TP+FN=0 (у test немає шахрайств).")

    pred = s >= t_star
    df = pd.DataFrame({"csv_row": ids, "label": y, "score": s, "amount": amount})
    df["error_type"] = np.where(pred & (y == 0), "FP", np.where(~pred & (y == 1), "FN", ""))
    df["score_minus_t"] = df["score"] - t_star if not math.isinf(t_star) else np.nan
    errs = df[df["error_type"] != ""].sort_values("csv_row")

    top = pd.concat(
        [errs[errs["error_type"] == k].head(5) for k in ("FP", "FN")]
    )[["csv_row", "error_type", "label", "score", "amount", "score_minus_t"]]

    top.to_csv(out / "test_error_examples.csv", index=False)
    print("\nПерші 5 FP і перші 5 FN (за номером рядка CSV):")
    print(top.to_string(index=False))

    for k in ("FP", "FN"):

        g = errs[errs["error_type"] == k]

        if len(g):

            print(f"{k}: n={len(g)}, медіана Amount={g['amount'].median():.2f}, "
                  f"медіана |s-t*|={g['score_minus_t'].abs().median():.4g}")


def main() -> None:

    ap = argparse.ArgumentParser()
    ap.add_argument("--saved", type=Path, required=True, help="Тека зі збереженими прогнозами")
    args = ap.parse_args()
    out = args.saved
    res = run_checks(verbose=False)

    for name, r in res.items():

        print(f"Перевірка {name}: {'OK' if r['ok'] else 'ПОМИЛКА'} (t*={r['t_star']}, C={r['C_min']})")

    if not all(r["ok"] for r in res.values()):

        raise SystemExit("Перевірки A/B/C не пройдено - test не оцінюється.")

    data = load_predictions(out)
    t_star = validation_part(data, out)
    test_part(data, t_star, out)


if __name__ == "__main__":

    main()
