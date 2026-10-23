from __future__ import annotations

import numpy as np
import pandas as pd

COST_FN = 10
COST_FP = 1


def confusion(y: np.ndarray, s: np.ndarray, t: float) -> tuple[int, int, int, int]:

    y = np.asarray(y)
    pred = np.asarray(s) >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))

    return tp, fp, fn, tn


def cost(fn: int, fp: int) -> int:

    return COST_FN * fn + COST_FP * fp


def precision(tp: int, fp: int) -> float:

    return tp / (tp + fp) if tp + fp > 0 else float("nan")


def recall(tp: int, fn: int) -> float:

    return tp / (tp + fn) if tp + fn > 0 else float("nan")


def candidate_table(y: np.ndarray, s: np.ndarray) -> pd.DataFrame:

    y = np.asarray(y).astype(np.int64)
    s = np.asarray(s, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    order = np.argsort(-s, kind="stable")
    s_sorted = s[order]
    y_sorted = y[order]
    cum_tp = np.cumsum(y_sorted)
    cum_fp = np.cumsum(1 - y_sorted)
    last = np.flatnonzero(np.r_[s_sorted[1:] != s_sorted[:-1], True])
    thr = np.r_[np.inf, s_sorted[last]]
    tp = np.r_[0, cum_tp[last]]
    fp = np.r_[0, cum_fp[last]]
    fn = n_pos - tp
    tn = n_neg - fp
    c = COST_FN * fn + COST_FP * fp

    return pd.DataFrame(

        {"threshold": thr, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": c}
    )


def best_threshold(table: pd.DataFrame) -> pd.Series:

    best = table[table["C"] == table["C"].min()]
    return best.loc[best["threshold"].idxmax()]


def candidate_table_brute(y, s) -> pd.DataFrame:

    cands = sorted({float(v) for v in s}, reverse=True)
    rows = []

    for t in [float("inf")] + cands:

        tp = fp = fn = tn = 0

        for yi, si in zip(y, s):

            pred = si >= t

            if pred and yi == 1:

                tp += 1

            elif pred and yi == 0:

                fp += 1

            elif not pred and yi == 1:

                fn += 1

            else:

                tn += 1

        rows.append((t, tp, fp, fn, tn, 10 * fn + fp))

    return pd.DataFrame(rows, columns=["threshold", "TP", "FP", "FN", "TN", "C"])


def best_threshold_brute(table: pd.DataFrame) -> tuple[float, int]:

    min_c = min(table["C"])
    t = max(th for th, c in zip(table["threshold"], table["C"]) if c == min_c)

    return float(t), int(min_c)
