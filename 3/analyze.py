"""ЛР3. Команда аналізу: журнали -> таблиця, криві, ефекти, перевірка вибору; потім тест вибраної моделі.

    uv run python analyze.py --data <тека, де лежать seg_train і seg_test>

Тут (і лише тут) завантажується test. Модель/поріг/режим за результатами test не змінюються.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score

from lab3_common import (CLASSES, check_data_dir, IMG_SIZE, REGIMES, RUNS, SPLIT_CSV, file_sha256, image_path,
                         labels_of, load_run_logs, load_split, make_eval_loader, macro_f1, md_table,
                         part_df, predict_logits, select_config)

RES = Path("results")
FIG = Path("figures")


def verify_logs(logs: dict, df: pd.DataFrame) -> None:
    """Хеш split.csv однаковий в усіх семи журналах і збігається з поточним файлом."""
    current = file_sha256(SPLIT_CSV)
    hashes = {k: v["split_csv_sha256"] for k, v in logs.items()}
    if len(set(hashes.values())) != 1:
        raise SystemExit(f"Різні хеші split.csv у журналах: {hashes}")
    if set(hashes.values()) != {current}:
        raise SystemExit("Хеш split.csv у журналах не збігається з поточним split.csv. "
                         "Запуски на різних поділах порівнювати не можна - перенавчіть усі сім.")
    # логіти validation збігаються з журналом
    y_val = labels_of(part_df(df, "validation"))
    for (regime, seed), log in logs.items():
        lg = np.load(Path("runs") / f"{regime}_seed{seed}" / "val_logits.npy")
        if lg.shape != (len(y_val), 6):
            raise SystemExit(f"{regime}_seed{seed}: форма val_logits {lg.shape}")
        if abs(macro_f1(y_val, lg) - log["val_macro_f1"][-1]) > 1e-9:
            raise SystemExit(f"{regime}_seed{seed}: macro-F1 з val_logits.npy не збігається з журналом")
    print("Хеш split.csv однаковий у всіх семи журналах і збігається з поточним; val_logits узгоджені.")


def runs_table_and_curves(logs: dict) -> pd.DataFrame:
    rows = []
    for i, (regime, seed) in enumerate(RUNS, 1):
        log = logs[(regime, seed)]
        rows.append({"#": i, "Режим": regime, "Зерно": seed,
                     "Втрата train, епоха 8": round(log["train_loss"][-1], 4),
                     "macro-F1 validation, епоха 8": round(log["val_macro_f1"][-1], 4),
                     "Час навчання, с": round(log["train_time_sec"], 1)})
    table = pd.DataFrame(rows)
    table.to_csv(RES / "runs_table.csv", index=False)
    (RES / "runs_table.md").write_text(md_table(table) + "\n", encoding="utf-8")

    colors = {"scratch": "tab:red", "frozen": "tab:blue", "unfreeze": "tab:green"}
    styles = {1: "-", 2: "--", 3: ":"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    epochs = np.arange(1, 9)
    for (regime, seed) in RUNS:
        log = logs[(regime, seed)]
        kw = dict(color=colors[regime], ls=styles[seed], marker="o", ms=3, label=f"{regime}, seed {seed}")
        ax[0].plot(epochs, log["train_loss"], **kw)
        ax[1].plot(epochs, log["val_macro_f1"], **kw)
    ax[0].set(xlabel="Епоха", ylabel="Втрата train", title="Втрата на train")
    ax[1].set(xlabel="Епоха", ylabel="macro-F1", title="macro-F1 на validation")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "runs_curves.png", dpi=150)
    plt.close(fig)
    return table


def effects(f_last: dict) -> dict:
    f_scr = f_last[("scratch", 1)]
    fr = [f_last[("frozen", s)] for s in (1, 2, 3)]
    un = [f_last[("unfreeze", s)] for s in (1, 2, 3)]
    f_fr, f_un = float(np.mean(fr)), float(np.mean(un))
    r_fr, r_un = max(fr) - min(fr), max(un) - min(un)
    r = max(r_fr, r_un)
    d_pre, d_unf = f_fr - f_scr, f_un - f_fr
    res = {"F_scratch_1": f_scr, "F_frozen_mean": f_fr, "R_frozen": r_fr,
           "F_unfreeze_mean": f_un, "R_unfreeze": r_un, "R": r,
           "delta_pre": d_pre, "delta_unf": d_unf, "delta_total": d_pre + d_unf,
           "check_total": f_un - f_scr}

    def verdict(d: float) -> str:
        return ("|Δ| ≤ R: різниця не відрізняється від розкиду, результату немає"
                if abs(d) <= r else "|Δ| > R: ефект вартий перевірки, але не доведений")
    res["verdict_pre"], res["verdict_unf"] = verdict(d_pre), verdict(d_unf)
    (RES / "effects.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tbl = pd.DataFrame([
        ("F̄_frozen", f"{f_fr:.4f}"), ("R_frozen", f"{r_fr:.4f}"),
        ("F̄_unfreeze", f"{f_un:.4f}"), ("R_unfreeze", f"{r_un:.4f}"),
        ("R = max(R_frozen, R_unfreeze)", f"{r:.4f}"), ("F_scratch,1", f"{f_scr:.4f}"),
        ("Δ_pre = F̄_frozen − F_scratch,1", f"{d_pre:+.4f}"),
        ("Δ_unf = F̄_unfreeze − F̄_frozen", f"{d_unf:+.4f}"),
        ("Δ_pre + Δ_unf", f"{d_pre + d_unf:+.4f}"),
    ], columns=["Величина", "Значення"])
    (RES / "effects.md").write_text(md_table(tbl) + "\n", encoding="utf-8")
    print("\n=== Розклад ефектів ===")
    print(tbl.to_string(index=False))
    print("Δ_pre:", res["verdict_pre"])
    print("Δ_unf:", res["verdict_unf"])
    return res


def load_selected_model(regime: str, seed: int) -> torch.nn.Module:
    ckpt = torch.load("lab3_model.pt", map_location="cpu", weights_only=True)
    if (ckpt["regime"], ckpt["seed"]) != (regime, seed):
        raise SystemExit(f"lab3_model.pt ({ckpt['regime']}, seed {ckpt['seed']}) не відповідає правилу "
                         f"вибору за журналами ({regime}, seed {seed}).")
    if list(ckpt["classes"]) != CLASSES or int(ckpt["img_size"]) != IMG_SIZE:
        raise SystemExit("lab3_model.pt: неправильні classes або img_size")
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 6)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def confusion_outputs(cm: np.ndarray) -> None:
    tbl = pd.DataFrame(cm, index=[f"true:{c}" for c in CLASSES], columns=[f"pred:{c}" for c in CLASSES])
    tbl.to_csv(RES / "confusion_matrix.csv")
    (RES / "confusion_matrix.md").write_text(md_table(tbl.reset_index().rename(columns={"index": ""})) + "\n",
                                             encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(6), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(6), CLASSES)
    ax.set(xlabel="Передбачений клас", ylabel="Справжній клас", title="Матриця плутанини (test)")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def five_errors(cm: np.ndarray, df_test: pd.DataFrame, y: np.ndarray, pred: np.ndarray,
                logits: np.ndarray, root: Path) -> pd.DataFrame:
    pairs = [(int(cm[i, j]), i, j) for i in range(6) for j in range(6) if i != j and cm[i, j] > 0]
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))            # частота спадає, потім true, потім pred
    files = {(i, j): sorted(df_test.loc[(y == i) & (pred == j), "file"]) for _, i, j in pairs}  # за іменем-рядком
    chosen = [(i, j, files[(i, j)][0], cnt) for cnt, i, j in pairs[:5]]
    k = 1
    while len(chosen) < 5:                                   # якщо різних пар менше п'яти
        added = False
        for cnt, i, j in pairs:
            if len(chosen) >= 5:
                break
            if len(files[(i, j)]) > k:
                chosen.append((i, j, files[(i, j)][k], cnt))
                added = True
        if not added:
            break
        k += 1
    idx = {f: n for n, f in enumerate(df_test["file"])}
    soft = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    rows = [{"Файл": f, "Справжній клас": CLASSES[i], "Передбачений клас": CLASSES[j],
             "Імовірність передбаченого класу": round(float(soft[idx[f], j]), 4),
             "Кількість таких помилок у test": cnt} for i, j, f, cnt in chosen]
    tbl = pd.DataFrame(rows)
    tbl.to_csv(RES / "five_errors.csv", index=False)
    (RES / "five_errors.md").write_text(md_table(tbl) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, len(chosen), figsize=(3.4 * len(chosen), 4.2))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, rows):
        with Image.open(image_path(root, r["Справжній клас"], r["Файл"], "test")) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"{r['Файл']}\n{r['Справжній клас']} → {r['Передбачений клас']}\n"
                     f"p={r['Імовірність передбаченого класу']:.3f}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "five_errors.png", dpi=150)
    plt.close(fig)
    return tbl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="Тека з seg_train і seg_test")
    ap.add_argument("--workdir", type=Path, default=Path("."),
                    help="Тека з split.csv, runs/ і lab3_model.pt (за замовчуванням поточна)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data_root = args.data.resolve()
    check_data_dir(data_root, ("seg_test",))                # test-зображення читаються лише тут
    os.chdir(args.workdir)
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    # ---- частина 1: лише журнали (test не чіпаємо)
    df = load_split(SPLIT_CSV)
    logs = load_run_logs(Path("runs"))
    verify_logs(logs, df)
    table = runs_table_and_curves(logs)
    print("\n=== Сім запусків ===")
    print(table.to_string(index=False))

    f_last = {k: v["val_macro_f1"][-1] for k, v in logs.items()}
    eff = effects(f_last)
    regime, seed, means = select_config(f_last)
    print(f"\nВибір за правилом: режим {regime} (середні {({r: round(m, 4) for r, m in means.items()})}), "
          f"запуск seed={seed}, F1(val)={f_last[(regime, seed)]:.4f}")
    if abs(eff["delta_unf"]) <= eff["R"]:
        print("Увага: |Δ_unf| ≤ R, тож вибір між frozen і unfreeze визначив шум (процедура та сама).")
    model = load_selected_model(regime, seed)                # перевірка lab3_model.pt проти журналів
    print("lab3_model.pt відповідає правилу вибору за журналами.")

    # ---- частина 2: test, один раз, на CPU
    loader = make_eval_loader(df, data_root, "test")
    logits = predict_logits(model.to("cpu"), loader, torch.device("cpu"))
    np.save(RES / "test_logits.npy", logits.astype(np.float32))
    df_test = part_df(df, "test")
    y = labels_of(df_test)
    pred = logits.argmax(1)
    f1_test = float(f1_score(y, pred, average="macro", labels=list(range(6)), zero_division=0))
    acc = float((pred == y).mean())
    n = len(y)
    se = float(np.sqrt(acc * (1 - acc) / n))
    f1_val = f_last[(regime, seed)]
    summary = {"regime": regime, "seed": seed, "macro_f1_validation": f1_val, "macro_f1_test": f1_test,
               "difference_val_minus_test": f1_val - f1_test, "accuracy_test": acc, "n_test": n,
               "standard_error_accuracy": se}
    (RES / "test_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    print("\n=== TEST (вибраний запуск, CPU) ===")
    print(f"macro-F1(test)={f1_test:.4f}; macro-F1(validation)={f1_val:.4f}; різниця val−test={f1_val - f1_test:+.4f}")
    print(f"accuracy(test)={acc:.4f}; SE=sqrt(p(1-p)/n)={se:.4f} (n={n})")

    cm = confusion_matrix(y, pred, labels=list(range(6)))
    confusion_outputs(cm)
    print("\nМатриця плутанини (рядки - справжній, стовпці - передбачений):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

    errors = five_errors(cm, df_test, y, pred, logits, data_root)
    print("\n=== П'ять характерних помилок ===")
    print(errors.to_string(index=False))
    print("\nКатегорії помилок (сцена містить обидва класи / ймовірна помилка розмітки / помилка моделі) "
          "визначте самі, переглянувши figures/five_errors.png.")


if __name__ == "__main__":
    main()
