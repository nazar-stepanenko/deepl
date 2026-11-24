"""ЛР3. Команда навчання: поділ, перевірки пунктів 1-2, сім запусків, журнали, вибір, lab3_model.pt.

    uv run python train.py --data <тека, де лежать seg_train і seg_test>

Test тут не використовується.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from checks import run_all_checks
from lab3_common import (CLASSES, check_data_dir, IMG_SIZE, REGIMES, RUNS, SPLIT_CSV, device_name, ensure_split,
                         file_sha256, load_run_logs, pick_device, select_config, train_run)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                    help="Тека з seg_train/seg_train/... і seg_test/seg_test/... (набір до репозиторію не додається)")
    ap.add_argument("--workdir", type=Path, default=Path("."),
                    help="Тека для результатів (split.csv, runs/, lab3_model.pt, figures/ ...). "
                         "Для повторного навчання вкажіть нову порожню теку, щоб не чіпати збережені результати")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--resume", action="store_true", help="Пропустити запуски, що вже мають run.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # кирилиця й Δ в будь-якій консолі

    root = args.data.resolve()                              # абсолютний шлях ДО зміни робочої теки
    check_data_dir(root)
    args.workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.workdir)                                  # усі вихідні файли - відносно workdir

    device = pick_device(args.device)
    print(f"Пристрій: {device_name(device)}")

    df = ensure_split(root, SPLIT_CSV)                 # split.csv створюється/перевіряється
    split_hash = file_sha256(SPLIT_CSV)
    print(f"split.csv SHA-256: {split_hash}")

    if not run_all_checks(df, root, device):           # пункти 1-2
        raise SystemExit("Перевірки не збіглися з очікуваним - знайдіть помилку, перш ніж іти далі.")

    runs_dir = Path("runs")
    for regime, seed in RUNS:                                # пункт 3: рівно сім запусків
        run_dir = runs_dir / f"{regime}_seed{seed}"
        if (run_dir / "run.json").exists():
            old = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            if not args.resume:
                raise SystemExit(f"{run_dir} уже існує. Для повторного навчання вкажіть нову теку: --workdir <тека>; для продовження перерваного - --resume.")
            if old["split_csv_sha256"] != split_hash:
                raise SystemExit(f"{run_dir}: інший поділ. Після зміни поділу перенавчіть усі сім запусків.")
            print(f"Пропуск {regime} seed={seed}: уже виконано")
            continue
        print(f"\n=== Запуск {regime}, seed={seed} ===")
        run = train_run(regime, seed, df, root, device, split_hash, run_dir)
        print(f"Готово: F1(val, епоха 8)={run['val_macro_f1'][-1]:.4f}, час {run['train_time_sec']:.1f} с")

    # пункт 5: вибір лише за validation
    logs = load_run_logs(runs_dir)
    f_last = {key: log["val_macro_f1"][-1] for key, log in logs.items()}
    regime, seed, means = select_config(f_last)
    print("\nСередні macro-F1 на validation за режимами:",
          {r: round(m, 4) for r, m in means.items()})
    print(f"Вибрано: {regime}, seed={seed}, F1(val)={f_last[(regime, seed)]:.4f}")

    state = torch.load(runs_dir / f"{regime}_seed{seed}" / "model.pt", map_location="cpu",
                       weights_only=True)
    torch.save({"state_dict": state, "regime": regime, "seed": seed,
                "classes": CLASSES, "img_size": IMG_SIZE}, "lab3_model.pt")
    Path("selection.json").write_text(json.dumps(
        {"regime": regime, "seed": seed, "val_macro_f1": f_last[(regime, seed)],
         "mean_val_macro_f1_by_regime": means}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Збережено lab3_model.pt і selection.json. Test не використовувався.")


if __name__ == "__main__":
    main()
