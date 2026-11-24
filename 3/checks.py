"""ЛР3, пункти 1-2: перевірки до навчання та перевірки режимів. Викликається з train.py."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lab3_common import (CLASSES, EVAL_MODULES, EXPECTED_SPLIT, EXPECTED_TOTAL, MEAN, REGIMES, STD,
                         TOP_MODULES, build_model, count_params, make_train_loader, md_table,
                         part_df, set_train_mode)

OUT = Path("checks")
FIG = Path("figures")

# Прогноз для досліду з навмисною помилкою (записується ДО запуску; змініть на власний, якщо думаєте інакше).
PREDICTION = (
    "Прогноз: у frozen з model.train() параметри замороженого кістяка НЕ зміняться (requires_grad=False, "
    "градієнтів немає і в оптимізаторі їх немає), а ковзні статистики BN (running_mean, running_var, "
    "num_batches_tracked) у bn1 та layer1-layer4 ЗМІНЯТЬСЯ, бо BN у режимі train оновлює їх під час "
    "прямого проходу. Із кривої втрати на train це майже непомітно: втрата все одно падатиме, "
    "голова навчається, а зсув статистик дає лише дрібну зміну ознак."
)

BN_KEYS = ("running_mean", "running_var", "num_batches_tracked")


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", " ")


# ------------------------------------------------------------------ 1. поділ
def check_split(df: pd.DataFrame) -> tuple[pd.DataFrame, bool, bool, bool]:
    rows, ok = [], True
    for label in CLASSES:
        row = {"Клас": label}
        for part in ("train", "validation", "test"):
            sub = df[(df["label"] == label) & (df["part"] == part)]
            n, s = len(sub), int(sum(int(Path(f).stem) for f in sub["file"]))
            good = (n, s) == EXPECTED_SPLIT[(label, part)]
            ok &= good
            row[part] = f"{n} / {_fmt(s)}" + ("" if good else "  ✗")
        rows.append(row)
    total = {"Клас": "Разом"}
    for part in ("train", "validation", "test"):
        sub = df[df["part"] == part]
        n, s = len(sub), int(sum(int(Path(f).stem) for f in sub["file"]))
        good = (n, s) == EXPECTED_TOTAL[part]
        ok &= good
        total[part] = f"{n} / {_fmt(s)}" + ("" if good else "  ✗")
    rows.append(total)

    # жоден файл не входить у дві частини
    sets = {p: set(df[df["part"] == p]["file"]) for p in ("train", "validation", "test")}
    disjoint = (not (sets["train"] & sets["validation"]) and not (sets["train"] & sets["test"])
                and not (sets["validation"] & sets["test"]) and df["file"].is_unique)
    return pd.DataFrame(rows), bool(ok and disjoint), ok, disjoint


# ------------------------------------------------------------------ 1. пакет після перетворень
def save_batch_figure(df: pd.DataFrame, root: Path) -> None:
    FIG.mkdir(exist_ok=True)
    torch.manual_seed(1)
    x, y = next(iter(make_train_loader(df, root)))
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    img = (x * std + mean).clamp(0, 1)                      # обернене нормування
    fig, axes = plt.subplots(4, 4, figsize=(10, 10.5))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(img[i].permute(1, 2, 0).numpy())
        ax.set_title(f"{int(y[i])} - {CLASSES[int(y[i])]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Перші 16 зображень навчального пакета (після аугментацій)")
    fig.tight_layout()
    fig.savefig(FIG / "train_batch.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 1. початкова втрата
def initial_loss(df: pd.DataFrame, root: Path, device: torch.device) -> pd.DataFrame:
    rows = []
    for regime in ("scratch", "frozen"):
        model, _ = build_model(regime, 1, device)            # кроки 1-5, зерно 1
        loader = make_train_loader(df, root)
        x, y = next(iter(loader))
        set_train_mode(model, regime)
        with torch.no_grad():
            loss = torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device)).item()
        rows.append({"Режим": regime, "L0 (перший пакет)": round(loss, 4),
                     "ln 6": round(math.log(6), 4), "L0 - ln 6": round(loss - math.log(6), 4)})
        del model                                            # цей екземпляр далі не використовується
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 2. параметри
EXPECTED_PARAMS = {
    "scratch": (11179590, 11179590, [(11179590, 3e-3)]),
    "frozen": (11179590, 3078, [(3078, 3e-3)]),
    "unfreeze": (11179590, 8396806, [(3078, 3e-3), (8393728, 3e-4)]),
}


def check_params(device: torch.device) -> tuple[pd.DataFrame, bool]:
    rows, ok = [], True
    for regime in REGIMES:
        model, opt = build_model(regime, 1, device)
        total, trainable = count_params(model)
        groups = [(sum(p.numel() for p in g["params"]), g["lr"]) for g in opt.param_groups]
        in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
        no_frozen_in_opt = all(p.requires_grad for g in opt.param_groups for p in g["params"])
        all_trainable_in_opt = in_opt == {id(p) for p in model.parameters() if p.requires_grad}
        e_total, e_train, e_groups = EXPECTED_PARAMS[regime]
        good = ((total, trainable, groups) == (e_total, e_train, e_groups)
                and no_frozen_in_opt and all_trainable_in_opt)
        ok &= good
        rows.append({"Режим": regime, "Усього": total, "requires_grad=True": trainable,
                     "Групи (n, lr)": "; ".join(f"{n} @ {lr:g}" for n, lr in groups),
                     "Заморожені в оптимізаторі": "ні" if no_frozen_in_opt else "ТАК",
                     "Збігається": "так" if good else "ні"})
    return pd.DataFrame(rows), bool(ok)


# ------------------------------------------------------------------ 2. що змінюється під час навчання
def module_changes(regime: str, device: torch.device, df: pd.DataFrame, root: Path,
                   use_set_train_mode: bool = True) -> tuple[pd.DataFrame, bool, dict]:
    model, opt = build_model(regime, 1, device)
    loader = make_train_loader(df, root)
    state0 = {k: v.detach().clone() for k, v in model.state_dict().items()}   # повна копія (з буферами)
    if use_set_train_mode:
        set_train_mode(model, regime)
    else:
        model.train()                                        # навмисна помилка: BN заморожених модулів теж у train
    loss_fn = torch.nn.CrossEntropyLoss()
    it = iter(loader)
    for _ in range(5):
        x, y = next(it)
        opt.zero_grad()
        loss_fn(model(x.to(device)), y.to(device)).backward()
        opt.step()
    state1 = model.state_dict()
    param_names = {n for n, _ in model.named_parameters()}

    rows, ok = [], True
    for mod in TOP_MODULES:
        keys = [k for k in state1 if k.split(".")[0] == mod]
        p_changed = any(not torch.equal(state0[k], state1[k]) for k in keys if k in param_names)
        bn_keys = [k for k in keys if k.rsplit(".", 1)[-1] in BN_KEYS]
        bn_changed = (any(not torch.equal(state0[k], state1[k]) for k in bn_keys)
                      if bn_keys else None)
        frozen = mod in EVAL_MODULES[regime]
        exp_p = not frozen
        exp_bn = None if not bn_keys else (not frozen)
        good = (p_changed == exp_p) and (bn_changed == exp_bn)
        ok &= good
        rows.append({"Режим": regime, "Модуль": mod,
                     "Параметри змінилися": "так" if p_changed else "ні",
                     "Ковзні статистики BN змінилися": "немає BN" if bn_changed is None
                     else ("так" if bn_changed else "ні"),
                     "Збігається з очікуваним": "так" if good else "ні"})
    # макс. абсолютна зміна статистик у layer1.0.bn1
    extra = {}
    for key in ("running_mean", "running_var"):
        k = f"layer1.0.bn1.{key}"
        extra[key] = float((state1[k] - state0[k]).abs().max().item())
    return pd.DataFrame(rows), bool(ok), extra


# ------------------------------------------------------------------ усі перевірки
def run_all_checks(df: pd.DataFrame, root: Path, device: torch.device) -> bool:
    OUT.mkdir(exist_ok=True)
    report: list[str] = []
    result: dict = {}

    # --- поділ
    table, ok_split, ok_counts, ok_disjoint = check_split(df)
    print("\n=== Контроль поділу (кількість / сума числових імен) ===")
    print(table.to_string(index=False))
    print(f"Усі 18 клітинок + підсумки збігаються: {ok_counts}; файли не перетинаються: {ok_disjoint}")
    report += ["## Контроль поділу", md_table(table),
               f"\nЗбіг з контрольною таблицею: **{'так' if ok_counts else 'НІ'}**; "
               f"жоден файл не входить у дві частини: **{'так' if ok_disjoint else 'НІ'}**.\n"]
    result["split"] = ok_split
    if not ok_split:
        _finish(report, result)
        return False                                         # без правильного поділу далі йти не можна

    # --- пакет
    save_batch_figure(df, root)
    print("Збережено figures/train_batch.png")

    # --- початкова втрата
    loss_table = initial_loss(df, root, device)
    print("\n=== Початкова втрата (до жодного кроку оптимізатора) ===")
    print(loss_table.to_string(index=False))
    report += ["## Початкова втрата", md_table(loss_table), ""]

    # --- параметри
    ptable, ok_params = check_params(device)
    print("\n=== Параметри й групи оптимізатора ===")
    print(ptable.to_string(index=False))
    report += ["## Параметри", md_table(ptable), ""]
    result["params"] = ok_params

    # --- що змінюється
    ok_changes = True
    for regime in ("frozen", "unfreeze"):
        t, ok, extra = module_changes(regime, device, df, root, True)
        ok_changes &= ok
        print(f"\n=== Що змінюється: {regime} (set_train_mode) ===")
        print(t.to_string(index=False))
        print(f"layer1.0.bn1: max|Δrunning_mean|={extra['running_mean']:.3e}, "
              f"max|Δrunning_var|={extra['running_var']:.3e}")
        report += [f"## Що змінюється: {regime}", md_table(t),
                   f"\nlayer1.0.bn1: max abs зміна running_mean = {extra['running_mean']:.3e}, "
                   f"running_var = {extra['running_var']:.3e}\n"]
    result["changes"] = bool(ok_changes)

    # --- дослід із навмисною помилкою
    print("\n=== Дослід із навмисною помилкою (frozen + model.train()) ===")
    print(PREDICTION)
    t, ok_exp, extra = module_changes("frozen", device, df, root, use_set_train_mode=False)
    print(t.to_string(index=False))
    print(f"layer1.0.bn1 після 5 кроків: max|Δrunning_mean|={extra['running_mean']:.6e}, "
          f"max|Δrunning_var|={extra['running_var']:.6e}")
    print("(\"ні\" у стовпці очікуваного тут правильне: це демонстрація помилки, а не збій перевірки)")
    report += ["## Дослід із навмисною помилкою (frozen + model.train())", PREDICTION, md_table(t),
               f"\nlayer1.0.bn1: max abs зміна running_mean = {extra['running_mean']:.6e}, "
               f"running_var = {extra['running_var']:.6e}\n"]
    result["deliberate_error_stats_changed"] = bool(extra["running_mean"] > 0 or extra["running_var"] > 0)
    result["deliberate_error_max_change"] = extra

    _finish(report, result)
    all_ok = bool(result["split"] and result["params"] and result["changes"])
    print(f"\nПеревірки пунктів 1-2 пройдено: {all_ok}")
    return all_ok


def _finish(report: list[str], result: dict) -> None:
    (OUT / "checks_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUT / "checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
