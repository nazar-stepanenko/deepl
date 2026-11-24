"""ЛР3. Спільний код: дані, поділ, модель, режими, цикл навчання, журнали."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights

# ----------------------------------------------------------------- константи
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]  # індекси 0..5
IMG_SIZE = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
TRAIN_PER_CLASS = 200
EPOCHS = 8
BATCH_TRAIN = 32
BATCH_EVAL = 64
WEIGHT_DECAY = 1e-4
LR_HEAD = 3e-3        # fc (у всіх режимах) і вся мережа в scratch
LR_BACKBONE = 3e-4    # layer4 у режимі unfreeze
SPLIT_CSV = Path("split.csv")
_CODE_DIR = Path(__file__).resolve().parent   # фіксуємо до можливої зміни робочої теки

REGIMES = ("scratch", "frozen", "unfreeze")
RUNS = [("scratch", 1), ("frozen", 1), ("frozen", 2), ("frozen", 3),
        ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3)]
TOP_MODULES = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc")
# що навчається (requires_grad=True)
TRAINABLE = {
    "scratch": TOP_MODULES,
    "frozen": ("fc",),
    "unfreeze": ("layer4", "fc"),
}
# модулі, які під час навчання переводяться в eval()
EVAL_MODULES = {
    "scratch": (),
    "frozen": ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4"),
    "unfreeze": ("conv1", "bn1", "layer1", "layer2", "layer3"),
}

# контрольна таблиця поділу: (клас, частина) -> (кількість, сума числових імен)
EXPECTED_SPLIT = {
    ("buildings", "train"): (200, 1996761), ("buildings", "validation"): (218, 4837323), ("buildings", "test"): (219, 4837559),
    ("forest", "train"): (200, 2032625), ("forest", "validation"): (237, 5276962), ("forest", "test"): (237, 5291326),
    ("glacier", "train"): (200, 2183923), ("glacier", "validation"): (276, 6135652), ("glacier", "test"): (277, 6157250),
    ("mountain", "train"): (200, 1950345), ("mountain", "validation"): (262, 5789535), ("mountain", "test"): (263, 5849765),
    ("sea", "train"): (200, 1987613), ("sea", "validation"): (255, 5655068), ("sea", "test"): (255, 5668011),
    ("street", "train"): (200, 1841817), ("street", "validation"): (250, 5541845), ("street", "test"): (251, 5547533),
}
EXPECTED_TOTAL = {"train": (1200, 11993084), "validation": (1498, 33236385), "test": (1502, 33351444)}


# ----------------------------------------------------------------- утиліти
def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple MPS"
    return f"CPU ({platform.processor() or platform.machine()})"


def sync(device: torch.device) -> None:
    """Дочекатися завершення асинхронних операцій перед вимірюванням часу."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_info() -> dict:
    """Хеш коміту й ознака незакомічених змін у відстежуваних файлах."""
    here = _CODE_DIR
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                         cwd=here, text=True, stderr=subprocess.DEVNULL)
        return {"commit": commit, "uncommitted_changes": bool(status.strip())}
    except Exception:  # немає git або це не репозиторій
        return {"commit": "unknown", "uncommitted_changes": None}


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------- поділ
def _class_files(folder: Path) -> list[str]:
    # sorted() за рядками: 0.jpg, 10006.jpg, 1001.jpg, ...
    return sorted(p.name for p in folder.iterdir() if p.suffix.lower() == ".jpg")


def check_data_dir(root: Path, parts=("seg_train", "seg_test")) -> None:
    """Зрозуміле повідомлення, якщо --data вказує не туди."""
    for part in parts:
        d = Path(root) / part / part
        if not d.is_dir():
            raise SystemExit(f"Не знайдено {d}.\n--data має вказувати на теку, де лежать "
                             "seg_train/seg_train/<клас>/*.jpg і seg_test/seg_test/<клас>/*.jpg.")


def build_split(data_root: Path) -> pd.DataFrame:
    """Поділ за процедурою з завдання. Повертає таблицю file,label,part."""
    train_dir = Path(data_root) / "seg_train" / "seg_train"
    test_dir = Path(data_root) / "seg_test" / "seg_test"
    rows: list[tuple[str, str, str]] = []

    rng = np.random.default_rng(0)               # ОДИН генератор для всіх класів по черзі
    for label in CLASSES:
        files = _class_files(train_dir / label)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        rows += [(files[i], label, "train") for i in idx[:TRAIN_PER_CLASS]]

    rng = np.random.default_rng(0)               # новий генератор для seg_test
    for label in CLASSES:
        files = _class_files(test_dir / label)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        cut = len(files) // 2
        rows += [(files[i], label, "validation") for i in idx[:cut]]
        rows += [(files[i], label, "test") for i in idx[cut:]]

    df = pd.DataFrame(rows, columns=["file", "label", "part"])
    # детермінований порядок рядків: частина, індекс класу, ім'я файлу (як рядок)
    df["_p"] = df["part"].map({"train": 0, "validation": 1, "test": 2})
    df["_c"] = df["label"].map({c: i for i, c in enumerate(CLASSES)})
    df = df.sort_values(["_p", "_c", "file"], kind="stable").drop(columns=["_p", "_c"])
    return df.reset_index(drop=True)


def ensure_split(data_root: Path, path: Path = SPLIT_CSV) -> pd.DataFrame:
    """Створити split.csv або переконатися, що наявний файл збігається з процедурою."""
    df = build_split(data_root)
    new_bytes = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if Path(path).exists():
        if Path(path).read_bytes() != new_bytes:
            raise SystemExit(f"{path} не збігається з поділом, побудованим з {data_root}. "
                             "Перевірте шлях до набору або видаліть split.csv і перенавчіть усі сім запусків.")
    else:
        Path(path).write_bytes(new_bytes)
    return df


def load_split(path: Path = SPLIT_CSV) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"file": str, "label": str, "part": str})


def part_df(df: pd.DataFrame, part: str) -> pd.DataFrame:
    return df[df["part"] == part].reset_index(drop=True)


def labels_of(df: pd.DataFrame) -> np.ndarray:
    index = {c: i for i, c in enumerate(CLASSES)}
    return df["label"].map(index).to_numpy(dtype=np.int64)


def image_path(root: Path, label: str, file: str, part: str) -> Path:
    base = "seg_train" if part == "train" else "seg_test"
    return Path(root) / base / base / label / file


# ----------------------------------------------------------------- дані
def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class SplitDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, transform):
        self.paths = [image_path(root, r.label, r.file, r.part) for r in df.itertuples()]
        self.labels = labels_of(df)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        return self.transform(img), int(self.labels[i])


def make_train_loader(df: pd.DataFrame, root: Path) -> DataLoader:
    ds = SplitDataset(part_df(df, "train"), root, train_transform())
    # без generator/sampler: випадковість береться з глобального генератора PyTorch
    return DataLoader(ds, batch_size=BATCH_TRAIN, shuffle=True, drop_last=False, num_workers=0)


def make_eval_loader(df: pd.DataFrame, root: Path, part: str) -> DataLoader:
    ds = SplitDataset(part_df(df, part), root, eval_transform())
    return DataLoader(ds, batch_size=BATCH_EVAL, shuffle=False, num_workers=0)


# ----------------------------------------------------------------- модель
def build_model(regime: str, seed: int, device: torch.device):
    """Кроки 1-5 з завдання. Повертає (model, optimizer)."""
    torch.manual_seed(seed)                                  # 1
    np.random.seed(seed)
    weights = None if regime == "scratch" else ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)    # 2
    model.fc = torch.nn.Linear(512, 6)                       # 3
    for p in model.parameters():                             # 4
        p.requires_grad = False
    for name in TRAINABLE[regime]:
        for p in getattr(model, name).parameters():
            p.requires_grad = True
    model = model.to(device)                                 # 5: пристрій, потім оптимізатор
    return model, make_optimizer(model, regime)


def make_optimizer(model, regime: str):
    head = list(model.fc.parameters())
    if regime == "scratch":
        groups = [{"params": list(model.parameters()), "lr": LR_HEAD}]
    elif regime == "frozen":
        groups = [{"params": head, "lr": LR_HEAD}]
    else:
        groups = [{"params": head, "lr": LR_HEAD},
                  {"params": list(model.layer4.parameters()), "lr": LR_BACKBONE}]
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def set_train_mode(model, regime: str) -> None:
    """model.train(), потім .eval() для заморожених модулів (щоб BN не оновлював статистики)."""
    model.train()
    for name in EVAL_MODULES[regime]:
        getattr(model, name).eval()


@torch.no_grad()
def predict_logits(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = [model(x.to(device)).float().cpu() for x, _ in loader]
    return torch.cat(out).numpy()


def macro_f1(y_true: np.ndarray, logits: np.ndarray) -> float:
    return float(f1_score(y_true, logits.argmax(1), average="macro",
                          labels=list(range(6)), zero_division=0))


def count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ----------------------------------------------------------------- один запуск
def train_run(regime: str, seed: int, df: pd.DataFrame, root: Path, device: torch.device,
              split_hash: str, run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    model, optimizer = build_model(regime, seed, device)
    train_loader = make_train_loader(df, root)
    val_loader = make_eval_loader(df, root, "validation")
    y_val = labels_of(part_df(df, "validation"))
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loss, val_f1 = [], []
    logits = None
    steps = 0
    sync(device)
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        set_train_mode(model, regime)
        total, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)       # втрата батча до кроку оптимізатора
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            seen += len(y)
            steps += 1
        assert seen == 1200, seen
        train_loss.append(total / seen)
        logits = predict_logits(model, val_loader, device)   # eval + no_grad
        val_f1.append(macro_f1(y_val, logits))
        print(f"  [{regime} seed={seed}] епоха {epoch}/{EPOCHS} "
              f"train_loss={train_loss[-1]:.4f} val_macroF1={val_f1[-1]:.4f}")
    sync(device)
    train_time = time.perf_counter() - t0
    assert steps == 304, steps

    np.save(run_dir / "val_logits.npy", logits.astype(np.float32))
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, run_dir / "model.pt")  # локально

    total_params, trainable = count_params(model)
    groups = [{**{k: v for k, v in g.items() if k != "params"},
               "n_params": sum(p.numel() for p in g["params"])} for g in optimizer.param_groups]
    run = {
        "regime": regime,
        "seed": seed,
        "settings": {
            "data": {"classes": CLASSES, "img_size": IMG_SIZE, "n_train": 1200,
                     "n_validation": int(len(y_val)), "split_csv": str(SPLIT_CSV)},
            "transforms": {
                "train": ["RandomResizedCrop(160, scale=(0.7, 1.0))", "RandomHorizontalFlip()",
                          "ToTensor()", "Normalize(mean, std)"],
                "eval": ["Resize((160, 160))", "ToTensor()", "Normalize(mean, std)"],
                "mean": MEAN, "std": STD,
            },
            "model": {"arch": "resnet18",
                      "weights": None if regime == "scratch" else "ResNet18_Weights.IMAGENET1K_V1",
                      "fc": "Linear(512, 6)", "total_params": total_params,
                      "trainable_params": trainable, "dtype": "float32",
                      "bn_momentum": model.bn1.momentum, "bn_eps": model.bn1.eps,
                      "trainable_modules": TRAINABLE[regime], "eval_modules": EVAL_MODULES[regime]},
            "optimizer": {"name": "AdamW", "weight_decay": WEIGHT_DECAY,
                          "defaults": optimizer.defaults, "groups": groups},
            "loss": "CrossEntropyLoss(reduction=mean), без ваг класів і згладжування",
            "loaders": {"train": {"batch_size": BATCH_TRAIN, "shuffle": True, "drop_last": False,
                                  "num_workers": 0, "generator": None, "sampler": None},
                        "eval": {"batch_size": BATCH_EVAL, "shuffle": False, "num_workers": 0}},
            "training": {"epochs": EPOCHS, "steps_per_epoch": 38, "total_steps": steps,
                         "last_batch_size": 16, "device": str(device),
                         "lr_schedule": None, "grad_clip": None, "amp": False,
                         "early_stopping": False, "checkpoint_selection": False},
        },
        "code": git_info(),
        "split_csv_sha256": split_hash,
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "torchvision": torchvision.__version__},
        "device": device_name(device),
        "train_loss": train_loss,
        "val_macro_f1": val_f1,
        "train_time_sec": train_time,
    }
    (run_dir / "run.json").write_text(
        json.dumps(jsonable(run), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return run


def load_run_logs(runs_dir: Path = Path("runs")) -> dict[tuple[str, int], dict]:
    logs = {}
    for regime, seed in RUNS:
        path = Path(runs_dir) / f"{regime}_seed{seed}" / "run.json"
        if not path.exists():
            raise SystemExit(f"Немає журналу {path}. Спершу запустіть навчання.")
        logs[(regime, seed)] = json.loads(path.read_text(encoding="utf-8"))
    return logs


def select_config(f_last: dict[tuple[str, int], float]) -> tuple[str, int, dict[str, float]]:
    """Режим з найбільшим середнім F; серед його запусків - найбільший F. Нічия: менший індекс/зерно."""
    means = {}
    for regime in REGIMES:
        vals = [f for (r, _), f in f_last.items() if r == regime]
        means[regime] = float(np.mean(vals))
    best_regime = max(REGIMES, key=lambda r: (means[r], -REGIMES.index(r)))
    seeds = sorted(s for (r, s) in f_last if r == best_regime)
    best_seed = max(seeds, key=lambda s: (f_last[(best_regime, s)], -s))
    return best_regime, best_seed, means
