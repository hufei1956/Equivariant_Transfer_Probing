# 最优先导入numpy，解决命名空间冲突
import numpy
numpy.set_printoptions(threshold=numpy.inf)
# 确保np和numpy指向同一个模块
import numpy as np
np = numpy

import os
import ast
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from escnn_v5_so2 import SO2ResNet18
from typing import Tuple, List

MODEL_KEY = "escnn_so2"  # 可选: "realwaste", "escnn_c4", "escnn_c8", "escnn_so2"
MODEL_FILES = {
    "realwaste": "realwaste_model.pth",
    "escnn_c4": "best_escnn_C4_realwaste.pth",
    "escnn_c8": "best_escnn_C8_realwaste.pth",
    "escnn_so2": "best_so2_v5_realwaste.pth",
}
# For SO2 models, look in outputs_escnn_v5 directory
if MODEL_KEY == "escnn_so2":
    MODEL_PATH = os.path.join("outputs_escnn_v5", MODEL_FILES.get(MODEL_KEY, "best_so2_v5_realwaste.pth"))
else:
    MODEL_PATH = os.path.join("model", MODEL_FILES.get(MODEL_KEY, "realwaste_model.pth"))
MODEL_CLASSES_FILE_PRIMARY = os.path.join("model", "realwaset_list.txt")
MODEL_CLASSES_FILE_FALLBACK = os.path.join("model", "realwaste_list.txt")
TRASHNET_DIR_NAME = "trashnet"
TEST_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0


def build_resnet18(num_classes: int) -> nn.Module:
    try:
        model = models.resnet18(weights=None)
    except AttributeError:
        model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class CustomToTensor:
    """
    自定义的 ToTensor 转换，避免使用 torchvision 的 ToTensor，解决 numpy 命名空间冲突
    """
    def __call__(self, pic):
        import torch
        # 手动实现 ToTensor 功能
        img = torch.tensor(list(pic.getdata()), dtype=torch.float32)
        img = img.view(pic.size[1], pic.size[0], 3).permute(2, 0, 1)
        return img / 255.0

def get_eval_transform(img_size: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        CustomToTensor(),  # 使用自定义的 ToTensor，避免 numpy 冲突
        transforms.Normalize(mean=mean, std=std),
    ])


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.size(0))


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    n_pos = labels.sum().item()
    if n_pos == 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    y = labels[order]
    tp = torch.cumsum(y, dim=0).float()
    fp = torch.cumsum(1 - y, dim=0).float()
    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / float(n_pos)
    for i in range(precision.numel() - 2, -1, -1):
        if precision[i] < precision[i + 1]:
            precision[i] = precision[i + 1]
    r = torch.cat([torch.zeros(1), recall, torch.ones(1)])
    p = torch.cat([torch.tensor([precision[0].item()]), precision, torch.tensor([0.0])])
    ap = torch.sum((r[1:] - r[:-1]) * p[1:]).item()
    return float(ap)


def compute_map(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, List[float]]:
    c = logits.size(1)
    probs = torch.softmax(logits, dim=1)
    aps = []
    for i in range(c):
        labels_i = (targets == i).to(torch.int64)
        scores_i = probs[:, i]
        aps.append(average_precision(scores_i, labels_i))
    mAP = float(sum(aps) / len(aps)) if aps else 0.0
    return mAP, aps


def find_case_insensitive_dir(base_dir: str, name: str) -> str:
    target_lower = name.lower()
    for entry in os.scandir(base_dir):
        if entry.is_dir() and entry.name.lower() == target_lower:
            return entry.path
    return os.path.join(base_dir, name)


class FilteredDataset(Dataset):
    def __init__(self, base: datasets.ImageFolder, keep_targets: set, remap: dict):
        self.base = base
        self.indices = [i for i, (_, t) in enumerate(base.samples) if t in keep_targets]
        self.remap = remap
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        x, t_base = self.base[self.indices[idx]]
        t_new = self.remap[t_base]
        return x, t_new


def load_class_order_from_file(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    try:
        parsed = ast.literal_eval(content)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip().lower() for x in parsed if str(x).strip()]
    except Exception:
        pass
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    out = []
    for ln in lines:
        if ln.startswith("[") and ln.endswith("]") and "," in ln:
            parts = ln[1:-1].split(",")
            out.extend([p.strip().strip("'").strip('"').lower() for p in parts if p.strip()])
        else:
            out.append(ln.lower())
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    ckpt_classes = []
    if isinstance(ckpt, dict) and "classes" in ckpt and isinstance(ckpt["classes"], (list, tuple)):
        ckpt_classes = [str(x).lower() for x in ckpt["classes"]]
    file_order = load_class_order_from_file(MODEL_CLASSES_FILE_PRIMARY)
    if not file_order:
        file_order = load_class_order_from_file(MODEL_CLASSES_FILE_FALLBACK)
    if len(ckpt_classes) > 0:
        model_class_order = ckpt_classes
    elif len(file_order) > 0:
        model_class_order = file_order
    else:
        if isinstance(state, dict) and "fc.weight" in state:
            out_feats = state["fc.weight"].shape[0]
            model_class_order = [str(i) for i in range(out_feats)]
        else:
            model_class_order = ["cardboard", "food organics", "glass", "metal", "miscellaneous trash", "paper", "plastic", "textile trash", "vegetation"]
    num_classes = len(model_class_order)
    is_escnn = (isinstance(ckpt, dict) and "group_type" in ckpt) or ("escnn" in os.path.basename(MODEL_PATH).lower())
    if is_escnn:
        group_type = None
        max_frequency = None
        base_channels = None
        if isinstance(ckpt, dict):
            group_type = ckpt.get("group_type", None)
            max_frequency = ckpt.get("max_frequency", None)
            base_channels = ckpt.get("base_channels", None)
        suffix = os.path.basename(MODEL_PATH).lower()
        if group_type is None:
            if "c4" in suffix:
                group_type = "C4"
            elif "c8" in suffix:
                group_type = "C8"
            elif "so2f" in suffix:
                group_type = "SO2"
                try:
                    max_frequency = int(suffix.split("so2f")[-1].split("_")[0])
                except Exception:
                    max_frequency = 4
        if max_frequency is None:
            max_frequency = 4
        if base_channels is None:
            base_channels = 16
        # For SO2 models, use the SO2ResNet18 class from escnn_v5_so2.py
        model = SO2ResNet18(
            in_channels=3,
            num_classes=num_classes,
            base_channels=int(base_channels),
            max_frequency=int(max_frequency),
            num_samples=16,
            image_size=IMG_SIZE,
            pool_method="norm"
        ).to(device)
    else:
        model = build_resnet18(num_classes=num_classes).to(device)
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state, strict=False)
    model.eval()

    root = find_case_insensitive_dir(os.getcwd(), TRASHNET_DIR_NAME)
    tf = get_eval_transform(IMG_SIZE)
    base = datasets.ImageFolder(root=root, transform=tf)

    name_lower = {cls.lower(): cls for cls in base.classes}
    test_lower = [c.lower() for c in TEST_CLASSES]
    present = [c for c in test_lower if c in name_lower]
    if len(present) == 0:
        raise RuntimeError("No matching classes found in dataset-resized for requested categories.")

    keep_targets = {base.class_to_idx[name_lower[c]] for c in present}
    remap = {base.class_to_idx[name_lower[c]]: i for i, c in enumerate(present)}
    ds = FilteredDataset(base, keep_targets, remap)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    order_to_index = {name: i for i, name in enumerate(model_class_order)}
    model_cols = [order_to_index[c] for c in present if c in order_to_index]

    all_logits = []
    all_targets = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            logits_sel = logits[:, model_cols].cpu()
            all_logits.append(logits_sel)
            all_targets.append(targets.cpu())

    if len(all_logits) == 0:
        raise RuntimeError("No samples found for selected classes.")

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    acc = accuracy_top1(logits, targets)
    mAP, aps = compute_map(logits, targets)

    print(f"TrashNet root: {root}")
    print(f"Evaluated classes (order): {present}")
    print(f"Samples: {logits.size(0)}")
    print(f"ACC: {acc:.4f}")
    print(f"mAP: {mAP:.4f}")
    for i, c in enumerate(present):
        print(f"AP[{c}]: {aps[i]:.4f}")


if __name__ == "__main__":
    main()

