import os
import sys
import ast
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from typing import Tuple, List

# 定义配置
MODEL_PATH = os.path.join("checkpoints", "best_resnet18_ablation_realwaste.pth")
TRASHNET_DIR_NAME = "ablation_study_data/trashnet_small"
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

def get_eval_transform(img_size: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
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
    # 如果路径本身存在，直接返回
    if os.path.exists(os.path.join(base_dir, name)):
        return os.path.join(base_dir, name)
    
    # 尝试不区分大小写查找
    # 这里处理多级目录
    parts = name.split('/')
    current_path = base_dir
    for part in parts:
        found = False
        if os.path.exists(os.path.join(current_path, part)):
             current_path = os.path.join(current_path, part)
             found = True
        else:
            for entry in os.scandir(current_path):
                if entry.is_dir() and entry.name.lower() == part.lower():
                    current_path = entry.path
                    found = True
                    break
        if not found:
            return os.path.join(base_dir, name) # Return original if not found
    return current_path

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    ckpt = torch.load(MODEL_PATH, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    
    # Determine class order
    ckpt_classes = []
    if isinstance(ckpt, dict) and "classes" in ckpt and isinstance(ckpt["classes"], (list, tuple)):
        ckpt_classes = [str(x).lower() for x in ckpt["classes"]]
    
    if len(ckpt_classes) > 0:
        model_class_order = ckpt_classes
    else:
        # Default classes used in training ablation (5 classes)
        model_class_order = ["cardboard", "glass", "metal", "paper", "plastic"]
    
    print(f"Model class order: {model_class_order}")
    
    # Build ResNet18 Model
    model = build_resnet18(num_classes=len(model_class_order)).to(device)
            
    try:
        model.load_state_dict(state)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Strict load failed: {e}. Trying strict=False")
        try:
            model.load_state_dict(state, strict=False)
            print("Model loaded with strict=False.")
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            return
            
    model.eval()

    # 处理数据集路径
    # TRASHNET_DIR_NAME 可以是相对路径
    if os.path.isabs(TRASHNET_DIR_NAME):
        root = TRASHNET_DIR_NAME
    else:
        root = os.path.join(os.getcwd(), TRASHNET_DIR_NAME)
        
    if not os.path.exists(root):
        # 尝试查找
        root = find_case_insensitive_dir(os.getcwd(), TRASHNET_DIR_NAME)
        
    if not os.path.exists(root):
        print(f"Error: TrashNet directory not found at {root}")
        return
        
    tf = get_eval_transform(IMG_SIZE)
    try:
        base = datasets.ImageFolder(root=root, transform=tf)
    except Exception as e:
        print(f"Error loading dataset from {root}: {e}")
        return

    name_lower = {cls.lower(): cls for cls in base.classes}
    test_lower = [c.lower() for c in TEST_CLASSES]
    present = [c for c in test_lower if c in name_lower]
    
    if len(present) == 0:
        print(f"Warning: No matching classes found. Available classes: {list(name_lower.keys())}")
        present = [c for c in test_lower if c in [cls.lower() for cls in base.classes]]
        
    if len(present) == 0:
         raise RuntimeError("No matching classes found in TrashNet for requested categories.")

    keep_targets = {base.class_to_idx[name_lower[c]] for c in present}
    remap = {base.class_to_idx[name_lower[c]]: i for i, c in enumerate(present)}
    ds = FilteredDataset(base, keep_targets, remap)
    print(f"Filtered dataset size: {len(ds)}")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    order_to_index = {name: i for i, name in enumerate(model_class_order)}
    model_cols = [order_to_index[c] for c in present if c in order_to_index]
    
    print(f"Mapping TrashNet classes {present} to Model indices {model_cols}")

    all_logits = []
    all_targets = []
    print("Starting inference...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            try:
                if i % 10 == 0: print(f"Batch {i}")
                images = images.to(device, non_blocking=True)
                logits = model(images)
                logits_sel = logits[:, model_cols].cpu()
                all_logits.append(logits_sel)
                all_targets.append(targets.cpu())
            except Exception as e:
                print(f"Error at batch {i}: {e}")
                break
    print("Inference done.")

    if len(all_logits) == 0:
        raise RuntimeError("No samples found for selected classes.")

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    acc = accuracy_top1(logits, targets)
    mAP, aps = compute_map(logits, targets)

    result_str = []
    result_str.append(f"TrashNet root: {root}")
    result_str.append(f"Evaluated classes (order): {present}")
    result_str.append(f"Samples: {logits.size(0)}")
    result_str.append(f"ACC: {acc:.4f}")
    result_str.append(f"mAP: {mAP:.4f}")
    for i, c in enumerate(present):
        result_str.append(f"AP[{c}]: {aps[i]:.4f}")
    
    print("\n".join(result_str))
    
    with open("ablation_rw_test_tn_results.txt", "w") as f:
        f.write("\n".join(result_str))

if __name__ == "__main__":
    main()
