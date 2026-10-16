import os
import json
import random
from typing import Tuple, List, Dict, Optional

# 显式导入numpy并作为np别名，确保torchvision使用正确的numpy类型
import numpy as np

import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

# 辅助函数：列出数据根目录下的所有类别文件夹（按字典序排序，保证映射稳定）
def _list_class_dirs(data_root: str) -> List[str]:
    # 列出数据根目录下的所有类别文件夹（按字典序排序，保证映射稳定）
    classes = [d.name for d in os.scandir(data_root) if d.is_dir()]
    classes.sort()
    return classes

# 自定义ToTensor转换，避免numpy命名空间冲突
class CustomToTensor:
    def __call__(self, pic):
        """Convert PIL Image to tensor without numpy conflicts."""
        import torch
        # 直接从PIL图像转换为tensor，绕过numpy
        img = torch.tensor(list(pic.getdata()), dtype=torch.float32)
        # 重塑为正确的形状：(H, W, C) -> (C, H, W)
        img = img.view(pic.size[1], pic.size[0], 3).permute(2, 0, 1)
        # 归一化到[0, 1]
        return img / 255.0

# 辅助函数：构建数据变换（训练集增强，验证集简单缩放）
def _build_transforms(img_size: int) -> Dict[str, transforms.Compose]:
    # 定义轻量的数据增强与评估变换
    # 训练：随机尺度裁剪 + 水平翻转；不做随机旋转（与旋转等变网络对比更公平）
    # 验证/测试：固定 Resize + CenterCrop，确保可复现
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        CustomToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        CustomToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])
    return {"train": train_tf, "eval": eval_tf}

# 辅助函数：按类别划分数据集索引（70/15/15 比例）
def _compute_split_indices_by_class(
    dataset: datasets.ImageFolder,
    ratio: Tuple[float, float, float],
    seed: int,
) -> Dict[str, List[int]]:
    # 按类别进行 70/15/15 划分，保证各类比例一致
    rng = random.Random(seed)
    class_to_indices: Dict[int, List[int]] = {c: [] for c in range(len(dataset.classes))}
    for idx, (_, target) in enumerate(dataset.samples):
        class_to_indices[target].append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for c, indices in class_to_indices.items():
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * ratio[0])
        n_val = int(n * ratio[1])
        # 剩余全部分配给 test，保证总数不丢失
        n_test = n - n_train - n_val
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    return {"train": train_idx, "val": val_idx, "test": test_idx}

# 辅助函数：将索引转换为相对路径（用于存盘）
def _indices_to_relpaths(dataset: datasets.ImageFolder, indices: List[int], data_root: str) -> List[str]:
    # 将索引转换为相对路径，方便存盘复现划分
    rels: List[str] = []
    for i in indices:
        path, _ = dataset.samples[i]
        rels.append(os.path.relpath(path, data_root).replace("\\", "/"))
    return rels

# 辅助函数：将相对路径恢复为当前 ImageFolder 的索引
def _relpaths_to_indices(dataset: datasets.ImageFolder, rels: List[str], data_root: str) -> List[int]:
    # 将相对路径恢复为当前 ImageFolder 的索引
    # 构建从相对路径到索引的映射
    path_to_idx: Dict[str, int] = {}
    for i, (path, _) in enumerate(dataset.samples):
        rel = os.path.relpath(path, data_root).replace("\\", "/")
        path_to_idx[rel] = i
    # 映射，不存在的样本会被跳过（稳妥处理）
    indices: List[int] = []
    for rel in rels:
        if rel in path_to_idx:
            indices.append(path_to_idx[rel])
    return indices

# 辅助函数：加载或创建数据集划分（按类别 70/15/15 划分）
def _load_or_create_split(
    data_root: str,
    split_file: str,
    base_dataset: datasets.ImageFolder,
    ratio: Tuple[float, float, float],
    seed: int,
) -> Dict[str, List[int]]:
    # 如果已有 split 文件则加载，否则创建并保存
    if os.path.exists(split_file):
        with open(split_file, "r", encoding="utf-8") as f:
            stored = json.load(f)
        train_idx = _relpaths_to_indices(base_dataset, stored["train"], data_root)
        val_idx = _relpaths_to_indices(base_dataset, stored["val"], data_root)
        test_idx = _relpaths_to_indices(base_dataset, stored["test"], data_root)
        return {"train": train_idx, "val": val_idx, "test": test_idx}
    else:
        splits = _compute_split_indices_by_class(base_dataset, ratio, seed)
        os.makedirs(os.path.dirname(split_file), exist_ok=True)
        payload = {
            "classes": base_dataset.classes,
            "ratio": ratio,
            "seed": seed,
            "train": _indices_to_relpaths(base_dataset, splits["train"], data_root),
            "val": _indices_to_relpaths(base_dataset, splits["val"], data_root),
            "test": _indices_to_relpaths(base_dataset, splits["test"], data_root),
        }
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return splits

# 主函数：创建 TrashNet 数据集（按类别 70/15/15 划分）
def create_trashnet_datasets(
    data_root: str,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    img_size: int = 224,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset, List[str], str]:
    """
    创建 TrashNet 的 train/val/test 数据集。
    返回：train_set, val_set, test_set, classes, split_file_path
    - data_root: 例如 './dataset-resized'
    - split_ratio: 划分比例 (train, val, test) = (0.7, 0.15, 0.15)
    - img_size: 输入到 ResNet 的图像尺度（默认 224）
    - seed: 随机种子，保证划分可复现
    """
    assert abs(sum(split_ratio) - 1.0) < 1e-6, "split_ratio 必须和为 1.0"
    assert os.path.isdir(data_root), f"数据目录不存在: {data_root}"

    # 列出类别，确保目录结构正确
    classes = _list_class_dirs(data_root)
    assert len(classes) > 0, f"未在 {data_root} 下发现类别文件夹"

    tfs = _build_transforms(img_size)

    # 基础数据集（用于生成划分索引和路径映射）
    base_dataset = datasets.ImageFolder(root=data_root, transform=None)
    split_basename = os.path.basename(os.path.normpath(data_root)) + "_split.json"
    split_file = os.path.join("splits", split_basename)
    splits = _load_or_create_split(data_root, split_file, base_dataset, split_ratio, seed)

    # 构造三份带不同 transform 的 ImageFolder，然后用 Subset 套上各自索引
    train_full = datasets.ImageFolder(root=data_root, transform=tfs["train"])
    eval_full = datasets.ImageFolder(root=data_root, transform=tfs["eval"])
    train_set = Subset(train_full, splits["train"])
    val_set = Subset(eval_full, splits["val"])
    test_set = Subset(eval_full, splits["test"])

    return train_set, val_set, test_set, train_full.classes, split_file


if __name__ == "__main__":
    # 简单自检：统计划分结果与类别映射
    tr, va, te, classes, split_path = create_trashnet_datasets(
        data_root="dataset-resized",
        split_ratio=(0.7, 0.15, 0.15),
        img_size=224,
        seed=42,
    )
    print(f"Classes: {classes}")
    print(f"Split file: {split_path}")
    print(f"Train/Val/Test sizes: {len(tr)}, {len(va)}, {len(te)}")

