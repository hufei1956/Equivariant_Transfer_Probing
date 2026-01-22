"""
D4 等变 ResNet-18 训练脚本 (v4)
基于 ESCNN 实现废弃物分类 (TrashNet)
训练流程参考 escnn_train_v2 风格

核心特点:
1. 使用 d4_resnet18.py 中的 D4 等变网络结构
2. 训练输出格式与 sre_conv_v1.py 保持一致 (简洁模式)
3. 支持 D4, C4, D8, C8 等不同群类型

参考文献:
- Weiler & Cesa (2019). General E(2)-Equivariant Steerable CNNs.
"""

import os
import sys
import csv
import time
import math
import argparse
from typing import Tuple, List, Optional
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# --- Monkey Patch for escnn vs sklearn compatibility ---
import numpy as np
import escnn.group._numerical as numerical

original_null = numerical.null

def patched_null(A, *args, **kwargs):
    # sklearn's randomized_svd no longer supports np.matrix, but escnn uses it.
    # We convert it to np.asarray just before passing to the function that eventually calls svd.
    if isinstance(A, np.matrix):
        A = np.asarray(A)
    return original_null(A, *args, **kwargs)

numerical.null = patched_null
# -----------------------------------------------------

# 导入模型 (假设 d4_resnet18.py 在同一目录)
try:
    from d4_resnet18 import (
        d4_resnet18, d4_resnet18_restricted, d4_resnet34,
        c4_resnet18, d8_resnet18, c8_resnet18,
        test_equivariance
    )
except ImportError:
    print("Warning: d4_resnet18.py not found. Please ensure it exists in the current directory.")
    sys.exit(1)

from dataset import create_trashnet_datasets

# ==================== 数据集配置 ====================
DATASET_NAME = "trashnet"  # 可选: "trashnet", "realwaste"
DATASET_ROOTS = {
    "trashnet": "dataset-resized",
    "realwaste": "RealWaste",
}

# ==================== 训练参数配置 ====================
TRAIN_CONFIG = {
    "epochs": 100,
    "batch_size": 16,
    "lr": 0.01,           # D4 ResNet 可能需要稍微大一点的学习率 (0.01 vs 0.001)
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "seed": 42,
    "img_size": 224,
    "output_dir": "outputs_d4_resnet18",
    "test_rotation": True,
    "base_width": 64,     # d4_resnet18 默认 base_width
}

# ==================== 模型构建 ====================

def build_d4_model(
    group: str,
    num_classes: int,
    base_width: int = 64,
    fixparams: bool = True
) -> nn.Module:
    """
    构建 D4/Cn 等变 ResNet 模型
    """
    group = group.upper()
    kwargs = {
        "num_classes": num_classes,
        "base_width": base_width,
        "fixparams": fixparams,
        "initial_stride": 2  # ImageNet 风格，适应 224x224
    }
    
    if group == "D4":
        return d4_resnet18(**kwargs)
    elif group == "D4R":  # D4 with restriction
        return d4_resnet18_restricted(**kwargs)
    elif group == "C4":
        return c4_resnet18(**kwargs)
    elif group == "D8":
        return d8_resnet18(**kwargs)
    elif group == "C8":
        return c8_resnet18(**kwargs)
    elif group == "D4_34":
        return d4_resnet34(**kwargs)
    else:
        raise ValueError(f"Unknown group: {group}")

# ==================== 评估函数 ====================

def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    assert scores.ndim == 1 and labels.ndim == 1
    N_pos = labels.sum().item()
    if N_pos == 0:
        return 0.0

    sorted_idx = torch.argsort(scores, descending=True)
    y = labels[sorted_idx]
    tp = torch.cumsum(y, dim=0).float()
    fp = torch.cumsum(1 - y, dim=0).float()
    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / float(N_pos)

    for i in range(precision.numel() - 2, -1, -1):
        if precision[i] < precision[i + 1]:
            precision[i] = precision[i + 1]

    r = torch.cat([torch.zeros(1), recall, torch.ones(1)])
    p = torch.cat([torch.tensor([precision[0].item()]), precision, torch.tensor([0.0])])
    ap = torch.sum((r[1:] - r[:-1]) * p[1:]).item()
    return float(ap)

def compute_map(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, List[float]]:
    num_classes = logits.size(1)
    probs = torch.softmax(logits, dim=1)
    aps: List[float] = []
    for c in range(num_classes):
        labels_c = (targets == c).to(torch.int64)
        scores_c = probs[:, c]
        ap_c = average_precision(scores_c, labels_c)
        aps.append(ap_c)
    mAP = float(sum(aps) / len(aps)) if aps else 0.0
    return mAP, aps

# ==================== 训练与评估 ====================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_grad_norm: Optional[float] = None,
    use_amp: bool = False,
    benchmark: bool = False,
) -> Tuple[float, float]:
    
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    num_batches = len(loader)
    
    for i, (images, targets) in enumerate(loader):
        batch_start = time.time()
        
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        
        scaler.scale(loss).backward()
        
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()

        batch_time = time.time() - batch_start
        
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size
        
        # 移除每个 batch 的打印，只在 benchmark 模式下保留
        if benchmark and (i % 10 == 0 or i == num_batches - 1):
            print(f"  Batch {i}/{num_batches} | Loss: {loss.item():.4f} | Time: {batch_time:.3f}s")
            
        if benchmark and i >= 5:
            print("  [Benchmark] 停止测试")
            break

    avg_loss = running_loss / max(1, total)
    acc = running_correct / max(1, total)
    return avg_loss, acc

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, List[float]]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    acc = accuracy_top1(logits, targets)
    mAP, aps = compute_map(logits, targets)
    return acc, mAP, aps

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_scheduler(optimizer: optim.Optimizer, scheduler_type: str, epochs: int, warmup_epochs: int = 0):
    if scheduler_type == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    elif scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif scheduler_type == "cosine_warmup":
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"不支持的调度器: {scheduler_type}")

# ==================== 旋转鲁棒性测试 ====================

class RotatedDataset(Dataset):
    def __init__(self, base_dataset, angle: float, img_size: int = 224):
        self.base_dataset = base_dataset
        self.angle = angle
        self.img_size = img_size
        self.eval_transform = transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        if hasattr(self.base_dataset, 'dataset'):
            real_idx = self.base_dataset.indices[idx]
            path, target = self.base_dataset.dataset.samples[real_idx]
        else:
            path, target = self.base_dataset.samples[idx]
        
        img = Image.open(path).convert('RGB')
        if self.angle != 0:
            img = TF.rotate(img, self.angle, interpolation=InterpolationMode.BILINEAR, fill=(0, 0, 0))
        img = self.eval_transform(img)
        return img, target

@torch.no_grad()
def test_rotation_robustness(
    model: nn.Module,
    test_set,
    device: torch.device,
    angles: List[int] = None,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict:
    if angles is None:
        angles = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180]
    
    model.eval()
    results = {}
    print("旋转鲁棒性测试 (在原始图像上旋转):")
    
    for angle in angles:
        rotated_dataset = RotatedDataset(test_set, angle, img_size)
        rotated_loader = DataLoader(rotated_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        correct = 0
        total = 0
        
        for images, targets in rotated_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        
        acc = correct / total
        results[angle] = acc
        print(f"  旋转 {angle:3d}°: 准确率 = {acc:.4f} ({acc*100:.2f}%)")
    
    return results

# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="D4 ResNet-18 训练脚本 (v4)")
    
    # 数据参数
    parser.add_argument("--data-root", type=str, default=None, help="数据集根目录 (默认根据 DATASET_NAME 自动选择)")
    parser.add_argument("--dataset-name", type=str, default=DATASET_NAME, choices=["trashnet", "realwaste"])
    parser.add_argument("--output-dir", type=str, default=TRAIN_CONFIG["output_dir"])
    
    # 模型参数
    parser.add_argument("--group", type=str, default="D4", 
                        choices=["D4", "D4R", "C4", "D8", "C8", "D4_34"],
                        help="群类型: D4, C4, D8, C8, D4R")
    parser.add_argument("--base-width", type=int, default=TRAIN_CONFIG["base_width"], help="基础通道数")
    parser.add_argument("--fixparams", action="store_true", default=True, help="调整通道数以保持参数量")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG["lr"])
    parser.add_argument("--momentum", type=float, default=TRAIN_CONFIG["momentum"])
    parser.add_argument("--weight-decay", type=float, default=TRAIN_CONFIG["weight_decay"])
    parser.add_argument("--seed", type=int, default=TRAIN_CONFIG["seed"], help="随机种子")
    parser.add_argument("--img-size", type=int, default=TRAIN_CONFIG["img_size"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Use AMP")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--benchmark", action="store_true", help="Run in benchmark mode (few batches)")
    
    # 策略
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["step", "cosine", "cosine_warmup"])
    parser.add_argument("--warmup-epochs", type=int, default=5)
    
    # 测试
    parser.add_argument("--test-rotation", action="store_true", default=True, help="Test rotation robustness")
    
    args = parser.parse_args()
    
    # 自动处理 data-root
    if args.data_root is None:
        if args.dataset_name in DATASET_ROOTS:
            args.data_root = DATASET_ROOTS[args.dataset_name]
        else:
            raise ValueError(f"未配置数据集 {args.dataset_name} 的默认路径，请在 DATASET_ROOTS 中添加或通过 --data-root 指定")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 数据集
    train_set, val_set, test_set, classes, _ = create_trashnet_datasets(
        data_root=args.data_root,
        split_ratio=(0.7, 0.15, 0.15),
        img_size=args.img_size,
        seed=args.seed,
    )
    
    print("=" * 70)
    print("D4 等变网络训练 (改进版 v4)")
    print("=" * 70)
    print(f"数据集: {args.dataset_name} -> {args.data_root}")
    print(f"类别: {classes}")
    print(f"训练/验证/测试: {len(train_set)}/{len(val_set)}/{len(test_set)}")
    print(f"对称群: {args.group}")
    print(f"基础通道数: {args.base_width}")
    print(f"学习率调度: {args.scheduler}")
    print("=" * 70)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 构建模型
    model = build_d4_model(
        group=args.group,
        num_classes=len(classes),
        base_width=args.base_width,
        fixparams=args.fixparams
    ).to(device)
    
    print(f"构建等变网络: {args.group}-ResNet18")
    
    # 统计参数量
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    print(f"\n模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print("=" * 70)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, args.scheduler, args.epochs, args.warmup_epochs)
    
    # 日志
    log_file = os.path.join("logs", f"d4_{args.group}_{args.dataset_name}_log.csv")
    with open(log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", "val_mAP", "lr"])
        
    best_val_map = 0.0
    best_epoch = 0
    start_time_all = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            max_grad_norm=args.max_grad_norm,
            use_amp=args.amp,
            benchmark=args.benchmark
        )
        
        val_acc, val_map, _ = evaluate(model, val_loader, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:03d} | loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_acc={val_acc:.4f} val_mAP={val_map:.4f} | lr={current_lr:.6f}")
        
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_acc:.4f}", f"{val_map:.4f}", f"{current_lr:.6f}"])
            
        if val_map > best_val_map:
            best_val_map = val_map
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join("checkpoints", f"best_d4_{args.group}_{args.dataset_name}.pth"))
            print(f"  -> 保存最佳模型 (val_mAP={val_map:.4f})")
            
    # 测试
    total_time = time.time() - start_time_all
    print("=" * 70)
    print(f"训练完成! 总耗时: {total_time/60:.1f} 分钟")
    print(f"最佳 epoch: {best_epoch}, 最佳 val_mAP: {best_val_map:.4f}")
    
    ckpt_path = os.path.join("checkpoints", f"best_d4_{args.group}_{args.dataset_name}.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
        print(f"已加载最佳模型: {ckpt_path}")
    print("=" * 70)
    
    test_acc, test_map, test_aps = evaluate(model, test_loader, device)
    print(f"[Test] top1_acc={test_acc:.4f} mAP={test_map:.4f}")
    print("各类别 AP:")
    for i, (cls_name, ap) in enumerate(zip(classes, test_aps)):
        print(f"  {cls_name}: {ap:.4f}")
    print("=" * 70)
    print("完成!")
    
    if args.test_rotation:
        test_rotation_robustness(model, test_set, device, img_size=args.img_size, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
