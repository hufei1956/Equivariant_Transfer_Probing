import os
import csv
import time
import argparse
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from dataset import create_trashnet_datasets

DATASET_NAME = "realwaste"
DATASET_ROOTS = {
    "trashnet": "ablation_study_data/trashnet_small",
    "realwaste": "ablation_study_data/realwaste_small",
}

def build_resnet18(num_classes: int, pretrained: bool = False) -> nn.Module:
    """
    构建 ResNet18 模型。
    为适配不同版本的 torchvision，这里同时兼容 weights / pretrained 两种写法。
    当 pretrained=False 时，不加载 ImageNet 权重；pretrained=True 则加载并微调。
    """
    try:
        # 新版 API（torchvision >= 0.13）
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except AttributeError:
        # 旧版 API
        model = models.resnet18(pretrained=pretrained)
    # 替换最后的全连接层为当前任务的类别数
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# 计算 Top-1 准确率
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    # 计算 Top-1 准确率
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

# 计算单类别的 AP（Average Precision）
def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    计算单类别的 AP（Average Precision）。
    - scores: 预测分数（越大越置信），形状 [N]
    - labels: 二值标签（0/1），形状 [N]
    实现流程：
      1) 按分数降序排序
      2) 累积求得 precision / recall 曲线
      3) 对 precision 做向后最大包络
      4) 对 PR 曲线积分得到 AP
    """
    assert scores.ndim == 1 and labels.ndim == 1
    N_pos = labels.sum().item()
    if N_pos == 0:
        return 0.0

    # 按分数排序
    sorted_idx = torch.argsort(scores, descending=True)
    y = labels[sorted_idx]
    tp = torch.cumsum(y, dim=0).float()
    fp = torch.cumsum(1 - y, dim=0).float()
    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / float(N_pos)

    # 将 precision 做非增包络（使其单调不增，便于积分）
    for i in range(precision.numel() - 2, -1, -1):
        if precision[i] < precision[i + 1]:
            precision[i] = precision[i + 1]

    # 在 (0,0) 与 (1,0) 两端补点，便于数值积分
    r = torch.cat([torch.zeros(1), recall, torch.ones(1)])
    p = torch.cat([torch.tensor([precision[0].item()]), precision, torch.tensor([0.0])])
    # 梯形法积分
    ap = torch.sum((r[1:] - r[:-1]) * p[1:]).item()
    return float(ap)

# 计算多类别的 mAP（one-vs-rest）
def compute_map(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, List[float]]:
    """
    计算多类别的 mAP（one-vs-rest）。
    - logits: [N, C] 未归一化分数（或概率），建议使用 softmax 后的置信度更稳定
    - targets: [N] 类别索引
    返回：mAP, 每类 AP 列表
    """
    num_classes = logits.size(1)
    probs = torch.softmax(logits, dim=1)
    aps: List[float] = []
    for c in range(num_classes):
        # 将第 c 类视为正类，构造二值标签
        labels_c = (targets == c).to(torch.int64)
        scores_c = probs[:, c]
        ap_c = average_precision(scores_c, labels_c)
        aps.append(ap_c)
    mAP = float(sum(aps) / len(aps)) if aps else 0.0
    return mAP, aps


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    # 训练一个 epoch，返回平均 loss 与训练集 top1 准确率
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        running_correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / max(1, total)
    acc = running_correct / max(1, total)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    # 在验证/测试集上评估 Top-1 准确率与 mAP
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
    mAP, _ = compute_map(logits, targets)
    return acc, mAP


def main():
    parser = argparse.ArgumentParser(description="TrashNet ResNet18 Baseline 训练脚本")
    parser.add_argument("--data-root", type=str, default="dataset-resized", help="数据集根目录")
    parser.add_argument("--output-dir", type=str, default="outputs", help="日志与模型输出目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--momentum", type=float, default=0.9, help="动量（SGD）")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--img-size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（影响划分）")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 线程数（Windows 建议 0）")
    parser.add_argument("--pretrained", action="store_true", help="是否加载 ImageNet 预训练权重进行微调")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 构造数据集与数据加载器
    selected_root = DATASET_ROOTS.get(DATASET_NAME.lower(), "dataset-resized")
    train_set, val_set, test_set, classes, split_path = create_trashnet_datasets(
        data_root=selected_root,
        split_ratio=(0.7, 0.15, 0.15),
        img_size=args.img_size,
        seed=args.seed,
    )
    print(f"数据集: {DATASET_NAME} -> {selected_root}")
    print(f"类别映射: {classes}")
    print(f"划分文件: {split_path}")
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # 构建模型与优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(num_classes=len(classes), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)

    # 日志文件（CSV）
    log_file = os.path.join("logs", f"train_ablation_{DATASET_NAME}_log.csv")
    with open(log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", "val_mAP", "lr"])

    best_val_acc = 0.0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_map = evaluate(model, val_loader, device)
        scheduler.step()

        # 记录日志
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.4f}", f"{val_acc:.4f}", f"{val_map:.4f}", f"{scheduler.get_last_lr()[0]:.6f}"])

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_acc={val_acc:.4f} val_mAP={val_map:.4f}")

        # 保存最佳模型（按验证准确率）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join("checkpoints", f"best_resnet18_ablation_{DATASET_NAME}.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_mAP": val_map,
                "classes": classes,
                "pretrained": args.pretrained,
            }, ckpt_path)
            print(f"已保存最佳模型到: {ckpt_path}")

    total_time = time.time() - start_time
    print(f"训练完成，总耗时 {total_time/60:.1f} 分钟")

    # 最终在测试集上评估
    test_acc, test_map = evaluate(model, test_loader, device)
    print(f"[Test] top1_acc={test_acc:.4f} mAP={test_map:.4f}")

    # 将测试结果追加到日志
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test", "", "", f"{test_acc:.4f}", f"{test_map:.4f}", ""])


if __name__ == "__main__":
    main()

