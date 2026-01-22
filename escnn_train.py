"""
ESCNN 等变神经网络训练脚本
基于 E(2)-Equivariant Steerable CNNs 实现废弃物分类
支持 C4, C8, D4, SO(2) 等不同旋转群

依赖安装: pip install escnn

参考文献:
- Weiler & Cesa (2019). General E(2)-Equivariant Steerable CNNs. NeurIPS.
"""

import os
import csv
import time
import argparse
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ESCNN 核心模块
from escnn import gspaces
from escnn import nn as enn

from dataset import create_trashnet_datasets

# ==================== 数据集配置 ====================
DATASET_NAME = "trashnet"  # 可选: "trashnet", "realwaste"
DATASET_ROOTS = {
    "trashnet": "dataset-resized",
    "realwaste": "RealWaste",
}


# ==================== 等变网络模块 ====================

class EquivariantBasicBlock(nn.Module):
    """
    等变版本的 ResNet BasicBlock
    包含两个 3x3 等变卷积和残差连接
    """
    expansion = 1

    def __init__(
        self,
        in_type: enn.FieldType,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        gspace = in_type.gspace
        
        # 定义输出类型
        out_type = enn.FieldType(gspace, out_channels * [gspace.regular_repr])
        
        # 第一个卷积: 3x3, 可能有 stride
        self.conv1 = enn.R2Conv(
            in_type, out_type,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = enn.InnerBatchNorm(out_type)
        self.relu1 = enn.ReLU(out_type, inplace=True)
        
        # 第二个卷积: 3x3, stride=1
        self.conv2 = enn.R2Conv(
            out_type, out_type,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = enn.InnerBatchNorm(out_type)
        self.relu2 = enn.ReLU(out_type, inplace=True)
        
        self.downsample = downsample
        self.out_type = out_type

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 残差连接
        out = enn.GeometricTensor(out.tensor + identity.tensor, out.type)
        out = self.relu2(out)
        
        return out


class EquivariantResNet18(nn.Module):
    """
    等变版本的 ResNet-18
    
    架构:
    - 输入层: 7x7 等变卷积
    - 4个 stage，每个包含 2 个 BasicBlock
    - Group Pooling: 将等变特征转换为不变特征
    - 全局平均池化 + 全连接层
    
    参数:
    - gspace: 对称群空间 (C4, C8, D4, SO2 等)
    - num_classes: 分类类别数
    - num_rot: 旋转离散化数量 (用于 SO2)
    """
    
    def __init__(
        self,
        gspace: gspaces.GSpace,
        num_classes: int,
        base_channels: int = 16,  # 基础通道数，控制模型大小
    ):
        super().__init__()
        self.gspace = gspace
        
        # 输入类型: 3通道 RGB 图像（标量场）
        self.in_type = enn.FieldType(gspace, 3 * [gspace.trivial_repr])
        
        # Stage 0: 初始卷积层 (类似 ResNet 的 conv1)
        # 输出通道数 = base_channels
        out_type_0 = enn.FieldType(gspace, base_channels * [gspace.regular_repr])
        self.conv1 = enn.R2Conv(
            self.in_type, out_type_0,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = enn.InnerBatchNorm(out_type_0)
        self.relu1 = enn.ReLU(out_type_0, inplace=True)
        self.maxpool = enn.PointwiseMaxPool(out_type_0, kernel_size=3, stride=2, padding=1)
        
        # Stage 1-4: 残差块
        self.layer1 = self._make_layer(out_type_0, base_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(self.layer1[-1].out_type, base_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(self.layer2[-1].out_type, base_channels * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(self.layer3[-1].out_type, base_channels * 8, blocks=2, stride=2)
        
        # Group Pooling: 将等变特征转换为不变特征
        self.gpool = enn.GroupPooling(self.layer4[-1].out_type)
        
        # 计算池化后的特征维度
        # GroupPooling 输出维度 = 通道数 (因为对群维度取最大值)
        gpool_out_dim = self.gpool.out_type.size
        
        # 全连接分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(gpool_out_dim, num_classes)
        
        # 初始化权重
        self._initialize_weights()

    def _make_layer(
        self,
        in_type: enn.FieldType,
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.ModuleList:
        """构建一个 stage 的残差块"""
        layers = nn.ModuleList()
        
        # 第一个块可能需要下采样
        downsample = None
        out_type = enn.FieldType(self.gspace, out_channels * [self.gspace.regular_repr])
        
        if stride != 1 or in_type.size != out_type.size:
            downsample = nn.Sequential(
                enn.R2Conv(in_type, out_type, kernel_size=1, stride=stride, bias=False),
                enn.InnerBatchNorm(out_type),
            )
        
        layers.append(EquivariantBasicBlock(in_type, out_channels, stride, downsample))
        
        # 后续块
        for _ in range(1, blocks):
            layers.append(EquivariantBasicBlock(layers[-1].out_type, out_channels))
        
        return layers

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, enn.R2Conv):
                # Kaiming 初始化
                nn.init.kaiming_normal_(m.weights.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, enn.InnerBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将输入包装为 GeometricTensor
        x = enn.GeometricTensor(x, self.in_type)
        
        # Stage 0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        
        # Stage 1-4
        for block in self.layer1:
            x = block(x)
        for block in self.layer2:
            x = block(x)
        for block in self.layer3:
            x = block(x)
        for block in self.layer4:
            x = block(x)
        
        # Group Pooling: 等变 -> 不变
        x = self.gpool(x)
        x = x.tensor  # 转回普通 Tensor
        
        # 全局平均池化 + 分类
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# ==================== 模型构建函数 ====================

def build_equivariant_resnet18(
    num_classes: int,
    group_type: str = "C8",
    num_rotations: int = 8,
    base_channels: int = 16,
) -> nn.Module:
    """
    构建等变 ResNet-18 模型
    
    参数:
    - num_classes: 分类类别数
    - group_type: 对称群类型
        - "C4": 4重旋转对称 (90°)
        - "C8": 8重旋转对称 (45°)
        - "C16": 16重旋转对称 (22.5°)
        - "D4": 4重旋转 + 翻转对称
        - "D8": 8重旋转 + 翻转对称
        - "SO2": 连续旋转对称 (离散化)
    - num_rotations: SO2 群的离散化数量
    - base_channels: 基础通道数
    
    返回:
    - model: 等变 ResNet-18 模型
    """
    group_type = group_type.upper()
    
    if group_type == "C4":
        gspace = gspaces.rot2dOnR2(N=4)
    elif group_type == "C8":
        gspace = gspaces.rot2dOnR2(N=8)
    elif group_type == "C16":
        gspace = gspaces.rot2dOnR2(N=16)
    elif group_type == "D4":
        gspace = gspaces.flipRot2dOnR2(N=4)
    elif group_type == "D8":
        gspace = gspaces.flipRot2dOnR2(N=8)
    elif group_type == "SO2":
        # 连续旋转群，离散化到 num_rotations 个方向
        gspace = gspaces.rot2dOnR2(N=-1, maximum_frequency=num_rotations)
    else:
        raise ValueError(f"不支持的群类型: {group_type}. 可选: C4, C8, C16, D4, D8, SO2")
    
    print(f"构建等变网络: group={group_type}, gspace={gspace}")
    
    model = EquivariantResNet18(
        gspace=gspace,
        num_classes=num_classes,
        base_channels=base_channels,
    )
    
    return model


# ==================== 评估函数 ====================

def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """计算 Top-1 准确率"""
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """计算单类别的 AP (Average Precision)"""
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
    """计算多类别的 mAP (one-vs-rest)"""
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
) -> Tuple[float, float]:
    """训练一个 epoch"""
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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, List[float]]:
    """在验证/测试集上评估"""
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
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """统计可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==================== 旋转鲁棒性测试 ====================

@torch.no_grad()
def test_rotation_robustness(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    angles: List[int] = None,
) -> dict:
    """
    测试模型对不同旋转角度的鲁棒性
    
    参数:
    - model: 待测试模型
    - loader: 测试数据加载器
    - device: 计算设备
    - angles: 测试的旋转角度列表 (度)
    
    返回:
    - results: {angle: accuracy} 字典
    """
    import torchvision.transforms.functional as TF
    
    if angles is None:
        angles = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180]
    
    model.eval()
    results = {}
    
    for angle in angles:
        correct = 0
        total = 0
        
        for images, targets in loader:
            # 对图像施加旋转
            if angle != 0:
                rotated = torch.stack([
                    TF.rotate(img, angle) for img in images
                ])
            else:
                rotated = images
            
            rotated = rotated.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            logits = model(rotated)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        
        acc = correct / total
        results[angle] = acc
        print(f"  旋转 {angle:3d}°: 准确率 = {acc:.4f} ({acc*100:.2f}%)")
    
    return results


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="ESCNN 等变 ResNet-18 训练脚本")
    
    # 数据参数
    parser.add_argument("--data-root", type=str, default=None, help="数据集根目录 (默认根据 dataset-name 自动选择)")
    parser.add_argument("--dataset-name", type=str, default="trashnet", choices=["trashnet", "realwaste"], help="数据集名称")
    parser.add_argument("--output-dir", type=str, default="outputs_escnn", help="输出目录")
    
    # 模型参数
    parser.add_argument("--group", type=str, default="C8", choices=["C4", "C8", "C16", "D4", "D8", "SO2"], help="对称群类型")
    parser.add_argument("--num-rotations", type=int, default=8, help="SO2 群的离散化数量")
    parser.add_argument("--base-channels", type=int, default=16, help="基础通道数 (控制模型大小)")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--momentum", type=float, default=0.9, help="动量 (SGD)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--img-size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 线程数")
    
    # 测试参数
    parser.add_argument("--test-rotation", action="store_true", help="训练后测试旋转鲁棒性")
    
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 选择数据集
    if args.data_root is None:
        args.data_root = DATASET_ROOTS.get(args.dataset_name.lower(), "dataset-resized")
    
    # 构造数据集
    train_set, val_set, test_set, classes, split_path = create_trashnet_datasets(
        data_root=args.data_root,
        split_ratio=(0.7, 0.15, 0.15),
        img_size=args.img_size,
        seed=args.seed,
    )
    
    print("=" * 60)
    print("ESCNN 等变网络训练")
    print("=" * 60)
    print(f"数据集: {args.dataset_name} -> {args.data_root}")
    print(f"类别: {classes}")
    print(f"划分文件: {split_path}")
    print(f"训练/验证/测试: {len(train_set)}/{len(val_set)}/{len(test_set)}")
    print(f"对称群: {args.group}")
    print("=" * 60)
    
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

    # 构建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = build_equivariant_resnet18(
        num_classes=len(classes),
        group_type=args.group,
        num_rotations=args.num_rotations,
        base_channels=args.base_channels,
    ).to(device)
    
    # 统计参数量
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print("=" * 60)

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, args.epochs // 3),
        gamma=0.1,
    )

    # 日志文件
    log_file = os.path.join("logs", f"escnn_{args.group}_{args.dataset_name}_log.csv")
    with open(log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", "val_mAP", "lr"])

    # 训练循环
    best_val_map = 0.0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_map, val_aps = evaluate(model, val_loader, device)
        scheduler.step()

        # 记录日志
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{train_acc:.4f}",
                f"{val_acc:.4f}",
                f"{val_map:.4f}",
                f"{scheduler.get_last_lr()[0]:.6f}",
            ])

        print(f"Epoch {epoch:03d} | loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_acc={val_acc:.4f} val_mAP={val_map:.4f}")

        # 保存最佳模型 (按 val_mAP)
        if val_map > best_val_map:
            best_val_map = val_map
            best_epoch = epoch
            ckpt_path = os.path.join("checkpoints", f"best_escnn_{args.group}_{args.dataset_name}.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_mAP": val_map,
                "classes": classes,
                "group_type": args.group,
                "base_channels": args.base_channels,
                "num_rotations": args.num_rotations,
            }, ckpt_path)
            print(f"  -> 保存最佳模型 (val_mAP={val_map:.4f})")

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"训练完成! 总耗时: {total_time/60:.1f} 分钟")
    print(f"最佳 epoch: {best_epoch}, 最佳 val_mAP: {best_val_map:.4f}")

    # 加载最佳模型进行测试
    ckpt_path = os.path.join("checkpoints", f"best_escnn_{args.group}_{args.dataset_name}.pth")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"已加载最佳模型: {ckpt_path}")

    # 测试集评估
    test_acc, test_map, test_aps = evaluate(model, test_loader, device)
    print("=" * 60)
    print(f"[Test] top1_acc={test_acc:.4f} mAP={test_map:.4f}")
    print("各类别 AP:")
    for i, (cls_name, ap) in enumerate(zip(classes, test_aps)):
        print(f"  {cls_name}: {ap:.4f}")

    # 将测试结果追加到日志
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test", "", "", f"{test_acc:.4f}", f"{test_map:.4f}", ""])

    # 旋转鲁棒性测试
    if args.test_rotation:
        print("=" * 60)
        print("旋转鲁棒性测试:")
        rotation_results = test_rotation_robustness(model, test_loader, device)
        
        # 保存旋转测试结果
        rotation_log = os.path.join("logs", f"rotation_test_{args.group}_{args.dataset_name}.csv")
        with open(rotation_log, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["angle", "accuracy"])
            for angle, acc in rotation_results.items():
                writer.writerow([angle, f"{acc:.4f}"])
        print(f"旋转测试结果已保存: {rotation_log}")

    print("=" * 60)
    print("完成!")


if __name__ == "__main__":
    main()
