"""
ESCNN 等变神经网络训练脚本 (改进版 v2)
基于 E(2)-Equivariant Steerable CNNs 实现废弃物分类

改进内容:
1. 残差加法增加类型安全检查
2. SO(2) 群参数说明更准确
3. GroupPooling 说明更清晰
4. 旋转鲁棒性测试在原始图像上进行
5. 增加表示类型选择说明
6. 权重初始化注释完善
7. 增加 cosine 学习率调度、label smoothing 等选项
8. 增加计算资源提示

依赖安装: pip install escnn

参考文献:
- Weiler & Cesa (2019). General E(2)-Equivariant Steerable CNNs. NeurIPS.
"""

import os
import csv
import time
import math
import argparse
from typing import Tuple, List, Optional, Literal
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ESCNN 核心模块
from escnn import gspaces
from escnn import nn as enn

from dataset import create_trashnet_datasets
def _field_type_regular_or_trivial(gspace, channels: int) -> enn.FieldType:
    grp = getattr(gspace, "fibergroup", None)
    order_attr = getattr(grp, "order", None)
    order = order_attr() if callable(order_attr) else order_attr
    if order == -1:
        maxf = None
        for attr in ("maximum_frequency", "_maximum_frequency", "_max_frequency", "max_frequency"):
            v = getattr(gspace, attr, None)
            if isinstance(v, int) and v > 0:
                maxf = v
                break
        if maxf is None:
            maxf = 4
        rep = gspace.irrep(0)
        for k in range(1, maxf + 1):
            rep = rep + gspace.irrep(k)
        return enn.FieldType(gspace, channels * [rep])
    try:
        rep = gspace.regular_repr
        return enn.FieldType(gspace, channels * [rep])
    except Exception:
        return enn.FieldType(gspace, channels * [gspace.trivial_repr])

def _is_continuous_so2(gspace) -> bool:
    grp = getattr(gspace, "fibergroup", None)
    order_attr = getattr(grp, "order", None)
    order = order_attr() if callable(order_attr) else order_attr
    return order == -1

def _build_nonlinearity(out_type: enn.FieldType):
    gspace = out_type.gspace
    if _is_continuous_so2(gspace):
        return enn.NormNonLinearity(out_type)
    else:
        return enn.ReLU(out_type, inplace=True)

# ==================== 数据集配置 ====================
DATASET_NAME = "trashnet"  # 可选: "trashnet", "realwaste"
DATASET_ROOTS = {
    "trashnet": "dataset-resized",
    "realwaste": "RealWaste",
}

GROUP_NAME = "SO2"
GROUP_CONFIGS = {
    "C4": {"group": "C4"},
    "C8": {"group": "C8"},
    "D4": {"group": "D4"},
    "SO2": {"group": "SO2", "max_frequency": 4},
}

# ==================== 等变网络模块 ====================

class EquivariantBasicBlock(nn.Module):
    """
    等变版本的 ResNet BasicBlock
    
    包含两个 3x3 等变卷积和残差连接。
    
    注意事项:
    - 残差连接要求 identity 和 out 的 FieldType 完全一致
    - 当 stride != 1 或通道数变化时，必须通过 downsample 调整 identity 的类型
    - 直接张量相加 (out.tensor + identity.tensor) 仅在类型匹配时有效
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
        
        out_type = _field_type_regular_or_trivial(gspace, out_channels)
        
        # 第一个卷积: 3x3, 可能有 stride
        self.conv1 = enn.R2Conv(
            in_type, out_type,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = enn.InnerBatchNorm(out_type) if not _is_continuous_so2(gspace) else None
        self.relu1 = _build_nonlinearity(out_type)
        
        # 第二个卷积: 3x3, stride=1
        self.conv2 = enn.R2Conv(
            out_type, out_type,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = enn.InnerBatchNorm(out_type) if not _is_continuous_so2(gspace) else None
        self.relu2 = _build_nonlinearity(out_type)
        
        self.downsample = downsample
        self.out_type = out_type
        self.in_type = in_type

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        identity = x
        
        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 残差连接 - 类型安全检查
        # 注意: 只有当 identity.type == out.type 时才能直接张量相加
        # 这里通过 downsample 已经保证了类型匹配
        assert identity.type == out.type, \
            f"残差连接类型不匹配: identity={identity.type}, out={out.type}"
        
        out = enn.GeometricTensor(out.tensor + identity.tensor, out.type)
        out = self.relu2(out)
        
        return out


class EquivariantResNet18(nn.Module):
    """
    等变版本的 ResNet-18
    
    架构:
    - 输入层: 7x7 等变卷积 (trivial_repr 输入)
    - 4个 stage，每个包含 2 个 BasicBlock (regular_repr 特征)
    - Group Pooling: 将等变特征投影到不变子空间
    - 全局平均池化 + 全连接层
    
    关于表示类型 (Representation Types):
    - trivial_repr: 标量表示，不随群作用变化 (用于输入RGB)
    - regular_repr: 正则表示，包含群的所有信息 (用于中间特征)
    - 混合使用 trivial + regular 可以更高效，但实现更复杂
    - 当前实现全部使用 regular_repr，简单有效
    
    关于 GroupPooling:
    - 作用: 将等变特征映射到群不变的特征空间
    - 数学上: 对每个空间位置，在群维度上取不变子空间的投影
    - 输出维度: 每个 regular_repr 通道变为 1 个标量 (群维度被消除)
    - 这保证了最终输出对旋转完全不变
    
    参数:
    - gspace: 对称群空间 (C4, C8, D4, SO2 等)
    - num_classes: 分类类别数
    - base_channels: 基础通道数，控制模型大小
    
    计算资源提示:
    - base_channels=16: 约 0.5-1M 参数，适合快速实验
    - base_channels=32: 约 2-4M 参数，接近标准 ResNet-18
    - 群越大 (如 C16, D8, SO2)，计算量越大
    - SO2 的 maximum_frequency 越高，计算量越大
    """
    
    def __init__(
        self,
        gspace: gspaces.GSpace,
        num_classes: int,
        base_channels: int = 16,
    ):
        super().__init__()
        self.gspace = gspace
        
        # 输入类型: 3通道 RGB 图像
        # 使用 trivial_repr 因为 RGB 值本身不随旋转变化
        self.in_type = enn.FieldType(gspace, 3 * [gspace.trivial_repr])
        
        # Stage 0: 初始卷积层
        out_type_0 = _field_type_regular_or_trivial(gspace, base_channels)
        self.conv1 = enn.R2Conv(
            self.in_type, out_type_0,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = enn.InnerBatchNorm(out_type_0) if not _is_continuous_so2(gspace) else None
        self.relu1 = _build_nonlinearity(out_type_0)
        self.maxpool = None if _is_continuous_so2(gspace) else enn.PointwiseMaxPool(out_type_0, kernel_size=3, stride=2, padding=1)
        
        # Stage 1-4: 残差块
        self.layer1 = self._make_layer(out_type_0, base_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(self.layer1[-1].out_type, base_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(self.layer2[-1].out_type, base_channels * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(self.layer3[-1].out_type, base_channels * 8, blocks=2, stride=2)
        
        # Group Pooling: 等变特征 -> 不变特征
        # 对于 regular_repr，GroupPooling 输出每个通道的群不变标量
        # 具体实现: 对群维度取平均 (或最大值，取决于 escnn 版本)
        self.gpool = enn.GroupPooling(self.layer4[-1].out_type)
        
        # GroupPooling 输出维度 = 通道数 (群维度被消除)
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
        
        out_type = _field_type_regular_or_trivial(self.gspace, out_channels)
        
        # 当 stride != 1 或通道数变化时，需要 downsample 来匹配类型
        downsample = None
        if stride != 1 or in_type.size != out_type.size:
            if _is_continuous_so2(self.gspace):
                downsample = enn.R2Conv(in_type, out_type, kernel_size=1, stride=stride, bias=False)
            else:
                downsample = nn.Sequential(
                    enn.R2Conv(in_type, out_type, kernel_size=1, stride=stride, bias=False),
                    enn.InnerBatchNorm(out_type),
                )
        
        layers.append(EquivariantBasicBlock(in_type, out_channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(EquivariantBasicBlock(layers[-1].out_type, out_channels))
        
        return layers

    def _initialize_weights(self):
        """
        权重初始化
        
        使用 Kaiming Normal 初始化，适合 ReLU 激活函数。
        
        注意:
        - 等变卷积的权重结构与普通卷积不同，但 Kaiming 初始化仍然适用
        - 替代方案: Xavier 初始化 (适合 Sigmoid/Tanh)
        - 对于等变网络，初始化策略的影响可能与普通 CNN 略有不同
        """
        for m in self.modules():
            if isinstance(m, enn.R2Conv):
                w = m.weights.data
                if w.ndim >= 2:
                    nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.normal_(w, 0.0, 0.02)
            elif isinstance(m, enn.InnerBatchNorm):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = enn.GeometricTensor(x, self.in_type)
        
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu1(x)
        if self.maxpool is not None:
            x = self.maxpool(x)
        
        for block in self.layer1:
            x = block(x)
        for block in self.layer2:
            x = block(x)
        for block in self.layer3:
            x = block(x)
        for block in self.layer4:
            x = block(x)
        
        x = self.gpool(x)
        x = x.tensor
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# ==================== 模型构建函数 ====================

def build_equivariant_resnet18(
    num_classes: int,
    group_type: str = "C8",
    maximum_frequency: int = 8,
    base_channels: int = 16,
) -> nn.Module:
    """
    构建等变 ResNet-18 模型
    
    参数:
    - num_classes: 分类类别数
    - group_type: 对称群类型
        - "C4", "C8", "C16": 离散旋转群，N 表示旋转次数 (360°/N 为最小旋转角)
        - "D4", "D8": 二面体群 = 旋转 + 翻转
        - "SO2": 连续旋转群 (需要傅里叶展开)
    - maximum_frequency: SO(2) 群的最大傅里叶频率
        - 注意: 这不是"离散化数量"！
        - 它控制傅里叶级数展开的最高频率
        - 频率越高，表示能力越强，但参数和计算量也越大
        - 典型值: 2-8，更高的值通常收益递减
        - 与离散群 C_n 的区别: C_n 直接对 n 个旋转离散采样，
          而 SO(2) + maximum_frequency 是在连续空间用有限频率逼近
    - base_channels: 基础通道数 (影响模型大小和计算量)
    
    计算资源参考:
    - C4 + base_channels=16: 最快，约 0.3M 参数
    - C8 + base_channels=16: 推荐起点，约 0.6M 参数
    - D8 + base_channels=16: 含翻转，约 1.2M 参数
    - SO2 + max_freq=4: 连续群，约 0.8M 参数
    - SO2 + max_freq=8: 更精确，约 1.5M 参数
    
    返回:
    - model: 等变 ResNet-18 模型
    """
    group_type = group_type.upper()
    
    if group_type == "C4":
        gspace = gspaces.rot2dOnR2(N=4)
        group_desc = "C4 (90° 旋转对称)"
    elif group_type == "C8":
        gspace = gspaces.rot2dOnR2(N=8)
        group_desc = "C8 (45° 旋转对称)"
    elif group_type == "C16":
        gspace = gspaces.rot2dOnR2(N=16)
        group_desc = "C16 (22.5° 旋转对称)"
    elif group_type == "D4":
        gspace = gspaces.flipRot2dOnR2(N=4)
        group_desc = "D4 (90° 旋转 + 翻转)"
    elif group_type == "D8":
        gspace = gspaces.flipRot2dOnR2(N=8)
        group_desc = "D8 (45° 旋转 + 翻转)"
    elif group_type == "SO2":
        # SO(2): 连续旋转群
        # N=-1 表示连续群，maximum_frequency 控制傅里叶展开的最高频率
        # 这与离散群的 N (旋转次数) 是完全不同的概念!
        gspace = gspaces.rot2dOnR2(N=-1, maximum_frequency=maximum_frequency)
        group_desc = f"SO(2) (连续旋转, max_freq={maximum_frequency})"
    else:
        raise ValueError(f"不支持的群类型: {group_type}. 可选: C4, C8, C16, D4, D8, SO2")
    
    print(f"构建等变网络: {group_desc}")
    print(f"  gspace: {gspace}")
    print(f"  base_channels: {base_channels}")
    
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
    max_grad_norm: Optional[float] = None,
) -> Tuple[float, float]:
    """
    训练一个 epoch
    
    参数:
    - max_grad_norm: 梯度裁剪阈值 (None 表示不裁剪)
    """
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
        
        # 梯度裁剪 (可选)
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
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


# ==================== 学习率调度器 ====================

def get_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    warmup_epochs: int = 0,
) -> optim.lr_scheduler._LRScheduler:
    """
    获取学习率调度器
    
    参数:
    - scheduler_type: "step", "cosine", "cosine_warmup"
    - epochs: 总训练轮数
    - warmup_epochs: warmup 轮数 (仅 cosine_warmup 使用)
    """
    if scheduler_type == "step":
        # 阶梯下降: 每 1/3 训练周期衰减 10 倍
        return optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    
    elif scheduler_type == "cosine":
        # 余弦退火: 平滑下降到接近 0
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    elif scheduler_type == "cosine_warmup":
        # 带 warmup 的余弦退火
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    else:
        raise ValueError(f"不支持的调度器: {scheduler_type}")


# ==================== 旋转鲁棒性测试 (改进版) ====================

class RotatedDataset(Dataset):
    """
    在原始 PIL 图像上应用旋转的数据集包装器
    
    改进说明:
    - 旋转在原始 PIL 图像上进行，然后再做归一化
    - 避免对归一化后的张量做几何变换引入的插值误差
    - 这样更准确地评估模型的真实旋转鲁棒性
    """
    def __init__(self, base_dataset, angle: float, img_size: int = 224):
        self.base_dataset = base_dataset
        self.angle = angle
        self.img_size = img_size
        
        # 标准评估变换 (不含旋转)
        self.eval_transform = transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # 获取原始样本 (如果是 Subset，需要处理)
        if hasattr(self.base_dataset, 'dataset'):
            # Subset 的情况
            real_idx = self.base_dataset.indices[idx]
            path, target = self.base_dataset.dataset.samples[real_idx]
        else:
            path, target = self.base_dataset.samples[idx]
        
        # 加载原始 PIL 图像
        img = Image.open(path).convert('RGB')
        
        # 在 PIL 图像上旋转 (使用双线性插值，填充为黑色)
        if self.angle != 0:
            img = TF.rotate(img, self.angle, interpolation=InterpolationMode.BILINEAR, fill=(0, 0, 0))
        
        # 然后应用评估变换
        img = self.eval_transform(img)
        
        return img, target


@torch.no_grad()
def test_rotation_robustness(
    model: nn.Module,
    test_set,  # 原始测试集 (用于获取图像路径)
    device: torch.device,
    angles: List[int] = None,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict:
    """
    测试模型对不同旋转角度的鲁棒性 (改进版)
    
    改进说明:
    - 旋转在原始 PIL 图像上进行，而不是归一化后的张量
    - 这避免了对标准化数值空间做几何变换引入的非线性插值误差
    - 结果更准确地反映模型的真实旋转鲁棒性
    
    参数:
    - model: 待测试模型
    - test_set: 测试数据集 (需要能获取原始图像路径)
    - device: 计算设备
    - angles: 测试的旋转角度列表 (度)
    
    返回:
    - results: {angle: accuracy} 字典
    """
    from torchvision import transforms
    
    if angles is None:
        angles = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180]
    
    model.eval()
    results = {}
    
    print("旋转鲁棒性测试 (在原始图像上旋转):")
    
    for angle in angles:
        # 创建旋转数据集
        rotated_dataset = RotatedDataset(test_set, angle, img_size)
        rotated_loader = DataLoader(
            rotated_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        
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
    parser = argparse.ArgumentParser(description="ESCNN 等变 ResNet-18 训练脚本 (改进版)")
    
    # 数据参数
    parser.add_argument("--data-root", type=str, default=None, help="数据集根目录")
    parser.add_argument("--dataset-name", type=str, default=DATASET_NAME, 
                        choices=["trashnet", "realwaste"], help="数据集名称")
    parser.add_argument("--output-dir", type=str, default="outputs_escnn", help="输出目录")
    
    # 模型参数
    parser.add_argument("--group", type=str, default=GROUP_NAME, 
                        choices=["C4", "C8", "C16", "D4", "D8", "SO2"], help="对称群类型")
    parser.add_argument("--max-frequency", type=int, default=GROUP_CONFIGS.get(GROUP_NAME, {}).get("max_frequency", 4), 
                        help="SO(2) 群的最大傅里叶频率 (注意: 不是离散化数量!)")
    parser.add_argument("--base-channels", type=int, default=16, 
                        help="基础通道数 (16: 快速实验, 32: 更大容量)")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--momentum", type=float, default=0.9, help="动量 (SGD)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--img-size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 线程数")
    
    # 训练策略
    parser.add_argument("--scheduler", type=str, default="cosine", 
                        choices=["step", "cosine", "cosine_warmup"], help="学习率调度器")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="warmup 轮数")
    parser.add_argument("--label-smoothing", type=float, default=0.0, 
                        help="标签平滑系数 (0.0-0.1)")
    parser.add_argument("--grad-clip", type=float, default=None, 
                        help="梯度裁剪阈值 (None 表示不裁剪)")
    
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
    
    print("=" * 70)
    print("ESCNN 等变网络训练 (改进版 v2)")
    print("=" * 70)
    print(f"数据集: {args.dataset_name} -> {args.data_root}")
    print(f"类别: {classes}")
    print(f"训练/验证/测试: {len(train_set)}/{len(val_set)}/{len(test_set)}")
    print(f"对称群: {args.group}")
    if args.group == "SO2":
        print(f"  最大傅里叶频率: {args.max_frequency}")
        print(f"  (注意: 这不是离散化数量，而是连续群的频率截断)")
    print(f"基础通道数: {args.base_channels}")
    print(f"学习率调度: {args.scheduler}")
    if args.label_smoothing > 0:
        print(f"标签平滑: {args.label_smoothing}")
    print("=" * 70)
    
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
    print(f"设备: {device}")
    
    model = build_equivariant_resnet18(
        num_classes=len(classes),
        group_type=args.group,
        maximum_frequency=args.max_frequency,
        base_channels=args.base_channels,
    ).to(device)
    
    # 统计参数量
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    print(f"\n模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 计算资源提示
    if args.group == "SO2" and args.max_frequency > 6:
        print("\n⚠️  警告: SO(2) 群的 max_frequency 较高，训练可能较慢")
    if args.base_channels > 24:
        print(f"⚠️  警告: base_channels={args.base_channels} 较大，内存占用可能较高")
    print("=" * 70)

    # 损失函数
    if args.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    
    # 学习率调度器
    scheduler = get_scheduler(
        optimizer, args.scheduler, args.epochs, args.warmup_epochs
    )

    # 日志文件
    suffix = args.group if args.group != "SO2" else f"SO2f{args.max_frequency}"
    log_file = os.path.join("logs", f"escnn_{suffix}_{args.dataset_name}_log.csv")
    with open(log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", "val_mAP", "lr"])

    # 训练循环
    best_val_map = 0.0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.grad_clip
        )
        val_acc, val_map, val_aps = evaluate(model, val_loader, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # 记录日志
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{train_acc:.4f}",
                f"{val_acc:.4f}",
                f"{val_map:.4f}",
                f"{current_lr:.6f}",
            ])

        print(f"Epoch {epoch:03d} | loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_acc={val_acc:.4f} val_mAP={val_map:.4f} | lr={current_lr:.6f}")

        # 保存最佳模型 (按 val_mAP)
        if val_map > best_val_map:
            best_val_map = val_map
            best_epoch = epoch
            ckpt_path = os.path.join("checkpoints", f"best_escnn_{suffix}_{args.dataset_name}.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_mAP": val_map,
                "classes": classes,
                "group_type": args.group,
                "base_channels": args.base_channels,
                "max_frequency": args.max_frequency,
            }, ckpt_path)
            print(f"  -> 保存最佳模型 (val_mAP={val_map:.4f})")

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"训练完成! 总耗时: {total_time/60:.1f} 分钟")
    print(f"最佳 epoch: {best_epoch}, 最佳 val_mAP: {best_val_map:.4f}")

    # 加载最佳模型进行测试
    ckpt_path = os.path.join("checkpoints", f"best_escnn_{suffix}_{args.dataset_name}.pth")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"已加载最佳模型: {ckpt_path}")

    # 测试集评估
    test_acc, test_map, test_aps = evaluate(model, test_loader, device)
    print("=" * 70)
    print(f"[Test] top1_acc={test_acc:.4f} mAP={test_map:.4f}")
    print("各类别 AP:")
    for i, (cls_name, ap) in enumerate(zip(classes, test_aps)):
        print(f"  {cls_name}: {ap:.4f}")

    # 将测试结果追加到日志
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test", "", "", f"{test_acc:.4f}", f"{test_map:.4f}", ""])

    # 旋转鲁棒性测试 (改进版: 在原始图像上旋转)
    if args.test_rotation:
        print("=" * 70)
        rotation_results = test_rotation_robustness(
            model, test_set, device,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        
        # 保存旋转测试结果
        rotation_log = os.path.join("logs", f"rotation_test_{suffix}_{args.dataset_name}.csv")
        with open(rotation_log, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["angle", "accuracy"])
            for angle, acc in rotation_results.items():
                writer.writerow([angle, f"{acc:.4f}"])
        print(f"旋转测试结果已保存: {rotation_log}")

    print("=" * 70)
    print("完成!")


if __name__ == "__main__":
    main()
