"""
ESCNN 训练脚本 v3 (针对 SO(2) 优化版)

核心改进:
1. 采用 FourierELU 作为非线性激活函数，解决 SO(2) 下的训练稳定性问题
2. 使用 IIDBatchNorm2d 配合频率 0 特征，确保颜色/亮度信息不丢失
3. 调整表示构建逻辑，适应 SO(2) 连续群特性
"""

import os
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

from escnn import gspaces
from escnn import nn as enn

from dataset import create_trashnet_datasets

# ==================== 核心组件构建 ====================

def _build_so2_field_type(gspace, channels: int, max_freq: int = 2) -> enn.FieldType:
    """
    构建 SO(2) 的 FieldType
    包含频率 0 (trivial) 到 max_freq 的所有频率，确保捕捉颜色和纹理方向
    """
    # 频率 0 (标量/颜色)
    rep = gspace.irrep(0)
    # 叠加高频特征 (纹理/边缘方向)
    for k in range(1, max_freq + 1):
        rep = rep + gspace.irrep(k)
    return enn.FieldType(gspace, channels * [rep])

def _build_activation(out_type: enn.FieldType, max_freq: int = 2):
    """
    构建激活函数
    对于 SO(2)，使用 FourierELU 是最稳健的选择
    """
    gspace = out_type.gspace
    N = max(8, 2 * max_freq + 4)
    
    # 检查是否为连续 SO(2)
    grp = getattr(gspace, "fibergroup", None)
    order = getattr(grp, "order", lambda: -1)()
    
    if order == -1: # SO(2)
        # 构造网格
        grp = gspace.fibergroup
        grid_obj = grp.grid(type='regular', N=N)
        
        # 提取唯一的 irreps
        unique_irreps = sorted(list(set(out_type.irreps)))
        
        # 使用另一种构造方式，确保 in_type 匹配
        # 我们手动指定 gspace, channels, irreps 来匹配 out_type
        # 同时通过 grid 参数传递网格
        
        # 计算通道数 (假设所有 irreps 都在每个通道中出现一次)
        # out_type.size 是总维度，len(out_type.irreps) 是总 irrep 数量
        # 对于 SO(2)，通常 FieldType 是 channels * [irrep_sum]
        # 所以 channels = out_type.size // irrep_sum.dim
        # 或者更简单：我们知道构建时是 channels * [rep]
        
        # 正确的做法：直接使用 out_type 作为 in_type
        # 但 FourierELU(in_type, grid, ...) 这种调用方式需要验证是否支持
        # 如果不支持，我们需要回退到 FourierELU(gspace, channels, irreps, ...) 并手动解决 type 不匹配问题
        
        # 尝试修正: 显式传递所有参数，并在最后手动覆盖 in_type
        # FourierELU 的构造函数不接受预构建的 grid 对象作为关键字参数
        # 它接受 *grid_args 和 **grid_kwargs 来内部调用 gspace.fibergroup.grid()
        # 所以我们应该传递构建 grid 所需的参数 (type='regular', N=N)
        # 修正: channels 参数必须正确计算
        # out_type.size 是特征总维度 (例如 16个通道 * 5维/通道 = 80)
        # out_type.irreps 是所有通道的 irreps 列表 (长度为 16*5 吗？不，是通道数)
        # out_type 是 channels * [irrep_sum]，其中 irrep_sum 是频率0到max_freq的和
        # irrep_sum 的维度是 1 + 2*max_freq (例如 1 + 2*2 = 5)
        # 所以 len(out_type.irreps) 是 irrep 的总数量 (不是通道数，也不是总维度)
        # wait, FieldType.irreps 是一个 irrep 对象的列表，其长度等于该 FieldType 可以分解为多少个不可约表示
        # 对于 SO(2)，每个通道包含频率 0, 1, 2...
        # 所以每个通道包含 max_freq + 1 个 irreps
        # 总 irreps 数量 = channels * (max_freq + 1)
        
        # FourierELU 的 channels 参数指的是 "spatial channels"，即特征图的通道数
        # 也就是我们在 build_so2_field_type 中传入的 base_channels
        
        # 让我们反推 channels:
        # out_type.size 是总维度 (例如 16 * 5 = 80)
        # len(out_type.irreps) 是 irrep 数量 (例如 16 * 3 = 48)
        # 这两个都不能直接作为 channels
        
        # 正确的方式：计算 irrep_sum 的维度
        # unique_irreps 是 irrep ID 的元组 (例如 (0,) 或 (1,))，而不是 irrep 对象本身
        # 在 escnn 中，irrep 通常由 id 标识
        # 我们需要通过 gspace.irrep(*id) 获取对象来访问 size 属性
        
        dim_per_channel = 0
        for irrep_id in unique_irreps:
            # unique_irreps 可能是 irrep 对象也可能是 id
            # 调试发现 unique_irreps 中的元素是 tuple，说明它们是 id
            # 我们使用 gspace.irrep(*irrep_id) 来获取对象
            irrep_obj = gspace.irrep(*irrep_id)
            dim_per_channel += irrep_obj.size
            
        channels = out_type.size // dim_per_channel
        
        act = enn.FourierELU(gspace, channels=channels, irreps=unique_irreps, type='regular', N=N, inplace=True)
        act.in_type = out_type # 强制覆盖
        return act
    else:
        return enn.ReLU(out_type, inplace=True)

def _build_norm_layer(out_type: enn.FieldType):
    """
    构建归一化层
    使用 IIDBatchNorm2d，因为它对所有频率独立归一化，兼容性最好
    InnerBatchNorm 需要 pointwise 支持，不适用于频域(Irreps)表示
    """
    return enn.IIDBatchNorm2d(out_type)

def _build_pool_layer(out_type: enn.FieldType):
    """构建池化层"""
    # 对于 FourierELU 网络，特征始终在频域
    # 使用 NormMaxPool 是合理的选择
    if hasattr(enn, "NormMaxPool"):
        return enn.NormMaxPool(out_type, kernel_size=2, stride=2)
    elif hasattr(enn, "PointwiseMaxPool"):
        return enn.PointwiseMaxPool(out_type, kernel_size=2, stride=2)
    else:
        return nn.AvgPool2d(2) # 降级方案

# ==================== 网络模块 ====================

class EquivariantBlock(nn.Module):
    def __init__(self, in_type, out_type, stride=1, max_freq=2):
        super().__init__()
        
        self.conv1 = enn.R2Conv(in_type, out_type, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = _build_norm_layer(out_type)
        # 注意：act1 的输出类型可能与 out_type 不同 (FieldType 对象实例不同)
        self.act1 = _build_activation(out_type, max_freq)
        
        # 使用 act1 的输出类型作为下一层的输入类型
        act1_out_type = self.act1.out_type
        
        self.conv2 = enn.R2Conv(act1_out_type, out_type, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = _build_norm_layer(out_type)
        self.act2 = _build_activation(out_type, max_freq)
        
        self.downsample = None
        if stride != 1 or in_type != out_type:
            self.downsample = nn.Sequential(
                enn.R2Conv(in_type, out_type, kernel_size=1, stride=stride, bias=False),
                _build_norm_layer(out_type)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out = out + identity
        out = self.act2(out)
        return out

class SO2ResNet18(nn.Module):
    def __init__(self, num_classes, base_channels=16, max_freq=2):
        super().__init__()
        
        # 定义空间: SO(2) 连续群
        # maximum_frequency 控制内部卷积核的复杂度，这里设为 max_freq
        self.gspace = gspaces.rot2dOnR2(N=-1, maximum_frequency=max_freq)
        self.max_freq = max_freq
        
        # 输入: RGB (3个标量场)
        self.in_type = enn.FieldType(self.gspace, 3 * [self.gspace.trivial_repr])
        
        # 初始层
        # 将 RGB 映射到混合频率空间 (Freq 0 + Freq 1 + ... + Freq max)
        out_type_0 = _build_so2_field_type(self.gspace, base_channels, max_freq)
        
        self.conv1 = enn.R2Conv(self.in_type, out_type_0, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = _build_norm_layer(out_type_0)
        self.act1 = _build_activation(out_type_0, max_freq)
        
        # 获取 act1 的真实输出类型
        act1_out_type = self.act1.out_type
        
        # 池化: 如果 NormMaxPool 不可用，可以用 stride=2 的卷积替代，这里尝试用 MaxPool
        self.maxpool = _build_pool_layer(act1_out_type)
        
        # ResNet Stages
        # 注意: layer1 的输入类型必须是 act1_out_type
        # layer1 的内部 block 会处理 out_type -> act_out_type 的转换
        # 但我们需要确保 _make_layer 正确处理输入
        
        self.layer1 = self._make_layer(act1_out_type, base_channels, 2, stride=1)
        # layer1[-1] 是 block，block 的最后是 act2
        self.layer2 = self._make_layer(self.layer1[-1].act2.out_type, base_channels*2, 2, stride=2)
        self.layer3 = self._make_layer(self.layer2[-1].act2.out_type, base_channels*4, 2, stride=2)
        self.layer4 = self._make_layer(self.layer3[-1].act2.out_type, base_channels*8, 2, stride=2)
        
        # 输出层: Group Pooling -> Flatten -> Linear
        last_type = self.layer4[-1].act2.out_type
        # SO(2) 连续群不支持 GroupPooling (需要 pointwise 非线性支持)
        # 使用 NormPool 代替，将特征向量的范数作为不变特征
        self.gpool = enn.NormPool(last_type) 
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.gpool.out_type.size, num_classes)
        
        # 初始化
        for m in self.modules():
            if isinstance(m, enn.R2Conv):
                # 广义 He 初始化
                enn.init.generalized_he_init(m.weights.data, m.basisexpansion, cache=True)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_type, channels, blocks, stride):
        layers = []
        # 目标输出类型 (卷积层的目标输出)
        out_type = _build_so2_field_type(self.gspace, channels, self.max_freq)
        
        # 第一个 block: in_type -> out_type
        layers.append(EquivariantBlock(in_type, out_type, stride, self.max_freq))
        
        # 后续 block: 前一个 block 的输出类型 -> out_type
        # 注意 EquivariantBlock 的输出是 act2(out_type)，即 FourierELU 的输出类型
        for _ in range(1, blocks):
            prev_out_type = layers[-1].act2.out_type
            layers.append(EquivariantBlock(prev_out_type, out_type, 1, self.max_freq))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        # 确保输入类型匹配
        if not hasattr(x, 'type'):
             x = enn.GeometricTensor(x, self.in_type)
        
        # 显式包装，防止中间过程丢失 GeometricTensor 属性
        if not isinstance(x, enn.GeometricTensor):
             x = enn.GeometricTensor(x, self.in_type)

        x = self.conv1(x)
        x = self.bn1(x)
        # 调试: 检查类型
        # print(f"Before act1: {x.type == self.act1.in_type}")
        x = self.act1(x)
        if self.maxpool:
            x = self.maxpool(x)
            
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.gpool(x)
        x = x.tensor
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ==================== 辅助函数 ====================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for i, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if i % 20 == 0:
            print(f"Batch {i}/{len(loader)} | Loss: {loss.item():.4f}")
            
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    return correct / total

# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16) # FourierELU 比较省显存，可以尝试 16
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--max-freq", type=int, default=2)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 数据集
    train_set, val_set, test_set, classes, _ = create_trashnet_datasets(
        data_root="dataset-resized", split_ratio=(0.7, 0.15, 0.15)
    )
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 模型
    print(f"Building SO(2) ResNet18 (FourierELU) | Base: {args.base_channels} | MaxFreq: {args.max_freq}")
    model = SO2ResNet18(len(classes), args.base_channels, args.max_freq).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1:02d} | Time: {time.time()-start:.1f}s | "
              f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_so2_v3.pth")
            print(f"  -> Best Model Saved (Acc: {best_acc:.4f})")

if __name__ == "__main__":
    main()
