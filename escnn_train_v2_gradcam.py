"""
ESCNN 等变神经网络训练脚本 (Grad-CAM 修复版)
基于 E(2)-Equivariant Steerable CNNs 实现废弃物分类

此版本专门修改用于解决 Grad-CAM 计算梯度时的 inplace 错误。
修改内容:
1. ReLU(inplace=False)
2. 尝试规避 BatchNorm 的 inplace 操作
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
        
        # SO(2) 连续群处理:
        # 为了避免 NormBatchNorm 对 trivial (频率0) 表示报错，
        # 我们这里恢复频率0，但在 NormBatchNorm 中使用特殊处理 (IIDBatchNorm2d)
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
        # NormNonLinearity 可能会有 inplace，尝试关闭
        return enn.ReLU(out_type, inplace=False) # Fallback
    else:
        # 关键修改: inplace=False
        return enn.ReLU(out_type, inplace=False)

def _build_norm_layer(out_type: enn.FieldType):
    """构建归一化层"""
    gspace = out_type.gspace
    if _is_continuous_so2(gspace):
        if hasattr(enn, "IIDBatchNorm2d"):
            return enn.IIDBatchNorm2d(out_type)
        elif hasattr(enn, "InnerBatchNorm"):
            return enn.InnerBatchNorm(out_type)
        else:
            print(f"Warning: No suitable BatchNorm found for SO(2). Skipping BN.")
            return None
    else:
        # 离散群使用 InnerBatchNorm
        return enn.InnerBatchNorm(out_type)

def _build_maxpool_layer(out_type: enn.FieldType, kernel_size=3, stride=2, padding=1):
    """构建最大池化层"""
    gspace = out_type.gspace
    if _is_continuous_so2(gspace):
        if hasattr(enn, "NormMaxPool"):
            return enn.NormMaxPool(out_type, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            return None
    else:
        return enn.PointwiseMaxPool(out_type, kernel_size=kernel_size, stride=stride, padding=padding)


# ==================== 训练参数配置 (手动修改) ====================
TRAIN_CONFIG = {
    "epochs": 50,
    "batch_size": 8,
    "lr": 0.005,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "base_channels": 16,
    "max_frequency": 2,
    "amp": False,
    "num_workers": 0,
    "benchmark": False,
    "seed": 42,
    "img_size": 224,
    "output_dir": "outputs_escnn",
    "test_rotation": False,
}

DATASET_NAME = "trashnet"
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
        
        self.conv1 = enn.R2Conv(
            in_type, out_type,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = _build_norm_layer(out_type)
        self.relu1 = _build_nonlinearity(out_type)
        
        self.conv2 = enn.R2Conv(
            out_type, out_type,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = _build_norm_layer(out_type)
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
        
        assert identity.type == out.type, \
            f"残差连接类型不匹配: identity={identity.type}, out={out.type}"
        
        # 避免 inplace 加法
        # out.tensor + identity.tensor 创建新 tensor，是安全的
        out = enn.GeometricTensor(out.tensor + identity.tensor, out.type)
        out = self.relu2(out)
        
        return out


class EquivariantResNet18(nn.Module):
    def __init__(
        self,
        gspace: gspaces.GSpace,
        num_classes: int,
        base_channels: int = 16,
    ):
        super().__init__()
        self.gspace = gspace
        
        self.in_type = enn.FieldType(gspace, 3 * [gspace.trivial_repr])
        
        out_type_0 = _field_type_regular_or_trivial(gspace, base_channels)
        self.conv1 = enn.R2Conv(
            self.in_type, out_type_0,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = _build_norm_layer(out_type_0)
        self.relu1 = _build_nonlinearity(out_type_0)
        self.maxpool = _build_maxpool_layer(out_type_0, kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(out_type_0, base_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(self.layer1[-1].out_type, base_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(self.layer2[-1].out_type, base_channels * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(self.layer3[-1].out_type, base_channels * 8, blocks=2, stride=2)
        
        if _is_continuous_so2(self.gspace):
            self.gpool = enn.NormPool(self.layer4[-1].out_type)
        else:
            self.gpool = enn.GroupPooling(self.layer4[-1].out_type)
        
        gpool_out_dim = self.gpool.out_type.size
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(gpool_out_dim, num_classes)
        
        self._initialize_weights()

    def _make_layer(
        self,
        in_type: enn.FieldType,
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.ModuleList:
        layers = nn.ModuleList()
        
        out_type = _field_type_regular_or_trivial(self.gspace, out_channels)
        
        downsample = None
        if stride != 1 or in_type.size != out_type.size:
            ds_layers = [enn.R2Conv(in_type, out_type, kernel_size=1, stride=stride, bias=False)]
            norm_layer = _build_norm_layer(out_type)
            if norm_layer is not None:
                ds_layers.append(norm_layer)
            
            if len(ds_layers) == 1:
                downsample = ds_layers[0]
            else:
                downsample = nn.Sequential(*ds_layers)
        
        layers.append(EquivariantBasicBlock(in_type, out_channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(EquivariantBasicBlock(layers[-1].out_type, out_channels))
        
        return layers

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, enn.R2Conv):
                w = m.weights.data
                if w.ndim >= 2:
                    nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.normal_(w, 0.0, 0.02)
            elif "BatchNorm" in m.__class__.__name__:
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


def build_equivariant_resnet18(
    num_classes: int,
    group_type: str = "C8",
    maximum_frequency: int = 8,
    base_channels: int = 16,
) -> nn.Module:
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
        gspace = gspaces.rot2dOnR2(N=-1, maximum_frequency=maximum_frequency)
    else:
        raise ValueError(f"不支持的群类型: {group_type}")
    
    model = EquivariantResNet18(
        gspace=gspace,
        num_classes=num_classes,
        base_channels=base_channels,
    )
    
    return model
