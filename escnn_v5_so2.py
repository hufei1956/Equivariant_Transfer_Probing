"""
ESCNN 训练脚本 v5 (SO(2) 优化版)
================================

基于 ResNet-18 架构的 SO(2) 等变网络训练脚本
参考 mlp.ipynb 中的最佳实践进行优化

关键改进:
1. 使用 mlp.ipynb 中的 FourierELU 配置方法
2. 改进层间类型匹配
3. 使用 SequentialModule 简化块设计
4. 改进初始化策略
5. 添加等变性测试功能
6. 保持 ResNet-18 架构不变

参考:
- mlp.ipynb 中的 ESCNN 最佳实践
- ESCNN 文档: https://quva-lab.github.io/escnn/
- 修复了 v4 版本中的初始化问题
"""

# 最优先导入numpy，解决命名空间冲突
import numpy
numpy.set_printoptions(threshold=numpy.inf)
import numpy as np
np = numpy

import os
import time
import argparse
from typing import Tuple, List, Optional
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

from escnn import gspaces
from escnn import nn as enn
from escnn.nn import FieldType, GeometricTensor
from escnn.group import SO2

from dataset import create_trashnet_datasets

# ==================== 训练参数配置 ====================
TRAIN_CONFIG = {
    # 训练超参数
    "epochs": 100,            # 训练轮数
    "batch_size": 16,         # 批大小 (显存不足可调小)
    "lr": 0.005,             # 学习率
    "momentum": 0.9,         # 动量
    "weight_decay": 1e-4,    # 权重衰减
    
    # 模型参数
    "base_channels": 16,     # 基础通道数
    "max_frequency": 4,      # SO2 最大频率 (保持较低以加速)
    "num_samples": 16,       # FourierELU 采样点数
    "pool_method": "norm",   # 不变性池化方法: "norm"                                或 "trivial"
    
    # 硬件加速与优化
    "amp": False,            # 是否开启混合精度训练 (SO2 建议关闭)
    "num_workers": 0,        # DataLoader 线程数 (Windows 建议设为 0)
    "benchmark": False,      # 是否仅运行基准测试
    
    # 其他
    "seed": 42,
    "img_size": 224,
    "output_dir": "outputs_escnn_v5",
    "test_rotation": False,  # 训练后是否测试旋转鲁棒性
}


# ==================== 数据集配置 ====================
DATASET_NAME = "trashnet"  # 可选: "trashnet", "realwaste"
DATASET_ROOTS = {
    "trashnet": "trashnet",
    "realwaste": "RealWaste",
}


# ==================== 模型定义 ====================

class SO2BasicBlock(nn.Module):
    """
    SO(2) 等变的 Basic Block (类似 ResNet BasicBlock)
    参考 mlp.ipynb 使用 SequentialModule 和改进的类型匹配
    """
    expansion = 1
    
    def __init__(
        self,
        gspace,
        in_type: FieldType,
        out_channels: int,
        max_freq: int,
        num_samples: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.gspace = gspace
        self.G = gspace.fibergroup
        
        # 使用 mlp.ipynb 中的 FourierELU 配置方式
        activation = enn.FourierELU(
            gspace,
            channels=out_channels,
            irreps=self.G.bl_regular_representation(L=max_freq).irreps,  # 频率 0 到 max_freq
            inplace=True,
            type='regular', N=num_samples,  # 参考 mlp.ipynb 配置
        )
        
        # 输出类型与激活函数的输入类型匹配
        out_type = activation.in_type
        
        # Conv1: 可能有下采样
        if stride > 1:
            # 参考 mlp.ipynb 使用 SequentialModule
            conv1_block = enn.SequentialModule(
                enn.PointwiseAvgPool(in_type, kernel_size=stride, stride=stride),
                enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
                enn.IIDBatchNorm2d(out_type),
                activation
            )
        else:
            conv1_block = enn.SequentialModule(
                enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
                enn.IIDBatchNorm2d(out_type),
                activation
            )
        
        # Conv2 块
        conv2_block = enn.SequentialModule(
            enn.R2Conv(out_type, out_type, kernel_size=3, padding=1, bias=False),
            enn.IIDBatchNorm2d(out_type)
        )
        
        self.conv1_block = conv1_block
        self.conv2_block = conv2_block
        self.downsample = downsample
        self.out_type = out_type
        
        # 输出激活函数
        self.act_out = enn.FourierELU(
            gspace,
            channels=out_channels,
            irreps=self.G.bl_regular_representation(L=max_freq).irreps,
            inplace=True,
            type='regular', N=num_samples,
        )
    
    def forward(self, x: GeometricTensor) -> GeometricTensor:
        identity = x
        
        # 主路径
        out = self.conv1_block(x)
        out = self.conv2_block(out)
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 加法需要类型匹配
        out = out + identity
        out = self.act_out(out)
        
        return out


class SO2ResNet18(nn.Module):
    """
    SO(2) 等变的 ResNet-18 网络
    参考 mlp.ipynb 中的架构设计
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        base_channels: int = 16,
        max_frequency: int = 4,
        num_samples: int = 16,
        image_size: int = 224,
        pool_method: str = 'norm'
    ):
        super().__init__()
        
        self.max_freq = max_frequency
        self.num_samples = num_samples
        self.pool_method = pool_method
        
        # SO(2) 群空间
        self.gspace = gspaces.rot2dOnR2(N=-1, maximum_frequency=max_frequency * 2)
        self.G: SO2 = self.gspace.fibergroup
        
        # 输入类型: 标量场 (trivial representation)
        self.input_type = enn.FieldType(self.gspace, in_channels * [self.gspace.trivial_repr])
        
        # Mask 处理边角
        self.mask = enn.MaskModule(self.input_type, image_size, margin=1)
        
        # ============ Stem ============
        # 参考 mlp.ipynb 配置 FourierELU
        stem_act = enn.FourierELU(
            self.gspace,
            channels=base_channels,
            irreps=self.G.bl_regular_representation(L=max_frequency).irreps,
            inplace=True,
            type='regular', N=num_samples,
        )
        
        # 参考 mlp.ipynb 使用 SequentialModule
        self.stem = enn.SequentialModule(
            enn.R2Conv(
                self.input_type, stem_act.in_type,
                kernel_size=7, stride=2, padding=3, bias=False
            ),
            enn.IIDBatchNorm2d(stem_act.in_type),
            stem_act,
            enn.PointwiseAvgPool(stem_act.out_type, kernel_size=3, stride=2, padding=1)
        )
        
        # 当前类型
        self.current_type = self.stem.out_type
        self.current_channels = base_channels
        
        # ============ Stages ============
        self.layer1 = self._make_layer(base_channels, 2, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 8, 2, stride=2)
        
        # ============ 不变性池化 ============
        if pool_method == 'norm':
            # 使用 NormPool - 取范数，丢失方向信息但保留幅度
            self.invariant_pool = enn.NormPool(self.current_type)
            final_features = self.invariant_pool.out_type.size
        else:  # trivial
            # 投影到标量空间 - 让网络学习如何组合特征
            final_channels = base_channels * 8
            self.trivial_type = enn.FieldType(
                self.gspace, final_channels * [self.gspace.trivial_repr]
            )
            self.to_trivial = enn.R2Conv(
                self.current_type, self.trivial_type,
                kernel_size=1, bias=False
            )
            self.invariant_pool = None
            final_features = final_channels
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_features, num_classes)
        )
        
        # 初始化
        self._initialize_weights()
    
    def _make_layer(self, channels: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        """创建一个 stage"""
        layers = []
        
        # 第一个 block 可能需要下采样
        downsample = None
        if stride != 1 or self.current_channels != channels:
            # 参考 mlp.ipynb 配置激活函数
            downsample_act = enn.FourierELU(
                self.gspace, channels=channels,
                irreps=self.G.bl_regular_representation(L=self.max_freq).irreps,
                inplace=True,
                type='regular', N=self.num_samples
            )
            new_type = downsample_act.in_type
            
            # 下采样模块: 池化 + 1x1 卷积
            if stride > 1:
                downsample = enn.SequentialModule(
                    enn.PointwiseAvgPool(self.current_type, kernel_size=stride, stride=stride),
                    enn.R2Conv(self.current_type, new_type, kernel_size=1, bias=False),
                    enn.IIDBatchNorm2d(new_type)
                )
            else:
                downsample = enn.SequentialModule(
                    enn.R2Conv(self.current_type, new_type, kernel_size=1, bias=False),
                    enn.IIDBatchNorm2d(new_type)
                )
        
        # 添加第一个 block
        layers.append(SO2BasicBlock(
            self.gspace, self.current_type, channels,
            self.max_freq, self.num_samples,
            stride=stride, downsample=downsample
        ))
        
        # 更新当前类型
        self.current_type = layers[-1].out_type
        self.current_channels = channels
        
        # 添加剩余 blocks
        for _ in range(1, num_blocks):
            layers.append(SO2BasicBlock(
                self.gspace, self.current_type, channels,
                self.max_freq, self.num_samples
            ))
            self.current_type = layers[-1].out_type
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化权重 - 参考 mlp.ipynb 和 ESCNN 最佳实践"""
        for m in self.modules():
            if isinstance(m, enn.R2Conv):
                # 使用 ESCNN 推荐的初始化
                if hasattr(m, 'weights') and m.weights is not None:
                    if m.weights.data.dim() >= 2:
                        # He 初始化
                        nn.init.kaiming_normal_(m.weights.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, enn.IIDBatchNorm2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 包装为 GeometricTensor
        x = self.input_type(x)
        x = self.mask(x)
        
        # Stem
        x = self.stem(x)
        
        # Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 不变性池化
        if self.pool_method == 'norm':
            x = self.invariant_pool(x)
        else:
            x = self.to_trivial(x)
        
        # 转回普通 tensor
        x = x.tensor
        
        # 全局池化 + 分类
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x


# ==================== 训练与验证 ====================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def compute_map(logits, targets):
    """
    计算多分类任务的 mAP (mean Average Precision)
    
    Args:
        logits: 模型输出的原始logits，形状为 [N, C]
        targets: 真实标签，形状为 [N]
    
    Returns:
        mAP: 所有类别的平均AP
        ap_scores: 每个类别的AP分数列表
    """
    from sklearn.metrics import average_precision_score
    import numpy as np
    
    # 将logits转换为概率
    probs = torch.softmax(logits, dim=1).numpy()
    targets = targets.numpy()
    
    # 转换为one-hot编码
    num_classes = probs.shape[1]
    targets_onehot = np.eye(num_classes)[targets]
    
    # 计算每个类别的AP
    ap_scores = []
    for i in range(num_classes):
        try:
            ap = average_precision_score(targets_onehot[:, i], probs[:, i])
            ap_scores.append(ap)
        except ValueError:
            # 如果某个类别没有正样本或负样本，AP可能无法计算
            ap_scores.append(0.0)
    
    # 计算mAP
    mAP = np.mean(ap_scores)
    
    return mAP, ap_scores


def evaluate(model, test_loader, device, num_classes):
    """评估模型"""
    model.eval()
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_logits.append(outputs.cpu())
            all_targets.append(labels.cpu())
    
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # 计算 accuracy
    preds = logits.argmax(dim=1)
    accuracy = (preds == targets).sum().item() / targets.size(0)
    
    # 计算真正的 mAP
    mAP, ap_scores = compute_map(logits, targets)
    
    return accuracy, mAP, ap_scores, preds.numpy(), targets.numpy()


# ==================== 混淆矩阵 ====================

def plot_confusion_matrix(targets, predictions, classes, output_path):
    """绘制混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(targets, predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    
    # 在每个格子中显示数值
    fmt = '.2f'
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(j, i, format(cm_normalized[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm_normalized[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存到: {output_path}")


# ==================== 等变性测试 ====================

def test_equivariance(model, device, num_tests=10):
    """
    测试模型的旋转等变性
    参考 mlp.ipynb 中的测试方法
    """
    model.eval()
    
    print("=" * 70)
    print("SO(2) 等变性测试")
    print("=" * 70)
    
    # 生成随机输入
    B = 5
    input_shape = (B, 3, 224, 224)
    x = torch.randn(*input_shape)
    
    # 包装为 GeometricTensor
    geo_x = model.input_type(x.to(device))
    
    # 原始输出
    with torch.no_grad():
        original_output = model(x.to(device))
    
    errors = []
    
    # 测试多个随机旋转
    for i in range(num_tests):
        # 采样随机旋转
        g = model.G.sample()
        
        # 旋转输入
        rotated_geo_x = g @ geo_x
        
        # 旋转后的输出
        rotated_x = rotated_geo_x.tensor
        with torch.no_grad():
            rotated_output = model(rotated_x)
        
        # 计算误差
        error = torch.linalg.norm(rotated_output - original_output, dim=1).mean().item()
        errors.append(error)
    
    avg_error = sum(errors) / len(errors)
    print(f"平均等变性误差: {avg_error:.6f}")
    print(f"误差范围: {min(errors):.6f} - {max(errors):.6f}")
    
    if avg_error < 0.1:
        print("✓ 模型具有良好的旋转等变性!")
    else:
        print("✗ 模型的旋转等变性需要改进")
    
    print("=" * 70)


# ==================== 主程序 ====================

def main():
    # 使用配置文件中的参数
    config = TRAIN_CONFIG
    
    # 数据集
    train_set, val_set, test_set, classes, _ = create_trashnet_datasets(
        data_root=DATASET_ROOTS[DATASET_NAME], split_ratio=(0.7, 0.15, 0.15)
    )
    
    # 获取数据集大小
    train_size = len(train_set)
    val_size = len(val_set)
    test_size = len(test_set)
    
    # 模型
    model = SO2ResNet18(
        in_channels=3,
        num_classes=len(classes),
        base_channels=config["base_channels"],
        max_frequency=config["max_frequency"],
        num_samples=config["num_samples"],
        image_size=config["img_size"],
        pool_method=config["pool_method"]
    )
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 创建输出目录
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # 优化器和学习率调度
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    
    # 打印训练信息
    print("=" * 70)
    print("ESCNN 等变网络训练 (v5 SO2 优化版)")
    print("=" * 70)
    print(f"数据集: {DATASET_NAME} -> {DATASET_ROOTS[DATASET_NAME]}")
    print(f"类别: {classes}")
    print(f"训练/验证/测试: {train_size}/{val_size}/{test_size}")
    print("对称群: SO2")
    print(f"基础通道数: {config['base_channels']}")
    print(f"最大频率: {config['max_frequency']}")
    print(f"FourierELU 采样点数: {config['num_samples']}")
    print(f"不变性池化: {config['pool_method']}")
    print("学习率调度: cosine")
    print("=" * 70)
    print(f"设备: {device}")
    print("构建等变网络: SO2 ResNet-18")
    print(f"  gspace: {model.gspace}")
    print(f"  参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print("=" * 70)
    
    # 初始化最佳指标
    best_mAP = 0.0
    best_epoch = 0
    
    # 数据加载器
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    
    # 等变性测试
    if config["test_rotation"]:
        test_equivariance(model, device)
    
    # 开始训练计时
    total_train_start = time.time()
    
    for epoch in range(config["epochs"]):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_mAP, _, _, _ = evaluate(model, val_loader, device, len(classes))
        scheduler.step()
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印epoch信息
        print(f"Epoch {epoch+1:03d} | loss={train_loss:.4f} acc={train_acc:.4f} | val_acc={val_acc:.4f} val_mAP={val_mAP:.4f} | lr={current_lr:.6f}")
        
        # 保存最佳模型
        if val_mAP > best_mAP:
            best_mAP = val_mAP
            best_epoch = epoch + 1
            model_path = f"{config['output_dir']}/best_so2_v5_{DATASET_NAME}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"  -> 保存最佳模型 (val_mAP={val_mAP:.4f})")
    
    # 训练完成
    total_train_time = time.time() - total_train_start
    print("=" * 70)
    print("训练完成! 总耗时: {:.1f} 分钟".format(total_train_time / 60))
    print(f"最佳 epoch: {best_epoch}, 最佳 val_mAP: {best_mAP:.4f}")
    print(f"已加载最佳模型: {model_path}")
    print("=" * 70)
    
    # 测试集验证
    print("[Test] 正在验证测试集...")
    model.load_state_dict(torch.load(model_path))
    test_acc, test_mAP, test_ap_scores, test_predicted, test_targets = evaluate(model, test_loader, device, len(classes))
    print(f"[Test] top1_acc={test_acc:.4f} mAP={test_mAP:.4f}")
    print("各类别 AP:")
    for cls, ap in zip(classes, test_ap_scores):
        print(f"  {cls}: {ap:.4f}")
    
    # 绘制混淆矩阵
    confusion_matrix_path = f"{config['output_dir']}/confusion_matrix_{DATASET_NAME}.png"
    plot_confusion_matrix(test_targets, test_predicted, classes, confusion_matrix_path)
    
    # 最终等变性测试
    if config["test_rotation"]:
        test_equivariance(model, device)
    
    print("=" * 70)
    print("完成!")


if __name__ == "__main__":
    main()