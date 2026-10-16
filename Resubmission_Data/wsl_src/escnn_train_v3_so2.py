"""
ESCNN 训练脚本 v3 (针对 SO(2) 优化版)

核心改进:
1. 采用 FourierELU 作为非线性激活函数，解决 SO(2) 下的训练稳定性问题
2. 使用 IIDBatchNorm2d 配合频率 0 特征，确保颜色/亮度信息不丢失
3. 调整表示构建逻辑，适应 SO(2) 连续群特性
"""

# 最优先导入numpy，解决命名空间冲突
import numpy
numpy.set_printoptions(threshold=numpy.inf)
# 确保np和numpy指向同一个模块
import numpy as np
np = numpy

# 检查np和numpy是否指向同一个模块
print(f"np id: {id(np)}")
print(f"numpy id: {id(numpy)}")
print(f"np.ndarray is numpy.ndarray: {np.ndarray is numpy.ndarray}")

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

# ==================== 训练参数配置 (手动修改) ====================
TRAIN_CONFIG = {
    # 训练超参数
    "epochs": 100,            # 训练轮数 (50轮通常足够)
    "batch_size": 16,         # 批大小 (显存不足可调小)
    "lr": 0.005,             # 学习率
    "momentum": 0.9,         # 动量
    "weight_decay": 1e-4,    # 权重衰减
    
    # 模型参数
    "base_channels": 16,     # 基础通道数 (恢复到 16)
    "max_frequency": 4,      # SO2 最大频率 (保持较低以加速)
    
    # 硬件加速与优化
    "amp": False,            # 是否开启混合精度训练 (SO2 的 Norm 操作在 float16 下可能不稳定，建议关闭)
    "num_workers": 0,        # DataLoader 线程数 (Windows 建议设为 0)
    "benchmark": False,      # 是否仅运行基准测试
    
    # 其他
    "seed": 42,
    "img_size": 224,
    "output_dir": "outputs_escnn",
    "test_rotation": False,  # 训练后是否测试旋转鲁棒性
}


# ==================== 数据集配置 ====================
DATASET_NAME = "trashnet"  # 可选: "trashnet", "realwaste"
DATASET_ROOTS = {
    "trashnet": "trashnet",
    "realwaste": "RealWaste",
}


# ==================== 核心组件构建 ====================

def _build_so2_field_type(gspace, channels: int, max_freq: int = 2) -> enn.FieldType:
    """
    构建 SO(2) 的 FieldType
    包含频率 0 (trivial) 到 max_freq 的所有频率，确保捕捉颜色和纹理方向
    使用 spectral_regular_representation 以兼容 FourierELU
    """
    # 为 SO(2) 创建频率列表 (0 到 max_freq)
    irreps_list = [(k,) for k in range(0, max_freq + 1)]
    
    # 使用 spectral_regular_representation 构建表示，这与 FourierELU 期望的格式匹配
    spectral_rep = gspace.fibergroup.spectral_regular_representation(*irreps_list)
    
    # 创建 FieldType
    return enn.FieldType(gspace, channels * [spectral_rep])

def _build_activation(out_type: enn.FieldType, max_freq: int = 2):
    """
    构建激活函数
    对于 SO(2)，使用 FourierELU 是最稳健的选择
    根据当前 ESCNN 版本的 API 要求，FourierELU 需要明确的 channels 和 irreps 参数
    """
    gspace = out_type.gspace
    N = max(8, 2 * max_freq + 4)
    
    # 检查是否为连续 SO(2)
    grp = getattr(gspace, "fibergroup", None)
    order = getattr(grp, "order", lambda: -1)()
    
    if order == -1: # SO(2)
        # 从 out_type 中获取第一个 representation，它应该是 spectral_regular_representation
        spectral_rep = out_type.representations[0]
        
        # 从 spectral_representation 中提取 irreps ID 列表
        # spectral_rep.irreps 包含了构成该表示的所有 irreps 的元组 ID
        irreps_list = spectral_rep.irreps
        
        # 计算每个通道的维度：每个 spectral_rep 的大小
        dim_per_channel = spectral_rep.size
        
        # 计算通道数：FieldType 的总大小除以单个 spectral_rep 的大小
        channels = out_type.size // dim_per_channel
        
        # 使用当前 ESCNN 版本要求的参数构建 FourierELU
        # 确保 irreps 参数是 irreps ID 列表（元组），这与 spectral_rep.irreps 格式匹配
        act = enn.FourierELU(
            gspace=gspace,
            channels=channels,
            irreps=irreps_list,
            type='regular',
            N=N,
            inplace=True
        )
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
    gspace = out_type.gspace
    
    # 检查是否为连续 SO(2)
    grp = getattr(gspace, "fibergroup", None)
    order = getattr(grp, "order", lambda: -1)()
    
    if order == -1: # SO(2)
        # 对于 SO2，使用更适合的池化策略
        # 参考 ESCNN 官方 SO2 实现
        # SO2 的 regular representation 不支持 pointwise 非线性操作
        # 因此，我们使用 NormPool，它不需要 pointwise 支持
        if hasattr(enn, "NormPool"):
            # NormPool 计算特征向量的范数，适用于所有表示
            # 这是 ESCNN 中 SO2 的推荐池化方式
            return enn.NormPool(out_type, kernel_size=2, stride=2)
        elif hasattr(enn, "PointwiseMaxPool"):
            # 只有当表示支持 pointwise 操作时才使用
            try:
                return enn.PointwiseMaxPool(out_type, kernel_size=2, stride=2)
            except AssertionError:
                # 如果不支持 pointwise 操作，回退到 NormPool
                print("PointwiseMaxPool 不支持当前表示，使用 NormPool")
                return enn.NormPool(out_type, kernel_size=2, stride=2)
        else:
            # 降级方案：使用常规的 AvgPool2d
            return nn.AvgPool2d(2)
    else: # 离散群（如 C4, C8）
        # 对于离散群，GroupPooling 更合适
        if hasattr(enn, "GroupPooling"):
            return enn.GroupPooling(out_type, kernel_size=2, stride=2)
        elif hasattr(enn, "PointwiseMaxPool"):
            return enn.PointwiseMaxPool(out_type, kernel_size=2, stride=2)
        else:
            return nn.AvgPool2d(2)

# ==================== 网络模块 ====================

class EquivariantBlock(nn.Module):
    def __init__(self, in_type, out_type, stride=1, max_freq=2):
        super().__init__()
        
        # 简化类型管理：所有层使用相同的输出类型
        # 直接使用 out_type 作为激活函数和后续层的类型
        
        self.conv1 = enn.R2Conv(in_type, out_type, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = _build_norm_layer(out_type)
        self.act1 = _build_activation(out_type, max_freq)  # 使用 SO(2) 专用激活函数
        
        self.conv2 = enn.R2Conv(out_type, out_type, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = _build_norm_layer(out_type)
        self.act2 = _build_activation(out_type, max_freq)  # 使用 SO(2) 专用激活函数
        
        self.downsample = None
        if stride != 1 or in_type != out_type:
            self.downsample = nn.Sequential(
                enn.R2Conv(in_type, out_type, kernel_size=1, stride=stride, bias=False),
                _build_norm_layer(out_type)
            )
        
        # 保存输入输出类型
        self.in_type = in_type
        self.out_type = out_type

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
        
        # 使用专门为 SO(2) 设计的激活函数
        self.act1 = _build_activation(out_type_0, max_freq)
        act1_out_type = out_type_0  # 激活函数的输出类型与输入类型相同
        
        # 池化: 使用 NormPool，它是 ESCNN 中 SO2 的推荐池化方式
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
        
        # 对于 SO(2)，使用更有效的池化策略
        # 参考 ESCNN 官方 SO2 实现：
        # 1. 对于频率 0 分量，它是标量，可以直接进行 GroupPooling
        # 2. 对于频率 > 0 分量，可以使用 NormPool 或直接丢弃
        # 3. 更优方案：使用 enn.GroupPooling 或自定义池化
        try:
            # 尝试使用 GroupPooling，如果支持的话
            self.gpool = enn.GroupPooling(last_type)
            print("使用 GroupPooling 进行全局池化")
        except Exception as e:
            # 如果 GroupPooling 不支持 SO2，使用改进的 NormPool 策略
            print(f"GroupPooling 不支持 SO2: {e}")
            print("使用改进的池化策略...")
            
            # 检查是否有频率 0 分量的专门池化
            # 对于 SO2，更好的做法是从特征中提取频率 0 分量
            # 然后只对这些分量进行池化，保留重要的标量信息
            
            # 方案1: 使用 NormPool，但确保它只作用于频率 0 分量
            # 方案2: 先提取频率 0 分量，再进行池化
            # 这里我们使用方案1，它更简单且效果良好
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
    
    for images, targets in loader:
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
            
    return total_loss / total, correct / total

# 计算单个类别的Precision-Recall曲线下面积(AP)
@torch.no_grad()
def calculate_ap(predictions, targets, num_classes):
    """计算每个类别的AP值"""
    from sklearn.metrics import average_precision_score
    import numpy as np
    
    # 转换为one-hot编码
    one_hot_targets = np.zeros((targets.shape[0], num_classes))
    one_hot_targets[np.arange(targets.shape[0]), targets.cpu().numpy()] = 1
    
    # 计算每个类别的AP
    ap_scores = []
    for cls in range(num_classes):
        if np.sum(one_hot_targets[:, cls]) == 0:
            # 该类别没有样本
            ap_scores.append(0.0)
            continue
        ap = average_precision_score(one_hot_targets[:, cls], predictions.cpu().numpy()[:, cls])
        ap_scores.append(ap)
    
    return np.array(ap_scores)

@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """评估模型，返回准确率、mAP、AP分数、预测和目标"""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    all_predicted_labels = []
    
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        
        # 保存预测结果和目标
        all_predictions.append(outputs)
        all_targets.append(targets)
        
        _, predicted = outputs.max(1)
        all_predicted_labels.append(predicted)
        
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    # 计算准确率
    accuracy = correct / total
    
    # 计算mAP
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_predicted_labels = torch.cat(all_predicted_labels, dim=0)
    
    ap_scores = calculate_ap(all_predictions, all_targets, num_classes)
    mAP = np.mean(ap_scores)
    
    return accuracy, mAP, ap_scores, all_predicted_labels.cpu().numpy(), all_targets.cpu().numpy()

# 绘制混淆矩阵
def plot_confusion_matrix(true_labels, predicted_labels, classes, output_path):
    """绘制混淆矩阵并保存到指定路径"""
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # 归一化
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 设置图形大小
    plt.figure(figsize=(10, 8))
    
    # 绘制热力图
    im = plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)
    
    # 设置类别标签
    tick_marks = np.arange(len(classes))
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
    plt.title('Confusion Matrix')
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存到: {output_path}")

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
    model = SO2ResNet18(len(classes), config["base_channels"], config["max_frequency"])
    
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
    print("ESCNN 等变网络训练 (改进版 v3)")
    print("=" * 70)
    print(f"数据集: {DATASET_NAME} -> {DATASET_ROOTS[DATASET_NAME]}")
    print(f"类别: {classes}")
    print(f"训练/验证/测试: {train_size}/{val_size}/{test_size}")
    print("对称群: SO2")
    print(f"基础通道数: {config['base_channels']}")
    print("学习率调度: cosine")
    print("=" * 70)
    print(f"设备: {device}")
    print("构建等变网络: SO2 (全方向旋转对称)")
    print(f"  gspace: {model.gspace}")
    print(f"  base_channels: {config['base_channels']}")
    print()
    print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print("=" * 70)
    
    # 初始化最佳指标
    best_mAP = 0.0
    best_epoch = 0
    
    # 数据加载器
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    
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
            model_path = f"{config['output_dir']}/best_so2_v3.pth"
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
    
    # 绘制混淆矩阵，文件名包含数据集名称
    confusion_matrix_path = f"{config['output_dir']}/confusion_matrix_{DATASET_NAME}.png"
    plot_confusion_matrix(test_targets, test_predicted, classes, confusion_matrix_path)
    
    print("=" * 70)
    print("完成!")

if __name__ == "__main__":
    main()
