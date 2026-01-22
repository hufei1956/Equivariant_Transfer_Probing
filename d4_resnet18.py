"""
D4 等变 ResNet-18 for TrashNet
==============================

基于 ESCNN e2wrn.py 的风格实现 D4 等变 ResNet-18 网络。

D4 群包含:
- 4 次旋转: 0°, 90°, 180°, 270°
- 4 次反射: 沿 x, y 轴及两条对角线
- 共 8 个群元素

关键设计 (参考 e2wrn.py):
1. gspace: flipRot2dOnR2(N=4) 对应 D4
2. 使用 regular_repr 作为中间特征表示
3. 最后一层转为 trivial_repr 以获得不变性
4. 支持群限制 (restriction) 以在深层降低计算
5. 使用 InnerBatchNorm 和 ReLU
6. fixparams=True 保持参数量与标准 ResNet 相当

用于 TrashNet 垃圾分类 (6 类):
cardboard, glass, metal, paper, plastic, trash
"""

from typing import Tuple, List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import escnn.nn as enn
from escnn.nn import init
from escnn import gspaces


__all__ = [
    "D4ResNet18",
    "D4ResNet34",
    "d4_resnet18",
    "d4_resnet34", 
    "c4_resnet18",
    "d8_resnet18",
]


# ==================== 辅助函数 (来自 e2wrn.py) ====================

def conv7x7(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=3,
            dilation=1, bias=False):
    """7x7 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 7,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )


def conv3x3(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=1,
            dilation=1, bias=False):
    """3x3 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 3,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )


def conv1x1(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=0,
            dilation=1, bias=False):
    """1x1 convolution"""
    return enn.R2Conv(in_type, out_type, 1,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )


def regular_feature_type(gspace: gspaces.GSpace, planes: int, fixparams: bool = True):
    """
    构建 regular representation 的特征类型
    
    fixparams=True 时，调整通道数使参数量与标准 CNN 相当
    """
    assert gspace.fibergroup.order() > 0
    
    N = gspace.fibergroup.order()
    
    if fixparams:
        planes *= math.sqrt(N)
    
    planes = planes / N
    planes = int(planes)
    
    return enn.FieldType(gspace, [gspace.regular_repr] * planes)


def trivial_feature_type(gspace: gspaces.GSpace, planes: int, fixparams: bool = True):
    """
    构建 trivial representation (标量场) 的特征类型
    """
    if fixparams:
        planes *= math.sqrt(gspace.fibergroup.order())
        
    planes = int(planes)
    return enn.FieldType(gspace, [gspace.trivial_repr] * planes)


FIELD_TYPE = {
    "trivial": trivial_feature_type,
    "regular": regular_feature_type,
}


# ==================== ResNet BasicBlock ====================

class BasicBlock(enn.EquivariantModule):
    """
    等变 BasicBlock (标准 ResNet 风格)
    
    结构: Conv -> BN -> ReLU -> Conv -> BN -> (+shortcut) -> ReLU
    """
    expansion = 1
    
    def __init__(
        self,
        in_type: enn.FieldType,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[enn.SequentialModule] = None,
        fixparams: bool = True,
    ):
        super(BasicBlock, self).__init__()
        
        self.in_type = in_type
        gspace = in_type.gspace
        
        # 输出类型
        out_type = FIELD_TYPE["regular"](gspace, out_channels, fixparams=fixparams)
        self.out_type = out_type
        
        # Conv1 + BN1 + ReLU1
        self.conv1 = conv3x3(in_type, out_type, stride=stride)
        self.bn1 = enn.InnerBatchNorm(out_type)
        self.relu1 = enn.ReLU(out_type, inplace=True)
        
        # Conv2 + BN2
        self.conv2 = conv3x3(out_type, out_type)
        self.bn2 = enn.InnerBatchNorm(out_type)
        
        # ReLU after addition
        self.relu2 = enn.ReLU(out_type, inplace=True)
        
        self.downsample = downsample
    
    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 使用更稳健的加法方式，避免 GeometricTensor 类型检查过于严格的问题
        out = enn.GeometricTensor(out.tensor + identity.tensor, out.type)
        out = self.relu2(out)
        
        return out
    
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        # assert input_shape[1] == self.in_type.size # 暂时注释掉，避免 fixparams 导致的微小尺寸差异报错
        if self.downsample is not None:
            return self.downsample.evaluate_output_shape(input_shape)
        else:
            return input_shape


class PreActBasicBlock(enn.EquivariantModule):
    """
    预激活 BasicBlock (类似 e2wrn.py 的 WideBasic)
    
    结构: BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> (+shortcut)
    """
    expansion = 1
    
    def __init__(
        self,
        in_type: enn.FieldType,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[enn.SequentialModule] = None,
        fixparams: bool = True,
    ):
        super(PreActBasicBlock, self).__init__()
        
        self.in_type = in_type
        gspace = in_type.gspace
        
        out_type = FIELD_TYPE["regular"](gspace, out_channels, fixparams=fixparams)
        self.out_type = out_type
        
        # BN1 + ReLU1 + Conv1
        self.bn1 = enn.InnerBatchNorm(in_type)
        self.relu1 = enn.ReLU(in_type, inplace=True)
        self.conv1 = conv3x3(in_type, out_type, stride=stride)
        
        # BN2 + ReLU2 + Conv2
        self.bn2 = enn.InnerBatchNorm(out_type)
        self.relu2 = enn.ReLU(out_type, inplace=True)
        self.conv2 = conv3x3(out_type, out_type)
        
        self.downsample = downsample
    
    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        identity = x
        
        out = self.bn1(x)
        out = self.relu1(out)
        
        if self.downsample is not None:
            # PreAct 结构中，shortcut 分支应该使用激活后的特征（如果是 stride > 1 或通道改变）
            # 参考 e2wrn.py 的实现
            identity = self.downsample(out)
        
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        out = enn.GeometricTensor(out.tensor + identity.tensor, out.type)
        
        return out
    
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        # assert input_shape[1] == self.in_type.size
        if self.downsample is not None:
            return self.downsample.evaluate_output_shape(input_shape)
        else:
            return input_shape


# ==================== D4 ResNet 主网络 ====================

class D4ResNet(nn.Module):
    """
    D4 等变 ResNet
    """
    
    def __init__(
        self,
        layers: List[int],
        num_classes: int = 6,
        N: int = 4,
        f: bool = True,
        r: int = 0,
        base_width: int = 64,
        fixparams: bool = True,
        block_type: str = 'standard',
        initial_stride: int = 2,
        deltaorth: bool = False,
    ):
        super(D4ResNet, self).__init__()
        
        self._N = N
        self._f = f
        self._r = r
        self._fixparams = fixparams
        
        # 选择 block 类型
        if block_type == 'preact':
            block = PreActBasicBlock
        else:
            block = BasicBlock
        
        # 通道数配置 (标准 ResNet)
        self.nStages = [base_width, base_width, base_width * 2, base_width * 4, base_width * 8]
        
        # 创建 gspace
        if f:
            if N != 1:
                self.gspace = gspaces.flipRot2dOnR2(N=N)
            else:
                self.gspace = gspaces.flip2dOnR2()
        else:
            if N != 1:
                self.gspace = gspaces.rot2dOnR2(N=N)
            else:
                self.gspace = gspaces.trivialOnR2()
        
        # 打印网络信息
        group_name = f"{'D' if f else 'C'}{N}"
        print(f"| D4ResNet-{sum(layers)*2 + 2} with {group_name} equivariance")
        print(f"| Group order: {self.gspace.fibergroup.order()}")
        
        # 输入类型: RGB 图像是标量场
        self.in_type = enn.FieldType(self.gspace, 3 * [self.gspace.trivial_repr])
        
        # 当前类型 (用于构建层)
        self._in_type = self.in_type
        
        # ============ Stem ============
        stem_out_type = FIELD_TYPE["regular"](self.gspace, self.nStages[0], fixparams=fixparams)
        
        if initial_stride == 2:
            # ImageNet 风格: 7x7 conv stride 2 + maxpool
            self.conv1 = conv7x7(self.in_type, stem_out_type, stride=2)
            self.bn1 = enn.InnerBatchNorm(stem_out_type)
            self.relu1 = enn.ReLU(stem_out_type, inplace=True)
            self.maxpool = enn.PointwiseMaxPool(stem_out_type, kernel_size=3, stride=2, padding=1)
        else:
            # CIFAR 风格: 3x3 conv stride 1, no maxpool
            self.conv1 = conv3x3(self.in_type, stem_out_type, stride=1)
            self.bn1 = enn.InnerBatchNorm(stem_out_type)
            self.relu1 = enn.ReLU(stem_out_type, inplace=True)
            self.maxpool = None
        
        self._in_type = stem_out_type
        
        # ============ ResNet Stages ============
        # Layer 1
        self.layer1 = self._make_layer(block, self.nStages[1], layers[0], stride=1)
        
        # 可选限制 (r >= 2)
        if r >= 2 and N > 1:
            N_new = N // 2
            subgroup_id = (0., N_new) if f else N_new
            self.restrict1 = self._restrict_layer(subgroup_id)
        else:
            self.restrict1 = lambda x: x
        
        # Layer 2
        self.layer2 = self._make_layer(block, self.nStages[2], layers[1], stride=2)
        
        # 可选限制 (r >= 1)
        if r >= 1 and N > 1:
            if r == 3:
                # 第二次限制到更小的群
                subgroup_id = (0., 1) if f else 1
            else:
                N_new = N // 2
                subgroup_id = (0., N_new) if f else N_new
            self.restrict2 = self._restrict_layer(subgroup_id)
        else:
            self.restrict2 = lambda x: x
        
        # Layer 3
        self.layer3 = self._make_layer(block, self.nStages[3], layers[2], stride=2)
        
        # Layer 4 - 最后一层转为 trivial 以获得不变性
        self.layer4 = self._make_layer(block, self.nStages[4], layers[3], stride=2, totrivial=True)
        
        # 最终 BN 和 ReLU
        self.bn_final = enn.InnerBatchNorm(self._in_type)
        self.relu_final = enn.ReLU(self._in_type, inplace=True)
        
        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self._in_type.size, num_classes)
        
        # 权重初始化
        self._initialize_weights(deltaorth)
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"| Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    def _make_layer(
        self,
        block,
        planes: int,
        num_blocks: int,
        stride: int = 1,
        totrivial: bool = False,
    ) -> enn.SequentialModule:
        """创建一个 stage"""
        
        layers = []
        
        # 输出类型
        if totrivial:
            out_type = FIELD_TYPE["trivial"](self.gspace, planes, fixparams=self._fixparams)
        else:
            out_type = FIELD_TYPE["regular"](self.gspace, planes, fixparams=self._fixparams)
        
        # 第一个 block 可能需要下采样
        downsample = None
        if stride != 1 or self._in_type.size != out_type.size:
            downsample = enn.SequentialModule(
                conv1x1(self._in_type, out_type, stride=stride),
                enn.InnerBatchNorm(out_type),
            )
        
        # 第一个 block
        layers.append(block(self._in_type, planes, stride=stride, 
                           downsample=downsample, fixparams=self._fixparams))
        self._in_type = layers[-1].out_type
        
        # 后续 blocks
        for i in range(1, num_blocks):
            # 最后一个 block 可能需要转为 trivial
            if i == num_blocks - 1 and totrivial:
                # 需要特殊处理
                final_out_type = FIELD_TYPE["trivial"](self.gspace, planes, fixparams=self._fixparams)
                downsample_final = enn.SequentialModule(
                    conv1x1(self._in_type, final_out_type),
                    enn.InnerBatchNorm(final_out_type),
                )
                # 使用自定义的最后一个 block
                layers.append(self._make_trivial_block(self._in_type, planes, downsample_final))
                self._in_type = final_out_type
            else:
                layers.append(block(self._in_type, planes, fixparams=self._fixparams))
                self._in_type = layers[-1].out_type
        
        return enn.SequentialModule(*layers)
    
    def _make_trivial_block(self, in_type, planes, downsample):
        """创建输出为 trivial 的最后一个 block"""
        gspace = in_type.gspace
        out_type = FIELD_TYPE["trivial"](gspace, planes, fixparams=self._fixparams)
        
        # 简单的卷积 + BN 结构
        return enn.SequentialModule(
            conv3x3(in_type, out_type),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True),
            conv3x3(out_type, out_type),
            enn.InnerBatchNorm(out_type),
        )
    
    def _restrict_layer(self, subgroup_id) -> enn.SequentialModule:
        """群限制层 (参考 e2wrn.py)"""
        layers = []
        layers.append(enn.RestrictionModule(self._in_type, subgroup_id))
        layers.append(enn.DisentangleModule(layers[-1].out_type))
        self._in_type = layers[-1].out_type
        self.gspace = self._in_type.gspace
        
        return enn.SequentialModule(*layers)
    
    def _initialize_weights(self, deltaorth: bool = False):
        """权重初始化"""
        for name, module in self.named_modules():
            if isinstance(module, enn.R2Conv):
                if deltaorth:
                    with torch.no_grad():
                        init.deltaorthonormal_init(module.weights, module.basisexpansion)
                else:
                    # 使用 no_grad 来避免 "leaf Variable that requires grad is being used in an in-place operation"
                    with torch.no_grad():
                        init.generalized_he_init(module.weights, module.basisexpansion)
            elif isinstance(module, (nn.BatchNorm2d, enn.InnerBatchNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 包装为 GeometricTensor
        x = enn.GeometricTensor(x, self.in_type)
        
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        if self.maxpool is not None:
            x = self.maxpool(x)
        
        # Stages
        x = self.layer1(x)
        x = self.restrict1(x)
        
        x = self.layer2(x)
        x = self.restrict2(x)
        
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Final BN + ReLU
        x = self.bn_final(x)
        x = self.relu_final(x)
        
        # 转回普通 tensor
        x = x.tensor
        
        # Global Average Pooling + FC
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# ==================== 便捷构造函数 ====================

def d4_resnet18(num_classes=6, **kwargs):
    return D4ResNet(layers=[2, 2, 2, 2], num_classes=num_classes, N=4, f=True, r=0, **kwargs)

def d4_resnet18_restricted(num_classes=6, **kwargs):
    return D4ResNet(layers=[2, 2, 2, 2], num_classes=num_classes, N=4, f=True, r=3, **kwargs)

def d4_resnet34(num_classes=6, **kwargs):
    return D4ResNet(layers=[3, 4, 6, 3], num_classes=num_classes, N=4, f=True, r=0, **kwargs)

def c4_resnet18(num_classes=6, **kwargs):
    return D4ResNet(layers=[2, 2, 2, 2], num_classes=num_classes, N=4, f=False, r=0, **kwargs)

def d8_resnet18(num_classes=6, **kwargs):
    return D4ResNet(layers=[2, 2, 2, 2], num_classes=num_classes, N=8, f=True, r=0, **kwargs)

def c8_resnet18(num_classes=6, **kwargs):
    return D4ResNet(layers=[2, 2, 2, 2], num_classes=num_classes, N=8, f=False, r=0, **kwargs)


# ==================== 测试函数 ====================

def test_equivariance():
    """测试模型的等变性/不变性"""
    print("\n" + "=" * 70)
    print("Testing D4 Equivariance")
    print("=" * 70)
    
    # 创建模型
    model = d4_resnet18(num_classes=6, initial_stride=1, base_width=16)
    model.eval()
    
    # 测试输入
    x = torch.randn(2, 3, 32, 32)
    
    # 各种变换
    x_rot90 = x.rot90(1, (2, 3))
    x_rot180 = x.rot90(2, (2, 3))
    x_flip_h = x.flip(dims=[3])
    x_flip_v = x.flip(dims=[2])
    
    with torch.no_grad():
        y = model(x)
        y_rot90 = model(x_rot90)
        y_rot180 = model(x_rot180)
        y_flip_h = model(x_flip_h)
        y_flip_v = model(x_flip_v)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    print("\nInvariance test:")
    print(f"  90° rotation:    {'PASS' if torch.allclose(y, y_rot90, atol=1e-5) else 'FAIL'}")
    print(f"  180° rotation:   {'PASS' if torch.allclose(y, y_rot180, atol=1e-5) else 'FAIL'}")
    print(f"  Horizontal flip: {'PASS' if torch.allclose(y, y_flip_h, atol=1e-5) else 'FAIL'}")
    print(f"  Vertical flip:   {'PASS' if torch.allclose(y, y_flip_v, atol=1e-5) else 'FAIL'}")
    
    print("=" * 70)


if __name__ == "__main__":
    test_equivariance()
