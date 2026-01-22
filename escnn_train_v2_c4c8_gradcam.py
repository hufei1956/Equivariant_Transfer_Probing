import torch
import torch.nn as nn
from typing import Optional

from escnn import gspaces
from escnn import nn as enn


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
    return enn.ReLU(out_type, inplace=False)


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
            in_type, out_type, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = enn.InnerBatchNorm(out_type) if not _is_continuous_so2(gspace) else None
        self.relu1 = _build_nonlinearity(out_type)

        self.conv2 = enn.R2Conv(
            out_type, out_type, kernel_size=3, stride=1, padding=1, bias=False
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

        assert identity.type == out.type

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
            self.in_type, out_type_0, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = enn.InnerBatchNorm(out_type_0) if not _is_continuous_so2(gspace) else None
        self.relu1 = _build_nonlinearity(out_type_0)
        self.maxpool = (
            None
            if _is_continuous_so2(gspace)
            else enn.PointwiseMaxPool(out_type_0, kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(out_type_0, base_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(self.layer1[-1].out_type, base_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(self.layer2[-1].out_type, base_channels * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(self.layer3[-1].out_type, base_channels * 8, blocks=2, stride=2)

        self.gpool = enn.GroupPooling(self.layer4[-1].out_type)
        gpool_out_dim = self.gpool.out_type.size

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(gpool_out_dim, num_classes)

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

    return EquivariantResNet18(gspace=gspace, num_classes=num_classes, base_channels=base_channels)

