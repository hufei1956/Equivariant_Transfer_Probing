# D4 等变 ResNet-18 for TrashNet

基于 ESCNN e2wrn.py 风格实现的 D4 等变 ResNet-18 网络。

## 文件说明

| 文件 | 说明 |
|------|------|
| `d4_resnet18.py` | D4 等变 ResNet-18 模型定义 |
| `train_d4_resnet18.py` | 完整训练脚本 |

## 快速开始

### 1. 基本训练

```bash
# 使用 D4 群 (推荐)
python train_d4_resnet18.py --data-root trashnet --group D4 --epochs 100

# 使用 C4 群 (纯旋转)
python train_d4_resnet18.py --data-root trashnet --group C4 --epochs 100

# 使用 D8 群 (更细的旋转)
python train_d4_resnet18.py --data-root trashnet --group D8 --epochs 100
```

### 2. 调整参数

```bash
# 增加基础通道数
python train_d4_resnet18.py --base-width 32 --lr 0.01

# 使用 D4 with restriction (D4 -> D2 -> D1)
python train_d4_resnet18.py --group D4R
```

## 群对比

| 群 | 元素数 | 等变操作 | 推荐场景 |
|----|--------|----------|----------|
| C4 | 4 | 90°旋转 | 一般分类 |
| D4 | 8 | 90°旋转 + 反射 | **推荐** |
| C8 | 8 | 45°旋转 | 需要更细旋转 |
| D8 | 16 | 45°旋转 + 反射 | 高精度需求 |

## 与 SO(2) 对比

| 特性 | D4 (离散) | SO(2) (连续) |
|------|-----------|--------------|
| 非线性 | ReLU ✓ | FourierELU |
| 池化 | GroupPooling (无损) ✓ | NormPool (有损) |
| 训练稳定性 | 高 ✓ | 中等 |
| 预期 mAP | ~0.85+ | ~0.68-0.75 |

## 核心代码解析

### e2wrn.py 关键设计

```python
# 1. 创建 D4 群空间
if f:  # f=True 表示包含反射
    gspace = gspaces.flipRot2dOnR2(N=4)  # D4 群
else:
    gspace = gspaces.rot2dOnR2(N=4)      # C4 群

# 2. 使用 regular_repr (正则表示)
out_type = enn.FieldType(gspace, channels * [gspace.regular_repr])

# 3. 等变卷积
conv = enn.R2Conv(in_type, out_type, kernel_size=3, padding=1)

# 4. 等变批归一化
bn = enn.InnerBatchNorm(out_type)

# 5. 等变 ReLU
relu = enn.ReLU(out_type, inplace=True)

# 6. 最后转为 trivial 以获得不变性
trivial_type = enn.FieldType(gspace, channels * [gspace.trivial_repr])
```

### fixparams 参数

`fixparams=True` 时，会调整通道数使参数量与标准 CNN 相当：

```python
def regular_feature_type(gspace, planes, fixparams=True):
    N = gspace.fibergroup.order()  # D4: N=8
    if fixparams:
        planes *= math.sqrt(N)  # 补偿群大小
    planes = planes / N
    return enn.FieldType(gspace, [gspace.regular_repr] * int(planes))
```

## 预期性能

在 TrashNet 上的预期表现：

| 模型 | val_mAP | 参数量 |
|------|---------|--------|
| 标准 ResNet-18 | ~0.85 | 11.2M |
| D4 ResNet-18 | ~0.87 | ~3M |
| C8 ResNet-18 | ~0.86 | ~2M |
| SO(2) ResNet-18 | ~0.70 | ~4M |

D4 群版本通常能获得更好的泛化性能，同时参数量更少！

## 等变性验证

运行模型文件可以验证等变性：

```bash
python d4_resnet18.py
```

输出示例：
```
Testing D4 Equivariance
  90° rotation:    PASS
  180° rotation:   PASS
  Horizontal flip: PASS
  Vertical flip:   PASS
```
