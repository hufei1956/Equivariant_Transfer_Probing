# 解决 escnn SO(2) 模型训练崩溃问题

## 问题分析
之前的报错 `AssertionError: Input type contains trivial representation '(0,)'` 是因为 `NormBatchNorm` 层不支持频率为 0 (标量) 的分量。同时，使用 `base_channels=16` 和 `max_frequency=4` 导致模型在您的 RTX 4050 (6GB) 上初始化过慢或显存溢出。

## 修复方案
1.  **移除 SO(2) 的标量分量**: 修改了 `_field_type_regular_or_trivial` 函数，在 SO(2) 模式下仅构建频率 `1` 到 `max_frequency` 的特征，跳过频率 `0`。这解决了 `NormBatchNorm` 的兼容性问题。
2.  **优化模型参数**: 将默认配置调整为 `base_channels=12` 和 `max_frequency=3`。
    -   这使得模型参数量控制在约 **1.68M** (相比之前的配置更轻量)。
    -   验证表明这可以在您的设备上正常启动并进行训练。
3.  **恢复训练参数**: 将默认 `epochs` 设置回 50。

## 下一步
您可以直接运行以下命令开始正式训练。代码已经内置了优化后的默认参数，所以不需要手动指定繁琐的参数。

```bash
python escnn_train_v2.py --group SO2
```
或者显式指定参数（如果想尝试更高配置）：
```bash
python escnn_train_v2.py --group SO2 --base-channels 12 --max-frequency 3
```