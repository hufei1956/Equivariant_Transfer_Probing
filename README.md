# Baseline and Equivariant ResNet-18 for Waste Classification

本目录提供 ResNet-18 基线与基于群等变卷积的 ResNet-18 变体（C4、C8、D4、SO(2)），用于在 TrashNet 与 RealWaste 两个垃圾分类数据集上进行训练、跨域评估和可视化分析（Grad-CAM）。

## 目录结构

- 训练与模型
  - 基线训练：[train.py](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/train.py)
  - 等变训练（D4/C4/C8/D8）：[train_d4_resnet18.py](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/train_d4_resnet18.py)、网络定义 [d4_resnet18.py](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/d4_resnet18.py)
  - ESCNN 训练（C4/C8/SO(2)）：[escnn_train_v2.py](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/escnn_train_v2.py)、[escnn_train_v3_so2.py](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/escnn_train_v3_so2.py)
  - 已训练模型：请见 [model/](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/model) 与 [checkpoints/](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/checkpoints)
    - 示例：best_escnn_C8_{trashnet|realwaste}.pth、best_d4_D4_{trashnet|realwaste}.pth、best_resnet18_ablation_{dataset}.pth
    - 类别顺序文件：trashnet_list.txt、realwaste_list.txt（测试与可视化对齐类别索引）

- 数据与划分
  - 数据集接口与划分：[dataset.py](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/dataset.py)
  - 训练/验证/测试划分文件：见 [splits/](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/splits)

- 测试与跨域评估
  - TrashNet 测试 GUI：[trashnet_test_self.py](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/trashnet_test_self.py)
  - 跨域评估（T→R / R→T）：[Trashnetmodel_test_realwaste.py](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/Trashnetmodel_test_realwaste.py)、[Realwastemodel_test_trashnet.py](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/Realwastemodel_test_trashnet.py)
  - 消融评估：[ablation_TN_Test_RW.py](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/ablation_TN_Test_RW.py)、[ablation_RW_Test_TN.py](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/ablation_RW_Test_TN.py)

- Grad-CAM 可视化
  - 基线可视化：[Grad_CAM/gradcam_visualization.py](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/Grad_CAM/gradcam_visualization.py)
  - C8 等变模型可视化：[Grad_CAM/gradcam_visualization_C8.py](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/Grad_CAM/gradcam_visualization_C8.py)
  - 标注解析与掩码：Label Studio 导出 JSON + 掩码生成 [Grad_CAM/download_pic_labelstudio.py](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/Grad_CAM/download_pic_labelstudio.py)
  - 说明文档：Grad-CAM 指南 [Grad_CAM/GradCAM_analysis_guide.md](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/Grad_CAM/GradCAM_analysis_guide.md)、流程说明 [Grad_CAM/figure3_workflow_complete.md](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/Grad_CAM/figure3_workflow_complete.md)
  - GUI 工具：交互式生成 3×6 热力图网格 [gui_visual.py](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/gui_visual.py)

## 环境与依赖

- Python ≥ 3.10（Windows 建议 num_workers=0）
- PyTorch、torchvision（注意 torchvision 预训练权重 API：`weights=None` 新版写法）
- escnn（群等变卷积库）：用于 C4/C8/D4/SO(2) 网络
- matplotlib、pillow、numpy、pandas
- 可选：pytorch-grad-cam（用于标准 CNN；ESCNN 已改为自定义 autograd 计算）

示例安装（conda 环境中）：
```
pip install torch torchvision escnn pillow matplotlib numpy pandas
```

## 数据准备

- TrashNet（`dataset-resized/`）与 RealWaste（`RealWaste/`）目录，建议统一到项目根下
- 首次运行训练脚本会在 [splits/](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/splits) 写入 70/15/15 划分 JSON，后续复用，保证可复现

## 训练

- ResNet-18 基线（示例）：
```
python train.py --dataset-name trashnet --epochs 100 --batch-size 32
```
模型保存至 `checkpoints/`（例如 `best_resnet18.pth`），日志写入 `logs/`。

- 等变训练（D4/C4/C8/D8）：
```
python train_d4_resnet18.py --group C8 --dataset-name trashnet --epochs 100
```
输出模型命名示例：`best_d4_D4_trashnet.pth` / `best_escnn_C8_trashnet.pth` 等。

- ESCNN 与 SO(2) 连续群（频率截断）：
```
python escnn_train_v3_so2.py --group SO2 --max-frequency 4 --dataset-name trashnet
```

## 测试与跨域评估

- 同域 GUI 测试：
```
python trashnet_test_self.py
```
从 `model/` 选择 `.pth` 模型，评估 Top-1 与 mAP。

- 跨域评估（T→R / R→T）：
```
python Trashnetmodel_test_realwaste.py
python Realwastemodel_test_trashnet.py
```

## 消融实验

- 训练小数据子集：
```
python train_ablation.py --dataset-name trashnet --subset small
```
保存为 `best_resnet18_ablation_{dataset}.pth`。

- 跨域消融评估：
```
python ablation_TN_Test_RW.py      # 用 TrashNet 消融模型测 RealWaste 小集
python ablation_RW_Test_TN.py      # 用 RealWaste 消融模型测 TrashNet 小集
```
结果写入 `ablation_*_results.txt`。

## Grad-CAM 可视化与 AC 指标

- 基线/等变 Grad-CAM：
```
python Grad_CAM/gradcam_visualization.py
python Grad_CAM/gradcam_visualization_C8.py --mode batch   # 导出 AC CSV
python Grad_CAM/gradcam_visualization_C8.py --mode figure  # 导出论文 Figure 3 热力图
```
导出位置见 [Grad_CAM/visualizations/](file:///d:/13_Project_Code/Project_CV_github/Multi_Mode_forCV_WHIT/4_Baseline_Tarshnet_resnet18/Grad_CAM/visualizations)（如 `ac_statistics_C8.csv`、`figure3_C8.png`）。

- GUI 热力图生成（3 行 6 列，按样本行排版）：
```
python gui_visual.py
```
选择两套模型与 6 张图片，生成 `heatmap_grid_YYYYMMDD_HHMMSS.png`。

## 模型文件与命名规范

- `model/`：已训练模型与类别列表文件（trashnet_list.txt / realwaste_list.txt）
- `checkpoints/`：训练过程保存的最佳权重
- 命名规则：`best_{架构/群}_{数据集}.pth`；部分脚本保存为 `best_resnet18_ablation_{dataset}.pth`

## 结果与日志

- 日志文件：`logs/*.csv`（每轮训练的 acc/mAP、学习率等）
- 训练记录：`训练记录/` 下按时间归档的文本记录，便于复盘

## 常见问题

- Windows 建议 DataLoader `num_workers=0`（避免多进程问题）
- torchvision 预训练权重请使用新版 API：`models.resnet18(weights=None)`
- ESCNN 兼容 Grad-CAM：本项目在 C8 可视化中使用 `autograd.grad` 手写 Grad-CAM，避免 inplace/backward 冲突

---

如需复现实验或扩展到更多群（如 C16/D8），可参考以上训练脚本与命名规范，保持类别顺序文件与评估脚本一致。*** End Patch***
*** End Patch
*** End Patch
*** End Patch
*** End Patch
