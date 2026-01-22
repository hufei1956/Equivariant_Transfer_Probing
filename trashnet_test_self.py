import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import ast

# 尝试导入自定义模型
try:
    from escnn_train_v2 import build_equivariant_resnet18
    from d4_resnet18 import d4_resnet18
    # 尝试添加 SRE 路径
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SRE-Conv'))
    from SRE_Conv.sre_resnet import sre_resnet18
except ImportError:
    pass

class TrashnetTesterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trashnet 模型测试工具")
        self.root.geometry("600x450")
        
        self.model_dir = "model"
        self.dataset_dir = "dataset-resized"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.setup_ui()
        self.load_model_list()

    def setup_ui(self):
        # 提示标签
        tk.Label(self.root, text="请选择 trashnet 生成的模型进行验证", font=("Arial", 12)).pack(pady=10)
        tk.Label(self.root, text="(注意：本工具仅用于测试 Trashnet 数据集)", font=("Arial", 10, "italic"), fg="red").pack(pady=5)

        # 模型选择下拉框
        frame_select = tk.Frame(self.root)
        frame_select.pack(pady=10)
        tk.Label(frame_select, text="选择模型文件:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar()
        self.combo_models = ttk.Combobox(frame_select, textvariable=self.model_var, width=40, state="readonly")
        self.combo_models.pack(side=tk.LEFT, padx=5)
        self.combo_models.bind("<<ComboboxSelected>>", self.on_model_selected)

        # 选中的模型显示
        self.lbl_selected = tk.Label(self.root, text="当前未选择模型", fg="blue")
        self.lbl_selected.pack(pady=5)

        # 开始按钮
        self.btn_start = tk.Button(self.root, text="开始检测", command=self.start_testing, state=tk.DISABLED, bg="#dddddd")
        self.btn_start.pack(pady=20, ipadx=20, ipady=5)

        # 状态显示
        self.lbl_status = tk.Label(self.root, text="", font=("Arial", 11))
        self.lbl_status.pack(pady=10)

        # 进度条
        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, length=400, mode='indeterminate')
        self.progress.pack(pady=10)

        # 结果显示区域
        self.result_text = tk.Text(self.root, height=8, width=50, bg="#f0f0f0", state=tk.DISABLED)
        self.result_text.pack(pady=10)

    def load_model_list(self):
        if not os.path.exists(self.model_dir):
            messagebox.showerror("错误", f"找不到模型文件夹: {self.model_dir}")
            return
        
        files = [f for f in os.listdir(self.model_dir) if f.endswith('.pth')]
        if not files:
            self.combo_models['values'] = ["未找到模型文件"]
            self.combo_models.current(0)
            self.btn_start.config(state=tk.DISABLED)
        else:
            self.combo_models['values'] = files
            self.combo_models.current(0)
            self.on_model_selected(None)

    def on_model_selected(self, event):
        selected = self.model_var.get()
        if selected and selected != "未找到模型文件":
            self.lbl_selected.config(text=f"已选择: {selected}")
            self.btn_start.config(state=tk.NORMAL, bg="#4CAF50", fg="white")
        else:
            self.lbl_selected.config(text="当前未选择模型")
            self.btn_start.config(state=tk.DISABLED, bg="#dddddd")

    def start_testing(self):
        model_name = self.model_var.get()
        if not model_name:
            return

        self.btn_start.config(state=tk.DISABLED)
        self.combo_models.config(state=tk.DISABLED)
        self.lbl_status.config(text="检测中... (正在加载模型和数据)", fg="orange")
        self.progress.start(10)
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)

        # 在新线程中运行测试，防止界面卡死
        threading.Thread(target=self.run_inference, args=(model_name,), daemon=True).start()

    def run_inference(self, model_filename):
        try:
            acc, map, error = self.evaluate_model(model_filename)
            
            # 更新 UI 需要在主线程
            self.root.after(0, self.update_ui_finish, acc, map, error)
        except Exception as e:
            self.root.after(0, self.update_ui_finish, None, None, str(e))

    def update_ui_finish(self, acc, map, error):
        self.progress.stop()
        self.btn_start.config(state=tk.NORMAL)
        self.combo_models.config(state="readonly")
        
        if error:
            self.lbl_status.config(text="检测失败", fg="red")
            messagebox.showerror("错误", f"检测过程中发生错误:\n{error}")
        else:
            self.lbl_status.config(text="检测完成!", fg="green")
            result_str = f"模型: {self.model_var.get()}\n\nACC (Top-1): {acc:.2%}\nmAP: {map:.4f}"
            self.result_text.config(state=tk.NORMAL)
            self.result_text.insert(tk.END, result_str)
            self.result_text.config(state=tk.DISABLED)

    def evaluate_model(self, model_filename):
        model_path = os.path.join(self.model_dir, model_filename)
        
        # 1. 确定类别
        classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        num_classes = len(classes)

        # 2. 构建模型结构
        # 简单的启发式规则判断模型类型
        lower_name = model_filename.lower()
        if "d4" in lower_name:
            try:
                model = d4_resnet18(num_classes=num_classes)
            except NameError:
                return None, None, "未找到 d4_resnet18 定义"
        elif "sre" in lower_name:
            try:
                model = sre_resnet18(num_classes=num_classes)
            except NameError:
                return None, None, "未找到 sre_resnet18 定义"
        elif "escnn" in lower_name:
            # 尝试推断参数
            group_type = "C4"
            if "c8" in lower_name: group_type = "C8"
            if "so2" in lower_name: group_type = "SO2"
            
            # 尝试加载 checkpoint 获取更准确的参数
            try:
                ckpt = torch.load(model_path, map_location='cpu')
                if isinstance(ckpt, dict):
                    group_type = ckpt.get("group_type", group_type)
            except:
                pass
                
            try:
                model = build_equivariant_resnet18(
                    num_classes=num_classes, 
                    group_type=str(group_type),
                    maximum_frequency=4, # 默认值
                    base_channels=16     # 默认值
                )
            except NameError:
                return None, None, "未找到 build_equivariant_resnet18 定义"
        else:
            # 默认 ResNet18
            try:
                model = models.resnet18(weights=None)
            except:
                model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        # 3. 加载权重
        model = model.to(self.device)
        try:
            ckpt = torch.load(model_path, map_location=self.device)
            state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            return None, None, f"加载权重失败: {e}"
        
        model.eval()

        # 4. 准备数据
        if not os.path.exists(self.dataset_dir):
            return None, None, f"找不到数据集目录: {self.dataset_dir}"

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        try:
            dataset = datasets.ImageFolder(root=self.dataset_dir, transform=transform)
        except Exception as e:
            return None, None, f"加载数据集失败: {e}"

        # 简单验证类别是否匹配
        if len(dataset.classes) != num_classes:
            return None, None, f"数据集类别数量({len(dataset.classes)})与模型({num_classes})不匹配"

        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

        # 5. 推理
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device)
                outputs = model(images)
                all_logits.append(outputs.cpu())
                all_targets.append(targets.cpu())

        logits = torch.cat(all_logits)
        targets = torch.cat(all_targets)

        # 6. 计算指标
        acc = self.calculate_accuracy(logits, targets)
        map_score = self.calculate_map(logits, targets)

        return acc, map_score, None

    def calculate_accuracy(self, logits, targets):
        preds = logits.argmax(dim=1)
        correct = (preds == targets).sum().item()
        return correct / len(targets)

    def calculate_map(self, logits, targets):
        # 简化版 mAP 计算
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        aps = []
        for i in range(num_classes):
            target_i = (targets == i).float()
            score_i = probs[:, i]
            
            if target_i.sum() == 0:
                aps.append(0)
                continue
                
            sorted_indices = torch.argsort(score_i, descending=True)
            sorted_targets = target_i[sorted_indices]
            
            tp = torch.cumsum(sorted_targets, dim=0)
            fp = torch.cumsum(1 - sorted_targets, dim=0)
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / target_i.sum()
            
            ap = torch.trapz(precision, recall).item()
            aps.append(ap)
            
        return sum(aps) / len(aps)

if __name__ == "__main__":
    root = tk.Tk()
    app = TrashnetTesterApp(root)
    root.mainloop()
