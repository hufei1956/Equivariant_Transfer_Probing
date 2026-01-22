import os
import ast
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms
from escnn_train_v2 import build_equivariant_resnet18


def build_resnet18(num_classes: int) -> nn.Module:
    try:
        model = models.resnet18(weights=None)
    except AttributeError:
        model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_eval_transform(img_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_class_order_from_file(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    try:
        parsed = ast.literal_eval(content)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip().lower() for x in parsed if str(x).strip()]
    except Exception:
        pass
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    out = []
    for ln in lines:
        if ln.startswith("[") and ln.endswith("]") and "," in ln:
            parts = ln[1:-1].split(",")
            out.extend([p.strip().strip("'").strip('"').lower() for p in parts if p.strip()])
        else:
            out.append(ln.lower())
    return out


def infer_image(model: nn.Module, img_path: str, transform, device, class_names):
    from PIL import Image as PILImage
    img = PILImage.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0)
        top_idx = int(torch.argmax(probs).item())
        top_prob = float(probs[top_idx].item())
        pred_name = class_names[top_idx] if 0 <= top_idx < len(class_names) else str(top_idx)
    return pred_name, top_prob


def build_model_from_checkpoint(model_path: str, device):
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    ckpt_classes = []
    if isinstance(ckpt, dict) and "classes" in ckpt and isinstance(ckpt["classes"], (list, tuple)):
        ckpt_classes = [str(x).lower() for x in ckpt["classes"]]
    base_dir = os.path.dirname(model_path)
    fallback_files = []
    fname = os.path.basename(model_path).lower()
    if "trashnet" in fname:
        fallback_files.append(os.path.join(base_dir, "trashnet_list.txt"))
    if "realwaste" in fname:
        fallback_files.append(os.path.join(base_dir, "realwaste_list.txt"))
        fallback_files.append(os.path.join(base_dir, "realwaset_list.txt"))
    fallback_files.append(os.path.join(base_dir, "classes.txt"))
    file_order = []
    for fp in fallback_files:
        file_order = load_class_order_from_file(fp)
        if file_order:
            break
    if len(ckpt_classes) > 0:
        class_order = ckpt_classes
    elif len(file_order) > 0:
        class_order = file_order
    else:
        if isinstance(state, dict) and "fc.weight" in state:
            out_feats = state["fc.weight"].shape[0]
            class_order = [str(i) for i in range(out_feats)]
        else:
            class_order = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    is_escnn = (isinstance(ckpt, dict) and "group_type" in ckpt) or ("escnn" in fname)
    if is_escnn:
        group_type = None
        max_frequency = None
        base_channels = None
        if isinstance(ckpt, dict):
            group_type = ckpt.get("group_type", None)
            max_frequency = ckpt.get("max_frequency", None)
            base_channels = ckpt.get("base_channels", None)
        if group_type is None:
            suffix = fname.split("best_escnn_")[-1].split("_")[0]
            if suffix.startswith("so2f"):
                group_type = "SO2"
                try:
                    max_frequency = int(suffix.replace("so2f", ""))
                except Exception:
                    max_frequency = 4
            else:
                group_type = suffix.upper()
        if max_frequency is None:
            max_frequency = 4
        if base_channels is None:
            base_channels = 16
        model = build_equivariant_resnet18(
            num_classes=len(class_order),
            group_type=str(group_type),
            maximum_frequency=int(max_frequency),
            base_channels=int(base_channels),
        ).to(device)
    else:
        model = build_resnet18(num_classes=len(class_order)).to(device)
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state, strict=False)
    model.eval()
    return model, class_order


class VisualApp:
    def __init__(self, root):
        self.root = root
        self.root.title("分类可视化")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_eval_transform(224)
        self.model_paths = self._scan_models()
        self.model = None
        self.class_names = []
        self.image_path = None
        self.left_img = None

        left_frame = tk.Frame(root, width=500, height=500)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_frame = tk.Frame(root, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas = tk.Label(left_frame, bg="#222")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        tk.Label(right_frame, text="选择模型文件").pack(anchor="w", padx=8, pady=4)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(right_frame, textvariable=self.model_var, values=self.model_paths, state="readonly")
        if self.model_paths:
            self.model_combo.current(0)
        self.model_combo.pack(fill=tk.X, padx=8)
        tk.Button(right_frame, text="加载模型", command=self.load_selected_model).pack(fill=tk.X, padx=8, pady=6)

        tk.Label(right_frame, text="选择图片").pack(anchor="w", padx=8, pady=4)
        tk.Button(right_frame, text="打开图片", command=self.open_image).pack(fill=tk.X, padx=8, pady=2)
        self.path_label = tk.Label(right_frame, text="", fg="#888")
        self.path_label.pack(fill=tk.X, padx=8)

        tk.Button(right_frame, text="预测", command=self.predict).pack(fill=tk.X, padx=8, pady=12)
        self.result_label = tk.Label(right_frame, text="结果", font=("Arial", 12))
        self.result_label.pack(fill=tk.X, padx=8, pady=4)

    def _scan_models(self):
        mdir = os.path.join(os.getcwd(), "model")
        paths = []
        if os.path.isdir(mdir):
            for fn in os.listdir(mdir):
                if fn.lower().endswith(".pth"):
                    paths.append(os.path.join(mdir, fn))
        return paths

    def load_selected_model(self):
        sel = self.model_var.get()
        if not sel or not os.path.isfile(sel):
            self.result_label.config(text="未选择模型文件")
            return
        self.model, self.class_names = build_model_from_checkpoint(sel, self.device)
        self.result_label.config(text=f"模型已加载：{os.path.basename(sel)}\n类别数：{len(self.class_names)}")

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp"), ("All files", "*.*")])
        if not path:
            return
        self.image_path = path
        self.path_label.config(text=path)
        img = Image.open(path).convert("RGB")
        w, h = img.size
        maxw, maxh = 500, 500
        scale = min(maxw / max(1, w), maxh / max(1, h))
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.LANCZOS)
        self.left_img = ImageTk.PhotoImage(img)
        self.canvas.config(image=self.left_img)

    def predict(self):
        if self.model is None:
            self.result_label.config(text="请先加载模型")
            return
        if not self.image_path:
            self.result_label.config(text="请先选择图片")
            return
        name, prob = infer_image(self.model, self.image_path, self.transform, self.device, self.class_names)
        self.result_label.config(text=f"预测类别：{name}\n置信度：{prob:.4f}")


def main():
    root = tk.Tk()
    app = VisualApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

