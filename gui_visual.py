import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from Grad_CAM.gradcam_visualization_C8 import GradCAMVisualizerC8

import matplotlib.cm as cm


def _load_checkpoint_classes(model_path: str) -> List[str]:
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "classes" in ckpt and isinstance(ckpt["classes"], (list, tuple)):
        return [str(x) for x in ckpt["classes"]]
    return []


def _parse_label_from_filename(filename: str) -> Optional[str]:
    m = re.search(r"(?:^|[-_])(tn|rw)_([a-zA-Z]+)", filename)
    if not m:
        return None
    return m.group(2).lower()


def _to_overlay(pil_img: Image.Image, cam_map_224: np.ndarray, alpha: float = 0.45, cmap_name: str = "jet") -> Image.Image:
    img = pil_img.convert("RGB")
    img_np = np.asarray(img).astype(np.float32) / 255.0

    cam_map_224 = np.asarray(cam_map_224, dtype=np.float32)
    cam_map_224 = np.clip(cam_map_224, 0.0, 1.0)
    cam_img = Image.fromarray((cam_map_224 * 255).astype(np.uint8)).resize(img.size, resample=Image.BILINEAR)
    cam_np = np.asarray(cam_img).astype(np.float32) / 255.0

    cmap = cm.get_cmap(cmap_name)
    heat = cmap(cam_np)[..., :3].astype(np.float32)
    overlay = img_np * (1.0 - alpha) + heat * alpha
    overlay = np.clip(overlay, 0.0, 1.0)
    return Image.fromarray((overlay * 255).astype(np.uint8))


def generate_heatmap_grid(
    model_a_path: str,
    model_b_path: str,
    image_paths: List[str],
    output_dir: str,
    device: str = "cuda",
    cell_size: int = 256,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    classes_a = _load_checkpoint_classes(model_a_path)
    classes_b = _load_checkpoint_classes(model_b_path)
    num_classes_a = len(classes_a) if classes_a else 6
    num_classes_b = len(classes_b) if classes_b else 9

    viz_a = GradCAMVisualizerC8(model_a_path, num_classes=num_classes_a, device=device)
    viz_b = GradCAMVisualizerC8(model_b_path, num_classes=num_classes_b, device=device)

    map_a = {c.lower(): i for i, c in enumerate(classes_a)}
    map_b = {c.lower(): i for i, c in enumerate(classes_b)}

    rows = 3
    cols = 6
    canvas = Image.new("RGB", (cols * cell_size, rows * cell_size), (255, 255, 255))

    selected = list(image_paths)[: 2 * rows]
    if len(selected) < 2 * rows:
        selected = selected + [None] * (2 * rows - len(selected))

    for i, img_path in enumerate(selected):
        r = i // 2
        block = (i % 2) * 3
        if img_path is None:
            continue

        pil_img = Image.open(img_path).convert("RGB")
        label = _parse_label_from_filename(os.path.basename(img_path))
        target_a = map_a.get(label) if label else None
        target_b = map_b.get(label) if label else None

        for sub in range(3):
            c = block + sub
            x0 = c * cell_size
            y0 = r * cell_size

            if sub == 0:
                out_img = pil_img
            elif sub == 1:
                cam_a, _, _, _ = viz_a.generate_cam(img_path, target_class=target_a)
                out_img = _to_overlay(pil_img, cam_a)
            else:
                cam_b, _, _, _ = viz_b.generate_cam(img_path, target_class=target_b)
                out_img = _to_overlay(pil_img, cam_b)

            out_img = out_img.resize((cell_size, cell_size), resample=Image.BILINEAR)
            canvas.paste(out_img, (x0, y0))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"heatmap_grid_{ts}.png")
    canvas.save(out_path)
    return out_path


class HeatmapGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Heatmap Generator")

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.default_output_dir = os.path.join(self.base_dir, "Grad_CAM", "visualizations")

        self.model_a_var = tk.StringVar(value=os.path.join(self.base_dir, "model", "best_escnn_C8_trashnet.pth"))
        self.model_b_var = tk.StringVar(value=os.path.join(self.base_dir, "model", "best_escnn_C8_realwaste.pth"))
        self.output_dir_var = tk.StringVar(value=self.default_output_dir)

        self.device_var = tk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")
        self.status_var = tk.StringVar(value="就绪")

        self.image_paths: List[str] = []
        self.preview_imgtk = None

        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        row = 0
        ttk.Label(frm, text="模型 A (TN-trained)").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.model_a_var, width=80).grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(frm, text="选择", command=self._choose_model_a).grid(row=row, column=2, sticky="e")

        row += 1
        ttk.Label(frm, text="模型 B (RW-trained)").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.model_b_var, width=80).grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(frm, text="选择", command=self._choose_model_b).grid(row=row, column=2, sticky="e")

        row += 1
        ttk.Label(frm, text="输出目录").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.output_dir_var, width=80).grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(frm, text="选择", command=self._choose_output_dir).grid(row=row, column=2, sticky="e")

        row += 1
        ttk.Label(frm, text="设备").grid(row=row, column=0, sticky="w")
        ttk.Combobox(frm, textvariable=self.device_var, values=["cuda", "cpu"], width=10, state="readonly").grid(row=row, column=1, sticky="w", padx=6)

        row += 1
        ttk.Label(frm, text="选择 6 张图片 (输出 3 行 6 列)").grid(row=row, column=0, sticky="w")
        btns = ttk.Frame(frm)
        btns.grid(row=row, column=1, sticky="w", padx=6)
        ttk.Button(btns, text="添加图片", command=self._choose_images).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(btns, text="清空", command=self._clear_images).grid(row=0, column=1)

        row += 1
        self.listbox = tk.Listbox(frm, height=8, width=100, selectmode=tk.SINGLE)
        self.listbox.grid(row=row, column=0, columnspan=3, sticky="ew")
        self.listbox.bind("<<ListboxSelect>>", self._on_select_image)

        row += 1
        action = ttk.Frame(frm)
        action.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Button(action, text="生成热力图", command=self._on_generate).grid(row=0, column=0, sticky="w")
        ttk.Label(action, textvariable=self.status_var).grid(row=0, column=1, sticky="w", padx=10)

        row += 1
        self.preview_label = ttk.Label(frm)
        self.preview_label.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(10, 0))

        frm.columnconfigure(1, weight=1)

    def _choose_model_a(self):
        path = filedialog.askopenfilename(title="选择模型 A", filetypes=[("PyTorch model", "*.pth"), ("All", "*.*")])
        if path:
            self.model_a_var.set(path)

    def _choose_model_b(self):
        path = filedialog.askopenfilename(title="选择模型 B", filetypes=[("PyTorch model", "*.pth"), ("All", "*.*")])
        if path:
            self.model_b_var.set(path)

    def _choose_output_dir(self):
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_dir_var.set(path)

    def _choose_images(self):
        paths = filedialog.askopenfilenames(title="选择图片 (最多 6 张)", filetypes=[("Images", "*.jpg;*.jpeg;*.png"), ("All", "*.*")])
        if not paths:
            return
        remain = 6 - len(self.image_paths)
        to_add = list(paths)[:max(0, remain)]
        self.image_paths.extend(to_add)
        self._refresh_listbox()

    def _clear_images(self):
        self.image_paths = []
        self._refresh_listbox()
        self.preview_label.configure(image="")
        self.preview_imgtk = None

    def _refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for p in self.image_paths:
            self.listbox.insert(tk.END, p)
        self.status_var.set(f"已选择图片: {len(self.image_paths)}/6")

    def _on_select_image(self, _evt=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if idx < 0 or idx >= len(self.image_paths):
            return
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((900, 350), resample=Image.BILINEAR)
            self.preview_imgtk = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=self.preview_imgtk)
        except Exception:
            return

    def _on_generate(self):
        if len(self.image_paths) != 6:
            messagebox.showerror("错误", "请先选择 6 张图片。")
            return
        model_a = self.model_a_var.get().strip()
        model_b = self.model_b_var.get().strip()
        out_dir = self.output_dir_var.get().strip()
        device = self.device_var.get().strip()

        if not os.path.exists(model_a):
            messagebox.showerror("错误", "模型 A 路径不存在。")
            return
        if not os.path.exists(model_b):
            messagebox.showerror("错误", "模型 B 路径不存在。")
            return
        if not out_dir:
            messagebox.showerror("错误", "输出目录为空。")
            return

        self.status_var.set("生成中...")
        self.root.update_idletasks()

        def worker():
            try:
                out_path = generate_heatmap_grid(
                    model_a_path=model_a,
                    model_b_path=model_b,
                    image_paths=self.image_paths,
                    output_dir=out_dir,
                    device=device,
                    cell_size=256,
                )
                self.root.after(0, lambda: self._on_done(out_path))
            except Exception as e:
                self.root.after(0, lambda: self._on_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self, out_path: str):
        self.status_var.set(f"已生成: {out_path}")
        try:
            img = Image.open(out_path).convert("RGB")
            img.thumbnail((900, 500), resample=Image.BILINEAR)
            self.preview_imgtk = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=self.preview_imgtk)
        except Exception:
            pass

    def _on_error(self, msg: str):
        self.status_var.set("失败")
        messagebox.showerror("生成失败", msg)


def main():
    root = tk.Tk()
    HeatmapGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
