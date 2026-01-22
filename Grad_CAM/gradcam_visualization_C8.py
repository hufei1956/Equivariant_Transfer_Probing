"""
Grad-CAM Visualization Code for ESCNN C8 Models (Paper Figure 3)
================================================================

This script generates the attention visualizations for ESCNN C8 models.
It compares attention patterns between TrashNet-trained (C8) and RealWaste-trained (C8) models.

Requirements:
    pip install torch torchvision pytorch-grad-cam pillow matplotlib numpy pandas opencv-python escnn --break-system-packages

Usage:
    python gradcam_visualization_C8.py
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json
import argparse
from pathlib import Path
from escnn import nn as enn
import re

# 添加项目根目录到 sys.path 以便导入 escnn_train_v2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(PROJECT_ROOT)

try:
    from escnn_train_v2_c4c8_gradcam import build_equivariant_resnet18
except ImportError as e:
    print(f"Error importing escnn_train_v2_c4c8_gradcam: {e}")
    sys.exit(1)

# --- Configuration ---
DEFAULT_TN_MODEL = os.path.join(PROJECT_ROOT, 'model', 'best_escnn_C8_trashnet.pth')
DEFAULT_RW_MODEL = os.path.join(PROJECT_ROOT, 'model', 'best_escnn_C8_realwaste.pth')
DEFAULT_SAMPLES_DIR = os.path.join(BASE_DIR, 'aligned_images')
DEFAULT_JSON_PATH = os.path.join(BASE_DIR, 'label', 'project-4-at-2026-01-16-15-58-932da740.json')
DEFAULT_MASKS_DIR = os.path.join(BASE_DIR, 'masks')
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, 'visualizations')

TARGET_CLASSES_5 = ["cardboard", "glass", "metal", "paper", "plastic"]


def parse_labelstudio_json(json_path, output_dir, images_dir):
    """
    Parse Label Studio JSON and generate binary masks for images.
    """
    print(f"Parsing annotations from {json_path}...")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    count = 0
    for task in tasks:
        # Get image filename
        img_url = task.get('data', {}).get('image') or task.get('data', {}).get('img')
        if not img_url:
            continue
            
        filename = os.path.basename(img_url)
        image_path = os.path.join(images_dir, filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {filename} not found in {images_dir}")
            continue
            
        # Open image to get dimensions
        try:
            with Image.open(image_path) as img:
                w, h = img.size
        except Exception as e:
            print(f"Error opening image {filename}: {e}")
            continue

        # Create mask
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        
        has_annotation = False
        for annotation in task.get('annotations', []):
            for result in annotation.get('result', []):
                # Handle Polygon Labels
                if result.get('type') == 'polygonlabels':
                    points = result.get('value', {}).get('points', [])
                    if not points:
                        continue
                    
                    pixels = []
                    for pt in points:
                        x = pt[0] * w / 100.0
                        y = pt[1] * h / 100.0
                        pixels.append((x, y))
                    
                    if len(pixels) > 2:
                        draw.polygon(pixels, outline=255, fill=255)
                        has_annotation = True
                
                # Handle Rectangle Labels
                elif result.get('type') == 'rectanglelabels':
                    value = result.get('value', {})
                    x = value.get('x', 0) * w / 100.0
                    y = value.get('y', 0) * h / 100.0
                    width = value.get('width', 0) * w / 100.0
                    height = value.get('height', 0) * h / 100.0
                    
                    draw.rectangle([x, y, x + width, y + height], outline=255, fill=255)
                    has_annotation = True

        if has_annotation:
            mask_filename = filename.rsplit('.', 1)[0] + '_mask.png'
            mask.save(os.path.join(output_dir, mask_filename))
            count += 1
            
    print(f"Generated {count} masks in {output_dir}")

def _load_checkpoint_classes(model_path: str):
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "classes" in ckpt and isinstance(ckpt["classes"], (list, tuple)):
        return [str(x) for x in ckpt["classes"]]
    return None

def _parse_label_from_filename(filename: str):
    m = re.search(r"(?:^|[-_])(tn|rw)_([a-zA-Z]+)", filename)
    if not m:
        return None
    return m.group(2).lower()

def _overlay_heatmap(pil_img: Image.Image, cam_map_224: np.ndarray, alpha: float = 0.45, cmap: str = "jet"):
    img = pil_img.convert("RGB")
    img_np = np.asarray(img).astype(np.float32) / 255.0

    cam = np.clip(cam_map_224, 0.0, 1.0)
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(img.size, resample=Image.BILINEAR)
    cam_np = np.asarray(cam_img).astype(np.float32) / 255.0

    cm = plt.get_cmap(cmap)
    heat = cm(cam_np)[..., :3].astype(np.float32)
    overlay = img_np * (1.0 - alpha) + heat * alpha
    overlay = np.clip(overlay, 0.0, 1.0)
    return img_np, overlay


class _EscnnCamWrapper(nn.Module):
    def __init__(self, escnn_model: nn.Module):
        super().__init__()
        self.m = escnn_model
        self.cam_identity = nn.Identity()

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = enn.GeometricTensor(x, self.m.in_type)

        x = self.m.conv1(x)
        if self.m.bn1 is not None:
            x = self.m.bn1(x)
        x = self.m.relu1(x)
        if self.m.maxpool is not None:
            x = self.m.maxpool(x)

        for block in self.m.layer1:
            x = block(x)
        for block in self.m.layer2:
            x = block(x)
        for block in self.m.layer3:
            x = block(x)
        for block in self.m.layer4:
            x = block(x)

        x = self.m.gpool(x)
        x = x.tensor
        feats = self.cam_identity(x)

        x = self.m.avgpool(feats)
        x = torch.flatten(x, 1)
        logits = self.m.fc(x)
        if return_features:
            return feats, logits
        return logits


class GradCAMVisualizerC8:
    """Generate Grad-CAM visualizations for ESCNN C8 models"""
    
    def __init__(self, model_path, num_classes, device='cuda'):
        """
        Args:
            model_path: Path to trained model checkpoint
            num_classes: Number of output classes (TrashNet: 6, RealWaste: 9)
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Loading C8 model from {model_path} on {self.device} (num_classes={num_classes})...", flush=True)

        self.model_path = model_path
        self.ckpt_classes = _load_checkpoint_classes(model_path) or []
        self.class_to_index = {c.lower(): i for i, c in enumerate(self.ckpt_classes)}

        # Load ESCNN C8 model
        # 注意：这里我们强制使用 C8 配置
        self.model = build_equivariant_resnet18(
            num_classes=num_classes,
            group_type="C8",
            maximum_frequency=8, # 默认值
            base_channels=16     # 默认值
        )
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            
            # 加载权重
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model weights: {e}")
            raise e
            
        self.model.to(self.device)
        self.model.eval()

        self.model = _EscnnCamWrapper(self.model).to(self.device)
        self.model.eval()
        print(f"Model ready: {os.path.basename(model_path)}", flush=True)
        
        # 强制将所有 ReLU 模块的 inplace 设置为 False
        for module in self.model.modules():
            if hasattr(module, 'inplace'):
                module.inplace = False
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def generate_cam(self, image_path, target_class=None):
        """
        Generate Grad-CAM heatmap for an image
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image.resize((224, 224))) / 255.0
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad_(True)
        
        # Get prediction
        feats, output = self.model(input_tensor, return_features=True)
        probs = torch.softmax(output, dim=1)
        confidence, pred_class = probs.max(dim=1)
        pred_class = int(pred_class.item())
        confidence = float(confidence.item())

        class_idx = pred_class if target_class is None else int(target_class)
        score = output[:, class_idx].sum()
        grads = torch.autograd.grad(score, feats, retain_graph=False, create_graph=False)[0]

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * feats).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_map = cam

        return cam_map, pred_class, confidence, image_np
    
    def compute_attention_concentration(self, cam_map, object_mask):
        """
        Compute attention concentration metric (AC)
        """
        # Ensure same size
        if cam_map.shape != object_mask.shape:
            from scipy.ndimage import zoom
            scale_factors = (object_mask.shape[0] / cam_map.shape[0],
                           object_mask.shape[1] / cam_map.shape[1])
            cam_map = zoom(cam_map, scale_factors, order=1)
        
        # Normalize CAM
        cam_normalized = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
        
        # Binarize mask
        binary_mask = (object_mask > 0).astype(float)
        
        # Compute AC
        attention_on_object = (cam_normalized * binary_mask).sum()
        total_attention = cam_normalized.sum()
        ac_score = attention_on_object / (total_attention + 1e-8)
        
        return ac_score


def batch_compute_ac_statistics(samples_dir, masks_dir, tn_model_path, 
                                rw_model_path, output_csv='ac_statistics_C8.csv', max_images: int = 0, device: str = "cuda"):
    """
    Compute AC statistics for all samples (for Table 5 in the paper)
    """
    import pandas as pd
    
    # 初始化模型 - 注意类别数不同
    # TrashNet 通常是 6 类 (glass, paper, cardboard, plastic, metal, trash)
    # RealWaste 通常是 9 类
    tn_viz = GradCAMVisualizerC8(tn_model_path, num_classes=6, device=device)
    rw_viz = GradCAMVisualizerC8(rw_model_path, num_classes=9, device=device)
    
    results = []
    
    files = sorted([f for f in os.listdir(samples_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if isinstance(max_images, int) and max_images > 0:
        files = files[:max_images]
    
    for image_file in files:
        image_path = os.path.join(samples_dir, image_file)
        
        # Look for mask (png or npy)
        mask_name_base = os.path.splitext(image_file)[0] + '_mask'
        mask_path = os.path.join(masks_dir, mask_name_base + '.png')
        
        if not os.path.exists(mask_path):
             mask_path = os.path.join(masks_dir, mask_name_base + '.npy')
             if not os.path.exists(mask_path):
                print(f"⚠️  Mask not found for {image_file}, skipping...")
                continue
        
        # Load mask
        if mask_path.endswith('.npy'):
            mask = np.load(mask_path)
        else:
            mask_img = Image.open(mask_path).convert('L').resize((224, 224), Image.NEAREST)
            mask = np.array(mask_img)
        
        # Generate CAMs and compute AC
        try:
            tn_cam, tn_pred, tn_conf, _ = tn_viz.generate_cam(image_path)
            rw_cam, rw_pred, rw_conf, _ = rw_viz.generate_cam(image_path)
            
            tn_ac = tn_viz.compute_attention_concentration(tn_cam, mask)
            rw_ac = rw_viz.compute_attention_concentration(rw_cam, mask)
            
            # Determine dataset origin
            dataset = 'TrashNet' if 'tn_' in image_file.lower() else 'RealWaste'
            
            results.append({
                'image': image_file,
                'dataset': dataset,
                'tn_pred': tn_pred,
                'tn_conf': tn_conf,
                'tn_ac': tn_ac,
                'rw_pred': rw_pred,
                'rw_conf': rw_conf,
                'rw_ac': rw_ac
            })
            
            print(f"✓ Processed {image_file}: TN_AC={tn_ac:.3f}, RW_AC={rw_ac:.3f}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            import traceback
            traceback.print_exc()
    
    if not results:
        print("No results generated.")
        return None

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved statistics: {output_csv}")
    
    # Compute summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS FOR TABLE 5 (C8 Models)")
    print("="*60)
    
    for dataset in ['TrashNet', 'RealWaste']:
        subset = df[df['dataset'] == dataset]
        if len(subset) == 0: continue
        
        print(f"\n{dataset} samples ({len(subset)}):")
        print(f"  TN Model (C8) - Mean AC: {subset['tn_ac'].mean():.3f} ± {subset['tn_ac'].std():.3f}")
        print(f"  RW Model (C8) - Mean AC: {subset['rw_ac'].mean():.3f} ± {subset['rw_ac'].std():.3f}")
    
    return df

def export_figure3(
    samples_dir: str,
    tn_model_path: str,
    rw_model_path: str,
    output_path: str,
    files_tn=None,
    files_rw=None,
    device: str = "cuda",
):
    tn_viz = GradCAMVisualizerC8(tn_model_path, num_classes=6, device=device)
    rw_viz = GradCAMVisualizerC8(rw_model_path, num_classes=9, device=device)

    all_files = sorted([f for f in os.listdir(samples_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    tn_all = [f for f in all_files if "tn_" in f.lower()]
    rw_all = [f for f in all_files if "rw_" in f.lower()]

    if files_tn is None:
        files_tn = tn_all[:3]
    if files_rw is None:
        files_rw = rw_all[:3]

    selected = [files_tn[:3], files_rw[:3]]
    row_names = ["TrashNet", "RealWaste"]

    fig, axes = plt.subplots(nrows=2, ncols=9, figsize=(18, 4), dpi=200)
    for r in range(2):
        for s in range(3):
            fname = selected[r][s] if s < len(selected[r]) else None
            for c in range(3):
                ax = axes[r, s * 3 + c]
                ax.axis("off")
                if fname is None:
                    continue
                img_path = os.path.join(samples_dir, fname)
                img = Image.open(img_path).convert("RGB")

                label = _parse_label_from_filename(fname)
                tn_target = tn_viz.class_to_index.get(label) if label else None
                rw_target = rw_viz.class_to_index.get(label) if label else None

                if c == 0:
                    ax.imshow(np.asarray(img))
                    if s == 0:
                        ax.set_ylabel(row_names[r], fontsize=10)
                    title = f"{label}" if label else fname
                    ax.set_title(title, fontsize=8)
                elif c == 1:
                    cam_tn, _, _, _ = tn_viz.generate_cam(img_path, target_class=tn_target)
                    _, overlay = _overlay_heatmap(img, cam_tn)
                    ax.imshow(overlay)
                    if r == 0 and s == 0:
                        ax.set_title("TN-trained", fontsize=8)
                else:
                    cam_rw, _, _, _ = rw_viz.generate_cam(img_path, target_class=rw_target)
                    _, overlay = _overlay_heatmap(img, cam_rw)
                    ax.imshow(overlay)
                    if r == 0 and s == 0:
                        ax.set_title("RW-trained", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations for ESCNN C8')
    parser.add_argument('--tn_model', default=DEFAULT_TN_MODEL, help='Path to TrashNet model (C8)')
    parser.add_argument('--rw_model', default=DEFAULT_RW_MODEL, help='Path to RealWaste model (C8)')
    parser.add_argument('--samples_dir', default=DEFAULT_SAMPLES_DIR, help='Directory with sample images')
    parser.add_argument('--json_path', default=DEFAULT_JSON_PATH, help='Path to Label Studio JSON')
    parser.add_argument('--masks_dir', default=DEFAULT_MASKS_DIR, help='Directory to save/load masks')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--max_images', type=int, default=0, help='Max number of images to process (0=all)')
    parser.add_argument('--mode', choices=['batch', 'figure'], default='batch', help='batch: export CSV; figure: export Figure3 heatmap')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')

    args = parser.parse_args()

    # Ensure directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.masks_dir, exist_ok=True)

    # Step 1: Generate masks from JSON if JSON exists
    if os.path.exists(args.json_path):
        parse_labelstudio_json(args.json_path, args.masks_dir, args.samples_dir)
    else:
        print(f"Info: JSON file not found at {args.json_path}. Assuming masks already exist in {args.masks_dir}")

    if args.mode == "batch":
        batch_compute_ac_statistics(
            samples_dir=args.samples_dir,
            masks_dir=args.masks_dir,
            tn_model_path=args.tn_model,
            rw_model_path=args.rw_model,
            output_csv=os.path.join(args.output_dir, 'ac_statistics_C8.csv'),
            max_images=args.max_images,
            device=args.device,
        )
    else:
        export_figure3(
            samples_dir=args.samples_dir,
            tn_model_path=args.tn_model,
            rw_model_path=args.rw_model,
            output_path=os.path.join(args.output_dir, "figure3_C8.png"),
            device=args.device,
        )

    print("\n✨ Done!")
